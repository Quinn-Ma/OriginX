"""Pure data conversion matching the official RoboCasa365 evaluation client.

This module consumes arrays and a frame reader; it does not download data or
silently choose a training split. See README.md for the dataset reader contract.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from PIL import Image

STATE_DIM = ACTION_DIM = 60
ACTION_LENGTH = 16
HISTORY_LENGTH = 4
HISTORY_INTERVAL = 2
CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)

# Each archive's meta/modality.json must agree before using these conversions.
STATE_SLICES = {
    "base_position": (0, 3),
    "base_rotation": (3, 7),
    "end_effector_position_relative": (7, 10),
    "end_effector_rotation_relative": (10, 14),
    "gripper_qpos": (14, 16),
}
ACTION_SLICES = {
    "base_motion": (0, 4),
    "control_mode": (4, 5),
    "end_effector_position": (5, 8),
    "end_effector_rotation": (8, 11),
    "gripper_close": (11, 12),
}


def validate_modality(modality: Mapping) -> None:
    """Fail closed if a LeRobot archive uses a different feature ordering."""
    for section, expected, source in (
        ("state", STATE_SLICES, "observation.state"),
        ("action", ACTION_SLICES, "action"),
    ):
        for name, bounds in expected.items():
            actual = modality[section][name]
            if (actual["start"], actual["end"]) != bounds or actual["original_key"] != source:
                raise ValueError(f"Unsupported {section}.{name} modality: {actual}")


def _finite_array(value, trailing_dim: int, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim < 1 or value.shape[-1] != trailing_dim:
        raise ValueError(f"{name} must end in dimension {trailing_dim}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Official evaluator convention, vectorized over leading dimensions.

    Quaternions are normalized and sign-canonicalized to w >= 0. The zero
    quaternion returns zero, matching the official evaluation implementation.
    """
    quaternion = _finite_array(quaternion, 4, "quaternion").astype(np.float64)
    flat = quaternion.reshape(-1, 4)
    result = np.zeros((len(flat), 3), dtype=np.float32)
    for index, q in enumerate(flat):
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            continue
        q = q / norm
        if q[3] < 0:
            q = -q
        sin_half = np.linalg.norm(q[:3])
        if sin_half < 1e-12:
            continue
        angle = 2.0 * np.arctan2(sin_half, np.clip(q[3], -1.0, 1.0))
        result[index] = q[:3] / sin_half * angle
    return result.reshape(quaternion.shape[:-1] + (3,))


def lerobot_state_to_policy(state: np.ndarray) -> np.ndarray:
    """Base-first quaternion 16D LeRobot state -> raw EE-first axis-angle 60D."""
    state = _finite_array(state, 16, "LeRobot state")
    raw = np.concatenate(
        [state[..., 7:10], quat_xyzw_to_axis_angle(state[..., 10:14]),
         state[..., 14:16], state[..., :3], quat_xyzw_to_axis_angle(state[..., 3:7])],
        axis=-1,
    )
    padded = np.zeros(state.shape[:-1] + (STATE_DIM,), dtype=np.float32)
    padded[..., :14] = raw
    return padded


def lerobot_action_to_policy(action: np.ndarray) -> np.ndarray:
    """Base4/mode1/EE6/gripper1 -> EE6/gripper1/base4/mode1, then zero-pad."""
    action = _finite_array(action, 12, "LeRobot action")
    padded = np.zeros(action.shape[:-1] + (ACTION_DIM,), dtype=np.float32)
    padded[..., :12] = np.concatenate([action[..., 5:12], action[..., :5]], axis=-1)
    return padded


def history_indices(timestep: int) -> tuple[int, ...]:
    if not isinstance(timestep, (int, np.integer)) or timestep < 0:
        raise ValueError("timestep must be a nonnegative integer")
    return tuple(max(0, timestep - offset) for offset in (6, 4, 2, 0))


def center_crop(image, crop_ratio: float = 0.95) -> Image.Image:
    """Exactly the center-crop-and-resize transform in the evaluation client."""
    if not 0 < crop_ratio <= 1:
        raise ValueError("crop_ratio must lie in (0, 1]")
    if not isinstance(image, Image.Image):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Frames must be RGB uint8 arrays with shape (H, W, 3)")
        image = Image.fromarray(image)
    if image.mode != "RGB":
        raise ValueError("Frames must be RGB")
    if crop_ratio >= 1:
        return image
    width, height = image.size
    crop_width, crop_height = max(1, int(width * crop_ratio)), max(1, int(height * crop_ratio))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), getattr(Image, "Resampling", Image).BILINEAR)


def build_messages(videos: Mapping[str, Sequence], instruction: str, crop_ratio=0.95) -> list[dict]:
    """Build the official three-video/no_cot prompt, without extra action tokens."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty text")
    processed = {}
    for key in CAMERA_KEYS:
        if len(videos[key]) != HISTORY_LENGTH:
            raise ValueError(f"{key} must contain exactly four history frames")
        processed[key] = [center_crop(frame, crop_ratio) for frame in videos[key]]
    return [
        {"role": "user", "content": [
            {"type": "text", "text": "Left camera: "},
            {"type": "video", "video": processed[CAMERA_KEYS[0]]},
            {"type": "text", "text": "\nRight camera: "},
            {"type": "video", "video": processed[CAMERA_KEYS[1]]},
            {"type": "text", "text": "\nWrist camera: "},
            {"type": "video", "video": processed[CAMERA_KEYS[2]]},
            {"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]


@dataclass
class EpisodeSample:
    messages: list[dict]
    state: np.ndarray  # (1, 4, 60), raw values
    action: np.ndarray  # (1, 16, 60), raw controller commands
    valid_steps: np.ndarray  # (1, 16), bool; padded tail is excluded from loss


def make_sample(
    states: np.ndarray,
    actions: np.ndarray,
    frame_reader: Callable[[str, tuple[int, ...]], Sequence],
    timestep: int,
    instruction: str,
    crop_ratio: float = 0.95,
) -> EpisodeSample:
    """Adapt one *episode-local* sample without loading the whole video.

    states/actions are aligned LeRobot columns of shape (T,16)/(T,12).
    frame_reader(camera_key, indices) must return RGB frames in requested order,
    including repeated boundary frames. It owns timestamp/path resolution.
    """
    states = _finite_array(states, 16, "LeRobot states")
    actions = _finite_array(actions, 12, "LeRobot actions")
    if states.ndim != 2 or actions.ndim != 2 or len(states) != len(actions):
        raise ValueError("Expected aligned episode arrays with shape (T,16) and (T,12)")
    indices = history_indices(timestep)
    if timestep >= len(states):
        raise ValueError("timestep is outside this episode")
    videos = {key: frame_reader(key, indices) for key in CAMERA_KEYS}
    valid_count = min(ACTION_LENGTH, len(actions) - timestep)
    future = lerobot_action_to_policy(actions[timestep:timestep + ACTION_LENGTH])
    if valid_count < ACTION_LENGTH:
        future = np.concatenate([future, np.repeat(future[-1:], ACTION_LENGTH - valid_count, axis=0)])
    return EpisodeSample(
        messages=build_messages(videos, instruction, crop_ratio),
        state=lerobot_state_to_policy(states[list(indices)])[None],
        action=future[None],
        valid_steps=(np.arange(ACTION_LENGTH) < valid_count)[None],
    )
