"""CPU-only boundary for the official RoboCasa GR00T policy.

This module neither imports GR00T nor loads a model. A real policy supplied by an
admitted service must be the pinned Gr00tPolicy with its official transforms.
"""
from collections.abc import Mapping
import math
import threading
import numpy as np

CAMERAS = ('video.robot0_agentview_left', 'video.robot0_agentview_right',
           'video.robot0_eye_in_hand')
STATE_DIMS = {
    'state.end_effector_position_relative': 3,
    'state.end_effector_rotation_relative': 4,
    'state.gripper_qpos': 2,
    'state.base_position': 3,
    'state.base_rotation': 4,
}
ACTION_DIMS = {
    'action.end_effector_position': 3,
    'action.end_effector_rotation': 3,
    'action.gripper_close': 1,
    'action.base_motion': 4,
    'action.control_mode': 1,
}
LANGUAGE = 'annotation.human.task_description'
CHUNK = 16
PROTOCOL = 'originx-gr00t-rng-v1'


def _numeric(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f'{name}: expected floating {shape}, got {array.shape}/{array.dtype}')
    if not np.isfinite(array).all():
        raise ValueError(f'{name}: nonfinite values')
    return np.ascontiguousarray(array, dtype=np.float32).copy()


def build_observation(native, instruction):
    """Preserve named raw state and corrected RGB; add one temporal dimension.

    Native RoboCasaGymEnv already converts state to float32 and flips rendered
    RGB. Neither operation, normalization, crop, nor resize is repeated here.
    Official GR00T transforms perform the latter operations inside get_action.
    """
    if not isinstance(native, Mapping):
        raise ValueError('A current named native observation is mandatory')
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4096:
        raise ValueError('Expected one nonempty bounded task instruction')
    result = {}
    for name in CAMERAS:
        image = np.asarray(native[name])
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError(f'{name}: expected RGB uint8 256x256')
        result[name] = np.ascontiguousarray(image)[None].copy()
    for name, dim in STATE_DIMS.items():
        result[name] = _numeric(native[name], (dim,), name)[None]
    result[LANGUAGE] = np.asarray([instruction])
    return result


def flatten_actions(decoded):
    """Map the official unnormalized five groups to the entry's (16,12) order.

    No clipping, thresholding, new normalization, or rotation conversion is
    added: existing convert_action and Gym controller perform final routing.
    """
    if not isinstance(decoded, Mapping) or set(decoded) != set(ACTION_DIMS):
        raise ValueError('Expected exactly the five decoded GR00T action groups')
    arrays = [_numeric(decoded[k], (CHUNK, d), k) for k, d in ACTION_DIMS.items()]
    return np.ascontiguousarray(np.concatenate(arrays, axis=-1), dtype=np.float32)


class PolicyAdapter:
    """Wrap an already admitted official policy; no policy factory side effect."""
    def __init__(self, policy):
        if not callable(getattr(policy, 'get_action', None)):
            raise TypeError('An official policy or explicit test double is required')
        self.policy = policy

    def infer_native(self, native, instruction):
        return flatten_actions(self.policy.get_action(build_observation(native, instruction)))


class LatestObservation:
    """Same-thread, read-only capture point for a Gym reset/step wrapper.

    The new executor calls capture immediately after each native observation.
    Only public policy fields are retained; simulator state never enters this
    holder or the inference request. This is not a simulator reset wrapper.
    """
    def __init__(self):
        self._thread = threading.get_ident()
        self._native = None
        self.steps = None

    def capture(self, native, *, steps):
        if threading.get_ident() != self._thread:
            raise RuntimeError('Observation capture changed execution thread')
        if type(steps) is not int or steps < 0 or (self.steps is not None and steps != self.steps+1):
            raise ValueError('Native observations must advance by one action step')
        if self.steps is None and steps != 0:
            raise ValueError('First capture must be the actual native reset observation')
        checked = build_observation(native, 'schema check')
        self._native = {k: v[0].copy() for k, v in checked.items() if k != LANGUAGE}
        self.steps = steps

    def current(self):
        if threading.get_ident() != self._thread or self._native is None:
            raise RuntimeError('No current same-thread native observation')
        return {k: v.copy() for k, v in self._native.items()}


class EntryBridge:
    """Adapter for the evaluator's infer(state_history, images, instruction).

    The 14D axis-angle history is checked against the tapped observation using
    the pinned entry.observation_to_state, then discarded. Reconstructing raw
    quaternions from this lossy representation is intentionally prohibited.
    The official one-frame GR00T context is preserved even when the caller
    maintains more frames for another policy.
    """
    def __init__(self, policy, latest, pack_state):
        self.adapter = PolicyAdapter(policy)
        self.latest = latest
        self.pack_state = pack_state
        self.calls = 0
        self.last_step = None

    def infer(self, state_history, image_history, instruction):
        native = self.latest.current()
        states = np.asarray(state_history)
        if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] != 14:
            raise ValueError('Evaluator state history must be T x 14; no padded XR1 state')
        expected = np.asarray(self.pack_state(native), dtype=np.float32)
        if expected.shape != (14,) or not np.array_equal(states[-1], expected):
            raise ValueError('Tapped native state does not match the current evaluator state')
        if set(image_history) != set(CAMERAS):
            raise ValueError('Exactly the three native camera streams are required')
        for key in CAMERAS:
            frames = np.asarray(image_history[key])
            if frames.ndim != 4 or frames.shape[0] < 1 or not np.array_equal(frames[-1], native[key]):
                raise ValueError('Tapped RGB does not match the current evaluator frame')
        expected_step = 0 if self.last_step is None else self.last_step + CHUNK
        if self.latest.steps != expected_step:
            raise ValueError('Expected one query every 16 actually executed actions')
        result = self.adapter.infer_native(native, instruction)
        self.calls += 1
        self.last_step = self.latest.steps
        return result


def intervention_step(horizon):
    if type(horizon) is not int or horizon <= 16:
        raise ValueError('Invalid official horizon')
    step = CHUNK * math.ceil(horizon / (2*CHUNK))
    if step >= horizon:
        raise ValueError('Intervention must leave budget')
    return step
