import ast
from pathlib import Path

import numpy as np
import pytest

from training_bridge.layout import (
    CAMERA_KEYS, ACTION_SLICES, STATE_SLICES, history_indices,
    lerobot_action_to_policy, lerobot_state_to_policy, make_sample,
    quat_xyzw_to_axis_angle, validate_modality,
)


def official_functions():
    path = Path(__file__).resolve().parents[2] / "reproduction/Xiaomi-Robotics-1/eval_robocasa365/entry.py"
    source = ast.parse(path.read_text())
    wanted = {"quat_xyzw_to_axis_angle", "observation_to_state", "sample_history"}
    selected = ast.Module(body=[node for node in source.body
                               if isinstance(node, ast.FunctionDef) and node.name in wanted], type_ignores=[])
    import collections
    from typing import Any
    scope = {"np": np, "collections": collections, "Any": Any}
    exec(compile(selected, str(path), "exec"), scope)
    return scope


def test_quaternion_matches_official_evaluator_random_sign_and_degeneracies():
    official = official_functions()["quat_xyzw_to_axis_angle"]
    random = np.random.default_rng(34).normal(size=(100, 4))
    special = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, -1],
                        [1, 0, 0, 0], [-1, 0, 0, 0], [1e-15, 0, 0, 1]])
    values = np.concatenate([random, special, -random])
    expected = np.stack([official(q) for q in values])
    np.testing.assert_array_equal(quat_xyzw_to_axis_angle(values), expected)


def test_state_matches_official_observation_layout_and_padding():
    state = np.random.default_rng(19).normal(size=(7, 16))
    official = official_functions()["observation_to_state"]
    expected = []
    for row in state:
        expected.append(official({
            "state.base_position": row[:3], "state.base_rotation": row[3:7],
            "state.end_effector_position_relative": row[7:10],
            "state.end_effector_rotation_relative": row[10:14],
            "state.gripper_qpos": row[14:16],
        }))
    output = lerobot_state_to_policy(state)
    np.testing.assert_array_equal(output[:, :14], expected)
    assert np.count_nonzero(output[:, 14:]) == 0


def test_action_order_matches_documented_controller_slices():
    output = lerobot_action_to_policy(np.arange(12, dtype=np.float32))
    np.testing.assert_array_equal(output[:3], [5, 6, 7])
    np.testing.assert_array_equal(output[3:6], [8, 9, 10])
    np.testing.assert_array_equal(output[6:7], [11])
    np.testing.assert_array_equal(output[7:11], [0, 1, 2, 3])
    np.testing.assert_array_equal(output[11:12], [4])
    assert np.count_nonzero(output[12:]) == 0


def test_history_and_episode_end_do_not_cross_episode():
    import collections
    official = official_functions()["sample_history"]
    for step in range(9):
        expected = official(collections.deque(np.arange(step + 1)), 4, 2)
        np.testing.assert_array_equal(history_indices(step), expected)
    states, actions = np.zeros((3, 16)), np.arange(36).reshape(3, 12)
    calls = []
    def reader(key, indices):
        calls.append((key, indices))
        return [np.zeros((32, 32, 3), dtype=np.uint8) for _ in indices]
    sample = make_sample(states, actions, reader, 2, "test task")
    assert calls == [(key, (0, 0, 0, 2)) for key in CAMERA_KEYS]
    assert sample.valid_steps.sum() == 1
    np.testing.assert_array_equal(sample.action[0], np.repeat(sample.action[0, :1], 16, axis=0))
    assert sample.state.shape == (1, 4, 60)


def test_schema_mismatch_and_invalid_frames_fail_closed():
    modality = {section: {key: {"start": lo, "end": hi, "original_key": source}
                         for key, (lo, hi) in fields.items()}
                for section, fields, source in [("state", STATE_SLICES, "observation.state"),
                                                 ("action", ACTION_SLICES, "action")]}
    validate_modality(modality)
    modality["action"]["base_motion"]["start"] = 1
    with pytest.raises(ValueError, match="Unsupported"):
        validate_modality(modality)
    with pytest.raises(ValueError, match="non-finite"):
        lerobot_state_to_policy(np.full(16, np.nan))
