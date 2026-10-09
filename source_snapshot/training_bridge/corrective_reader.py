"""Allowed-observation adapters for verified successful simulation recordings.

Policy targets are already EE-first controller actions in collector recordings;
they must not pass through the base-first LeRobot action permutation.
"""

from bisect import bisect_right
from collections import OrderedDict
import copy
import json
from pathlib import Path

import numpy as np

from recovery.collector import CAMERAS, INSTRUCTION, STATE_SHAPES, allowed_observation

try:
    from .layout import (ACTION_DIM, ACTION_LENGTH, CAMERA_KEYS, EpisodeSample,
                         build_messages, lerobot_state_to_policy)
    from .provenance import dataset_file, sha256_file
except ImportError:
    from layout import (ACTION_DIM, ACTION_LENGTH, CAMERA_KEYS, EpisodeSample,
                        build_messages, lerobot_state_to_policy)
    from provenance import dataset_file, sha256_file


class CorrectiveBranchReader:
    """Read only success-validated continuation actions, with real prefix context.

    The recovery producer owns replay/success validation. This reader invokes
    its agreed public verifier before any sample becomes available. Missing
    producer implementation, smoke data, failed branches, or bad evidence are
    errors, never a reason to accept an unverified alternative.
    """

    def __init__(self, directory):
        from recovery.continuation import verify_successful_continuation

        verified = verify_successful_continuation(directory)
        self.root = Path(verified["root"]).resolve()
        self.parent_root = Path(verified["parent_root"]).resolve()
        metadata = verified["metadata"]
        self.task = metadata["spec"]["task"]
        self.registry_sha256 = metadata["spec"]["registry_sha256"]
        self.task_source_sha256 = metadata["spec"]["task_source_sha256"]
        self.branch_step = metadata["branch_step"]
        self.actions = np.array(verified["actions"], dtype=np.float32, copy=True)
        if (type(self.branch_step) is not int or self.branch_step < 0 or
                self.actions.ndim != 2 or self.actions.shape[1] != 12 or
                not len(self.actions) or not np.isfinite(self.actions).all()):
            raise ValueError("Verified continuation returned an invalid sample range")
        self.actions.setflags(write=False)
        self.records, self.parent_records = verified["records"], verified["parent_records"]
        if len(self.records) != len(self.actions) + 1 or self.branch_step >= len(self.parent_records):
            raise ValueError("Verified continuation returned incomplete observation indices")
        self.manifest_sha256 = sha256_file(self.root / "manifest.json")
        self.parent_manifest_sha256 = sha256_file(self.parent_root / "manifest.json")
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.actions)

    def inventory(self):
        return {"source": "verified_successful_continuation", "root": str(self.root), "task": self.task,
                "steps": len(self), "branch_step": self.branch_step,
                "manifest_sha256": self.manifest_sha256, "parent_root": str(self.parent_root),
                "parent_manifest_sha256": self.parent_manifest_sha256,
                "registry_sha256": self.registry_sha256, "task_source_sha256": self.task_source_sha256}

    def clear_caches(self):
        self._cache.clear()

    def sample_identity(self, timestep):
        return {"sample_index": timestep, "continuation_step": timestep,
                "global_step": self.branch_step + timestep}

    def _observation(self, global_step):
        if global_step < self.branch_step:
            root, records, step = self.parent_root, self.parent_records, global_step
        else:
            root, records, step = self.root, self.records, global_step - self.branch_step
        record = records[step]
        path = dataset_file(root, record["policy_observation"])
        if not path.is_relative_to(root / "policy_observations"):
            raise ValueError("Corrective observation is outside the permitted input directory")
        # Rehash even cached frames, so on-disk mutation cannot silently survive
        # because a previous sample happened to populate the decode cache.
        if sha256_file(path) != record["policy_observation_sha256"]:
            raise ValueError("Corrective policy observation changed")
        key = str(path)
        if key not in self._cache:
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(CAMERAS) | set(STATE_SHAPES) | {INSTRUCTION}:
                    raise ValueError("Corrective observation contains unsupported fields")
                observation = {name: saved[name] for name in saved.files}
                observation[INSTRUCTION] = observation[INSTRUCTION].item()
            self._cache[key] = allowed_observation(observation)
        self._cache.move_to_end(key)
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return self._cache[key]

    def sample(self, timestep):
        if type(timestep) is not int or not 0 <= timestep < len(self):
            raise ValueError("Corrective timestep is outside the successful continuation")
        if (sha256_file(self.root / "manifest.json") != self.manifest_sha256 or
                sha256_file(self.parent_root / "manifest.json") != self.parent_manifest_sha256):
            raise ValueError("Corrective or parent manifest changed after verification")
        global_step = self.branch_step + timestep
        history = [self._observation(max(0, global_step - offset)) for offset in (6, 4, 2, 0)]
        return make_corrective_sample(history, self.actions[timestep:timestep + ACTION_LENGTH])


class LiveInterventionReader:
    """Success-filtered self-imitation spans from one continuous live trajectory.

    This makes no failed-parent, exact-replay, or counterfactual recovery claim.
    The producer verifies the real seed intervention, final official success,
    and the eligible progress/success spans before exposing this interface.
    """

    def __init__(self, directory):
        from recovery.live_intervention import verify_live_intervention_success

        verified = verify_live_intervention_success(directory)
        self.root = Path(verified["root"]).resolve()
        metadata = verified["metadata"]
        if metadata.get("source_kind") != "live_intervention_success_v1":
            raise ValueError("Expected the verified live-intervention source kind")
        self.task = metadata["spec"]["task"]
        self.actions = np.array(verified["actions"], dtype=np.float32, copy=True)
        self.actions.setflags(write=False)
        self.records = verified["records"]
        self.spans = copy.deepcopy(verified["spans"])
        if (self.actions.ndim != 2 or self.actions.shape[1] != 12 or not len(self.actions) or
                not np.isfinite(self.actions).all() or len(self.records) != len(self.actions) + 1 or not self.spans):
            raise ValueError("Verified live recording returned incomplete data or no eligible spans")
        self.ends, previous_stop, count = [], 0, 0
        for span in self.spans:
            start, stop = span["start_step"], span["stop_step"]
            if type(start) is not int or type(stop) is not int or not previous_stop <= start < stop <= len(self.actions):
                raise ValueError("Live intervention action spans must be ordered, disjoint and nonempty")
            count += stop - start
            self.ends.append(count)
            previous_stop = stop
        self.manifest_sha256 = sha256_file(self.root / "manifest.json")
        self.registry_sha256 = metadata["spec"]["registry_sha256"]
        self.task_source_sha256 = metadata["spec"]["task_source_sha256"]
        self._cache = OrderedDict()

    def __len__(self):
        return self.ends[-1]

    def _location(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise ValueError("Sample index is outside the verified live intervention spans")
        span_index = bisect_right(self.ends, index)
        previous = self.ends[span_index - 1] if span_index else 0
        return span_index, self.spans[span_index]["start_step"] + index - previous

    def sample_identity(self, index):
        span_index, timestep = self._location(index)
        return {"sample_index": index, "global_step": timestep, "span_index": span_index,
                "intervention_index": self.spans[span_index]["intervention_index"]}

    def inventory(self):
        return {"source": "verified_live_intervention_success", "source_kind": "live_intervention_success_v1",
                "root": str(self.root), "task": self.task, "steps": len(self), "trajectory_steps": len(self.actions),
                "manifest_sha256": self.manifest_sha256, "registry_sha256": self.registry_sha256,
                "task_source_sha256": self.task_source_sha256,
                "spans": [{key: span[key] for key in ("start_step", "stop_step", "intervention_index")}
                          for span in self.spans]}

    def clear_caches(self):
        self._cache.clear()

    def _observation(self, timestep):
        record = self.records[timestep]
        path = dataset_file(self.root, record["policy_observation"])
        if not path.is_relative_to(self.root / "policy_observations"):
            raise ValueError("Live observation is outside the permitted input directory")
        if sha256_file(path) != record["policy_observation_sha256"]:
            raise ValueError("Live policy observation changed")
        if timestep not in self._cache:
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(CAMERAS) | set(STATE_SHAPES) | {INSTRUCTION}:
                    raise ValueError("Live observation contains unsupported fields")
                observation = {key: saved[key] for key in saved.files}
                observation[INSTRUCTION] = observation[INSTRUCTION].item()
            self._cache[timestep] = allowed_observation(observation)
        self._cache.move_to_end(timestep)
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return self._cache[timestep]

    def sample(self, index):
        span_index, timestep = self._location(index)
        if sha256_file(self.root / "manifest.json") != self.manifest_sha256:
            raise ValueError("Live manifest changed after success verification")
        history = [self._observation(max(0, timestep - offset)) for offset in (6, 4, 2, 0)]
        stop = min(timestep + ACTION_LENGTH, self.spans[span_index]["stop_step"])
        return make_corrective_sample(history, self.actions[timestep:stop])


def open_corrective_reader(directory):
    """Dispatch explicit source kinds; never fall back after failed verification."""
    metadata = json.loads((Path(directory) / "manifest.json").read_text())
    if metadata.get("source_kind") == "live_intervention_success_v1":
        return LiveInterventionReader(directory)
    if metadata.get("recording_kind") == "continuation" and "source_kind" not in metadata:
        return CorrectiveBranchReader(directory)
    raise ValueError("Unsupported or unverified corrective recording source")


def make_corrective_sample(history, future_actions):
    """Convert exactly four chronological, permission-filtered history frames.

    The caller selects [-6,-4,-2,0] from parent plus continuation observations.
    Only future continuation actions are accepted, with episode-tail masking.
    This pure adapter is not the source-provenance gate.
    """
    if tuple(CAMERAS) != tuple(CAMERA_KEYS) or len(history) != 4:
        raise ValueError("Expected the official three-camera, four-frame history")
    expected_keys = set(CAMERAS) | set(STATE_SHAPES) | {INSTRUCTION}
    if any(set(observation) != expected_keys for observation in history):
        raise ValueError("Corrective history must contain only the allowed observation fields")
    history = [allowed_observation(observation) for observation in history]
    if len({observation[INSTRUCTION] for observation in history}) != 1:
        raise ValueError("Corrective history crosses instruction or episode boundaries")
    base_first_state = np.stack([
        np.concatenate([observation["state.base_position"], observation["state.base_rotation"],
                        observation["state.end_effector_position_relative"],
                        observation["state.end_effector_rotation_relative"], observation["state.gripper_qpos"]])
        for observation in history
    ])
    future_actions = np.asarray(future_actions)
    if (future_actions.ndim != 2 or future_actions.shape[1] != 12 or
            not 1 <= len(future_actions) <= ACTION_LENGTH or not np.isfinite(future_actions).all()):
        raise ValueError("Expected one to sixteen finite EE-first corrective controller actions")
    action = np.zeros((ACTION_LENGTH, ACTION_DIM), dtype=np.float32)
    count = len(future_actions)
    action[:count, :12] = future_actions
    action[count:] = action[count - 1]
    return EpisodeSample(
        messages=build_messages({key: [observation[key] for observation in history] for key in CAMERAS},
                                history[-1][INSTRUCTION]),
        state=lerobot_state_to_policy(base_first_state)[None], action=action[None],
        valid_steps=(np.arange(ACTION_LENGTH) < count)[None],
    )
