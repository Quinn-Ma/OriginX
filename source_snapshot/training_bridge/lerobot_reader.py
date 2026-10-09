"""Read verified pretrain-human LeRobot v2 episode files and timestamped video.

Supported schema is detected from actual info.json and modality.json; shared
v3 files/video offsets are explicitly unsupported. No network or simulator use.
"""

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from string import Formatter

import numpy as np

try:
    from .layout import CAMERA_KEYS, make_sample, validate_modality
    from .provenance import dataset_file, verify_provenance, sha256_file
except ImportError:
    from layout import CAMERA_KEYS, make_sample, validate_modality
    from provenance import dataset_file, verify_provenance, sha256_file


def _jsonl_by_id(path, key):
    result = {}
    for line in Path(path).read_text().splitlines():
        item = json.loads(line)
        identifier = item[key]
        if not isinstance(identifier, int) or identifier < 0 or identifier in result:
            raise ValueError(f"Invalid or duplicate {key} in {path}")
        result[identifier] = item
    if not result:
        raise ValueError(f"Empty metadata file: {path}")
    return result


def _scalar_column(table, key):
    values = []
    for value in table[key]:
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(f"Expected scalar/one-element values in {key}")
        values.append(array.item())
    return np.asarray(values)


def nearest_video_indices(frame_timestamps, requested_timestamps, tolerance):
    """Nearest actual video PTS, preserving sparse order and duplicate requests."""
    starts = np.asarray(frame_timestamps, dtype=np.float64)
    requested = np.asarray(requested_timestamps, dtype=np.float64)
    if starts.ndim != 1 or not len(starts) or not np.isfinite(starts).all() or (np.diff(starts) <= 0).any():
        raise ValueError("Video frame timestamps must be finite and strictly increasing")
    if not np.isfinite(requested).all():
        raise ValueError("Requested timestamps must be finite")
    right = np.searchsorted(starts, requested, side="left").clip(0, len(starts) - 1)
    left = np.maximum(right - 1, 0)
    # Match numpy.argmin tie behavior: the earlier frame wins.
    choose_right = np.abs(starts[right] - requested) < np.abs(starts[left] - requested)
    indices = np.where(choose_right, right, left)
    if (np.abs(starts[indices] - requested) > tolerance).any():
        raise ValueError("Video timestamps do not align with requested observation rows")
    return indices.astype(np.int64)


@dataclass
class EpisodeArrays:
    states: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    instructions: list[str]
    parquet_sha256: str


class LeRobotPretrainReader:
    """Bounded CPU reader; callers explicitly select episodes before sampling."""

    def __init__(self, dataset_root, provenance, registry_path, box_links_path, *,
                 episode_ids, episode_cache_size=2, video_cache_size=6, video_factory=None):
        self.root = Path(dataset_root).resolve()
        self.provenance = verify_provenance(self.root, provenance, registry_path, box_links_path)
        self.info = json.loads(dataset_file(self.root, "meta/info.json").read_text())
        self.modality = json.loads(dataset_file(self.root, "meta/modality.json").read_text())
        validate_modality(self.modality)
        self.episodes = _jsonl_by_id(dataset_file(self.root, "meta/episodes.jsonl"), "episode_index")
        self.tasks = _jsonl_by_id(dataset_file(self.root, "meta/tasks.jsonl"), "task_index")
        if not str(self.info.get("codebase_version", "")).startswith("v2."):
            raise ValueError("Only verified LeRobot v2 episode files are supported; v3 offsets are not implemented")
        if self.info.get("robot_type") != "PandaOmron":
            raise ValueError("Unexpected robot embodiment")
        if self.info["total_episodes"] != len(self.episodes):
            raise ValueError("Episode count differs from info.json")
        if sum(ep["length"] for ep in self.episodes.values()) != self.info["total_frames"]:
            raise ValueError("Episode lengths do not sum to info.json total_frames")
        self.chunks_size = self.info["chunks_size"]
        if not isinstance(self.chunks_size, int) or self.chunks_size <= 0:
            raise ValueError("Invalid chunks_size")
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Invalid declared video FPS")
        for field in ("data_path", "video_path"):
            fields = {name for _, name, _, _ in Formatter().parse(self.info[field]) if name is not None}
            if "episode_index" not in fields or not fields <= {"episode_index", "episode_chunk", "video_key"}:
                raise ValueError(f"Unsupported shared/unknown path pattern: {field}")
        self.annotation_key = self.modality["annotation"]["human.task_description"]["original_key"]
        features = self.info["features"]
        if features["observation.state"]["shape"] != [16] or features["action"]["shape"] != [12]:
            raise ValueError("Unexpected state/action feature shape")
        self.video_keys, self.video_shapes = {}, {}
        for camera in CAMERA_KEYS:
            video_key = self.modality["video"][camera.removeprefix("video.")]["original_key"]
            feature = features[video_key]
            if feature["dtype"] != "video" or feature["names"] != ["height", "width", "channel"]:
                raise ValueError(f"Unsupported video feature schema: {video_key}")
            shape = tuple(feature["shape"])
            if len(shape) != 3 or shape[2] != 3:
                raise ValueError("Expected three-channel RGB video")
            declared_fps = [details["video.fps"] for name in ("video_info", "info")
                            if (details := feature.get(name)) and "video.fps" in details]
            if not declared_fps or not all(np.isclose(float(fps), self.fps) for fps in declared_fps):
                raise ValueError("Per-video FPS disagrees with dataset FPS")
            self.video_keys[camera], self.video_shapes[camera] = video_key, shape
        ids = list(episode_ids)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, int) or i not in self.episodes for i in ids):
            raise ValueError("Select explicit unique episode IDs present in this archive")
        self.episode_ids = tuple(ids)
        if episode_cache_size < 1 or video_cache_size < 1:
            raise ValueError("Cache capacities must be positive")
        self._episode_cache_size, self._video_cache_size = episode_cache_size, video_cache_size
        self._episodes_cache, self._videos_cache = OrderedDict(), OrderedDict()
        self._video_factory = video_factory

    def _episode_path(self, field, episode_id, video_key=""):
        relative = self.info[field].format(episode_index=episode_id,
                                          episode_chunk=episode_id // self.chunks_size,
                                          video_key=video_key)
        return dataset_file(self.root, relative)

    def _check_episode(self, episode_id):
        if episode_id not in self.episode_ids:
            raise ValueError("Episode was not selected in this reader's explicit episode manifest")

    @staticmethod
    def _cache(cache, key, value, capacity):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > capacity:
            cache.popitem(last=False)
        return value

    def load_episode(self, episode_id):
        self._check_episode(episode_id)
        if episode_id in self._episodes_cache:
            self._episodes_cache.move_to_end(episode_id)
            return self._episodes_cache[episode_id]
        import pandas as pd
        path = self._episode_path("data_path", episode_id)
        columns = ["observation.state", "action", "timestamp", "frame_index", "episode_index",
                   self.annotation_key, "annotation.human.task_name"]
        table = pd.read_parquet(path, columns=columns)
        expected_length = self.episodes[episode_id]["length"]
        if len(table) != expected_length or expected_length <= 0:
            raise ValueError("Parquet row count disagrees with episode metadata")
        if not np.array_equal(_scalar_column(table, "frame_index"), np.arange(expected_length)):
            raise ValueError("Episode frame_index is not contiguous and zero-based")
        if not np.array_equal(_scalar_column(table, "episode_index"), np.full(expected_length, episode_id)):
            raise ValueError("Parquet contains a different episode")
        timestamps = _scalar_column(table, "timestamp").astype(np.float64)
        expected_times = np.arange(expected_length, dtype=np.float64) / self.fps
        if not np.isfinite(timestamps).all() or not np.allclose(timestamps, expected_times, rtol=0, atol=1e-4):
            raise ValueError("Observation timestamps do not match declared episode-local sampling rate")
        # Keep the recorded quaternion precision until layout.py converts it
        # to axis-angle; the official evaluator also computes rotations in FP64.
        states = np.stack(table["observation.state"]).astype(np.float64)
        actions = np.stack(table["action"]).astype(np.float32)
        if states.shape != (expected_length, 16) or actions.shape != (expected_length, 12):
            raise ValueError("Parquet state/action array shape disagrees with metadata")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite state/action values")
        descriptions = _scalar_column(table, self.annotation_key)
        if not np.issubdtype(descriptions.dtype, np.integer):
            raise ValueError("Task-description IDs must be integers")
        instructions = []
        for index in descriptions:
            if index not in self.tasks or not isinstance(self.tasks[index]["task"], str):
                raise ValueError("Missing task-description mapping")
            instruction = self.tasks[index]["task"]
            if instruction not in self.episodes[episode_id]["tasks"]:
                raise ValueError("Per-frame instruction disagrees with episode metadata")
            instructions.append(instruction)
        if len(set(instructions)) != 1:
            raise ValueError("Expected a single global task instruction within an episode")
        task_names = _scalar_column(table, "annotation.human.task_name")
        if not np.issubdtype(task_names.dtype, np.integer):
            raise ValueError("Task-name IDs must be integers")
        if any(index not in self.tasks or self.tasks[index]["task"] != self.provenance["task"] for index in task_names):
            raise ValueError("Parquet task-name identity disagrees with provenance")
        data = EpisodeArrays(states, actions, timestamps, instructions, sha256_file(path))
        return self._cache(self._episodes_cache, episode_id, data, self._episode_cache_size)

    def frames(self, episode_id, camera_key, row_indices):
        self._check_episode(episode_id)
        episode = self.load_episode(episode_id)
        indices = np.asarray(row_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or (indices < 0).any() or (indices >= len(episode.states)).any():
            raise ValueError("Video requests must be valid episode-local row indices")
        if camera_key not in self.video_keys:
            raise ValueError("Unsupported camera")
        key = (episode_id, camera_key)
        if key not in self._videos_cache:
            factory = self._video_factory
            if factory is None:
                from decord import VideoReader
                factory = lambda path: VideoReader(str(path), num_threads=1)
            video = factory(self._episode_path("video_path", episode_id, self.video_keys[camera_key]))
            if len(video) != len(episode.states):
                raise ValueError("Decoded video frame count disagrees with the episode")
            if not np.isclose(video.get_avg_fps(), self.fps, rtol=1e-3, atol=1e-3):
                raise ValueError("Decoded video FPS disagrees with metadata")
            pts = np.asarray(video.get_frame_timestamp(list(range(len(video)))))[:, 0]
            # Validate the full episode's temporal alignment, not just one sample.
            aligned = nearest_video_indices(pts, episode.timestamps, 0.5 / self.fps + 1e-4)
            if not np.array_equal(aligned, np.arange(len(episode.states))):
                raise ValueError("Video frames are not aligned one-to-one with parquet observations")
            self._cache(self._videos_cache, key, (video, pts), self._video_cache_size)
        self._videos_cache.move_to_end(key)
        video, pts = self._videos_cache[key]
        decoded_indices = nearest_video_indices(pts, episode.timestamps[indices], 0.5 / self.fps + 1e-4)
        result = video.get_batch(decoded_indices.tolist())
        result = result.asnumpy() if hasattr(result, "asnumpy") else np.asarray(result)
        expected_shape = (len(indices),) + self.video_shapes[camera_key]
        if result.shape != expected_shape or result.dtype != np.uint8:
            raise ValueError("Decoded frames do not match the declared RGB shape/dtype")
        return list(result)

    def sample(self, episode_id, timestep):
        episode = self.load_episode(episode_id)
        if not isinstance(timestep, int) or not 0 <= timestep < len(episode.states):
            raise ValueError("Timestep is outside the selected episode")
        return make_sample(episode.states, episode.actions,
                           lambda camera, indices: self.frames(episode_id, camera, indices),
                           timestep, episode.instructions[timestep])

    def inventory(self):
        return {"schema": self.info["codebase_version"], "task": self.provenance["task"],
                "archive_episodes": len(self.episodes), "archive_frames": self.info["total_frames"],
                "selected_episode_ids": list(self.episode_ids), "fps": self.fps,
                "video_keys": self.video_keys, "provenance": self.provenance,
                "note": "Archive episode count may exceed nominal 100; selection is explicit. next.done is not a success label."}

    def clear_caches(self):
        """Release decoder references before cycling to other tasks in a mixture."""
        self._videos_cache.clear()
        self._episodes_cache.clear()
