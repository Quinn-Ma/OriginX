"""Unpromoted candidate: omit RGB renders never sampled by fixed 16/4/2 policy.

No observable is disabled and no simulation step, observation sampling time,
filter, or RNG call is omitted. Use QueryCheckedClient with DecimatedGym.
Requires a real-scene, per-step state/RNG and query-pixel parity gate first.
"""
from collections import deque
from functools import wraps

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


def needed(step):
    if type(step) is not int or step < 0:
        raise ValueError("Expected nonnegative episode control-step index")
    return step == 0 or step % 16 in (0, 10, 12, 14)


class DecimatedEnv:
    def __init__(self, env, *, enabled=True):
        self.env = env
        self.enabled = enabled
        self.step_index = 0
        self._active_step = None
        self._hooks = []
        self.counts = {"sensor_calls": 0, "rendered": 0, "cached": 0}
        self.origins = {c: deque(maxlen=7) for c in CAMERAS}
        self._latest = {c: 0 for c in CAMERAS}

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def core(self):
        return self.env.unwrapped.env

    def _restore(self):
        for obs, original, wrapped in self._hooks:
            if obs._sensor is not wrapped:
                raise RuntimeError("Camera sensor changed while candidate was active")
            obs._sensor = original
        self._hooks.clear()

    def _install(self):
        import numpy as np
        from robosuite.utils.observables import NO_CORRUPTION, NO_FILTER, NO_DELAY

        core = self.core
        if core.viewer_get_obs or core.has_renderer or not core.use_camera_obs:
            raise ValueError("Candidate requires ordinary offscreen RGB observables")
        if tuple(core.camera_names) != CAMERAS:
            raise ValueError("Unexpected camera set/order")
        if any(core.camera_depths) or any(x is not None for x in core.camera_segmentations):
            raise ValueError("Depth/segmentation camera dependencies are unsupported")
        names = {c + "_image" for c in CAMERAS}
        if {n for n, o in core._observables.items() if o.modality == "image"} != names:
            raise ValueError("Unexpected image observable/dependency")
        for camera in CAMERAS:
            obs = core._observables[camera + "_image"]
            if not obs.is_active() or not obs.is_enabled():
                raise ValueError("Expected enabled, active image observable")
            if (obs._corrupter is not NO_CORRUPTION or obs._filter is not NO_FILTER
                    or obs._delayer is not NO_DELAY or obs._sensor.__name__ != "camera_rgb"):
                raise ValueError("Unsupported sensor/filter/corruption/delay effects")
            if obs._sampling_timestep != 1.0 / core.control_freq:
                raise ValueError("Unexpected camera sampling frequency")
            original = obs._sensor
            cached = [np.array(obs.obs, copy=True)]
            if cached[0].shape != (256, 256, 3) or cached[0].dtype != np.uint8:
                raise ValueError("Unexpected camera size/type")

            def build(original, cached, camera):
                @wraps(original)
                def sensor(cache):
                    step = self._active_step
                    # Reset/setup/any call outside a known evaluator step must render.
                    real = step is None or not self.enabled or needed(step)
                    self.counts["sensor_calls"] += 1
                    if real:
                        cached[0] = original(cache)
                        self.counts["rendered"] += 1
                        if step is not None:
                            self._latest[camera] = step
                    else:
                        self.counts["cached"] += 1
                    return cached[0]
                return sensor

            wrapped = build(original, cached, camera)
            # Do not call set_sensor: its validity check performs an extra render.
            # The original sensor's modality and all Observable state are preserved.
            obs._sensor = wrapped
            self._hooks.append((obs, original, wrapped))

    def reset(self, *args, **kwargs):
        self._restore()
        self._active_step = None
        result = self.env.reset(*args, **kwargs)
        self.step_index = 0
        for camera in CAMERAS:
            self.origins[camera].clear()
            self.origins[camera].append(0)
            self._latest[camera] = 0
        self._install()
        return result

    def step(self, *args, **kwargs):
        if not self._hooks:
            raise RuntimeError("reset() is required before stepping")
        self._active_step = self.step_index + 1
        try:
            result = self.env.step(*args, **kwargs)
        finally:
            self._active_step = None
        self.step_index += 1
        for camera in CAMERAS:
            self.origins[camera].append(self._latest[camera])
        return result

    def assert_query_ready(self):
        if self.step_index % 16:
            raise RuntimeError("Policy queried outside the fixed replan schedule")
        expected = [max(0, self.step_index - offset) for offset in (6, 4, 2, 0)]
        for camera in CAMERAS:
            history = list(self.origins[camera])
            actual = [history[max(0, len(history) - 1 - offset)] for offset in (6, 4, 2, 0)]
            if actual != expected:
                raise RuntimeError(f"Stale image would reach policy: {camera}: {actual} != {expected}")

    def close(self):
        self._restore()
        return self.env.close()


class DecimatedGym:
    def __init__(self, gym, cfg):
        if (cfg.replan_steps, cfg.obs_history, cfg.obs_interval) != (16, 4, 2):
            raise ValueError("Only the verified fixed 16/4/2 schedule is supported")
        if cfg.save_videos or cfg.save_failure_videos:
            raise ValueError("Video saving would consume otherwise unused images")
        self.gym = gym
        self.current = None

    def make(self, *args, **kwargs):
        self.current = DecimatedEnv(self.gym.make(*args, **kwargs))
        return self.current


class QueryCheckedClient:
    def __init__(self, client, gym):
        self.client, self.gym = client, gym

    def __getattr__(self, name):
        return getattr(self.client, name)

    def infer(self, *args, **kwargs):
        if self.gym.current is None:
            raise RuntimeError("No current candidate environment")
        self.gym.current.assert_query_ready()
        return self.client.infer(*args, **kwargs)
