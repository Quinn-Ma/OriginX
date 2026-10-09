"""One execution thread owns all model access and global random state.

Socket threads pass bytes only. Request tensors are deserialized, validated,
converted exactly as in official deploy/server.py, forwarded, and serialized
inside this worker. No batching, compilation, logits optimization, or cache
continuation is introduced.
"""
from __future__ import annotations

from concurrent.futures import Future
import hashlib
import json
import pickle
import queue
import random
import threading

import numpy as np

PROTOCOL = "robocasa-multiplex-rng-v1"
CONTROL = "_multiplex_control"
ALLOWED = frozenset(("task_id", "state", "action_mask", "input_ids", "attention_mask",
                     "pixel_values_videos", "video_grid_thw", "pixel_values", "image_grid_thw"))
REQUIRED = frozenset(("task_id", "state", "action_mask", "input_ids", "attention_mask",
                      "pixel_values_videos", "video_grid_thw"))


class RngBank:
    def __init__(self, torch):
        self.torch = torch
        self.cuda = torch.cuda.is_available()

    def capture(self):
        t = self.torch
        return (random.getstate(), np.random.get_state(), t.get_rng_state().clone(),
                [s.clone() for s in t.cuda.get_rng_state_all()] if self.cuda else [])

    def restore(self, state):
        t = self.torch
        random.setstate(state[0]); np.random.set_state(state[1]); t.set_rng_state(state[2])
        if self.cuda:
            t.cuda.set_rng_state_all(state[3])

    def seed(self, seed):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Seed must be an integer in [0,2**63)")
        random.seed(seed); np.random.seed(seed % 2**32)
        self.torch.manual_seed(seed)
        if self.cuda:
            self.torch.cuda.manual_seed_all(seed)

    @staticmethod
    def hashes(state):
        def h(value):
            return hashlib.sha256(value).hexdigest()
        return {"python_rng_sha256": h(pickle.dumps(state[0], protocol=5)),
                "numpy_rng_sha256": h(pickle.dumps(state[1], protocol=5)),
                "torch_cpu_rng_sha256": h(state[2].cpu().numpy().tobytes()),
                "torch_cuda_rng_sha256": [h(s.cpu().numpy().tobytes()) for s in state[3]]}


def validate_request(request, torch):
    if not isinstance(request, dict) or not REQUIRED <= request.keys() or request.keys() - ALLOWED:
        raise ValueError("Only complete official processor requests; no cache/generation/logits overrides")
    if request["task_id"] != "robocasa365":
        raise ValueError("Only robocasa365 requests")
    for name, value in request.items():
        if name == "task_id":
            continue
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError("Request values must be CPU tensors")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Nonfinite processor tensor")
    if tuple(request["state"].shape) != (1, 4, 60) or tuple(request["action_mask"].shape) != (1, 16, 60):
        raise ValueError("Only batch1, history4, full16x60 actions")
    ids = request["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or not 2 <= ids.shape[1] <= 16384:
        raise ValueError("Expected a complete batch1 multimodal prompt")
    if ids.is_floating_point() or ids.dtype == torch.bool or request["attention_mask"].shape != ids.shape:
        raise ValueError("Invalid token ids/mask")
    if not request["state"].is_floating_point() or not request["action_mask"].is_floating_point():
        raise ValueError("State and action mask must be floating point")


def official_forward(model, request, torch):
    """The exact reference server tensor conversion and full model call."""
    data = {key: (value.to(device=model.device, dtype=model.dtype)
                  if isinstance(value, torch.Tensor) and value.is_floating_point()
                  else value.to(device=model.device) if isinstance(value, torch.Tensor)
                  else value) for key, value in request.items()}
    return model(**data).actions.cpu()


class SerialExecutor:
    def __init__(self, model, torch, identity, *, capacity=8):
        if capacity < 1:
            raise ValueError("Positive bounded capacity required")
        self.model, self.torch = model, torch
        self._config_hash = self._model_config_hash()
        self.identity = dict(identity, protocol=PROTOCOL, exclusive_connection=False,
                             shared_model=True, isolated_rng_stream=True,
                             serial_forward=True, rng_isolation="python_numpy_torch_cpu_all_cuda_per_connection",
                             full_request_only=True, inference_optimization="none",
                             model_config_runtime_sha256=self._config_hash)
        self._queue = queue.Queue(maxsize=capacity)
        self._sessions = {}
        self._closed = False
        self._started = threading.Event()
        self._startup_error = None
        self._thread = threading.Thread(target=self._work, name="xr1-serial-forward", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=30) or self._startup_error is not None:
            raise RuntimeError("Serial worker initialization failed") from self._startup_error

    def _model_config_hash(self):
        config = getattr(self.model, "config", None)
        if config is None:
            return None  # CPU fake models only.
        return hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True, default=str).encode()).hexdigest()

    def submit(self, connection, payload=None, *, close=False):
        if self._closed:
            raise RuntimeError("Executor is closed")
        future = Future()
        self._queue.put((connection, payload, close, future), timeout=30)
        return future

    def close(self):
        if not self._closed:
            self._closed = True
            self._queue.put(None, timeout=30)
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                raise RuntimeError("Inference executor did not finish; process owner must terminate")

    def _work(self):
        try:
            self.rng = RngBank(self.torch)
            self.identity.update(model_execution_thread_name=threading.current_thread().name,
                                 model_execution_thread_ident=threading.get_ident())
        except Exception as error:
            self._startup_error = error
            return
        finally:
            self._started.set()
        while True:
            item = self._queue.get()
            if item is None:
                return
            connection, payload, close, future = item
            try:
                if close:
                    self._sessions.pop(connection, None)
                    result = None
                else:
                    result = self._handle(connection, payload)
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)

    def _handle(self, connection, payload):
        # This method is never invoked from socket threads.
        if threading.current_thread() is not self._thread:
            raise RuntimeError("Model execution must remain on the sole worker")
        session = self._sessions.setdefault(connection, {"rng": None, "requests": 0, "poisoned": False})
        ambient = self.rng.capture()
        try:
            if session["poisoned"]:
                raise RuntimeError("Failed session is closed; reconnect, never silently retry")
            if session["rng"] is not None:
                self.rng.restore(session["rng"])
            request = pickle.loads(payload)
            if not isinstance(request, dict):
                raise ValueError("Dictionary request required")
            command = request.get(CONTROL)
            if command is not None:
                if request.get("protocol") != PROTOCOL or command not in {"hello", "reset_rng", "rng_state"}:
                    raise ValueError("Unsupported multiplex control")
                nonce = request.get("request_nonce")
                if not isinstance(nonce, str) or not nonce:
                    raise ValueError("Acknowledgement nonce required")
                response = {"ok": True, "request_nonce": nonce, "connection_id": connection, **self.identity}
                if command == "reset_rng":
                    self.rng.seed(request["seed"])
                    session.update(rng=self.rng.capture(), requests=0, seed=request["seed"])
                if command in {"reset_rng", "rng_state"}:
                    if session["rng"] is None:
                        raise ValueError("RNG state unavailable before initialization")
                    response.update(seed=session["seed"], requests_since_reset=session["requests"],
                                    **self.rng.hashes(session["rng"]))
            else:
                if session["rng"] is None:
                    raise ValueError("An acknowledged episode RNG initialization is required")
                validate_request(request, self.torch)
                if self._model_config_hash() != self._config_hash:
                    raise ValueError("Shared model config changed")
                response = official_forward(self.model, request, self.torch)
                if self._model_config_hash() != self._config_hash:
                    raise ValueError("Model forward changed shared configuration")
                if tuple(response.shape) != (1, 16, 60) or not self.torch.isfinite(response).all():
                    raise ValueError("Unexpected official model output")
                session["rng"] = self.rng.capture()
                session["requests"] += 1
            return pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            session["poisoned"] = True
            raise
        finally:
            self.rng.restore(ambient)
