"""Single-device accumulation and atomic step-boundary checkpoints.

This module has no dataset selection, model loading, or automatic execution.
Only trusted local checkpoints written by this code should be loaded.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch

try:
    from .bridge import TRAINABLE_MODULES
except ImportError:
    from bridge import TRAINABLE_MODULES


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def rng_state(device):
    device = torch.device(device)
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


class SingleDeviceTrainer:
    """Microbatch-one training; checkpoint only after complete optimizer steps."""

    def __init__(self, bridge, optimizer, provider, *, identity, device,
                 accumulate=1, max_grad_norm=1.0):
        if accumulate < 1 or max_grad_norm <= 0:
            raise ValueError("accumulate and max_grad_norm must be positive")
        self.bridge, self.optimizer, self.provider = bridge, optimizer, provider
        if any(p.dtype != torch.float32 for p in bridge.trainable_parameters()):
            raise ValueError("Training requires FP32 master parameters in every trainable component")
        self.identity, self.device = identity, torch.device(device)
        self.accumulate, self.max_grad_norm = accumulate, max_grad_norm
        self.global_step = 0
        self._at_step_boundary = True

    def step(self):
        self.bridge.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._at_step_boundary = False
        losses, samples = [], []
        for _ in range(self.accumulate):
            inputs, actions, valid_steps, sample_id = self.provider.next_batch()
            result = self.bridge(inputs, actions, valid_steps)
            loss = result["loss"]
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite training loss; optimizer was not stepped")
            (loss / self.accumulate).backward()
            losses.append(loss.detach().float().item())
            samples.append(sample_id)
        parameters = list(self.bridge.trainable_parameters())
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        self._at_step_boundary = True
        return {"step": self.global_step, "loss": sum(losses) / len(losses),
                "gradient_norm_before_clip": norm.item(), "samples": samples}

    def save(self, path):
        if not self._at_step_boundary:
            raise RuntimeError("Refusing an incomplete accumulation checkpoint")
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1, "identity": self.identity, "global_step": self.global_step,
            "accumulate": self.accumulate, "max_grad_norm": self.max_grad_norm,
            "heads": {name: _to_cpu(getattr(self.bridge.model, name).state_dict()) for name in TRAINABLE_MODULES},
            "optimizer": _to_cpu(self.optimizer.state_dict()),
            "provider": self.provider.state_dict(), "rng": rng_state(self.device),
        }
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            # Make the rename durable on Linux filesystems after a host crash.
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def load(self, path):
        # These are our own local training checkpoints, containing Python/NumPy
        # RNG state as well as tensors; never accept an untrusted downloaded .pt.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 1 or checkpoint.get("identity") != self.identity:
            raise ValueError("Checkpoint data/base/code/optimizer identity differs from this run")
        if checkpoint["accumulate"] != self.accumulate or checkpoint["max_grad_norm"] != self.max_grad_norm:
            raise ValueError("Checkpoint accumulation/clipping settings differ")
        for name in TRAINABLE_MODULES:
            getattr(self.bridge.model, name).load_state_dict(checkpoint["heads"][name], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.provider.load_state_dict(checkpoint["provider"])
        self.global_step = int(checkpoint["global_step"])
        self._at_step_boundary = True
        self.optimizer.zero_grad(set_to_none=True)
        # Restore last: constructing/loading models and processors can consume RNG.
        restore_rng(checkpoint["rng"], self.device)
        return self.global_step
