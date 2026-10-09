"""Isolated adapter-only action LoRA prototype; not a full HF exporter.

No device selection, model loading, training, or filesystem writes on import.
The original weight/bias names survive replacement. This has only CPU fixture
validation; run real-model precision/inference parity before any real training.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

VERSION = 1


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError("Only ordinary dense state tensors are supported")
        value = value.detach().cpu().contiguous()
        digest.update(_json([name, list(value.shape), str(value.dtype)]))
        raw = value.reshape(-1).view(torch.uint8).numpy()
        for start in range(0, raw.size, 4 * 1024 * 1024):
            digest.update(raw[start:start + 4 * 1024 * 1024].tobytes())
    return digest.hexdigest()


def _seed_for(seed, name):
    return int.from_bytes(hashlib.sha256(_json([seed, name])).digest()[:8], "little") % (2**63)


class ActionLoRALinear(nn.Module):
    """Frozen original linear plus FP32 trainable low-rank residual."""
    def __init__(self, original, rank, alpha, seed):
        super().__init__()
        if type(original) is not nn.Linear:
            raise ValueError("Expected an exact torch.nn.Linear, not a quantized/custom module")
        self.in_features, self.out_features = original.in_features, original.out_features
        # Reuse original Parameter objects and names, with no constructor RNG.
        self.weight = original.weight
        self.bias = original.bias
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.adapter_enabled = True
        generator = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.empty((rank, self.in_features), dtype=torch.float32, device="cpu")
        bound = 1 / math.sqrt(self.in_features)
        a.uniform_(-bound, bound, generator=generator)
        self.lora_A = nn.Parameter(a.to(device=original.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(
            (self.out_features, rank), dtype=torch.float32, device=original.weight.device))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.train(original.training)

    def forward(self, inputs):
        base = F.linear(inputs, self.weight, self.bias)
        if not self.adapter_enabled:
            return base
        residual = F.linear(F.linear(inputs.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base + (residual * self.scaling).to(base.dtype)


@dataclass
class AdapterHandle:
    model: nn.Module
    metadata: dict

    @property
    def adapter_names(self):
        return {f"{item['name']}.{suffix}" for item in self.metadata["targets"]
                for suffix in ("lora_A", "lora_B")}

    def base_fingerprint(self):
        return _tensor_digest((name, value) for name, value in self.model.state_dict().items()
                              if name not in self.adapter_names)

    def assert_integrity(self):
        modules = dict(self.model.named_modules())
        for target in self.metadata["targets"]:
            module = modules.get(target["name"])
            if type(module) is not ActionLoRALinear:
                raise ValueError("LoRA target module has changed")
            if (module.rank != self.metadata["rank"] or module.alpha != self.metadata["alpha"] or
                    module.scaling != self.metadata["alpha"] / self.metadata["rank"]):
                raise ValueError("LoRA recipe has changed")
            expected = {"lora_A": (module.rank, target["in_features"]),
                        "lora_B": (target["out_features"], module.rank)}
            for leaf, shape in expected.items():
                parameter = getattr(module, leaf)
                if (tuple(parameter.shape) != shape or parameter.dtype != torch.float32 or
                        parameter.device != module.weight.device):
                    raise ValueError("Adapter parameter geometry/dtype/device changed")
        trainable = {name for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        if trainable != self.adapter_names:
            raise ValueError("Only the declared adapter tensors may require gradients")
        if self.base_fingerprint() != self.metadata["base_state_sha256"]:
            raise ValueError("Original model tensor identity changed")


def inject_action_lora(model, *, base_identity, rank=8, alpha=None, seed=0):
    """Freeze all original parameters and replace only exact action attention paths.

    base_identity is a caller-supplied immutable checkpoint identity string;
    the independent full original tensor SHA additionally verifies its contents.
    Fingerprinting a real 5B model has nontrivial CPU-copy cost and is not yet
    measured. No optimizer is created; no inference/service launcher is provided.
    """
    if not isinstance(base_identity, str) or not base_identity.strip():
        raise ValueError("Provide a nonempty immutable source checkpoint identity")
    if type(rank) is not int or rank < 1:
        raise ValueError("rank must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    alpha = float(rank if alpha is None else alpha)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    layers = getattr(getattr(model, "dit", None), "layers", None)
    if type(layers) is not nn.ModuleList or not len(layers):
        raise ValueError("Expected model.dit.layers ModuleList")
    if any(isinstance(m, ActionLoRALinear) for m in model.modules()):
        raise ValueError("Adapters are already installed")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Clear existing gradients explicitly before injection")
    targets, replacements, original_ids = [], [], set()
    for index, layer in enumerate(layers):
        attention = getattr(layer, "attn", None)
        qkv, out = getattr(attention, "qkv_proj", None), getattr(attention, "o_proj", None)
        if type(qkv) is not nn.Linear or type(out) is not nn.Linear:
            raise ValueError("Both exact qkv_proj/o_proj linears are required in every action layer")
        hidden = qkv.in_features
        if (qkv.out_features != hidden * 3 or qkv.bias is None or
                out.in_features != hidden or out.out_features != hidden or out.bias is not None):
            raise ValueError("Unexpected released action attention geometry/bias")
        for leaf, module in (("qkv_proj", qkv), ("o_proj", out)):
            if module.weight.device.type == "meta" or not module.weight.is_floating_point():
                raise ValueError("Materialized floating-point original weights are required")
            if id(module.weight) in original_ids:
                raise ValueError("Aliased attention target weights are unsupported")
            original_ids.add(id(module.weight))
            name = f"dit.layers.{index}.attn.{leaf}"
            targets.append({"name": name, "in_features": module.in_features,
                            "out_features": module.out_features, "bias": module.bias is not None,
                            "base_dtype": str(module.weight.dtype)})
            replacements.append((attention, leaf, module, name))
    metadata = {"format": "xr1-action-lora-adapter-only", "version": VERSION,
                "base_identity": base_identity, "base_state_sha256": _tensor_digest(model.state_dict().items()),
                "rank": rank, "alpha": alpha, "seed": seed, "targets": targets,
                "adapter_dtype": "torch.float32"}
    # All structural/identity validation precedes model mutation.
    prepared = [(parent, leaf, ActionLoRALinear(module, rank, alpha, _seed_for(seed, name)))
                for parent, leaf, module, name in replacements]
    model.requires_grad_(False)
    for parent, leaf, replacement in prepared:
        setattr(parent, leaf, replacement)
    return AdapterHandle(model, metadata)


def _payload_digest(metadata, tensors):
    return hashlib.sha256(_json(metadata) + _tensor_digest(tensors.items()).encode()).hexdigest()


def save_adapter(handle, path):
    """Create a new adapter-only .pt file atomically; never overwrite."""
    handle.assert_integrity()
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    state = handle.model.state_dict()
    tensors = {name: state[name].detach().cpu().clone() for name in sorted(handle.adapter_names)}
    if not all(torch.isfinite(value).all().item() for value in tensors.values()):
        raise ValueError("Adapter contains nonfinite values")
    metadata = json.loads(_json(handle.metadata))
    payload = {"metadata": metadata, "tensors": tensors,
               "sha256": _payload_digest(metadata, tensors)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".adapter-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
        # Hard-link publication is atomic and refuses a concurrent overwrite.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "adapter_parameters": sum(v.numel() for v in tensors.values()),
            "payload_sha256": payload["sha256"], "base_state_sha256": metadata["base_state_sha256"]}


def load_adapter(handle, path):
    """Load only after all identity, configuration, schema, and hash checks pass."""
    handle.assert_integrity()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "tensors", "sha256"}:
        raise ValueError("Invalid adapter payload schema")
    if payload["metadata"] != handle.metadata:
        raise ValueError("Adapter source model or configuration identity mismatch")
    tensors = payload["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != handle.adapter_names:
        raise ValueError("Invalid adapter tensor names")
    state = handle.model.state_dict()
    for name, value in tensors.items():
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                value.shape != state[name].shape or not torch.isfinite(value).all().item()):
            raise ValueError("Invalid adapter shape/dtype/value")
    if payload["sha256"] != _payload_digest(payload["metadata"], tensors):
        raise ValueError("Adapter payload hash mismatch")
    with torch.no_grad():
        for name, value in tensors.items():
            state[name].copy_(value)
    return {"loaded_tensors": len(tensors), "payload_sha256": payload["sha256"]}


@contextmanager
def adapters_disabled(handle):
    """Temporarily bypass adapters, without changing parameters or autograd flags.

    Single-threaded use only. Existing student graphs remain valid because the
    flag chooses the next forward path; it does not alter saved tensors.
    """
    modules = dict(handle.model.named_modules())
    selected = [modules[target["name"]] for target in handle.metadata["targets"]]
    if any(type(module) is not ActionLoRALinear or not module.adapter_enabled for module in selected):
        raise RuntimeError("Expected enabled adapters before a temporary teacher bypass")
    try:
        for module in selected:
            module.adapter_enabled = False
        yield
    finally:
        for module in selected:
            module.adapter_enabled = True
