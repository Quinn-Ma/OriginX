"""Flow loss for the released HF inference model; no model source edits.

The baseline has no supervised training forward. We reuse its component
methods with its exact inference mask and positional encoding. This is a new
training objective implementation, not a reproduction of unpublished settings.
"""

import json
import shutil
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

try:
    from .layout import ACTION_DIM, ACTION_LENGTH, HISTORY_LENGTH, STATE_DIM
except ImportError:
    from layout import ACTION_DIM, ACTION_LENGTH, HISTORY_LENGTH, STATE_DIM

TRAINABLE_MODULES = (
    "dit", "state_projector", "action_projector", "action_output_layer",
    "t_embedder", "t_projector", "sink",
)


def validate_processor(processor) -> None:
    """Refuse a different robot or scaling instead of silently retraining it."""
    stats = processor.action_config["robocasa365"]
    mean, std = stats["mean"], stats["std"]
    expected = torch.zeros((1, ACTION_LENGTH, ACTION_DIM), dtype=torch.float32)
    expected[..., :12] = 1
    if mean.shape != expected.shape or std.shape != expected.shape:
        raise ValueError("Expected the released 16x60 RoboCasa365 action statistics")
    if not torch.equal(mean.cpu().float(), torch.zeros_like(expected)):
        raise ValueError("This bridge requires the released identity action mean")
    if not torch.equal(std.cpu().float(), expected):
        raise ValueError("This bridge requires identity scale on exactly the first 12 actions")


def encode_sample(sample, processor, device, floating_dtype=None):
    """Official processor inputs plus supervised target; one sample only."""
    validate_processor(processor)
    inputs = dict(processor.apply_chat_template(
        sample.messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_resize=False,
        state=sample.state,
        robot_type="robocasa365",
    ))
    device = torch.device(device)
    if floating_dtype is None:
        floating_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # Match the official server: cast every floating input to model dtype,
    # while preserving integer token IDs, grid tensors and integer masks.
    inputs = {k: (v.to(device=device, dtype=floating_dtype) if v.is_floating_point() else v.to(device))
              if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    actions = torch.as_tensor(sample.action, device=device, dtype=torch.float32)
    valid_steps = torch.as_tensor(sample.valid_steps, device=device, dtype=torch.bool)
    return inputs, actions, valid_steps


class FrozenVLMFlowBridge(nn.Module):
    """Train the HF DiT and its projections, with a frozen VLM.

    FP32 trainable weights are the default; CUDA forward uses BF16 autocast.
    This permits FP32 optimizer states with ordinary AdamW. The supplied smoke
    script does not perform an optimizer step. The model itself remains an
    ordinary HF model and can still be saved independently of this wrapper.
    """

    def __init__(self, model, fp32_trainable: bool = True):
        super().__init__()
        self.model = model
        config = model.config
        if (config.state_dim, config.action_dim, config.state_length) != (60, 60, 4):
            raise ValueError("Expected released RoboCasa365 state_dim=60, action_dim=60, state_length=4")
        model.requires_grad_(False)
        for name in TRAINABLE_MODULES:
            module = getattr(model, name)
            if fp32_trainable:
                module.float()
            module.requires_grad_(True)
        self.model.vlm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Calling wrapper.train() must not reactivate frozen VLM dropout.
        self.model.vlm.eval()
        return self

    def trainable_parameters(self):
        return (p for p in self.model.parameters() if p.requires_grad)

    def _validate_batch(self, inputs, actions, valid_steps):
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (ACTION_LENGTH, ACTION_DIM):
            raise ValueError("actions must have shape (B,16,60)")
        batch_size = actions.shape[0]
        if tuple(inputs["state"].shape) != (batch_size, HISTORY_LENGTH, STATE_DIM):
            raise ValueError("state must have shape (B,4,60)")
        if tuple(valid_steps.shape) != (batch_size, ACTION_LENGTH):
            raise ValueError("valid_steps must have shape (B,16)")
        if valid_steps.dtype != torch.bool:
            raise ValueError("valid_steps must have boolean dtype")
        if not valid_steps.any(dim=1).all():
            raise ValueError("Each sample needs at least one valid future action")
        mask = inputs["action_mask"]
        if tuple(mask.shape) != tuple(actions.shape):
            raise ValueError("action_mask must match (B,16,60); the HF processor defaults to B=1")
        expected = torch.zeros_like(mask)
        expected[..., :12] = 1
        if not torch.equal(mask, expected):
            raise ValueError("action_mask must select exactly the first 12 dimensions at all 16 steps")
        if not torch.isfinite(actions).all() or not torch.isfinite(inputs["state"]).all():
            raise ValueError("state/action tensors must be finite")
        if not torch.equal(inputs["state"][..., 14:], torch.zeros_like(inputs["state"][..., 14:])):
            raise ValueError("RoboCasa365 state dimensions 14:60 must be zero padding")
        if inputs["input_ids"].shape[0] != batch_size:
            raise ValueError("Token batch size does not match action batch size")
        if "attention_mask" not in inputs or inputs["attention_mask"].shape != inputs["input_ids"].shape:
            raise ValueError("An explicit (B,S) VLM attention_mask is required")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.device != actions.device:
                raise ValueError(f"{key} is on a different device from the target")
        if valid_steps.device != actions.device:
            raise ValueError("valid_steps is on a different device from the target")

    def forward(self, inputs, actions, valid_steps, *, noise=None, times=None):
        """Return masked flow MSE and diagnostics; no optimizer or data side effects.

        Defaults: uniform t in [0,1), all-60D standard normal noise, and loss
        only on valid future steps × first 12 action dimensions. Inactive noisy
        coordinates are still sampled, as in baseline inference, then masked
        by the unchanged HF dit_forward. Temporal padding affects loss only.
        """
        self._validate_batch(inputs, actions, valid_steps)
        self.model.vlm.eval()
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if actions.is_cuda else nullcontext()
        with autocast:
            return self._loss(inputs, actions, valid_steps, noise, times)

    def _loss(self, inputs, actions, valid_steps, noise, times):
        # no_grad (not inference_mode): autograd must save constant K/V tensors
        # when computing gradients through attention in the trainable DiT.
        vlm_inputs = {k: v for k, v in inputs.items() if k not in {"state", "action_mask", "task_id"}}
        with torch.no_grad():
            context = self.model.vlm.model(**vlm_inputs, use_cache=True)

        mask = inputs["action_mask"]
        state = inputs["state"]
        batch_size, action_length, _ = mask.shape
        query_length = 1 + state.shape[1] + action_length
        positions = (
            torch.arange(query_length, device=mask.device).view(1, 1, -1).repeat(3, batch_size, 1)
            + context.position_ids.max(dim=-1)[0][..., None] + 1
        )
        position_embeds = self.model.rotary_emb(mask, positions)
        cache_mask = inputs["attention_mask"][:, None, :].expand(-1, query_length, -1)
        causal_mask = torch.tril(torch.ones(batch_size, query_length, query_length, device=mask.device))
        attention_mask = torch.cat([cache_mask, causal_mask], dim=-1)[:, None].bool()

        # Preserve the checkpoint input arithmetic dtype (BF16 on H100).
        target = actions.to(dtype=mask.dtype)
        if noise is None:
            noise = torch.randn_like(mask)
        else:
            if noise.shape != mask.shape or noise.device != mask.device or not torch.isfinite(noise).all():
                raise ValueError("noise must be finite, on-device and have shape (B,16,60)")
            noise = noise.to(dtype=mask.dtype)
        if times is None:
            times = torch.rand((batch_size, 1, 1), device=mask.device, dtype=mask.dtype)
        else:
            if tuple(times.shape) != (batch_size, 1, 1) or times.device != mask.device:
                raise ValueError("times must be on-device with shape (B,1,1)")
            if not torch.isfinite(times).all() or (times < 0).any() or (times > 1).any():
                raise ValueError("times must be finite and between 0 and 1")
            times = times.to(dtype=mask.dtype)
        noisy_action = (1 - times) * noise + times * target
        state_embed = self.model.state_projector(state)
        prediction = self.model.dit_forward(
            noisy_action=noisy_action,
            t=times,
            action_mask=mask,
            state_embed=state_embed,
            position_embeds=position_embeds,
            past_key_values=context.past_key_values,
            attn_mask=attention_mask,
        )
        if prediction.shape != actions.shape:
            raise ValueError("HF dit_forward returned an unexpected shape")
        loss_mask = mask.bool() & valid_steps[..., None]
        velocity_target = target.float() - noise.float()
        selected_error = (prediction.float() - velocity_target)[loss_mask]
        loss = selected_error.square().mean()
        return {"loss": loss, "active_values": loss_mask.sum().detach(),
                "time_mean": times.float().mean().detach()}


def gradient_report(bridge: FrozenVLMFlowBridge) -> dict:
    """Assert that gradients reach every trainable component and never the VLM."""
    vlm = list(bridge.model.vlm.parameters())
    if any(p.requires_grad or p.grad is not None for p in vlm):
        raise AssertionError("VLM was not completely frozen")
    report = {}
    for name in TRAINABLE_MODULES:
        gradients = [p.grad for p in getattr(bridge.model, name).parameters() if p.requires_grad]
        if not gradients or any(g is None for g in gradients):
            raise AssertionError(f"Missing gradient in {name}")
        if not all(torch.isfinite(g).all().item() for g in gradients):
            raise AssertionError(f"Non-finite gradient in {name}")
        norm_sq = sum(g.detach().double().square().sum().item() for g in gradients)
        if norm_sq <= 0:
            raise AssertionError(f"Zero gradient in {name}")
        report[name] = {"gradient_l2": norm_sq ** 0.5, "tensors": len(gradients)}
    report["vlm"] = {"frozen": True, "gradient_tensors": 0}
    return report


def export_hf(model, processor, source_dir, output_dir) -> Path:
    """Save the underlying model, with HF-compatible names and inference assets.

    No training-wrapper prefix is introduced. Never overwrites a nonempty
    directory or the source. The caller chooses export dtype before calling;
    smoke_gpu.py deliberately casts to BF16 after the gradient-only check.
    """
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if not source_dir.is_dir() or source_dir == output_dir:
        raise ValueError("Use a complete local source snapshot and a separate export directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty export: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(output_dir)
    # save_pretrained normally copies auto_map code. Carry across any missing
    # tokenizer/video/template assets without replacing the newly saved config
    # or copying a stale weight index after resharding.
    for source in source_dir.iterdir():
        if not source.is_file() or source.name.endswith(".index.json"):
            continue
        if source.suffix not in {".json", ".py", ".jinja", ".txt", ".model"}:
            continue
        destination = output_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    required = ("config.json", "processor_config.json", "preprocessor_config.json")
    if not all((output_dir / name).is_file() for name in required):
        raise RuntimeError("Export is missing required HF model/processor metadata")
    (output_dir / "training_bridge_export.json").write_text(json.dumps({
        "source": str(source_dir),
        "model_format": "underlying AutoModel.save_pretrained; no wrapper prefix",
        "objective": "masked flow MSE, uniform time, frozen VLM",
        "action_horizon": 16,
        "loss_dimensions": list(range(12)),
        "history": [4, 2],
    }, indent=2) + "\n")
    return output_dir
