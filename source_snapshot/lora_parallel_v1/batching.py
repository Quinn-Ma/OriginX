"""Collate already encoded, individual RoboCasa365 samples without reprocessing.

This is the released Qwen3-VL video packing layout, not an image/batch tokenizer.
Only token-axis right padding is added. Pixel values are concatenated verbatim
in sample -> camera -> patch order. No RNG, target-to-prompt, normalization,
processor call, tensor recast or device move takes place here.
"""
from collections.abc import Mapping
import torch


BATCHING_POLICY = {
    "kind": "independent_encoded_qwen3vl_microbatch_v1",
    "allowed_microbatch": [1, 2, 4, 8, 16],
    "token_padding_side": "right",
    "video_packing": "sample_then_camera_original_patch_axis",
    "cameras_per_sample": 3,
    "state_shape": [4, 60],
    "action_shape": [16, 60],
    "targets_never_added_to_inputs": True,
    "random_draws": "none; caller uses original slot_draws once per identity.slot_seed",
    "loss_contract": "caller averages per-sample masked losses, not pooled active values",
}

_REQUIRED = {"input_ids", "attention_mask", "state", "action_mask",
             "pixel_values_videos", "video_grid_thw"}
_OPTIONAL = {"mm_token_type_ids", "task_id"}
_INTEGER = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def pad_token_id_from_processor(processor):
    """Require the actual tokenizer's explicit padding ID; no invented fallback."""
    value = processor.tokenizer.pad_token_id
    if type(value) is not int or value < 0:
        raise ValueError("The released tokenizer must provide an explicit pad_token_id")
    return value


def _shape(tensor, shape, key):
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{key} must be a tensor of shape {tuple(shape)}")


def collate_encoded_samples(samples, *, pad_token_id, video_token_id=151656,
                            spatial_merge_size=2):
    """Return (inputs, actions[B,16,60], valid[B,16], ordered identities).

    ``samples`` consists of the original provider's (inputs, actions, valid,
    identity) tuples. All must already share a device and each corresponding
    tensor's dtype. Optional task_id is metadata [1] or [1,1], and remains
    [B] or [B,1]; the bridge must continue excluding it from the VLM input.
    Padding IDs are meaningful only where attention_mask=0. The bridge must
    derive each sample's action RoPE offset using valid prefix positions.
    """
    samples = list(samples)
    if len(samples) not in (1, 2, 4, 8, 16):
        raise ValueError("This run permits only microbatch sizes 1, 2, 4, 8, or 16")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be the tokenizer's nonnegative integer ID")
    if type(video_token_id) is not int or video_token_id < 0 or video_token_id == pad_token_id:
        raise ValueError("video_token_id must be a non-padding integer ID")
    if type(spatial_merge_size) is not int or spatial_merge_size != 2:
        raise ValueError("This released model uses spatial_merge_size=2")
    keys = None
    reference = None
    lengths = []
    previous_slot = None
    identities = []
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, (list, tuple)) or len(sample) != 4:
            raise ValueError("Each item must be a provider (inputs,actions,valid,identity) tuple")
        inputs, actions, valid, identity = sample
        if not isinstance(inputs, Mapping) or not _REQUIRED <= inputs.keys():
            raise ValueError("Missing released video processor input keys")
        if set(inputs) - _REQUIRED - _OPTIONAL:
            raise ValueError("Unknown processor keys are refused (including supervision/labels)")
        if keys is None:
            keys = set(inputs)
        elif set(inputs) != keys:
            raise ValueError("All samples must have exactly the same encoded keys")
        if not isinstance(identity, Mapping):
            raise ValueError("Each sample must retain its provider identity")
        slot, seed = identity.get("slot_index"), identity.get("slot_seed")
        if type(slot) is not int or slot < 0 or type(seed) is not int or seed < 0:
            raise ValueError("Each sample needs its original integer slot_index and slot_seed")
        if previous_slot is not None and slot != previous_slot + 1:
            raise ValueError("Microbatch samples must be consecutive in committed slot order")
        previous_slot = slot
        identities.append(dict(identity))
        ids = inputs["input_ids"]
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("input_ids must have shape (1,S), S>=1")
        if ids.dtype not in _INTEGER:
            raise ValueError("input_ids must have integer dtype")
        length = ids.shape[1]
        lengths.append(length)
        _shape(inputs["attention_mask"], (1, length), "attention_mask")
        mask = inputs["attention_mask"]
        if mask.dtype not in _INTEGER | {torch.bool}:
            raise ValueError("attention_mask must be an integer/bool tensor")
        if not torch.all((mask == 0) | (mask == 1)).item() or not mask[0, 0].item():
            raise ValueError("attention_mask must contain nonempty binary right-padded tokens")
        if length > 1 and torch.any(mask[0, 1:].to(torch.int8) > mask[0, :-1].to(torch.int8)).item():
            raise ValueError("Preexisting left/interior token padding is refused")
        _shape(inputs["state"], (1, 4, 60), "state")
        _shape(inputs["action_mask"], (1, 16, 60), "action_mask")
        _shape(actions, (1, 16, 60), "actions")
        _shape(valid, (1, 16), "valid")
        if actions.dtype != torch.float32 or valid.dtype != torch.bool:
            raise ValueError("Original action FP32 and valid bool dtypes must be retained")
        if not inputs["state"].is_floating_point() or not inputs["action_mask"].is_floating_point():
            raise ValueError("state/action_mask must retain their floating input dtype")
        grid = inputs["video_grid_thw"]
        _shape(grid, (3, 3), "video_grid_thw (three ordered cameras)")
        if grid.dtype not in _INTEGER or not torch.all(grid > 0).item():
            raise ValueError("video_grid_thw must contain positive integer grids")
        if torch.any(grid[:, 1:] % spatial_merge_size).item():
            raise ValueError("Video spatial grid must be divisible by the vision merge size")
        pixels = inputs["pixel_values_videos"]
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 2 or not pixels.is_floating_point():
            raise ValueError("pixel_values_videos must retain the flattened (patches,features) layout")
        patch_count = int(grid.to(torch.int64).prod(dim=1).sum().item())
        if pixels.shape[0] != patch_count:
            raise ValueError("Video grid patch count does not match unchanged pixel rows")
        token_count = int(((ids == video_token_id) & mask.bool()).sum().item())
        if token_count != patch_count // spatial_merge_size**2:
            raise ValueError("Video placeholder token count differs from the encoded camera grids")
        if "mm_token_type_ids" in inputs:
            _shape(inputs["mm_token_type_ids"], (1, length), "mm_token_type_ids")
            if inputs["mm_token_type_ids"].dtype not in _INTEGER:
                raise ValueError("mm_token_type_ids must have integer dtype")
        if "task_id" in inputs:
            task = inputs["task_id"]
            if not isinstance(task, torch.Tensor) or tuple(task.shape) not in ((1,), (1, 1)) or task.dtype not in _INTEGER:
                raise ValueError("Optional task_id must be integer tensor [1] or [1,1]")
        tensors = dict(inputs, _supervised_actions=actions, _valid_steps=valid)
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise ValueError("All encoded values must already be tensors")
        if any(value.device != actions.device for value in tensors.values()):
            raise ValueError("All sample tensors must share the target's device")
        if reference is None:
            reference = tensors
        else:
            for key, value in tensors.items():
                previous = reference[key]
                if value.device != previous.device or value.dtype != previous.dtype:
                    raise ValueError(f"{key}: mixed device/dtype would change original input arithmetic")
                if key == "pixel_values_videos" and value.shape[1:] != previous.shape[1:]:
                    raise ValueError("All video patch feature dimensions must agree")
                if key == "task_id" and value.shape != previous.shape:
                    raise ValueError("Optional task_id shapes must agree")

    maximum = max(lengths)
    batched = {}
    for key in samples[0][0]:
        values = [sample[0][key] for sample in samples]
        if key in {"input_ids", "attention_mask", "mm_token_type_ids"}:
            fill = pad_token_id if key == "input_ids" else 0
            rows = []
            for value, length in zip(values, lengths):
                if length == maximum:
                    rows.append(value)
                else:
                    rows.append(torch.cat((value, value.new_full((1, maximum - length), fill)), dim=1))
            batched[key] = torch.cat(rows, dim=0)
        else:
            batched[key] = torch.cat(values, dim=0)
    return (batched, torch.cat([sample[1] for sample in samples], dim=0),
            torch.cat([sample[2] for sample in samples], dim=0), identities)
