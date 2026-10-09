#!/usr/bin/env python3
"""H100 gradient smoke; synthetic, verified human or corrective sample, no update."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from bridge import TRAINABLE_MODULES, FrozenVLMFlowBridge, encode_sample, export_hf, gradient_report
from layout import CAMERA_KEYS, make_sample


def synthetic_sample():
    frames = {}
    for camera, key in enumerate(CAMERA_KEYS):
        images = []
        for frame in range(32):
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            image[..., camera] = 80 + frame * 4
            image[64:192, 64:192, (camera + 1) % 3] = 160
            images.append(image)
        frames[key] = images
    states = np.zeros((32, 16), dtype=np.float32)
    states[:, 6] = 1  # base quaternion xyzw
    states[:, 13] = 1  # EE quaternion xyzw
    states[:, 7:10] = [0.4, 0.1, 0.8]
    states[:, 14:16] = [0.02, -0.02]
    actions = np.zeros((32, 12), dtype=np.float32)
    actions[:, 5] = 0.01  # an EE controller command in LeRobot ordering
    actions[:, 11] = -1
    return make_sample(states, actions,
                       lambda key, indices: [frames[key][i] for i in indices],
                       timestep=8, instruction="Close the blender lid.")


def check_processor_parity(before, after):
    if before.keys() != after.keys():
        raise AssertionError("Reloaded processor produced different input keys")
    for key in before:
        a, b = before[key], after[key]
        if isinstance(a, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                raise AssertionError(f"Reloaded processor changed {key}")
        elif a != b:
            raise AssertionError(f"Reloaded processor changed {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Complete local HF RoboCasa365 snapshot")
    parser.add_argument("--export-dir", type=Path, help="Optional NEW directory for real HF save/reload parity")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=193)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", type=Path, help="Optional complete local pretrain-human LeRobot dataset")
    source.add_argument("--corrective-dir", type=Path, help="Verified successful simulation recording")
    parser.add_argument("--corrective-index", type=int, default=0, help="Index among eligible corrective actions")
    parser.add_argument("--acquisition", type=Path, help="Downloader's download_complete.json for that dataset")
    parser.add_argument("--registry", type=Path, help="Official robocasa/utils/dataset_registry.py")
    parser.add_argument("--box-links", type=Path, help="Official box_links_ds.json")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=8)
    parser.add_argument("--attn-implementation", default="flash_attention_2", choices=["flash_attention_2", "sdpa"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA GPU; use tests/ for CPU checks")
    if not args.model.is_dir():
        raise ValueError("--model must be a local complete snapshot; no automatic downloads")
    if args.report.exists():
        raise FileExistsError(f"Refusing to replace report {args.report}")
    data_report = None
    if args.corrective_dir is not None:
        if any(value is not None for value in (args.acquisition, args.registry, args.box_links)):
            raise ValueError("Human acquisition options cannot accompany a corrective recording")
        from corrective_reader import open_corrective_reader
        reader = open_corrective_reader(args.corrective_dir)
        sample = reader.sample(args.corrective_index)
        data_report = {**reader.inventory(), "sample": reader.sample_identity(args.corrective_index)}
    elif args.dataset is not None:
        if any(value is None for value in (args.acquisition, args.registry, args.box_links)):
            raise ValueError("Real data requires --acquisition, --registry and --box-links")
        from provenance import provenance_from_acquisition
        from lerobot_reader import LeRobotPretrainReader
        provenance = provenance_from_acquisition(args.dataset, args.acquisition, args.registry, args.box_links)
        reader = LeRobotPretrainReader(args.dataset, provenance, args.registry, args.box_links,
                                      episode_ids=[args.episode])
        sample = reader.sample(args.episode, args.timestep)
        data_report = {**reader.inventory(), "sample_episode": args.episode, "sample_timestep": args.timestep,
                       "parquet_sha256": reader.load_episode(args.episode).parquet_sha256}
    else:
        if any(value is not None for value in (args.acquisition, args.registry, args.box_links)):
            raise ValueError("Dataset provenance options require --dataset")
        sample = synthetic_sample()

    from transformers import AutoModel, AutoProcessor

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True,
                                             use_fast=False, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                     attn_implementation=args.attn_implementation,
                                     dtype=torch.bfloat16, local_files_only=True).to(device).to(torch.bfloat16)
    # Match deploy/server.py exactly, including its explicit whole-model BF16
    # cast of floating buffers (some RoPE buffers are otherwise initialized FP32).
    # The checkpoint's Python module initializes its own persistent RNG at import.
    torch.manual_seed(args.seed)
    inputs, actions, valid_steps = encode_sample(sample, processor, device)
    bridge = FrozenVLMFlowBridge(model, fp32_trainable=True).train()
    result = bridge(inputs, actions, valid_steps)
    if not torch.isfinite(result["loss"]).item():
        raise AssertionError("Non-finite flow loss")
    result["loss"].backward()
    report = {
        "synthetic_only": args.dataset is None and args.corrective_dir is None, "optimizer_steps": 0, "data": data_report,
        "loss": result["loss"].detach().item(),
        "active_loss_values": int(result["active_values"]),
        "gradients": gradient_report(bridge),
        "trainable_parameters": sum(p.numel() for p in bridge.trainable_parameters()),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "export_reload": {"status": "not_requested"},
    }
    print(json.dumps({"gradient_smoke": "passed", "loss": report["loss"],
                      "peak_cuda_gib": report["peak_cuda_memory_bytes"] / 2**30}), flush=True)

    if args.export_dir is not None:
        # No optimizer update occurred. Deliberately return heads to the
        # released inference dtype before saving and comparing inference.
        model.zero_grad(set_to_none=True)
        for name in TRAINABLE_MODULES:
            getattr(model, name).to(torch.bfloat16)
        # Only heads changed precision for training. The inference startup cast
        # above and after reload handles all nonpersistent buffers identically.
        model.eval()
        torch.manual_seed(args.seed)
        with torch.no_grad():
            reference = model(**inputs, logits_to_keep=1).actions.detach().cpu()
        export_hf(model, processor, args.model, args.export_dir)
        del result, bridge, model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded_processor = AutoProcessor.from_pretrained(
            args.export_dir, trust_remote_code=True, use_fast=False, local_files_only=True)
        restored_inputs, _, _ = encode_sample(sample, reloaded_processor, device)
        check_processor_parity(inputs, restored_inputs)
        reloaded = AutoModel.from_pretrained(
            args.export_dir, trust_remote_code=True,
            attn_implementation=args.attn_implementation,
            dtype=torch.bfloat16, local_files_only=True).to(device).to(torch.bfloat16).eval()
        # Loading custom code can seed RNG; always reset immediately before inference.
        torch.manual_seed(args.seed)
        with torch.no_grad():
            restored = reloaded(**restored_inputs, logits_to_keep=1).actions.detach().cpu()
        torch.testing.assert_close(restored, reference, rtol=0, atol=0)
        report["export_reload"] = {
            "status": "passed", "exact_action_parity": True,
            "processor_inputs_equal": True, "path": str(args.export_dir.resolve()),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {args.report}", flush=True)


if __name__ == "__main__":
    main()
