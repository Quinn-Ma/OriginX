"""Explicit single-GPU 3:1 Human300/self-imitation candidate, at most 200 updates.

Invoke as a module from a pinned parent directory containing training_bridge/
and recovery/. Nothing launches on import. The released base is always loaded
fresh; only this candidate's own step-boundary checkpoint may be resumed.
"""

import argparse
import copy
import fcntl
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .bridge import FrozenVLMFlowBridge
from .mixed_manifest import build_sample_provider
from .provenance import sha256_file
from .train_core import SingleDeviceTrainer
from .train_single import _json_hash, runtime_identity


def code_identity():
    import recovery

    training = Path(__file__).resolve().parent
    collection = Path(recovery.__file__).resolve().parent
    return ({path.name: sha256_file(path) for path in sorted(training.glob("*.py"))},
            {path.name: sha256_file(path) for path in sorted(collection.glob("*.py"))})


def effective_manifest(source):
    """Flatten recipe keys for the same checkpoint/export schema as the control."""
    result = copy.deepcopy(source)
    recipe = result["recipe"]
    if (result.get("initialization") != "same_released_base_as_human_control" or
            recipe.get("microbatch") != 1 or recipe.get("gradient_accumulation") != 8 or
            recipe.get("planned_optimizer_steps") != 200):
        raise ValueError("Require the released-base, accumulation-eight, 200-update matched control recipe")
    for name, value in recipe.items():
        if name in result and result[name] != value:
            raise ValueError(f"Conflicting mixed recipe field: {name}")
        result[name] = value
    return result


def check_run_boundary(output, resume, max_steps):
    if type(max_steps) is not int or not 1 <= max_steps <= 200:
        raise ValueError("Mixed run must remain within 1..200 total optimizer updates")
    checkpoints = list(output.glob("checkpoint-step-*.pt"))
    if resume is None and checkpoints:
        raise FileExistsError("Existing candidate checkpoints require explicit resume or a new output directory")
    if resume is not None:
        if not (output / "run_identity.json").is_file():
            raise ValueError("Resume requires the original candidate output directory")
        if not checkpoints or resume.resolve() != max(checkpoints).resolve():
            raise ValueError("Resume must use this candidate's latest completed checkpoint")


def run(args):
    from transformers import AutoModel, AutoProcessor

    check_run_boundary(args.output, args.resume, args.max_steps)
    before_manifest = sha256_file(args.manifest)
    # Gate all real source data before loading a model or taking any optimizer step.
    device = torch.device(args.device)
    provider = build_sample_provider(args.manifest, processor=None, device=device)
    if sha256_file(args.manifest) != before_manifest:
        raise ValueError("Mixed experiment manifest changed during source validation")
    source = json.loads(args.manifest.read_text())
    manifest = effective_manifest(source)
    training_code, recovery_code = code_identity()
    torch.cuda.set_device(device)
    runtime = runtime_identity()
    control = json.loads(Path(source["human_run_identity"]).read_text())
    control_runtime = control["identity"].get("runtime")
    if control_runtime is not None and control_runtime != runtime:
        raise ValueError("Candidate runtime differs from the pinned Human-only control runtime")
    identity = {
        "manifest_sha256": _json_hash(manifest), "source_mixed_manifest_sha256": before_manifest,
        "base_checkpoint": source["base_checkpoint"], "training_code": training_code,
        "recovery_code": recovery_code,
        "data": [reader.inventory() for reader in provider.providers["human"].readers],
        "corrective_data": [reader.inventory() for reader in provider.providers["corrective"].readers],
        "runtime": runtime,
        "control_runtime_comparison": "identical" if control_runtime is not None else "not_recorded_in_control_identity",
        "device_model": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "precision": "FP32 trainable masters, BF16 CUDA autocast, frozen VLM",
        "optimizer": manifest["optimizer"], "lr_schedule": "constant",
        "mixture": source["mixture"],
    }
    job_path = args.output / "run_identity.json"
    if job_path.exists():
        if json.loads(job_path.read_text())["identity"] != identity:
            raise ValueError("Candidate data, source, code, runtime or optimizer identity changed")
    else:
        job_path.write_text(json.dumps({"identity": identity, "manifest": manifest,
                                        "source_manifest_path": str(args.manifest.resolve())}, indent=2) + "\n")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    processor = AutoProcessor.from_pretrained(manifest["base_model"], trust_remote_code=True,
                                             use_fast=False, local_files_only=True)
    model = AutoModel.from_pretrained(manifest["base_model"], trust_remote_code=True,
                                     attn_implementation="flash_attention_2", dtype=torch.bfloat16,
                                     local_files_only=True).to(device).to(torch.bfloat16)
    bridge = FrozenVLMFlowBridge(model, fp32_trainable=True)
    for child in provider.providers.values():
        child.processor = processor
    config = manifest["optimizer"]
    optimizer = torch.optim.AdamW(bridge.trainable_parameters(), lr=config["lr"], betas=tuple(config["betas"]),
                                 eps=config["eps"], weight_decay=config["weight_decay"], foreach=False)
    # HF custom-code import seeds globally; the declared seed must be restored last.
    random.seed(manifest["seed"])
    np.random.seed(manifest["seed"])
    torch.manual_seed(manifest["seed"])
    trainer = SingleDeviceTrainer(bridge, optimizer, provider, identity=identity, device=device,
                                  accumulate=manifest["gradient_accumulation"], max_grad_norm=manifest["max_grad_norm"])
    if args.resume is not None:
        trainer.load(args.resume)
        if (args.resume.name != f"checkpoint-step-{trainer.global_step:08d}.pt" or
                not 1 <= trainer.global_step <= manifest["planned_optimizer_steps"]):
            raise ValueError("Resume checkpoint filename/step violates the candidate's bounded plan")
    if trainer.global_step >= args.max_steps:
        print(f"No work: candidate already has {trainer.global_step} optimizer steps.", flush=True)
        return
    estimated_bytes = int(sum(parameter.numel() for parameter in bridge.trainable_parameters()) * 12 * 1.1)
    def require_checkpoint_space():
        if shutil.disk_usage(args.output).free < estimated_bytes + args.reserve_gib * 2**30:
            raise OSError("Candidate checkpoint would violate the disk reserve")
    require_checkpoint_space()
    with (args.output / "metrics.jsonl").open("a") as log:
        log.write(json.dumps({"event": "start_or_resume", "step": trainer.global_step, "mixture": "HHHC"}) + "\n")
        log.flush()
        while trainer.global_step < args.max_steps:
            begin = time.monotonic()
            event = trainer.step()
            sources = [item["source"] for item in event["samples"]]
            if sources.count("human") != 6 or sources.count("corrective") != 2:
                raise RuntimeError("Accumulated candidate update did not contain exactly 6 human and 2 corrective samples")
            event.update(wall_seconds=time.monotonic() - begin, lr=optimizer.param_groups[0]["lr"])
            log.write(json.dumps(event) + "\n")
            log.flush()
            print(json.dumps({key: value for key, value in event.items() if key != "samples"}), flush=True)
            if trainer.global_step % args.checkpoint_every == 0 or trainer.global_step == args.max_steps:
                require_checkpoint_space()
                checkpoint = args.output / f"checkpoint-step-{trainer.global_step:08d}.pt"
                trainer.save(checkpoint)
                print(json.dumps({"checkpoint": str(checkpoint), "step": trainer.global_step}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True, help="Total candidate updates including resumed steps, at most 200")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reserve-gib", type=float, default=50)
    args = parser.parse_args()
    if (args.checkpoint_every < 1 or not math.isfinite(args.reserve_gib) or args.reserve_gib < 0 or
            not 1 <= args.max_steps <= 200):
        raise ValueError("Invalid bounded run/checkpoint settings")
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("An explicitly scheduled CUDA device is required")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".training.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
