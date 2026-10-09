#!/usr/bin/env python3
"""Explicit single-GPU pretrain-human training. No jobs run on import."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import time
from collections import OrderedDict
from pathlib import Path

# Must precede CUDA initialization for deterministic cuBLAS behavior.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

try:
    from .bridge import FrozenVLMFlowBridge, encode_sample
    from .build_manifest import human300_tasks
    from .lerobot_reader import LeRobotPretrainReader
    from .provenance import provenance_from_acquisition, sha256_file
    from .train_core import SingleDeviceTrainer
except ImportError:
    from bridge import FrozenVLMFlowBridge, encode_sample
    from build_manifest import human300_tasks
    from lerobot_reader import LeRobotPretrainReader
    from provenance import provenance_from_acquisition, sha256_file
    from train_core import SingleDeviceTrainer


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checkpoint_identity(source):
    """Pin full source weights and processor/custom-code assets, not only its path."""
    source = Path(source).resolve()
    index = source / "model.safetensors.index.json"
    if index.exists():
        weights = set(json.loads(index.read_text())["weight_map"].values())
    elif (source / "model.safetensors").is_file():
        weights = {"model.safetensors"}
    else:
        raise ValueError("Expected a complete safetensors HF checkpoint")
    metadata = {p.name for p in source.iterdir() if p.is_file() and p.suffix in {".py", ".json", ".jinja", ".txt", ".model"}}
    hashes = {}
    for name in sorted(weights | metadata):
        if Path(name).name != name or not (source / name).is_file():
            raise ValueError(f"Invalid/missing checkpoint asset: {name}")
        hashes[name] = sha256_file(source / name)
    return hashes


def runtime_identity():
    versions = {}
    for distribution in ("torch", "transformers", "flash-attn", "numpy", "pandas", "pyarrow", "decord"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "distribution_metadata_unavailable"
    return {"python": platform.python_version(), "packages": versions,
            "torch_cuda": torch.version.cuda, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "strict_deterministic_algorithms": True, "tf32": False,
            "attention_implementation": "flash_attention_2"}


class SamplePool:
    """Explicit task-uniform, episode-uniform, frame-uniform single samples."""

    def __init__(self, readers, processor, device, seed):
        if not readers or len({r.provenance["task"] for r in readers}) != len(readers):
            raise ValueError("One explicitly identified dataset per task is required")
        self.readers, self.processor, self.device = readers, processor, device
        self.random = random.Random(seed)
        self.draws = 0
        self.observed_parquet_hashes = {}
        self.active_readers = OrderedDict()

    def next_batch(self):
        reader_index = self.random.randrange(len(self.readers))
        reader = self.readers[reader_index]
        self.active_readers[reader_index] = True
        self.active_readers.move_to_end(reader_index)
        while len(self.active_readers) > 2:
            old, _ = self.active_readers.popitem(last=False)
            self.readers[old].clear_caches()
        episode_id = reader.episode_ids[self.random.randrange(len(reader.episode_ids))]
        episode = reader.load_episode(episode_id)
        timestep = self.random.randrange(len(episode.states))
        key = f"{reader.provenance['task']}/{episode_id}"
        if key in self.observed_parquet_hashes and self.observed_parquet_hashes[key] != episode.parquet_sha256:
            raise ValueError(f"Previously observed parquet changed: {key}")
        self.observed_parquet_hashes[key] = episode.parquet_sha256
        sample = reader.sample(episode_id, timestep)
        inputs, actions, valid = encode_sample(sample, self.processor, self.device)
        self.draws += 1
        return inputs, actions, valid, {"task": reader.provenance["task"], "episode": episode_id,
                                       "timestep": timestep, "draw": self.draws}

    def state_dict(self):
        return {"rng": self.random.getstate(), "draws": self.draws,
                "observed_parquet_hashes": dict(self.observed_parquet_hashes)}

    def load_state_dict(self, state):
        self.random.setstate(state["rng"])
        self.draws = int(state["draws"])
        self.observed_parquet_hashes = dict(state["observed_parquet_hashes"])


def read_manifest(path):
    value = json.loads(Path(path).read_text())
    if value.get("format_version") != 1 or value.get("purpose") != "pretrain_human_training":
        raise ValueError("Require an explicit pretrain_human_training manifest v1")
    if value.get("sampler") != "task_uniform_episode_uniform_frame":
        raise ValueError("Unsupported sampling policy")
    if not isinstance(value.get("seed"), int) or not 0 <= value["seed"] < 2**32:
        raise ValueError("Require a uint32 fixed seed")
    if not isinstance(value.get("gradient_accumulation"), int) or value["gradient_accumulation"] <= 0:
        raise ValueError("Require positive gradient_accumulation")
    if not math.isfinite(value.get("max_grad_norm", float("nan"))) or value["max_grad_norm"] <= 0:
        raise ValueError("Require finite positive max_grad_norm")
    optimizer = value["optimizer"]
    if optimizer.get("type") != "AdamW":
        raise ValueError("Only explicit AdamW is implemented")
    for key in ("lr", "eps", "weight_decay"):
        if not math.isfinite(optimizer[key]) or optimizer[key] < 0 or (key != "weight_decay" and optimizer[key] == 0):
            raise ValueError(f"Invalid optimizer {key}")
    if len(optimizer["betas"]) != 2 or any(not 0 <= beta < 1 for beta in optimizer["betas"]):
        raise ValueError("Invalid optimizer betas")
    if not value.get("datasets"):
        raise ValueError("Require explicit dataset/episode selection")
    if value.get("microbatch") != 1:
        raise ValueError("This trainer supports microbatch=1 only")
    if len({item["task"] for item in value["datasets"]}) != len(value["datasets"]):
        raise ValueError("Duplicate task entries in training manifest")
    if value.get("scope") == "human300":
        if {item["task"] for item in value["datasets"]} != set(human300_tasks(value["registry"])):
            raise ValueError("Human300 scope requires all 300 registry tasks exactly once")
    elif value.get("scope") != "explicit_pilot_tasks":
        raise ValueError("Unsupported or missing training scope")
    if type(value.get("planned_optimizer_steps")) is not int or value["planned_optimizer_steps"] <= 0:
        raise ValueError("Require positive planned_optimizer_steps")
    for item in value["datasets"]:
        ids = item.get("episode_ids", [])
        if not ids or len(ids) > 100 or any(type(index) is not int or index < 0 for index in ids) or len(set(ids)) != len(ids):
            raise ValueError("Human-only training selects at most 100 explicit episodes per task")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True, help="Total optimizer steps, including resumed steps")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", type=Path, help="Trusted local step-boundary checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reserve-gib", type=float, default=50)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.checkpoint_every <= 0 or args.reserve_gib < 0:
        raise ValueError("Invalid run limits")
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("This entry point requires an explicitly scheduled CUDA GPU")
    manifest = read_manifest(args.manifest)
    if args.max_steps > manifest["planned_optimizer_steps"]:
        raise ValueError("Requested optimizer steps exceed the explicitly planned experiment")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".training.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, manifest)


def run(args, manifest):
    from transformers import AutoModel, AutoProcessor

    job_file = args.output / "run_identity.json"
    existing_checkpoints = list(args.output.glob("checkpoint-step-*.pt"))
    if args.resume is None and existing_checkpoints:
        raise FileExistsError("Existing run detected; use a new output or an explicit checkpoint to resume")
    if args.resume is not None and not job_file.exists():
        raise ValueError("Resume requires the original output directory and run_identity.json")
    if args.resume is not None and (not existing_checkpoints or args.resume.resolve() != max(existing_checkpoints).resolve()):
        raise ValueError("Resume in this output directory must use its latest completed checkpoint")
    readers = []
    for item in manifest["datasets"]:
        if item.get("role") != "train":
            raise ValueError("Only datasets explicitly assigned role=train may enter this run")
        provenance = provenance_from_acquisition(item["root"], item["acquisition"],
                                                manifest["registry"], manifest["box_links"])
        if item["task"] != provenance["task"]:
            raise ValueError("Manifest task disagrees with acquisition provenance")
        if item.get("archive_sha256") != provenance["archive_sha256"]:
            raise ValueError("Manifest archive SHA256 disagrees with acquisition provenance")
        readers.append(LeRobotPretrainReader(item["root"], provenance, manifest["registry"],
                                            manifest["box_links"], episode_ids=item["episode_ids"]))
    active_code = ("bridge.py", "layout.py", "lerobot_reader.py", "provenance.py", "train_core.py", "train_single.py", "build_manifest.py")
    identity = {"manifest_sha256": _json_hash(manifest),
                "base_checkpoint": checkpoint_identity(manifest["base_model"]),
                "training_code": {name: sha256_file(Path(__file__).parent / name) for name in active_code},
                "data": [reader.inventory() for reader in readers],
                "runtime": runtime_identity(),
                "precision": "FP32 trainable masters, BF16 CUDA autocast, frozen VLM",
                "optimizer": manifest["optimizer"], "lr_schedule": "constant"}
    if job_file.exists():
        if json.loads(job_file.read_text())["identity"] != identity:
            raise ValueError("Data, source checkpoint, code or optimizer changed since this run began")
    else:
        job_file.write_text(json.dumps({"identity": identity, "manifest": manifest}, indent=2) + "\n")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Strict mode surfaces unsupported nondeterministic operators instead of
    # silently promising reproducibility; FlashAttention support is GPU-tested later.
    torch.use_deterministic_algorithms(True)
    processor = AutoProcessor.from_pretrained(manifest["base_model"], trust_remote_code=True,
                                             use_fast=False, local_files_only=True)
    model = AutoModel.from_pretrained(manifest["base_model"], trust_remote_code=True,
                                     attn_implementation="flash_attention_2", dtype=torch.bfloat16,
                                     local_files_only=True).to(device).to(torch.bfloat16)
    bridge = FrozenVLMFlowBridge(model, fp32_trainable=True)
    optimizer_config = manifest["optimizer"]
    optimizer = torch.optim.AdamW(bridge.trainable_parameters(), lr=optimizer_config["lr"],
                                 betas=tuple(optimizer_config["betas"]), eps=optimizer_config["eps"],
                                 weight_decay=optimizer_config["weight_decay"], foreach=False)
    # Restore deterministic seeds after the custom checkpoint module's import seeding.
    random.seed(manifest["seed"])
    np.random.seed(manifest["seed"])
    torch.manual_seed(manifest["seed"])
    provider = SamplePool(readers, processor, device, manifest["seed"])
    trainer = SingleDeviceTrainer(bridge, optimizer, provider, identity=identity, device=device,
                                  accumulate=manifest["gradient_accumulation"], max_grad_norm=manifest["max_grad_norm"])
    if args.resume is not None:
        trainer.load(args.resume)
    if trainer.global_step >= args.max_steps:
        print(f"No work: checkpoint already has {trainer.global_step} optimizer steps.")
        return
    # FP32 masters + two Adam moment tensors, with 10% serialization headroom.
    estimated_checkpoint = int(sum(p.numel() for p in bridge.trainable_parameters()) * 12 * 1.1)
    if shutil.disk_usage(args.output).free < estimated_checkpoint + args.reserve_gib * 2**30:
        raise OSError("Insufficient disk reserve for even one checkpoint; no optimizer step was taken")
    def save_checkpoint():
        if shutil.disk_usage(args.output).free < estimated_checkpoint + args.reserve_gib * 2**30:
            raise OSError("Checkpoint would violate disk reserve; previous checkpoint remains intact")
        path = args.output / f"checkpoint-step-{trainer.global_step:08d}.pt"
        trainer.save(path)
        print(json.dumps({"checkpoint": str(path), "step": trainer.global_step}), flush=True)

    with (args.output / "metrics.jsonl").open("a") as log:
        log.write(json.dumps({"event": "start_or_resume", "step": trainer.global_step}) + "\n")
        log.flush()
        while trainer.global_step < args.max_steps:
            begin = time.monotonic()
            result = trainer.step()
            result["wall_seconds"] = time.monotonic() - begin
            result["lr"] = optimizer.param_groups[0]["lr"]
            log.write(json.dumps(result) + "\n")
            log.flush()
            print(json.dumps({key: value for key, value in result.items() if key != "samples"}), flush=True)
            if trainer.global_step % args.checkpoint_every == 0 or trainer.global_step == args.max_steps:
                save_checkpoint()


if __name__ == "__main__":
    main()
