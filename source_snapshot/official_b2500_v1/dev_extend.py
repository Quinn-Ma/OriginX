#!/usr/bin/env python3
"""Evaluate one new model against immutable copies of the fixed dev reference.

Only the new model is rendered. The frozen dev_paired.py episode implementation
and summary function are used unchanged. No resume, retries, episode filtering,
model-server launch, remote execution, or partial-score publication.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

try:
    from . import dev_paired as original
except ImportError:
    import dev_paired as original


FROZEN_RUNNER_SHA256 = "de47225b8fe5146d997d2f4936d87d89b7bffcc4316ed29c1839945a19759c29"
CAMERAS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")
RUNTIME_PACKAGES = ("torch", "numpy", "mujoco", "robocasa", "robosuite", "transformers")
PROTOCOL = "robocasa-recovery-rng-v1"


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _int(value, low, high):
    return type(value) is int and low <= value <= high


def _stable_bytes(path):
    before = original.stamp(path)
    content = Path(path).read_bytes()
    if original.stamp(path) != before:
        raise ValueError(f"Source changed while reading: {path}")
    return content, hashlib.sha256(content).hexdigest(), before


def _source_files(manifest, runner):
    repo, recovery = Path(manifest["repo"]), Path(manifest["recovery_parent"])
    return {
        repo / "eval_robocasa365/entry.py": manifest["official_entry_sha256"],
        repo / "deploy/client.py": manifest["official_client_sha256"],
        repo / "deploy/server.py": manifest["official_server_sha256"],
        recovery / "recovery/policy.py": manifest["recovery_policy_sha256"],
        recovery / "recovery/seeded_server.py": manifest["recovery_server_sha256"],
        Path(manifest["protocol"]["registry"]): manifest["protocol"]["registry_sha256"],
        runner: manifest["runner_sha256"],
    }


def validate_reference_run(directory, expected_manifest_sha256, reference_name="base", *, runner=None):
    """Pin the original protocol, frozen evaluator, sources and complete base assets."""
    root = Path(directory).resolve(strict=True)
    runner = Path(runner or original.__file__).resolve(strict=True)
    if not _digest(expected_manifest_sha256):
        raise ValueError("An explicit SHA256 of the original run manifest is required")
    payload, digest, _ = _stable_bytes(root / "run_manifest.json")
    if digest != expected_manifest_sha256:
        raise ValueError("Original run manifest SHA256 differs from the declared reference")
    manifest = json.loads(payload)
    if (manifest.get("format_version") != 1 or manifest.get("reference") != reference_name
            or manifest.get("runner_sha256") != FROZEN_RUNNER_SHA256
            or original.sha(runner) != FROZEN_RUNNER_SHA256):
        raise ValueError("Reference is not the fixed original development run/evaluator")
    protocol = original.task_protocol(Path(manifest["package_root"]))
    if manifest.get("protocol") != protocol:
        raise ValueError("Original reference protocol or task registry changed")
    models = manifest.get("models", [])
    names = [model["name"] for model in models]
    if len(models) < 2 or len(set(names)) != len(names) or names.count(reference_name) != 1:
        raise ValueError("Original reference model identity is ambiguous")
    model = next(item for item in models if item["name"] == reference_name)
    # No reference inference is launched; -1 avoids imposing an irrelevant new
    # renderer placement restriction on an already-recorded model.
    observed = original.model_record(model, renderer_gpu=-1)
    if any(observed[key] != model.get(key) for key in ("asset_sha256", "asset_stamps", "model_hash", "model_path")):
        raise ValueError("Reference checkpoint assets or model hash differ from its original manifest")
    sources = _source_files(manifest, runner)
    if any(not _digest(digest) or original.sha(path) != digest for path, digest in sources.items()):
        raise ValueError("Reference evaluator, simulator registry or RNG adapter source changed")
    return {"root": root, "manifest": manifest, "manifest_sha256": expected_manifest_sha256,
            "model": copy.deepcopy(model), "runner": runner, "sources": sources}


def assert_reference_unchanged(reference):
    if original.sha(reference["root"] / "run_manifest.json") != reference["manifest_sha256"]:
        raise ValueError("Original run manifest changed after pinning")
    if any(original.sha(path) != digest for path, digest in reference["sources"].items()):
        raise ValueError("Pinned reference protocol/evaluator source changed")
    original.assert_assets(reference["model"])


def validate_episode_record(record, model, task, seed, manifest):
    """Check a normal completed official episode, without filtering success or pairing."""
    horizon = manifest["protocol"]["horizons"][task]
    if any(record.get(key) != value for key, value in
           {"model": model["name"], "model_hash": model["model_hash"], "task": task,
            "seed": seed, "global_seed": seed, "policy_seed": seed, "horizon": horizon}.items()):
        raise ValueError("Episode model, task, seed or full official horizon differs")
    if (type(record.get("success")) is not bool or not _int(record.get("steps"), 1, horizon)
            or record.get("error") not in (None, "")):
        raise ValueError("Episode contains an error or invalid outcome/length")
    official = record.get("official_episode", {})
    if (official.get("episode") != 0 or official.get("global_episode_index") != 0
            or official.get("seed") != seed or type(official.get("success")) is not bool
            or official["success"] != record["success"] or official.get("steps") != record["steps"]):
        raise ValueError("Official episode result disagrees with the reference record")
    initial = record.get("initial_observation", {})
    if (set(initial) != {"rgb", "proprioception", "physics_state", "instruction"}
            or set(initial["rgb"]) != set(CAMERAS)
            or any(not _digest(value) for value in initial["rgb"].values())
            or not _digest(initial["proprioception"]) or not _digest(initial["physics_state"])
            or not isinstance(initial["instruction"], str) or not initial["instruction"].strip()):
        raise ValueError("Episode lacks the original observation/physics pairing fingerprints")
    hello, ack = record.get("server_identity", {}), record.get("rng_ack", {})
    for message in (hello, ack):
        if (message.get("ok") is not True or message.get("protocol") != PROTOCOL
                or message.get("exclusive_connection") is not True
                or not message.get("server_instance") or not message.get("connection_id")
                or Path(message.get("model_path", "")).resolve() != Path(model["model_path"])
                or message.get("model_config_sha256") != model["asset_sha256"]["config.json"]
                or message.get("official_server_sha256") != manifest["official_server_sha256"]
                or message.get("recovery_server_sha256") != manifest["recovery_server_sha256"]):
            raise ValueError("Episode server identity differs from the declared model/code")
    if (ack.get("seed") != seed or ack.get("requests_since_reset") != 0
            or ack.get("server_instance") != hello["server_instance"]
            or ack.get("connection_id") != hello["connection_id"]
            or not _digest(ack.get("torch_cpu_rng_sha256"))
            or not isinstance(ack.get("torch_cuda_rng_sha256"), list) or not ack["torch_cuda_rng_sha256"]
            or any(not _digest(value) for value in ack["torch_cuda_rng_sha256"])):
        raise ValueError("Episode lacks the actual same-connection policy RNG-reset evidence")
    return record


def read_reference_episode(reference, task, seed):
    """Return None only while that exact original episode has not completed."""
    if task not in original.TASKS or seed not in original.SEEDS:
        raise ValueError("Reference episode is outside the fixed development plan")
    assert_reference_unchanged(reference)
    directory = reference["root"] / "episodes" / task / str(seed) / reference["model"]["name"]
    if directory.exists() and not directory.resolve().is_relative_to(reference["root"]):
        raise ValueError("Reference episode directory escapes the pinned run")
    exit_path = directory / "process_exit.json"
    if not exit_path.exists():
        return None
    if not exit_path.resolve().is_relative_to(reference["root"]):
        raise ValueError("Reference process exit artifact escapes the pinned run")
    exit_bytes, exit_hash, exit_stamp = _stable_bytes(exit_path)
    exit_record = json.loads(exit_bytes)
    if type(exit_record.get("returncode")) is not int or exit_record["returncode"] != 0:
        raise ValueError("Reference episode exited with an error; no replacement or retry")
    files = {"process_exit.json": (exit_bytes, exit_hash, exit_stamp)}
    for filename in ("result.json", "process.log"):
        path = directory / filename
        if not path.is_file() or not path.resolve().is_relative_to(reference["root"]):
            raise ValueError(f"Completed reference episode is missing an in-run artifact: {filename}")
        files[filename] = _stable_bytes(path)
    record = validate_episode_record(json.loads(files["result.json"][0]), reference["model"], task, seed, reference["manifest"])
    if any(original.stamp(directory / name) != state[2] for name, state in files.items()):
        raise ValueError("Reference episode changed during validation")
    return {"record": record, "directory": directory, "files": files}


def wait_reference_episode(reference, task, seed, *, timeout_seconds=5400, poll_seconds=30,
                           _clock=time.monotonic, _sleep=time.sleep, _read=read_reference_episode):
    if not 0 <= timeout_seconds <= 5400 or not 0 < poll_seconds <= 30:
        raise ValueError("Reference wait must be bounded by 90 minutes with polls at most 30 seconds")
    deadline = _clock() + timeout_seconds
    while True:
        result = _read(reference, task, seed)
        if result is not None:
            return result
        remaining = deadline - _clock()
        if remaining <= 0:
            raise TimeoutError(f"Reference episode did not complete before its deadline: {task}/{seed}")
        _sleep(min(poll_seconds, remaining))


def copy_reference_episode(reference, snapshot, destination):
    """Copy bytes, never link mutable source artifacts; pin all source/copy hashes."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for filename, (content, digest, source_stamp) in snapshot["files"].items():
        source = snapshot["directory"] / filename
        if original.stamp(source) != source_stamp or original.sha(source) != digest:
            raise ValueError("Reference episode changed before immutable copy")
        target = destination / filename
        target.write_bytes(content)
        if original.sha(target) != digest:
            raise ValueError("Copied reference bytes differ from source")
        target.chmod(0o444)
        files[filename] = {"source": str(source.resolve()), "sha256": digest, "source_stamp": source_stamp}
    provenance = {"format_version": 1, "kind": "copied_original_reference_episode",
                  "reference_run": str(reference["root"]), "reference_model": reference["model"]["name"],
                  "reference_run_manifest_sha256": reference["manifest_sha256"],
                  "hash_origin": "pinned_when_original_process_exit_zero_was_observed",
                  "files": files}
    original.atomic_json(destination / "source_provenance.json", provenance)
    (destination / "source_provenance.json").chmod(0o444)
    return {"record": snapshot["record"], "directory": destination, "provenance": provenance,
            "provenance_sha256": original.sha(destination / "source_provenance.json")}


def assert_reference_copy(copy_record, reference):
    assert_reference_unchanged(reference)
    directory, provenance = copy_record["directory"], copy_record["provenance"]
    if original.sha(directory / "source_provenance.json") != copy_record["provenance_sha256"]:
        raise ValueError("Copied reference provenance changed")
    for filename, evidence in provenance["files"].items():
        if (original.sha(directory / filename) != evidence["sha256"]
                or original.stamp(Path(evidence["source"])) != evidence["source_stamp"]
                or original.sha(Path(evidence["source"])) != evidence["sha256"]):
            raise ValueError("Pinned reference source or immutable copy changed")


def _terminate_owned(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def run(args):
    if args.output.exists():
        raise FileExistsError("Extension output must be new; resume and overwrite are unsupported")
    if args.renderer_gpu < 0 or not 0 <= args.wait_seconds <= 5400 or not 0 < args.poll_seconds <= 30:
        raise ValueError("Invalid renderer or bounded reference wait")
    reference = validate_reference_run(args.reference_run, args.reference_manifest_sha256, args.reference)
    runtime = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    if runtime != reference["manifest"]["simulator_packages"]:
        raise ValueError("Extension simulator/runtime packages differ from the reference run")
    plan = json.loads(args.plan.read_text())
    if plan.get("format_version") != 1 or len(plan.get("models", [])) != 1:
        raise ValueError("Extension plan must specify exactly one new model")
    model = original.model_record(plan["models"][0], args.renderer_gpu)
    if (model["name"] in {item["name"] for item in reference["manifest"]["models"]}
            or model["model_hash"] == reference["model"]["model_hash"]
            or model["server_port"] == reference["model"]["server_port"]):
        raise ValueError("Require a new model name/checkpoint and a separate dedicated inference port")
    if any(args.output.resolve().is_relative_to(path) for path in
           (reference["root"], Path(reference["model"]["model_path"]), Path(model["model_path"]))):
        raise ValueError("Extension output must be outside reference run and both model snapshots")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    code = output / "code"
    code.mkdir()
    runner = code / "dev_paired.py"
    runner.write_bytes(reference["runner"].read_bytes())
    runner.chmod(0o444)
    (code / "dev_extend.py").write_bytes(Path(__file__).read_bytes())
    (code / "dev_extend.py").chmod(0o444)
    extension_sha = original.sha(__file__)
    original.atomic_json(output / "reference_run_manifest.json", reference["manifest"])
    # The parsed manifest is recorded for readability. The original bytes are
    # also retained because its user-supplied SHA pins formatting as well.
    (output / "reference_run_manifest.original.json").write_bytes((reference["root"] / "run_manifest.json").read_bytes())
    if original.sha(output / "reference_run_manifest.original.json") != reference["manifest_sha256"]:
        raise ValueError("Original manifest changed while copying")
    manifest = copy.deepcopy(reference["manifest"])
    manifest.update(models=[reference["model"], model], reference=args.reference, renderer_gpu=args.renderer_gpu,
                    simulator_packages=runtime, created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    extension={"kind": "one_new_model_with_reused_fixed_reference_v1",
                               "reference_run": str(reference["root"]), "reference_model": args.reference,
                               "reference_run_manifest_sha256": reference["manifest_sha256"],
                               "extension_runner_sha256": extension_sha, "new_model_plan_sha256": original.sha(args.plan),
                               "reference_wait_seconds_per_episode": args.wait_seconds,
                               "reference_rendered_again": False, "resume_supported": False})
    original.atomic_json(output / "run_manifest.json", manifest)
    manifest_sha = original.sha(output / "run_manifest.json")
    status = {"status": "prepared", "copied_reference_episodes": 0, "new_model_episodes": 0,
              "expected_pairs": len(original.TASKS) * len(original.SEEDS)}
    def update(**fields):
        status.update(fields, updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        original.atomic_json(output / "extension_status.json", status)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    lock_path = Path(tempfile.gettempdir()) / f"robocasa-dev-render-{os.getuid()}-gpu-{args.renderer_gpu}.lock"
    child, reference_copies, records, candidates = None, [], [], []
    env = dict(os.environ, PYTHONHASHSEED="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES=str(args.renderer_gpu), MUJOCO_EGL_DEVICE_ID=str(args.renderer_gpu),
               MUJOCO_GL="egl", TOKENIZERS_PARALLELISM="false")
    try:
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for task in original.TASKS:
                for seed in original.SEEDS:
                    update(status="waiting_for_reference", task=task, seed=seed)
                    snapshot = wait_reference_episode(reference, task, seed, timeout_seconds=args.wait_seconds,
                                                      poll_seconds=args.poll_seconds)
                    copied = copy_reference_episode(reference, snapshot, output / "episodes" / task / str(seed) / args.reference)
                    reference_copies.append(copied)
                    records.append(copied["record"])
                    update(copied_reference_episodes=len(reference_copies))
                    if (original.sha(__file__) != extension_sha or original.sha(runner) != FROZEN_RUNNER_SHA256
                            or original.sha(output / "run_manifest.json") != manifest_sha):
                        raise ValueError("Frozen extension/evaluator/run manifest changed")
                    assert_reference_copy(copied, reference)
                    original.assert_assets(model)
                    if original.gpu_pids(args.renderer_gpu):
                        raise RuntimeError("Renderer GPU is occupied; refusing concurrent evaluation")
                    destination = output / "episodes" / task / str(seed) / model["name"]
                    destination.mkdir(parents=True, exist_ok=False)
                    command = [sys.executable, "-u", str(runner), "episode", "--output", str(output),
                               "--model", model["name"], "--task", task, "--seed", str(seed)]
                    update(status="running_new_model", model=model["name"], command=command)
                    with (destination / "process.log").open("xb") as log:
                        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                        try:
                            returncode = child.wait()
                        except BaseException:
                            _terminate_owned(child)
                            raise
                    original.atomic_json(destination / "process_exit.json", {"returncode": returncode})
                    if returncode:
                        raise RuntimeError(f"New-model episode exited {returncode}; artifacts preserved; no retry")
                    content, digest, _ = _stable_bytes(destination / "result.json")
                    record = validate_episode_record(json.loads(content), model, task, seed, manifest)
                    candidates.append((destination / "result.json", digest))
                    records.append(record)
                    update(new_model_episodes=len(candidates), status="episode_complete")
                    print(json.dumps({key: record[key] for key in ("model", "task", "seed", "success", "steps")}), flush=True)
            for copied in reference_copies:
                assert_reference_copy(copied, reference)
            original.assert_assets(model)
            if (original.sha(output / "run_manifest.json") != manifest_sha or original.sha(__file__) != extension_sha
                    or original.sha(runner) != FROZEN_RUNNER_SHA256
                    or any(original.sha(path) != digest for path, digest in candidates)):
                raise ValueError("Pinned candidate results, run manifest or sources changed before summary")
            summary = original.paired_summary(records, [args.reference, model["name"]], args.reference)
            original.atomic_json(output / "summary.json", summary)
            update(status="complete", pairs=len(original.TASKS) * len(original.SEEDS))
            print(json.dumps(summary), flush=True)
    except BaseException as error:
        _terminate_owned(child)
        update(status="failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True,
                        help="Original base-control200-paired-v1 directory, possibly still running")
    parser.add_argument("--reference-manifest-sha256", required=True)
    parser.add_argument("--reference", default="base")
    parser.add_argument("--plan", type=Path, required=True, help="format_version 1, models list containing one NEW model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--renderer-gpu", type=int, default=5)
    parser.add_argument("--wait-seconds", type=float, default=5400)
    parser.add_argument("--poll-seconds", type=float, default=30)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
