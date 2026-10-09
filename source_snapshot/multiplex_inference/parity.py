"""Explicit GPU engineering gate: one model, serial reference then two sockets.

Uses two exact first_request.{json,npz} fixtures from the official EvalClient.
Requires different token lengths and interleaves both input sequences, comparing
every full60D action tensor and all four kinds of per-client RNG state exactly.
The temporary server exits after the gate. This is not a rollout or speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np

from .client import WireClient
from .core import RngBank, SerialExecutor, official_forward, validate_request
from .server import checkpoint_identity, load_official_server, serve, sha256


def load_request_fixture(metadata_path, torch):
    metadata_path = Path(metadata_path).resolve(strict=True)
    metadata = json.loads(metadata_path.read_text())
    fields = metadata["fields"]
    descriptor_hash = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (metadata.get("format_version") != 1
            or metadata.get("capture_point") != "client_before_serialization_and_server_dtype_conversion"
            or descriptor_hash != metadata.get("request_sha256")):
        raise ValueError("Unexpected or changed official request capture")
    dtypes = {str(getattr(torch, name)): getattr(torch, name) for name in
              ("bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64")}
    request = {}
    with np.load(metadata_path.with_suffix(".npz"), allow_pickle=False) as arrays:
        for key, descriptor in fields.items():
            if descriptor["kind"] == "json":
                request[key] = descriptor["value"]
                continue
            if descriptor["kind"] != "torch.Tensor" or descriptor["dtype"] not in dtypes:
                raise ValueError("Expected original Torch tensor or JSON field")
            raw = np.asarray(arrays[descriptor["raw_key"]])
            if raw.dtype != np.uint8 or raw.ndim != 1 or hashlib.sha256(raw.tobytes()).hexdigest() != descriptor["raw_sha256"]:
                raise ValueError("Changed raw request tensor")
            shape = descriptor["shape"]
            if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
                raise ValueError("Invalid request tensor shape")
            request[key] = torch.from_numpy(raw.copy()).view(dtypes[descriptor["dtype"]]).reshape(shape)
    validate_request(request, torch)
    return request


def tensor_hash(value, torch):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def check_sequences(model, requests, *, torch, identity, port):
    """Serial reference generated before starting any socket/execution thread."""
    if len(requests) != 2 or requests[0]["input_ids"].shape[1] == requests[1]["input_ids"].shape[1]:
        raise ValueError("Parity requires two different instruction/token sequence lengths")
    bank = RngBank(torch)
    ambient = bank.capture()
    sequences = {"a": [0, 1, 0], "b": [1, 0, 1]}
    seeds = {"a": 940001, "b": 940002}
    reference, trace = {}, []
    try:
        for name, indices in sequences.items():
            bank.seed(seeds[name])
            reference[name] = []
            for index in indices:
                action = official_forward(model, requests[index], torch)
                reference[name].append((action.clone(), bank.hashes(bank.capture())))
    finally:
        bank.restore(ambient)
    executor = SerialExecutor(model, torch, identity, capacity=8)
    stop, ready = threading.Event(), threading.Event()
    errors = []
    ready_info = {}
    def serve_thread():
        try:
            serve(executor, host="127.0.0.1", port=port, max_clients=2, stop=stop,
                  ready_identity=executor.identity, ready_event=ready, ready_info=ready_info)
        except BaseException as exc:
            errors.append(exc)
            ready.set()
    host_thread = threading.Thread(target=serve_thread, name="parity-socket-listener", daemon=True)
    clients = {}
    try:
        host_thread.start()
        if not ready.wait(timeout=15) or errors:
            raise RuntimeError("Temporary parity endpoint did not bind") from (errors[0] if errors else None)
        for name in sequences:
            clients[name] = WireClient(ready_info["port"])
            clients[name].reset(seeds[name])
        if clients["a"].hello["connection_id"] == clients["b"].hello["connection_id"]:
            raise RuntimeError("Expected two simultaneously live independent connections")
        for sequence_index in range(3):
            for name in ("a", "b"):
                input_index = sequences[name][sequence_index]
                begin = time.perf_counter()
                action = clients[name].request(requests[input_index])
                elapsed = time.perf_counter() - begin
                actual_rng = clients[name].control("rng_state")
                expected, expected_rng = reference[name][sequence_index]
                equal = action.dtype == expected.dtype and action.shape == expected.shape and torch.equal(action, expected)
                rng_equal = all(actual_rng.get(key) == value for key, value in expected_rng.items())
                row = {"client": name, "sequence_index": sequence_index, "input_index": input_index,
                       "token_length": int(requests[input_index]["input_ids"].shape[1]),
                       "full_action_exact": equal, "all_rng_exact": rng_equal,
                       "max_abs": float((action.float() - expected.float()).abs().max()),
                       "action_sha256": tensor_hash(action, torch),
                       "reference_sha256": tensor_hash(expected, torch),
                       "actual_rng": actual_rng, "reference_rng": expected_rng,
                       "socket_request_seconds": elapsed}
                trace.append(row)
                if not equal or not rng_equal or actual_rng["requests_since_reset"] != sequence_index + 1:
                    raise RuntimeError("Interleaved request differs from the seeded serial reference")
        ambient_equal = bank.hashes(bank.capture()) == bank.hashes(ambient)
        if not ambient_equal:
            raise RuntimeError("Multiplex requests changed ambient process RNG")
        return {"passed": True, "trace": trace, "ambient_rng_unchanged": True,
                "sequences": sequences, "seeds": seeds, "temporary_port": ready_info["port"],
                "reference": "Exact official tensor conversion+fullforward; same Torch seeding as seeded_server plus explicit isolated Python/NumPy seeds",
                "note": "Six serial and six interleaved full-model forwards; no logits optimization or concurrent forward; not a speed benchmark"}
    except Exception as exc:
        return {"passed": False, "error_type": type(exc).__name__, "error": str(exc), "trace": trace}
    finally:
        for client in clients.values():
            client.close()
        stop.set()
        host_thread.join(timeout=15)
        executor.close()
        bank.restore(ambient)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xr1-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request-a", type=Path, required=True, help="Exact first_request.json fixture")
    parser.add_argument("--request-b", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0, help="Temporary localhost port; default0 lets the OS select an unused port")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=16)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Use a new parity report path")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Parent must reserve one visible GPU explicitly")
    if torch.cuda.mem_get_info()[0] < args.minimum_free_gib * 2**30:
        raise RuntimeError("Insufficient unreserved GPU memory for the engineering gate")
    identity, stamps = checkpoint_identity(args.model)
    server, source = load_official_server(args.xr1_repo, args.model, args.port)
    identity.update(server_instance="parity-only", official_server_sha256=sha256(source))
    inputs = [load_request_fixture(p, torch) for p in (args.request_a, args.request_b)]
    torch.cuda.reset_peak_memory_stats()
    report = check_sequences(server.model, inputs, torch=torch, identity=identity, port=args.port)
    report.update(model=identity, fixtures=[{"path": str(p.resolve()), "sha256": sha256(p),
                                           "npz_sha256": sha256(p.with_suffix(".npz")),
                                           "token_length": int(value["input_ids"].shape[1])}
                                          for p, value in zip((args.request_a, args.request_b), inputs)],
                  source_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                  peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30)
    if stamps != {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in stamps}:
        report.update(passed=False, error="Checkpoint changed during parity")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.report)}), flush=True)
    if not report["passed"]:
        raise RuntimeError("Multiplex model parity failed; do not start collection")


if __name__ == "__main__":
    main()
