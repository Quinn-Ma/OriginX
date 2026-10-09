"""New localhost-only server; never replace an active evaluation endpoint."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import secrets
import signal
import socket
import struct
import threading

from .core import SerialExecutor

MAX_PAYLOAD = 64 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recv_exact(conn, length):
    data = bytearray()
    while len(data) < length:
        part = conn.recv(length - len(data))
        if not part:
            raise EOFError("Connection closed")
        data.extend(part)
    return bytes(data)


def checkpoint_identity(directory):
    root = Path(directory).resolve(strict=True)
    config = root / "config.json"
    if not config.is_file() or not list(root.glob("*.safetensors")):
        raise ValueError("A complete local safetensors checkpoint is required")
    paths = sorted(p for p in root.iterdir() if p.is_file() and not p.name.startswith("."))
    stamps = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    hashes = {p.name: sha256(p) for p in paths}
    if stamps != {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}:
        raise ValueError("Checkpoint changed during identity audit")
    for index in root.glob("*.safetensors.index.json"):
        names = set(json.loads(index.read_text())["weight_map"].values())
        if any(Path(name).name != name or name not in hashes for name in names):
            raise ValueError("Missing or unexpected checkpoint shard")
    return {"model_path": str(root), "model_config_sha256": hashes["config.json"],
            "model_assets_sha256": hashes}, stamps


def load_official_server(repo, model, port):
    source = Path(repo).resolve(strict=True) / "deploy/server.py"
    spec = importlib.util.spec_from_file_location("_multiplex_official_server", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = module.Server(str(Path(model).resolve(strict=True)), "127.0.0.1", port)
    if server.model.training:
        raise RuntimeError("Expected the official loader's eval model")
    return server, source


def serve(executor, *, host, port, max_clients, stop, ready_identity, ready_event=None, ready_info=None):
    if host != "127.0.0.1" or not 0 <= port <= 65535 or not 1 <= max_clients <= 8:
        raise ValueError("Require localhost, explicit port, and1..8 clients")
    slots = threading.BoundedSemaphore(max_clients)
    live, lock = set(), threading.Lock()
    workers = []

    def connection_worker(conn):
        connection = secrets.token_hex(16)  # OS entropy, never policy/global RNG.
        try:
            conn.settimeout(300)
            while not stop.is_set():
                length = struct.unpack(">I", recv_exact(conn, 4))[0]
                if not 0 < length <= MAX_PAYLOAD:
                    raise ValueError("Request size outside bounded protocol")
                payload = recv_exact(conn, length)
                # Bytes only on I/O threads; no pickle/torch/model/RNG here.
                response = executor.submit(connection, payload).result()
                conn.sendall(struct.pack(">I", len(response)) + response)
        except (EOFError, ConnectionError):
            pass
        except Exception as error:
            print(json.dumps({"connection_id": connection, "error_type": type(error).__name__,
                              "error": str(error), "action": "close_without_retry"}), flush=True)
        finally:
            try:
                executor.submit(connection, close=True).result(timeout=30)
            except Exception:
                pass
            with lock:
                live.discard(conn)
            conn.close(); slots.release()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))  # A busy endpoint fails; never replaces it.
            listener.listen(max_clients); listener.settimeout(1)
            bound_port = listener.getsockname()[1]
            print(json.dumps({"ready": True, "host": host, "port": bound_port,
                              "max_clients": max_clients, **ready_identity}), flush=True)
            if ready_info is not None:
                ready_info.update(port=bound_port)
            if ready_event is not None:
                ready_event.set()
            while not stop.is_set():
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    continue
                if not slots.acquire(blocking=False):
                    conn.close()
                    continue
                with lock:
                    live.add(conn)
                thread = threading.Thread(target=connection_worker, args=(conn,), daemon=True)
                workers.append(thread); thread.start()
    finally:
        stop.set()
        with lock:
            for conn in list(live):
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        for thread in workers:
            thread.join(timeout=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xr1-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-clients", type=int, default=4)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Explicitly reserve exactly one visible CUDA device for this new server")
    model_identity, stamps = checkpoint_identity(args.model)
    server, official_source = load_official_server(args.xr1_repo, args.model, args.port)
    if stamps != {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in stamps}:
        raise RuntimeError("Checkpoint changed while loading")
    identity = {**model_identity, "server_instance": secrets.token_hex(16),
                "official_server_sha256": sha256(official_source),
                "multiplex_sources_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "torch_version": torch.__version__, "cuda_visible_device_count": 1}
    executor = SerialExecutor(server.model, torch, identity, capacity=args.max_clients * 2)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    try:
        serve(executor, host="127.0.0.1", port=args.port, max_clients=args.max_clients,
              stop=stop, ready_identity=executor.identity)
    finally:
        executor.close()


if __name__ == "__main__":
    main()
