"""Explicit opt-in client for the new protocol; old exclusive clients reject it."""
import importlib.util
import inspect
from pathlib import Path
import pickle
import secrets
import socket
import struct

from .core import CONTROL, PROTOCOL
from .server import MAX_PAYLOAD, recv_exact, sha256


class WireClient:
    def __init__(self, port, *, timeout=180, _socket=None):
        self.socket = _socket or socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.socket.settimeout(timeout)
        try:
            self.hello = self.control("hello")
            if (self.hello.get("protocol") != PROTOCOL or self.hello.get("exclusive_connection") is not False
                    or self.hello.get("serial_forward") is not True
                    or self.hello.get("rng_isolation") != "python_numpy_torch_cpu_all_cuda_per_connection"):
                raise ValueError("Expected the dedicated serial-forward multiplex endpoint")
        except Exception:
            self.close()
            raise

    def send(self, data):
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        if not 0 < len(payload) <= MAX_PAYLOAD:
            raise ValueError("Request exceeds maximum payload")
        self.socket.sendall(struct.pack(">I", len(payload)) + payload)

    def receive(self):
        length = struct.unpack(">I", recv_exact(self.socket, 4))[0]
        if not 0 < length <= MAX_PAYLOAD:
            raise ValueError("Unexpected response size")
        return pickle.loads(recv_exact(self.socket, length))

    def request(self, data):
        self.send(data)
        return self.receive()

    def control(self, command, **kwargs):
        nonce = secrets.token_hex(16)
        response = self.request({CONTROL: command, "protocol": PROTOCOL, "request_nonce": nonce, **kwargs})
        if not isinstance(response, dict) or response.get("ok") is not True or response.get("request_nonce") != nonce:
            raise RuntimeError("Missing server acknowledgement")
        if hasattr(self, "hello") and any(response.get(k) != self.hello.get(k)
                                          for k in ("connection_id", "server_instance", "model_config_sha256")):
            raise RuntimeError("Server identity changed midconnection")
        return response

    def reset(self, seed):
        result = self.control("reset_rng", seed=seed)
        if result["seed"] != seed or result["requests_since_reset"] != 0:
            raise RuntimeError("Inconsistent reset acknowledgement")
        return result

    def close(self):
        self.socket.close()


class MultiplexEvalClient:
    """Unchanged official EvalClient preprocessing and action decoding.

    Call reset(seed) exactly once when beginning each episode. The server allows
    explicit resets for engineering parity, but this wrapper refuses a second
    reset to prevent an accidental within-episode restart; create a new client.
    """
    def __init__(self, xr1_repo, model_path, port, *, timeout=180):
        root = Path(xr1_repo).resolve(strict=True)
        source = root / "eval_robocasa365/entry.py"
        spec = importlib.util.spec_from_file_location("_multiplex_official_entry", source)
        entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
        original = entry.Client
        if Path(inspect.getfile(original)).resolve() != root / "deploy/client.py":
            raise ValueError("Official client import came from unexpected source")
        class BoundedClient(original):
            def _connect_with_retry(self, max_retries=None, retry_interval=1):
                return super()._connect_with_retry(max_retries=1, retry_interval=1)
        entry.Client = BoundedClient
        self.eval = entry.EvalClient(str(model_path), "127.0.0.1", port, "robocasa365", .95)
        try:
            self.wire = WireClient(port, timeout=timeout, _socket=self.eval.client.client_socket)
            if (self.wire.hello.get("model_path") != str(Path(model_path).resolve())
                    or self.wire.hello.get("model_config_sha256") != sha256(Path(model_path) / "config.json")):
                raise ValueError("Multiplex checkpoint differs from processor")
            # Only framing changes: robust fragmented headers, same exact
            # official processor request and processor.decode_action afterward.
            self.eval.client._send_with_length_prefix = self.wire.send
            self.eval.client._recv_with_length_prefix = self.wire.receive
        except Exception:
            self.close()
            raise
        self._initialized = False

    def reset(self, seed):
        if self._initialized:
            raise ValueError("One episode per connection; no mid-episode reseed")
        result = self.wire.reset(seed)
        self._initialized = True
        return result

    def infer(self, *args, **kwargs):
        if not self._initialized:
            raise RuntimeError("Initialize this episode before inference")
        return self.eval.infer(*args, **kwargs)

    def close(self):
        self.eval.close()
