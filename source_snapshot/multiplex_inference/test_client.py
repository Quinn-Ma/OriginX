import hashlib
import json
import pickle
import struct

import numpy as np
import pytest
import torch

from multiplex_inference.client import WireClient
from multiplex_inference.core import SerialExecutor
from multiplex_inference.parity import load_request_fixture
from multiplex_inference.test_core import FakeModel, request


class FragmentedMemorySocket:
    """Real wire serialization/queue path without forbidden local TCP syscalls."""
    def __init__(self, executor, connection):
        self.executor, self.connection, self.data, self.closed = executor, connection, b"", False

    def settimeout(self, _):
        pass

    def sendall(self, payload):
        n = struct.unpack(">I", payload[:4])[0]
        assert n == len(payload) - 4
        response = self.executor.submit(self.connection, payload[4:]).result(timeout=10)
        self.data += struct.pack(">I", len(response)) + response

    def recv(self, n):
        # Fragment the header, then allow large tensor payload chunks.
        count = 1 if n <= 4 else min(n, 8192)
        value, self.data = self.data[:count], self.data[count:]
        return value

    def close(self):
        self.closed = True
        self.executor.submit(self.connection, close=True).result(timeout=10)


def test_two_wire_clients_preserve_rng_and_fragmented_headers():
    executor = SerialExecutor(FakeModel(), torch, {"server_instance": "fake", "model_config_sha256": "hash"})
    clients = []
    try:
        for name in ("a", "b"):
            clients.append(WireClient(0, _socket=FragmentedMemorySocket(executor, name)))
            assert clients[-1].hello["isolated_rng_stream"]
            assert clients[-1].hello["shared_model"] and not clients[-1].hello["exclusive_connection"]
            clients[-1].reset(14)
        a = clients[0].request(request(7))
        b = clients[1].request(request(7))
        assert torch.equal(a, b)
        assert clients[0].control("rng_state")["requests_since_reset"] == 1
    finally:
        for client in clients:
            client.close()
        executor.close()


def test_real_format_fixture_roundtrip_bfloat16_and_raw_hash_guard(tmp_path):
    values = request(7)
    values["state"] = values["state"].bfloat16()
    fields, arrays = {}, {}
    for index, (key, value) in enumerate(sorted(values.items())):
        if not isinstance(value, torch.Tensor):
            fields[key] = {"kind": "json", "value": value}
            continue
        name = f"field_{index}_raw"
        raw = value.contiguous().reshape(-1).view(torch.uint8).numpy()
        arrays[name] = raw
        fields[key] = {"kind": "torch.Tensor", "dtype": str(value.dtype), "shape": list(value.shape),
                       "raw_key": name, "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}
    path = tmp_path / "first_request.json"
    path.write_text(json.dumps({"format_version": 1,
                               "capture_point": "client_before_serialization_and_server_dtype_conversion",
                               "fields": fields, "request_sha256": hashlib.sha256(json.dumps(
                                   fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}))
    np.savez(path.with_suffix(".npz"), **arrays)
    loaded = load_request_fixture(path, torch)
    assert loaded["state"].dtype == torch.bfloat16 and torch.equal(loaded["state"], values["state"])
    bad = next(iter(arrays))
    arrays[bad] = arrays[bad].copy(); arrays[bad][0] ^= 1
    np.savez(path.with_suffix(".npz"), **arrays)
    with pytest.raises(ValueError, match="raw request"):
        load_request_fixture(path, torch)
