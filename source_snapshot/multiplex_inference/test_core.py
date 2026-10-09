import pickle
import random
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from multiplex_inference.core import (CONTROL, PROTOCOL, RngBank, SerialExecutor,
                                      official_forward, validate_request)


def request(length, marker=0):
    return {"task_id": "robocasa365", "input_ids": torch.arange(length)[None],
            "attention_mask": torch.ones(1, length, dtype=torch.long),
            "state": torch.full((1, 4, 60), float(marker)),
            "action_mask": torch.ones(1, 16, 60),
            "pixel_values_videos": torch.zeros(12, 3),
            "video_grid_thw": torch.ones(3, 3, dtype=torch.long)}


class FakeModel:
    device, dtype = torch.device("cpu"), torch.float32

    def __init__(self):
        self.threads = []
        self.rope_deltas = None
        self.active, self.maximum_active = 0, 0

    def __call__(self, **data):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            self.threads.append(threading.current_thread().name)
            assert "past_key_values" not in data and "cache_position" not in data
            # Simulate full-request overwrite of mutable Qwen RoPE metadata.
            self.rope_deltas = data["input_ids"].shape[1]
            x = torch.rand(1, 16, 60) + random.random() + np.random.rand() + self.rope_deltas
            if float(data["state"][0, 0, 0]) == -99:
                raise RuntimeError("injected model failure after consuming RNG")
            time.sleep(.001)
            return SimpleNamespace(actions=x)
        finally:
            self.active -= 1


def invoke(executor, connection, data):
    return pickle.loads(executor.submit(connection, pickle.dumps(data, protocol=5)).result(timeout=10))


def control(executor, connection, command, **kwargs):
    return invoke(executor, connection, {CONTROL: command, "protocol": PROTOCOL,
                                         "request_nonce": "nonce", **kwargs})


def test_interleaved_clients_match_serial_reference_actions_and_all_rngs():
    bank = RngBank(torch)
    ambient = bank.capture()
    sequences = {"a": [request(7), request(13), request(7)],
                 "b": [request(13), request(7), request(13)]}
    seeds = {"a": 19, "b": 23}
    reference = {}
    model = FakeModel()
    try:
        for client, sequence in sequences.items():
            bank.seed(seeds[client])
            reference[client] = [(official_forward(model, value, torch), bank.hashes(bank.capture()))
                                 for value in sequence]
        bank.restore(ambient)
        before = bank.hashes(bank.capture())
        model.threads.clear()
        executor = SerialExecutor(model, torch, {"server_instance": "test"})
        try:
            for client in sequences:
                control(executor, client, "reset_rng", seed=seeds[client])
            for index in range(3):
                for client in ("a", "b"):
                    actual = invoke(executor, client, sequences[client][index])
                    expected, rng = reference[client][index]
                    assert torch.equal(actual, expected)
                    state = control(executor, client, "rng_state")
                    assert all(state[key] == value for key, value in rng.items())
                    assert state["requests_since_reset"] == index + 1
            assert set(model.threads) == {"xr1-serial-forward"}
            assert model.maximum_active == 1
            assert bank.hashes(bank.capture()) == before
        finally:
            executor.close()
    finally:
        bank.restore(ambient)


def test_concurrent_submission_single_forward_and_poison_isolation():
    executor = SerialExecutor(FakeModel(), torch, {"server_instance": "test"})
    try:
        for key in ("a", "b", "bad"):
            control(executor, key, "reset_rng", seed=24)
        pending = [executor.submit(k, pickle.dumps(request(7), protocol=5)) for k in ("a", "b")]
        aa, bb = [pickle.loads(f.result(timeout=10)) for f in pending]
        assert torch.equal(aa, bb) and executor.model.maximum_active == 1
        before = control(executor, "a", "rng_state")
        with pytest.raises(RuntimeError, match="injected"):
            invoke(executor, "bad", request(7, -99))
        after = control(executor, "a", "rng_state")
        assert before == after
        with pytest.raises(RuntimeError, match="Failed session"):
            control(executor, "bad", "reset_rng", seed=24)
        executor.submit("bad", close=True).result(timeout=10)
        assert "bad" not in executor._sessions
    finally:
        executor.close()


@pytest.mark.parametrize("extra", ["past_key_values", "cache_position", "position_ids",
                                   "inputs_embeds", "num_steps", "logits_to_keep", "use_cache"])
def test_generation_or_optimization_overrides_rejected(extra):
    data = request(7)
    data[extra] = None
    with pytest.raises(ValueError, match="complete official"):
        validate_request(data, torch)


def test_no_inference_before_seed_and_bad_shape_rejected():
    executor = SerialExecutor(FakeModel(), torch, {"server_instance": "test"})
    try:
        hello = control(executor, "x", "hello")
        assert hello["exclusive_connection"] is False and hello["serial_forward"] is True
        with pytest.raises(ValueError, match="initialization"):
            invoke(executor, "x", request(7))
        data = request(7)
        data["state"] = data["state"][:, :3]
        with pytest.raises(ValueError, match="history4"):
            validate_request(data, torch)
    finally:
        executor.close()
