"""Independent GR00T C/V intervention and same-socket RNG evidence.

Generic Astra queue transport uses its existing request/response v1 schemas;
experiment records and GR00T policy/RNG identity have independent schemas.
This module never imports or changes the frozen Xiaomi assistance runtime.
"""

import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import threading
import time

try:
    from .adapter import STATE_DIMS, build_observation
except ImportError:
    from adapter import STATE_DIMS, build_observation


CAMERAS = (
    ("left", "video.robot0_agentview_left"),
    ("right", "video.robot0_agentview_right"),
    ("wrist", "video.robot0_eye_in_hand"),
)
ARMS = ("C", "V")
GENERATED_ARMS = ("V",)
INTERVAL = 120.0
POLL_SECONDS = 0.5
STATE_FIELDS = (
    "server_instance", "connection_id", "protocol", "model_config_sha256",
    "seed", "requests_since_reset", "python_rng_sha256", "numpy_rng_sha256",
    "torch_cpu_rng_sha256", "torch_cuda_rng_sha256",
)
OPAQUE_ID = re.compile(r"(?:rq-)?[0-9a-f]{32}\Z")


class InfrastructureError(RuntimeError):
    """A policy/observation protocol failure, never a clean language fallback."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def array_sha(value):
    """Fingerprint array shape, dtype and C-order content, not instructions."""
    import numpy as np

    array = np.asarray(value)
    return hashlib.sha256(
        str((array.shape, array.dtype.str)).encode() + array.tobytes(order="C")
    ).hexdigest()


def instruction_sha(instruction):
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()


def append_instruction(original, subgoal):
    return (original + "\nImmediate next actions: " + subgoal
            + "\nThen complete the original task.")


def checked_native_inputs(latest, pack_state, state_history, image_history, step):
    """Validate the same returned Gym observation before publishing any request.

    The 14D evaluator state is only an equality check. Native quaternions are
    copied from the original named observation, never reconstructed from it.
    """
    import numpy as np
    if latest.steps != step:
        raise InfrastructureError("Native tap/query action position differs")
    native = latest.current()
    build_observation(native, "schema check")
    states = np.asarray(state_history)
    expected = np.asarray(pack_state(native), dtype=np.float32)
    if (states.ndim != 2 or states.shape[0] < 1 or states.shape[1] != 14
            or expected.shape != (14,) or not np.array_equal(states[-1], expected)):
        raise InfrastructureError("Evaluator state differs from current native tap")
    if set(image_history) != {key for _, key in CAMERAS}:
        raise InfrastructureError("Unexpected evaluator camera fields")
    for _, key in CAMERAS:
        frames = np.asarray(image_history[key])
        if (frames.ndim != 4 or frames.shape[0] < 1
                or not np.array_equal(frames[-1], native[key])):
            raise InfrastructureError("Evaluator RGB differs from current native tap")
    return native


def verify_delivery_receipt(folder, response, request_id, digest, token):
    """Require the broker's actual CLI receipt before applying generated text.

    Full stdout/model/tool/answer verification remains the downstream audit's
    responsibility; this binding prevents applying an unreceipted response.
    """
    if (response.get("cli_receipt_file") != "cli_receipt.json"
            or sha(folder / "cli_receipt.json") != response.get("cli_receipt_sha256")):
        raise ValueError("Generated response has no matching actual CLI receipt")
    receipt = json.loads((folder / "cli_receipt.json").read_text(encoding="utf-8"))
    expected = dict(schema="astra_cli_receipt_v1", cli_executed=True, model="gpt-6-astra",
                    model_requested="gpt-6-astra", reasoning_effort="high", status="ok",
                    returncode=0, no_tool_calls=True, output_validated=True)
    if any(receipt.get(k) != v for k,v in expected.items()):
        raise ValueError("Generated response CLI identity/effort/status invalid")
    matches = [r for r in receipt.get("requests", []) if r.get("request_id") == request_id]
    if (len(matches) != 1 or matches[0].get("request_sha256") != digest
            or matches[0].get("request_token") != token):
        raise ValueError("Generated response CLI request binding differs")
    return receipt


def validate_response(value, request_id, request_sha256, request_token):
    if not isinstance(value, dict):
        raise ValueError("Response must be an object")
    expected = dict(
        schema="astra_observation_response_v1", request_id=request_id,
        request_sha256=request_sha256, request_token=request_token,
        model="gpt-6-astra",
    )
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Response identity/model/request digest differs")
    if value.get("status") not in ("ok", "error"):
        raise ValueError("Response status must be ok/error")
    for key in ("reasoning_effort", "effort"):
        if key in value and value[key] != "high":
            raise ValueError("Response reasoning effort differs")
    if value["status"] == "ok":
        text = value.get("subgoal_instruction")
        rationale = value.get("rationale", "")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1500:
            raise ValueError("Response needs a nonempty bounded subgoal_instruction")
        if not isinstance(rationale, str) or len(rationale) > 1500:
            raise ValueError("Response rationale exceeds the frozen bound")
        if any(ord(c) < 32 and c not in "\n\t" for c in text + rationale):
            raise ValueError("Response contains unsupported control characters")
    return value


def normalized_state(response):
    if not isinstance(response, dict):
        raise InfrastructureError("Keepalive response must be an object")
    state = {key: response[key] for key in STATE_FIELDS}
    if state["protocol"] != "originx-gr00t-rng-v1":
        raise InfrastructureError("Unexpected keepalive protocol")
    if type(state["seed"]) is not int or type(state["requests_since_reset"]) is not int:
        raise InfrastructureError("Keepalive seed/count must be exact integers")
    for key in ("server_instance", "connection_id"):
        if not isinstance(state[key], str) or not state[key]:
            raise InfrastructureError("Missing keepalive connection identity")
    hashes = [state[key] for key in (
        "model_config_sha256", "python_rng_sha256", "numpy_rng_sha256",
        "torch_cpu_rng_sha256",
    )]
    cuda = state["torch_cuda_rng_sha256"]
    if not isinstance(cuda, list) or not cuda:
        raise InfrastructureError("Missing CUDA RNG fingerprints")
    hashes += cuda
    if any(not isinstance(x, str) or re.fullmatch(r"[0-9a-f]{64}", x) is None
           for x in hashes):
        raise InfrastructureError("Malformed keepalive RNG/model digest")
    # Do not retain a mutable list owned by the transport response.
    state["torch_cuda_rng_sha256"] = list(cuda)
    return state


class AssistanceClient:
    def __init__(self, client, *, request_root, episode_output, request_id,
                 case_id, arm, horizon, wait_seconds=1200,
                 intervention_fraction=0.5, policy_id="gr00t_n15_robocasa_multitask120000",
                 latest_observation=None, pack_state=None):
        if arm not in ARMS:
            raise ValueError("Arm must be C or V")
        if intervention_fraction != 0.5:
            raise ValueError("This protocol fixes the midpoint query intervention")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("Horizon must be a positive integer")
        if not isinstance(wait_seconds, (int, float)) or not math.isfinite(wait_seconds) or wait_seconds <= 0:
            raise ValueError("Response wait must be finite and positive")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("A local case identity is required")
        if policy_id is not None and (not isinstance(policy_id, str) or not policy_id):
            raise ValueError("Policy identity must be a nonempty string when supplied")
        if latest_observation is None or not callable(pack_state):
            raise ValueError("Raw native-observation tap and pinned entry state checker are required")
        if policy_id != "gr00t_n15_robocasa_multitask120000":
            raise ValueError("Wrong independent policy identity")
        self.latest_observation, self.pack_state = latest_observation, pack_state
        self.client = client
        self.request_root = Path(request_root)
        self.episode_output = Path(episode_output)
        self.episode_output.mkdir(parents=True, exist_ok=True)
        # Old call sites may pass task/seed/arm-bearing IDs. Never export them.
        self.local_request_id = request_id
        self.request_id = (request_id if isinstance(request_id, str)
                           and OPAQUE_ID.fullmatch(request_id)
                           else "rq-" + secrets.token_hex(16))
        self.case_id, self.arm, self.policy_id = case_id, arm, policy_id
        self.horizon, self.wait_seconds = horizon, float(wait_seconds)
        self.trigger_step = 16 * math.ceil(horizon / 32)
        if self.trigger_step >= horizon:
            raise ValueError("Intervention must leave some official horizon")
        self.calls = 0
        self.instruction = None
        self._original_instruction = None
        self._infer_thread = None
        self.record = dict(
            schema="originx_gr00t_assistance_v1", arm=arm, policy_id=policy_id,
            case_id=case_id, request_id=self.request_id,
            trigger_step=self.trigger_step, trigger_reached=False,
            requested=False, status="not_reached", applied=False,
            fallback=False, wait_seconds=0.0,
            intervention_rule="tau = 16 * ceil(H / 32); only an actual query triggers",
            generated_failure_policy="continue_original_instruction",
            latency_matched=False,
        )

    def __getattr__(self, name):
        return getattr(self.client, name)

    def _save(self):
        write(self.episode_output / "assistance.json", self.record)

    def _keepalive(self, phase):
        row = dict(
            schema="originx_gr00t_same_socket_keepalive_v1", phase=phase,
            sequence=getattr(self, "_keepalive_count", 0), control="rng_state",
            reconnect=False, reset=False,
        )
        started = time.monotonic()
        try:
            thread = threading.get_ident()
            if self._infer_thread != thread:
                raise InfrastructureError("Keepalive must use the inference thread")
            wire = self.client.policy.wire
            if getattr(self, "_wait_wire", wire) is not wire:
                raise InfrastructureError("Keepalive wire changed; reconnect forbidden")
            self._wait_wire = wire
            response = wire.control("rng_state")
            state = normalized_state(response)
            for key in ("connection_id", "server_instance", "model_config_sha256", "protocol"):
                if state[key] != wire.hello[key]:
                    raise InfrastructureError("Keepalive identity differs from admitted hello: " + key)
            if state["seed"] != self.client.seed or state["requests_since_reset"] != self.client.infer_calls:
                raise InfrastructureError("Keepalive changed policy seed/request count")
            baseline = getattr(self, "_keepalive_baseline", None)
            if baseline is not None and state != baseline:
                raise InfrastructureError("Keepalive changed same-connection RNG/count/identity")
            if baseline is None:
                self._keepalive_baseline = state
            row.update(
                status="ok", same_thread=True, state=state, response=response,
                response_sha256=hashlib.sha256(
                    json.dumps(response, sort_keys=True).encode()
                ).hexdigest(),
            )
            self._keepalive_count = row["sequence"] + 1
            self.record["keepalive"] = dict(
                status="ok", interval_seconds=INTERVAL, calls=self._keepalive_count,
                same_socket=True, same_thread=True, rng_and_count_unchanged=True,
                reconnect=False, reset=False,
            )
        except Exception as error:
            row.update(status="error", error=repr(error))
            self.record.setdefault("first_error", repr(error))
            self.record.setdefault("first_keepalive_error", repr(error))
            self.record.update(status="keepalive_error", applied=False, fallback=False)
            self.record["keepalive"] = dict(
                status="error", error=repr(error), interval_seconds=INTERVAL,
                reconnect=False, reset=False,
            )
            if isinstance(error, InfrastructureError):
                raise
            raise InfrastructureError("Keepalive validation/transport failed") from error
        finally:
            row["wall_seconds"] = time.monotonic() - started
            with (self.episode_output / "policy-keepalive.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            self._save()

    def _fallback(self, reason, error=None):
        self.instruction = None
        self.record.update(status=reason, fallback=True, fallback_reason=reason,
                           applied=False, application_kind="original_fallback")
        if error is not None:
            self.record.setdefault("first_error", repr(error))

    def _request(self, images, instruction, step):
        folder = self.request_root / self.request_id
        try:
            folder.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise InfrastructureError("Request ID collision; refusing to overwrite evidence") from error
        cameras = []
        if self.arm == "V":
            import numpy as np
            from PIL import Image

            for name, key in CAMERAS:
                array = np.asarray(images[key][-1])
                if array.shape != (256, 256, 3) or array.dtype != np.uint8:
                    raise InfrastructureError("Expected current official uint8 256x256 RGB")
                image_path = folder / (name + ".png")
                Image.fromarray(np.ascontiguousarray(array)).save(image_path, format="PNG")
                cameras.append(dict(name=name, file=image_path.name, sha256=sha(image_path)))
        # The opaque case alias and routing metadata are not part of the broker's
        # LLM whitelist. The real case mapping stays outside the request directory.
        request = dict(
            schema="astra_observation_request_v1", request_id=self.request_id,
            request_token=secrets.token_hex(16), case_id=self.request_id,
            arm=self.arm, policy_id=self.policy_id,
            modalities=["text", "rgb"] if self.arm == "V" else ["text"],
            step=step, horizon=self.horizon, remaining_steps=self.horizon - step,
            original_instruction=instruction, cameras=cameras,
            permitted_inputs=(["original_instruction", "current_rgb", "step_budget"]
                              if self.arm == "V" else ["original_instruction", "step_budget"]),
            oracle_inputs_included=False,
        )
        mapping_path = self.episode_output / "request_mapping.json"
        write(mapping_path, dict(
            schema="gr00t_confirmatory_request_mapping_v1", request_id=self.request_id,
            local_request_id=self.local_request_id, case_id=self.case_id,
            policy_id=self.policy_id, arm=self.arm,
        ))
        request_path = folder / "request.json"
        write(request_path, request)
        digest = sha(request_path)
        self.record.update(
            requested=True, status="waiting", request_path=str(request_path),
            request_sha256=digest, request_token=request["request_token"],
            mapping_path=str(mapping_path), modalities=request["modalities"],
        )
        self._save()
        started = time.monotonic()
        next_keepalive = started + INTERVAL
        response_path = folder / "response.json"
        try:
            while time.monotonic() - started < self.wait_seconds:
                if time.monotonic() >= next_keepalive:
                    self._keepalive("periodic")
                    next_keepalive = time.monotonic() + INTERVAL
                if response_path.exists():
                    # Invalid/misrouted advice is not used, but the policy socket
                    # can still be valid. Its exit keepalive is mandatory.
                    try:
                        response = validate_response(
                            json.loads(response_path.read_text(encoding="utf-8")),
                            self.request_id, digest, request["request_token"],
                        )
                        if response["status"] == "ok":
                            verify_delivery_receipt(folder, response, self.request_id,
                                                    digest, request["request_token"])
                        self.record.update(response_sha256=sha(response_path), response=response)
                    except (ValueError, TypeError, UnicodeError) as error:
                        self._fallback("invalid_response", error)
                        break
                    if response["status"] == "error":
                        self._fallback("broker_error", response.get("error", "broker error"))
                    else:
                        self.instruction = append_instruction(
                            instruction, response["subgoal_instruction"].strip()
                        )
                        self.record.update(status="prepared", advice_validated=True,
                                           application_kind="generated", policy_instruction=self.instruction)
                    break
                time.sleep(POLL_SECONDS)
            else:
                self._fallback("timeout")
        finally:
            self.record["wait_seconds"] = time.monotonic() - started

    def _intervene(self, images, instruction, step):
        self.record.update(trigger_reached=True, step=step, original_instruction=instruction)
        self._keepalive("enter")
        try:
            if self.arm == "C":
                self.record.update(status="unchanged", application_kind="identity_sham")
            else:
                try:
                    self._request(images, instruction, step)
                except OSError as error:
                    self._fallback("transport_error", error)
        finally:
            # No forward can occur after a delivery failure until this succeeds.
            self._keepalive("exit")
        self._save()

    def infer(self, state_history, image_history, instruction):
        if self.record["status"] == "keepalive_error":
            raise InfrastructureError("Episode has a terminal keepalive failure; continuation forbidden")
        thread = threading.get_ident()
        if self._infer_thread is None:
            self._infer_thread = thread
        elif self._infer_thread != thread:
            raise InfrastructureError("Inference/keepalive thread changed")
        if not isinstance(instruction, str) or not instruction.strip():
            raise InfrastructureError("Original task instruction must be nonempty text")
        if self._original_instruction is None:
            self._original_instruction = instruction
        elif instruction != self._original_instruction:
            raise InfrastructureError("Original task instruction changed during the episode")
        step = self.calls * 16
        import numpy as np
        native = checked_native_inputs(self.latest_observation, self.pack_state,
                                       state_history, image_history, step)
        raw_state = np.concatenate([np.asarray(native[key], dtype=np.float32)
                                    for key in STATE_DIMS])
        current_images = {key: np.asarray(native[key])[None] for _, key in CAMERAS}
        named_state_hashes = {key: array_sha(native[key]) for key in STATE_DIMS}
        current_image_hashes = {key: array_sha(current_images[key]) for _, key in CAMERAS}
        input_trace = dict(
            step=step, state_sha256=array_sha(raw_state),
            native_state_sha256=named_state_hashes,
            state_representation="native_named_float32_16D_with_raw_quaternions",
            evaluator_state_history_sha256=array_sha(state_history),
            image_sha256=current_image_hashes, policy_context_frames=1,
            raw_observation_sha256=hashlib.sha256(json.dumps(
                dict(state=named_state_hashes, images=current_image_hashes),
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            original_instruction_sha256=instruction_sha(instruction),
        )
        at_trigger = step == self.trigger_step
        if at_trigger:
            if self.record["trigger_reached"]:
                raise InfrastructureError("Trigger already attempted; in-place retry forbidden")
            self.record["trigger_observation"] = input_trace.copy()
            self._intervene(current_images, instruction, step)
        actual_instruction = instruction if self.instruction is None else self.instruction
        action = self.client.infer(state_history, image_history, actual_instruction)
        action = np.asarray(action)
        if action.shape != (16, 12) or action.dtype != np.float32 or not np.isfinite(action).all():
            raise InfrastructureError("GR00T must return finite decoded float32 16x12 actions")
        self.calls += 1
        trace = dict(
            **input_trace, at_trigger=at_trigger,
            instruction_sha256=instruction_sha(actual_instruction),
            action_sha256=array_sha(action),
        )
        with (self.episode_output / "query_trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(trace, sort_keys=True, allow_nan=False) + "\n")
        if at_trigger:
            applied = self.instruction is not None
            if applied:
                self.record.update(status="applied", applied=True)
            receipt = dict(
                schema="originx_gr00t_intervention_receipt_v1", arm=self.arm,
                policy_id=self.policy_id, case_id=self.case_id, step=step,
                trigger_query_completed=True, applied=applied,
                fallback=self.record["fallback"], application_kind=self.record["application_kind"],
                instruction_unchanged=actual_instruction == instruction,
                original_instruction_sha256=instruction_sha(instruction),
                policy_instruction_sha256=instruction_sha(actual_instruction),
                action_sha256=trace["action_sha256"],
                request_id=self.request_id if self.record["requested"] else None,
                request_sha256=self.record.get("request_sha256"),
                response_sha256=self.record.get("response_sha256"),
            )
            write(self.episode_output / "intervention_receipt.json", receipt)
            self.record["application_receipt"] = receipt
            self._save()
        return action

    def finish(self):
        self.record["policy_queries"] = self.calls
        self._save()
        return self.record
