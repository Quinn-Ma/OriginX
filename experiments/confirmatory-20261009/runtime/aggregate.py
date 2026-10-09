"""Audit new-confirmation raw evidence, then call the independent analyzer.

Usage: python aggregate.py --config <new-output/config.json>
Optional --broker-state supplies the local durable broker directory, including
CLI calls that never published a response. Without that inventory, usage is
explicitly a lower-bound record of observed receipts, not a complete zero-cost
account. No raw evidence, source, model or historical report is changed.

The default output is <config.output>/analysis/{report.json,normalized_records.json,
authority.json,cli_usage.json}. --output-directory may write a *new* local report
folder; it cannot point into either historical native/Astra campaign. This option
does not remap remote source/evidence paths: aggregation must have those files
available, or their absence is reported as an authority failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

try:
    from .analysis import analyze
    from .protocol import validate_manifest
    from .assistance import GENERIC_INSTRUCTION, STATE_FIELDS, append_instruction
except ImportError:
    from analysis import analyze
    from protocol import validate_manifest
    from assistance import GENERIC_INSTRUCTION, STATE_FIELDS, append_instruction


MODEL = "gpt-6-astra"
COUNTER_SHA = "77f992d01aa1ae5f21ed7f170d0aab9c303c32148f4b427651c04912be8ce68a"
CAMERAS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")
CAMERA_NAMES = ("left", "right", "wrist")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
EVENT_ORDER = ("reset_requested", "reset_completed", "initial_fingerprints_recorded",
               "assignment_selected", "server_connected", "rng_ack", "first_infer")
HELLO_FIELDS = ("server_instance", "protocol", "exclusive_connection", "shared_model", "isolated_rng_stream",
                "serial_forward", "rng_isolation", "model_path", "model_config_sha256", "model_assets_sha256",
                "official_server_sha256", "multiplex_sources_sha256", "model_config_runtime_sha256",
                "model_execution_thread_name", "model_execution_thread_ident", "torch_version",
                "cuda_visible_device_count", "full_request_only", "inference_optimization")
USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_sha(value):
    return text_sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def array_sha(value):
    import numpy as np
    value = np.asarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes(order="C")).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _initial(episode, raw):
    import numpy as np
    initial = read(episode / "initial_scene.json")
    require(initial == raw["initial_scene"], "initial_scene_file_result_differ")
    require(sha(episode / "initial.xml") == initial["xml_sha256"], "initial_xml_hash_differs")
    require(set(initial["rgb"]) == set(CAMERAS), "initial_camera_set_differs")
    require(isinstance(initial["instruction"], str) and initial["instruction"].strip(), "missing_initial_instruction")
    with np.load(episode / "initial_observation.npz", allow_pickle=False) as arrays:
        for name in CAMERAS:
            image = arrays[name]
            require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[-1] == 3, "invalid_initial_rgb")
            require(array_sha(image) == initial["rgb"][name], "initial_rgb_array_hash_differs")
        for name in ("proprioception", "physics_state"):
            require(array_sha(arrays[name]) == initial[name], "initial_" + name + "_array_hash_differs")
    result = {"rgb:" + name: initial["rgb"][name] for name in CAMERAS}
    result.update({name: initial[name] for name in ("proprioception", "physics_state", "xml_sha256")})
    result["instruction_sha256"] = text_sha(initial["instruction"])
    for name in ("layout_id", "style_id"):
        require(type(initial[name]) is int, "initial_layout_style_must_be_integer")
        result[name] = str(initial[name])
    require(all(isinstance(v, str) and v for v in result.values()), "malformed_initial_fingerprint")
    return result, initial["instruction"]


def _traces(episode, steps, original):
    raw = jsonl(episode / "query_trace.jsonl")
    require([q.get("step") for q in raw] == list(range(0, steps, 16)), "query_steps_or_count_differs")
    normalized = []
    for q in raw:
        require(q["original_instruction_sha256"] == text_sha(original), "original_instruction_changes")
        require(set(q["image_sha256"]) == set(CAMERAS), "query_camera_set_differs")
        hashes = [q[k] for k in ("state_sha256", "instruction_sha256", "action_sha256")]
        hashes += list(q["image_sha256"].values())
        require(all(isinstance(v, str) and HEX64.fullmatch(v) for v in hashes), "malformed_query_digest")
        normalized.append(dict(step=q["step"], state_hash=q["state_sha256"],
                               raw_observation_hash=json_sha(q["image_sha256"]),
                               instruction_hash=q["instruction_sha256"], action_hash=q["action_sha256"]))
    return raw, normalized


def _keepalive(episode, assistance, raw, tau):
    rows = jsonl(episode / "policy-keepalive.jsonl")
    require(len(rows) >= 2 and rows[0]["phase"] == "enter" and rows[-1]["phase"] == "exit", "missing_keepalive_enter_exit")
    require(all(x["phase"] == "periodic" for x in rows[1:-1]), "unexpected_keepalive_phase")
    baseline = None
    for i, row in enumerate(rows):
        require(row["sequence"] == i and row["status"] == "ok" and row["same_thread"] is True
                and row["control"] == "rng_state" and row["reconnect"] is False and row["reset"] is False,
                "invalid_keepalive_record")
        response = row["response"]
        require(text_sha(json.dumps(response, sort_keys=True)) == row["response_sha256"], "keepalive_response_hash_differs")
        state = {k: response[k] for k in STATE_FIELDS}
        require(state == row["state"], "keepalive_normalization_differs")
        if baseline is None:
            baseline = state
        require(state == baseline, "keepalive_rng_or_connection_changed")
        require(state["seed"] == raw["job"]["policy_seed"] and state["requests_since_reset"] == tau // 16,
                "keepalive_seed_query_count_differs")
        for name in ("connection_id", "server_instance", "model_config_sha256", "protocol"):
            require(state[name] == raw["server_identity"][name], "keepalive_server_identity_differs")
    summary = assistance["keepalive"]
    require(summary.get("status") == "ok" and summary.get("calls") == len(rows)
            and summary.get("same_socket") is True and summary.get("same_thread") is True
            and summary.get("rng_and_count_unchanged") is True and summary.get("reconnect") is False
            and summary.get("reset") is False, "keepalive_summary_differs")


def _request(output, job, assistance, original, tau):
    folder = output / "requests" / job["request_id"]
    request = read(folder / "request.json")
    digest = sha(folder / "request.json")
    require(request.get("schema") == "astra_observation_request_v1"
            and request.get("request_id") == job["request_id"] and request.get("case_id") == job["request_id"]
            and request.get("arm") == job["arm"] and request.get("policy_id") == job["policy_id"]
            and request.get("step") == tau and request.get("horizon") == job["horizon"]
            and request.get("remaining_steps") == job["horizon"] - tau
            and request.get("original_instruction") == original and request.get("oracle_inputs_included") is False,
            "request_identity_or_budget_differs")
    token = request.get("request_token")
    require(isinstance(token, str) and re.fullmatch(r"[0-9a-f]{32}", token), "request_token_invalid")
    require(assistance.get("request_token") == token and assistance.get("request_sha256") == digest,
            "recorded_request_digest_or_token_differs")
    if job["arm"] == "L":
        require(request["modalities"] == ["text"] and request["cameras"] == []
                and request["permitted_inputs"] == ["original_instruction", "step_budget"], "vision_blind_request_contains_images")
    else:
        require(request["modalities"] == ["text", "rgb"]
                and request["permitted_inputs"] == ["original_instruction", "current_rgb", "step_budget"], "visual_request_modalities_differ")
        require([x["name"] for x in request["cameras"]] == list(CAMERA_NAMES), "request_camera_set_differs")
        from PIL import Image
        for camera in request["cameras"]:
            require(camera["file"] == camera["name"] + ".png" and sha(folder / camera["file"]) == camera["sha256"], "request_camera_digest_differs")
            with Image.open(folder / camera["file"]) as image:
                require(image.size == (256, 256) and image.mode == "RGB" and image.format == "PNG", "request_camera_format_differs")
    return folder, request, digest


def _inspect_events(events):
    messages, models, problems, usages = [], set(), [], []
    for event in events:
        require(isinstance(event, dict), "cli_event_not_object")
        if event.get("type") in ("error", "turn.failed"):
            problems.append("cli_trace_error")
        if re.search(r"tool|command|function_call", str(event.get("type", "")), re.I):
            problems.append("cli_tool_event")
        for obj in (event, event.get("session", {}), event.get("turn", {})):
            if isinstance(obj, dict) and isinstance(obj.get("model"), str):
                models.add(obj["model"])
        item = event.get("item")
        if isinstance(item, dict):
            if item.get("type") not in ("agent_message", "reasoning"):
                problems.append("cli_forbidden_item:" + str(item.get("type")))
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                try:
                    messages.append(json.loads(item["text"]))
                except (ValueError, KeyError):
                    pass
        if event.get("type") == "turn.completed":
            usages.append(event.get("usage", {}))
    totals = {key: None for key in USAGE_KEYS}
    for usage in usages:
        for key in USAGE_KEYS:
            amount = usage.get(key)
            if key == "reasoning_tokens" and amount is None:
                amount = usage.get("reasoning_output_tokens")
                details = usage.get("output_tokens_details")
                if amount is None and isinstance(details, dict):
                    amount = details.get("reasoning_tokens")
            if type(amount) is int and amount >= 0:
                totals[key] = (totals[key] or 0) + amount
    return dict(messages=messages, models=sorted(models), problems=problems,
                completed=bool(usages), usage_totals=totals)


def _command(receipt, visual):
    cmd = receipt.get("command", [])
    require(isinstance(cmd, list) and cmd and all(isinstance(x, str) for x in cmd), "cli_command_missing")
    name = cmd[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    require(name in ("codex", "codex.exe", "codex.cmd", "codex.bat"), "not_codex_executable")
    require(cmd.count("--model") == 1 and cmd[cmd.index("--model") + 1] == MODEL and "exec" in cmd and "--json" in cmd,
            "cli_model_or_mode_differs")
    efforts = [x for x in cmd if x.startswith("model_reasoning_effort=")]
    require(efforts == ["model_reasoning_effort=high"], "cli_effort_not_uniquely_high")
    require(cmd.count("-i") == (1 if visual else 0) and "--image" not in cmd, "cli_modality_differs")


def _generated_evidence(output, job, assistance, original, tau, all_jobs):
    folder, request, request_sha = _request(output, job, assistance, original, tau)
    response = read(folder / "response.json")
    response_sha = sha(folder / "response.json")
    require(response == assistance.get("response") and response_sha == assistance.get("response_sha256"), "applied_response_record_differs")
    require(response.get("schema") == "astra_observation_response_v1" and response.get("status") == "ok"
            and response.get("model") == MODEL and response.get("request_id") == job["request_id"]
            and response.get("request_token") == request["request_token"] and response.get("request_sha256") == request_sha,
            "response_identity_differs")
    require(response.get("cli_receipt_file") == "cli_receipt.json", "missing_cli_receipt")
    receipt = read(folder / "cli_receipt.json")
    require(sha(folder / "cli_receipt.json") == response.get("cli_receipt_sha256"), "receipt_digest_differs")
    require(receipt.get("schema") == "astra_cli_receipt_v1" and receipt.get("cli_executed") is True
            and receipt.get("model") == receipt.get("model_requested") == MODEL
            and receipt.get("reasoning_effort") == "high" and receipt.get("status") == "ok"
            and receipt.get("returncode") == 0 and receipt.get("no_tool_calls") is True
            and receipt.get("output_validated") is True, "cli_receipt_not_valid_success")
    _command(receipt, job["arm"] == "V")
    require(receipt.get("modalities") == request["modalities"], "receipt_modalities_differ")
    bindings = receipt.get("requests", [])
    ids = [x["request_id"] for x in bindings]
    require(1 <= len(ids) <= 8 and len(ids) == len(set(ids)) and ids.count(job["request_id"]) == 1,
            "cli_batch_request_identity_invalid")
    expected_group = (job["policy_id"], job["arm"], job["task"], job["batch_group"])
    for bound in bindings:
        other = all_jobs[bound["request_id"]]
        require((other["policy_id"], other["arm"], other["task"], other["batch_group"]) == expected_group,
                "batch_crosses_policy_arm_task_or_group")
    binding = next(x for x in bindings if x["request_id"] == job["request_id"])
    require(binding["request_sha256"] == request_sha and binding["request_token"] == request["request_token"], "cli_request_binding_differs")
    require(receipt.get("stdout_file") == "cli_stdout.jsonl" and receipt.get("raw_answer_file") == "cli_final_output.json",
            "cli_artifact_names_differ")
    require(sha(folder / "cli_stdout.jsonl") == receipt.get("stdout_sha256")
            and sha(folder / "cli_final_output.json") == receipt.get("raw_answer_sha256"), "cli_artifact_hash_differs")
    inspected = _inspect_events(jsonl(folder / "cli_stdout.jsonl"))
    final = read(folder / "cli_final_output.json")
    require(not inspected["problems"] and (not inspected["models"] or inspected["models"] == [MODEL]), "cli_trace_error_tool_or_model")
    require(inspected["completed"] and final in inspected["messages"], "answer_absent_from_completed_trace")
    require(all(type(inspected["usage_totals"][k]) is int for k in ("input_tokens", "output_tokens")), "completed_trace_lacks_usage")
    require(isinstance(final, dict) and set(final) == {"requests"} and isinstance(final["requests"], list), "final_schema_differs")
    items = final["requests"]
    require(len(items) == len(ids) and {x["request_id"] for x in items} == set(ids), "final_ids_differ")
    for item in items:
        require(set(item) == {"request_id", "request_token", "subgoal_instruction", "rationale"}, "final_item_fields_differ")
        b = next(x for x in bindings if x["request_id"] == item["request_id"])
        require(item["request_token"] == b["request_token"], "final_token_differ")
        advice, rationale = item["subgoal_instruction"], item["rationale"]
        require(isinstance(advice, str) and 1 <= len(advice.strip()) <= 1500
                and isinstance(rationale, str) and len(rationale) <= 1500
                and not any(ord(c) < 32 and c not in "\n\t" for c in advice + rationale), "final_text_invalid")
    item = next(x for x in items if x["request_id"] == job["request_id"])
    require(item["subgoal_instruction"].strip() == response["subgoal_instruction"].strip()
            and item["rationale"] == response["rationale"], "published_text_differs_from_cli_answer")
    for k in USAGE_KEYS:
        require(receipt.get("usage_totals", {}).get(k) == inspected["usage_totals"][k], "receipt_usage_differs_from_trace:" + k)
    return append_instruction(original, response["subgoal_instruction"].strip()), dict(
        request_sha256=request_sha, response_sha256=response_sha, cli_receipt_sha256=response["cli_receipt_sha256"],
        model_requested=MODEL, reasoning_effort="high", models_reported=inspected["models"],
        provider_signed_model_attestation=False, actual_cli_and_application_verified=True)


def _assistance(output, episode, job, raw, traces, original, all_jobs, config):
    a = raw["assistance"]
    require(read(episode / "assistance.json") == a, "assistance_file_result_differ")
    tau = 16 * math.ceil(job["horizon"] / 32)
    require(a.get("schema") == "confirmatory_assistance_v1" and a.get("policy_id") == job["policy_id"]
            and a.get("case_id") == job["case_id"] and a.get("arm") == job["arm"]
            and a.get("trigger_step") == tau and a.get("policy_queries") == len(traces), "assistance_identity_or_query_count_differs")
    trigger = [q for q in traces if q["step"] == tau]
    result = dict(triggered=False, delivered=False, fallback=False, evidence_valid=True)
    if not trigger:
        require(raw["steps"] <= tau and a.get("trigger_reached") is False and a.get("requested") is False
                and a.get("applied") is False and a.get("fallback") is False and a.get("status") == "not_reached",
                "early_terminal_assistance_claim_invalid")
        require(not (episode / "intervention_receipt.json").exists(), "early_terminal_has_receipt")
        return result
    require(len(trigger) == 1 and a.get("trigger_reached") is True and a.get("step") == tau, "trigger_record_differs")
    trigger = trigger[0]
    expected_observation = {k: trigger[k] for k in ("step", "state_sha256", "image_sha256", "original_instruction_sha256")}
    require(a["trigger_observation"] == expected_observation, "trigger_observation_record_differs")
    _keepalive(episode, a, raw, tau)
    receipt = read(episode / "intervention_receipt.json")
    require(receipt == a["application_receipt"] and receipt.get("schema") == "confirmatory_intervention_receipt_v1"
            and receipt.get("arm") == job["arm"] and receipt.get("policy_id") == job["policy_id"]
            and receipt.get("case_id") == job["case_id"] and receipt.get("step") == tau
            and receipt.get("trigger_query_completed") is True and receipt.get("action_sha256") == trigger["action_sha256"],
            "application_receipt_invalid")
    result["triggered"] = True
    expected_instruction = original
    if job["arm"] == "C":
        require(a.get("applied") is False and a.get("fallback") is False and a.get("requested") is False
                and a.get("application_kind") == "identity_sham" and a.get("status") == "unchanged", "sham_not_unchanged")
    elif job["arm"] in ("R", "G"):
        subgoal = original.strip() if job["arm"] == "R" else GENERIC_INSTRUCTION
        expected_instruction = append_instruction(original, subgoal)
        require(a.get("applied") is True and a.get("fallback") is False and a.get("requested") is False
                and a.get("application_kind") == "static" and a.get("status") == "applied"
                and a.get("subgoal_instruction") == subgoal, "static_instruction_contract_differs")
        result["delivered"] = True
    elif a.get("fallback") is True:
        require(a.get("applied") is False and a.get("application_kind") == "original_fallback"
                and a.get("status") in ("timeout", "broker_error", "invalid_response", "transport_error"), "unclean_fallback")
        if a.get("requested") is True:
            _request(output, job, a, original, tau)
        if a["status"] == "timeout":
            require(a.get("wait_seconds", -1) >= config["response_wait_seconds"], "timeout_before_registered_wait")
        elif a["status"] in ("invalid_response", "transport_error"):
            require(bool(a.get("first_error")), "fallback_lacks_error_evidence")
        if a["status"] == "broker_error":
            require(a.get("response", {}).get("status") == "error", "broker_fallback_lacks_error_response")
        result["fallback"] = True
    else:
        require(a.get("applied") is True and a.get("status") == "applied" and a.get("requested") is True
                and a.get("application_kind") == "generated", "missing_generated_application")
        expected_instruction, details = _generated_evidence(output, job, a, original, tau, all_jobs)
        result.update(delivered=True, cli_evidence=details)
    expected_hash = text_sha(expected_instruction)
    require(receipt.get("original_instruction_sha256") == text_sha(original)
            and receipt.get("policy_instruction_sha256") == expected_hash
            and receipt.get("instruction_unchanged") == (expected_instruction == original)
            and receipt.get("applied") == result["delivered"] and receipt.get("fallback") == result["fallback"],
            "application_instruction_or_flags_differ")
    if result["delivered"]:
        require(a.get("policy_instruction") == expected_instruction, "assistance_policy_instruction_differs")
    require(all(q["instruction_sha256"] == (text_sha(original) if q["step"] < tau else expected_hash) for q in traces),
            "instruction_not_applied_or_not_persistent")
    result["wait_seconds"] = a.get("wait_seconds")
    return result


def normalize_episode(config, config_sha, job, output, all_jobs, authority_ok=True):
    base = dict(case_id=job["case_id"], task_name=job["task"], policy_id=job["policy_id"], arm=job["arm"],
                seed=job["env_seed"], horizon=job["horizon"], episode_id=job["id"], query_interval=16)
    episode = output / "episodes" / job["id"]
    paths = [p for p in (episode / "result.json", episode / "outcome.json") if p.exists()]
    normalized = dict(base, record_present=bool(paths), valid=False, success=None, steps=None,
                      terminal_reason="missing_record", initial_fingerprint={}, trace=[], assistance={})
    evidence = dict(episode_id=job["id"], valid=False, result_paths=[str(p) for p in paths])
    if not paths:
        evidence["reason"] = "missing_record"
        return normalized, evidence
    try:
        require(len(paths) == 1, "ambiguous_multiple_primary_result_files")
        path = paths[0]
        raw = read(path)
        evidence["result_sha256"] = sha(path)
        require(authority_ok, "config_source_authority_failed")
        require(raw.get("job") == job and raw.get("config_sha256") == config_sha
                and raw.get("manifest_sha256") == config["manifest_sha256"], "result_authority_differs")
        require(raw.get("status") == "completed", "infrastructure_or_incomplete_episode")
        require(type(raw.get("success")) is bool and type(raw.get("steps")) is int, "terminal_outcome_types_invalid")
        stats, steps = raw["stats"], raw["steps"]
        require(len(stats["episodes"]) == 1 and 1 <= steps <= job["horizon"], "invalid_official_episode_count_or_steps")
        ep = stats["episodes"][0]
        require(ep["seed"] == job["env_seed"] and ep["episode"] == ep["global_episode_index"] == 0
                and ep["success"] is raw["success"] and ep["steps"] == steps and stats["horizon"] == job["horizon"], "official_stats_differ")
        reason = raw["terminal_reason"]
        require((raw["success"] and reason == "success") or (not raw["success"] and
                ((reason == "horizon" and steps == job["horizon"]) or reason in ("terminated", "truncated"))), "terminal_reason_does_not_explain_early_stop")
        before = raw["native_reset_before"]
        require(before == raw["native_reset_after"] and before.get("native_reset") is True
                and before.get("reset_patch_applied") is False and before.get("source_sha256") == COUNTER_SHA
                and before.get("pythonhashseed") == "0", "native_reset_guard_invalid")
        require(raw["initial_replay"].get("natural_reset") is True and raw["initial_replay"].get("state_restored") is False,
                "initial_state_was_restored")
        guard_path = episode / "readback-guard" / "summary.json"
        guard = read(guard_path)
        require(sha(guard_path) == raw["guard_report_sha256"] and guard.get("status") == "closed"
                and guard.get("failure") is None and set(guard["gl_error_counts"]) <= {"after_draw:1281"}
                and guard["raw_frames_checked"] > 0 and guard["sentinel_checks"] > 0, "readback_guard_invalid")
        require(tuple(e["event"] for e in raw["events"]) == EVENT_ORDER
                and [e["sequence"] for e in raw["events"]] == list(range(len(EVENT_ORDER))), "activation_order_differs")
        index = raw["endpoint_index"]
        require(type(index) is int and 0 <= index < len(config["models"]), "invalid_endpoint_index")
        model = config["models"][index]
        require(model["policy_id"] == job["policy_id"], "wrong_policy_endpoint")
        require(sha(model["server_manifest"]) == model["server_manifest_sha256"], "server_manifest_changed")
        actual, expected = raw["server_identity"], model["hello"]
        require(all(actual.get(k) == expected.get(k) for k in HELLO_FIELDS), "server_hello_changed")
        require(actual.get("protocol") == "robocasa-multiplex-rng-v1" and actual.get("isolated_rng_stream") is True,
                "server_rng_contract_invalid")
        if job["policy_id"] == "B":
            require(bool(expected.get("stage_serving_identity")) and actual.get("stage_serving_identity") == expected["stage_serving_identity"]
                    and actual.get("loader_kind") == expected.get("loader_kind") == "completed_frozen_A_plus_online_stage_branch_v2", "B_identity_differs")
        else:
            require(not actual.get("stage_serving_identity") and not actual.get("lora_serving_identity"), "original_policy_has_adaptations")
        ack, after = raw["rng_ack"], raw["rng_after"]
        queries = (steps + 15) // 16
        require(ack["seed"] == after["seed"] == job["policy_seed"] and ack["requests_since_reset"] == 0
                and after["requests_since_reset"] == queries, "policy_rng_seed_or_query_count_differs")
        for state in (ack, after):
            require(all(state.get(k) == actual.get(k) for k in ("server_instance", "connection_id")), "rng_connection_changed")
        fingerprint, original = _initial(episode, raw)
        traces, normalized_trace = _traces(episode, steps, original)
        normalized.update(initial_fingerprint=fingerprint, trace=normalized_trace,
                          success=raw["success"], steps=steps, terminal_reason=reason)
        normalized["assistance"] = _assistance(output, episode, job, raw, traces, original, all_jobs, config)
        normalized["valid"] = True
        evidence.update(valid=True, reason="verified", query_count=queries, initial_verified=True,
                        assistance_verified=True, trace_sha256=sha(episode / "query_trace.jsonl"))
    except Exception as error:
        normalized["valid"] = False
        normalized["adapter_error"] = str(error)
        evidence.update(reason=str(error), error_type=type(error).__name__)
    return normalized, evidence


def collect_usage(output, all_jobs, config_sha, manifest_sha, broker_state=None):
    candidates = [(p, p.parent, False) for p in (output / "requests").glob("*/cli_receipt.json")]
    inventory_complete, inventory_errors, interrupted = False, [], []
    if broker_state is not None:
        broker_state = Path(broker_state)
        identity = read(broker_state / "identity.json")
        require(identity.get("remote_output") == str(output) and identity.get("config_sha256") == config_sha
                and identity.get("manifest_sha256") == manifest_sha and identity.get("model") == MODEL
                and identity.get("reasoning_effort") == "high", "broker_state_belongs_to_different_campaign")
        candidates += [(p, p.parent, True) for p in (broker_state / "batches").glob("*/receipt.json")]
        for started in (broker_state / "batches").glob("*/cli_started.json"):
            if not (started.parent / "receipt.json").exists():
                interrupted.append(str(started))
        completion_path = output / "completion.json"
        status_path = broker_state / "broker_status.json"
        terminal_workers = completion_path.exists() and read(completion_path).get("all_children_drained") is True
        terminal_broker = status_path.exists() and read(status_path).get("state") in ("finished", "stopped")
        inventory_complete = not interrupted and terminal_workers and terminal_broker
    calls: dict[str, dict[str, Any]] = {}
    seen_request_ids = set()
    for receipt_path, folder, local in candidates:
        try:
            r = read(receipt_path)
            require(r.get("schema") == "astra_cli_receipt_v1", "unknown_cli_receipt_schema")
            bindings = r.get("requests", [])
            require(bindings and all(x.get("request_id") in all_jobs for x in bindings), "receipt_not_bound_to_manifest")
            if r.get("cli_executed") is not True:
                continue
            stdout = folder / ("stdout.jsonl" if local else "cli_stdout.jsonl")
            require(stdout.exists() and sha(stdout) == r.get("stdout_sha256"), "usage_stdout_missing_or_digest_differs")
            inspected = _inspect_events(jsonl(stdout))
            key = json_sha({k: r.get(k) for k in ("started_at", "command", "requests", "stdout_sha256", "batch_manifest_sha256")})
            usage = inspected["usage_totals"]
            require(all(r.get("usage_totals", {}).get(k) == usage[k] for k in USAGE_KEYS), "usage_receipt_trace_disagreement")
            seen_request_ids.update(x["request_id"] for x in bindings)
            call = dict(invocation_key=key, status=r.get("status"), request_ids=[x["request_id"] for x in bindings],
                        receipt_sha256=sha(receipt_path), usage=usage, latency_seconds=r.get("latency_seconds"),
                        source_paths=[str(receipt_path)], transcript_problems=inspected["problems"])
            if key in calls:
                require(calls[key]["usage"] == usage and calls[key]["latency_seconds"] == call["latency_seconds"], "duplicate_invocation_receipts_conflict")
                calls[key]["source_paths"].append(str(receipt_path))
            else:
                calls[key] = call
        except Exception as error:
            inventory_errors.append(dict(path=str(receipt_path), error=str(error)))
    totals = {k: None for k in USAGE_KEYS}
    unknown_counters = {k: 0 for k in USAGE_KEYS}
    latency_sum, latency_unknown = 0.0, 0
    for call in calls.values():
        for key, value in call["usage"].items():
            if type(value) is int:
                totals[key] = (totals[key] or 0) + value
            else:
                unknown_counters[key] += 1
        latency = call["latency_seconds"]
        if isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0:
            latency_sum += latency
        else:
            latency_unknown += 1
    requested = {p.parent.name for p in (output / "requests").glob("*/request.json") if p.parent.name in all_jobs}
    return dict(schema="originx_confirmatory_cli_usage_v1", unique_observed_cli_invocations=len(calls),
                observed_status_counts=dict(Counter(c["status"] for c in calls.values())), known_usage_totals=totals,
                observed_calls_missing_counters=unknown_counters, known_invocation_latency_sum_seconds=latency_sum,
                observed_calls_missing_latency=latency_unknown, request_files=len(requested),
                requests_without_observed_cli_receipt=sorted(requested - seen_request_ids),
                broker_inventory_provided=broker_state is not None,
                inventory_complete=inventory_complete and not inventory_errors,
                complete_input_output_token_accounting=(inventory_complete and not inventory_errors
                                                        and not unknown_counters["input_tokens"] and not unknown_counters["output_tokens"]),
                interrupted_cli_starts_without_receipt=interrupted, invalid_receipts=inventory_errors,
                cost_usd=None, reasoning_tokens_included_in_output_not_added_again=True,
                interpretation="Known observed usage; unpublished/interrupted calls may be unknown. No inferred zero token usage or monetary price.",
                calls=sorted(calls.values(), key=lambda x: x["invocation_key"]))


def aggregate(config_path, *, broker_state=None, output_directory=None, bootstrap_samples=2000):
    config_path = Path(config_path).resolve()
    config = read(config_path)
    require(config.get("schema") == "originx_confirmatory_config_v1", "not_new_confirmatory_config")
    output = Path(config["output"]).resolve()
    root = Path(config["root"]).resolve()
    require(output.parent == root / "results" and output.name.startswith("originx-confirmatory-"), "refuse_historical_or_foreign_output")
    require(config_path.parent == output, "config_not_in_its_own_output")
    destination = output / "analysis" if output_directory is None else Path(output_directory).resolve()
    require(not any(x in str(destination).lower() for x in ("astra-native-reset-full", "astra-native-reset-continuation", "native-reset-b2500")),
            "refuse_overwriting_historical_analysis")
    config_sha = sha(config_path)
    manifest_path = Path(config["manifest"])
    require(manifest_path.resolve().parent == output and sha(manifest_path) == config["manifest_sha256"], "manifest_authority_changed")
    manifest = read(manifest_path)
    validate_manifest(manifest)
    require(type(config.get("development")) is bool and config["development"] == manifest["development"], "development_mode_differs")
    jobs = manifest["jobs"]
    all_jobs = {j["request_id"]: j for j in jobs}
    require(len(all_jobs) == len(jobs), "duplicate_request_identity")
    source_checks = []
    require(bool(config.get("source_sha256")), "empty_source_manifest")
    for name, expected in sorted(config["source_sha256"].items()):
        check = dict(path=name, expected_sha256=expected, valid=False)
        try:
            actual = sha(name)
            check.update(actual_sha256=actual, valid=actual == expected)
        except OSError as error:
            check["error"] = str(error)
        source_checks.append(check)
    authority_ok = all(x["valid"] for x in source_checks)
    normalized, audits = [], []
    for job in jobs:
        record, audit = normalize_episode(config, config_sha, job, output, all_jobs, authority_ok)
        normalized.append(record)
        audits.append(audit)
    rows = [dict(case_id=j["case_id"], task_name=j["task"], policy_id=j["policy_id"], arm=j["arm"],
                 seed=j["env_seed"], horizon=j["horizon"], episode_id=j["id"], query_interval=16) for j in jobs]
    report = analyze(rows, [r for r in normalized if r["record_present"]], N=manifest["case_count"],
                     bootstrap_samples=bootstrap_samples)
    usage = collect_usage(output, all_jobs, config_sha, config["manifest_sha256"], broker_state)
    report.update(config_sha256=config_sha, manifest_sha256=config["manifest_sha256"],
                  development=manifest["development"], scored_confirmation=not manifest["development"],
                  authority_valid=authority_ok, raw_evidence_valid=sum(x["valid"] for x in audits),
                  raw_evidence_invalid_reasons=dict(Counter(x["reason"] for x in audits if not x["valid"])),
                  cli_usage={k: v for k, v in usage.items() if k != "calls"},
                  inference_note="No complex unvalidated p-values. Conditional task-block CIs; McNemar is unadjusted sensitivity only.")
    write(destination / "authority.json", dict(schema="originx_confirmatory_aggregation_authority_v1", config_path=str(config_path),
          config_sha256=config_sha, manifest_sha256=config["manifest_sha256"], source_checks=source_checks, episode_checks=audits))
    write(destination / "normalized_records.json", normalized)
    write(destination / "normalized_manifest.json", rows)
    write(destination / "cli_usage.json", usage)
    write(destination / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--broker-state", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    report = aggregate(args.config, broker_state=args.broker_state, output_directory=args.output_directory,
                       bootstrap_samples=args.bootstrap_samples)
    print(json.dumps(dict(development=report["development"], cases=report["N_per_policy"],
                          planned_arm_outcomes=report["planned_arm_outcomes"], present=report["present_arm_records"],
                          raw_evidence_valid=report["raw_evidence_valid"], authority_valid=report["authority_valid"])))


if __name__ == "__main__":
    main()
