"""Audit independent GR00T raw evidence before the frozen pure paired analyzer.

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

import importlib.util
import sys
sys.dont_write_bytecode = True

try:
    from .runner import validate_manifest, make_manifest, POLICY_ID, NAMESPACE, OUTPUT_NAMES
    from .assistance import STATE_FIELDS, append_instruction
    from .adapter import STATE_DIMS
except ImportError:
    from runner import validate_manifest, make_manifest, POLICY_ID, NAMESPACE, OUTPUT_NAMES
    from assistance import STATE_FIELDS, append_instruction
    from adapter import STATE_DIMS

ANALYSIS_SHA = "406b8916ba0d8b69f250362fe5ce0f8a73bb3c463f5f7a168c2c946a277c4009"
# Study ownership and request IDs belong to v2. The unchanged server, public
# checkpoint, official implementation and its seven source pins remain v1.
SERVICE_NAMESPACE = 'originx_gr00t_confirmation_20261009'
SERVICE_SOURCE_FILES = ('__init__.py','adapter.py','wire.py','core.py','server.py','client.py','parity_probe.py')

def load_analysis(path):
    require(sha(path) == ANALYSIS_SHA, "frozen_pure_analysis_source_changed")
    spec = importlib.util.spec_from_file_location("gr00t_frozen_pure_analysis", path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.analyze



MODEL = "gpt-6-astra"
COUNTER_SHA = "77f992d01aa1ae5f21ed7f170d0aab9c303c32148f4b427651c04912be8ce68a"
CAMERAS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")
CAMERA_NAMES = ("left", "right", "wrist")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
EVENT_ORDER = ("reset_requested", "reset_completed", "initial_fingerprints_recorded",
               "assignment_selected", "server_connected", "rng_ack", "first_infer")
PROTOCOL = 'originx-gr00t-rng-v1'
GPU_UUID = 'GPU-35278907-aae0-4ccc-e0fc-c25cb3a8bd17'
MODEL_REVISION = 'c484448aba1a9b60a04c9b0ca117241518ea69f3'
SOURCE_COMMIT = '9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10'
CHECKPOINT_PINS = {
    'config.json': '6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526',
    'model.safetensors.index.json': 'bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746',
    'experiment_cfg/metadata.json': '8be0fc606c9220356bad497feefc4ae4daaa05670acccc31caecaa7ae5590b69',
    'model-00001-of-00002.safetensors': '08f1891947973e2e5ec2422201cd90261806f77f2634f9ec0477c27aa5a4fe42',
    'model-00002-of-00002.safetensors': 'deb9c9cf40cd8983a7779af85341f6344db15f65bf3307073a0e3c3085450435',
}
USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


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
            require(image.dtype == np.uint8 and image.shape == (256, 256, 3), "invalid_initial_rgb")
            require(array_sha(image) == initial["rgb"][name], "initial_rgb_array_hash_differs")
        for name in ("proprioception", "physics_state"):
            require(array_sha(arrays[name]) == initial[name], "initial_" + name + "_array_hash_differs")
        require(arrays['proprioception'].shape == (14,), 'evaluator_initial_proprioception_not_14D')
    require(initial.get('native_state_representation') == 'named_float32_16D_with_raw_quaternions'
            and initial.get('native_observation_file') == 'initial_native_observation.npz'
            and sha(episode / 'initial_native_observation.npz') == initial.get('native_observation_sha256'),
            'native_initial_observation_file_or_representation_differs')
    require(set(initial['native_state_sha256']) == set(STATE_DIMS), 'native_initial_state_fields_differ')
    with np.load(episode / 'initial_native_observation.npz', allow_pickle=False) as arrays:
        require(set(arrays.files) == set(CAMERAS) | set(STATE_DIMS), 'native_initial_npz_fields_differ')
        for name in CAMERAS:
            require(arrays[name].shape == (256, 256, 3) and arrays[name].dtype == np.uint8
                    and array_sha(arrays[name]) == initial['rgb'][name], 'native_initial_RGB_differs')
        for name, dimension in STATE_DIMS.items():
            require(arrays[name].shape == (dimension,) and arrays[name].dtype == np.float32
                    and np.isfinite(arrays[name]).all()
                    and array_sha(arrays[name]) == initial['native_state_sha256'][name],
                    'native_initial_state_array_hash_or_shape_differs:' + name)
    result = {"rgb:" + name: initial["rgb"][name] for name in CAMERAS}
    result.update({name: initial[name] for name in ("proprioception", "physics_state", "xml_sha256")})
    result["instruction_sha256"] = text_sha(initial["instruction"])
    result.update({'native:' + name: digest for name, digest in initial['native_state_sha256'].items()})
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
        require(q.get('state_representation') == 'native_named_float32_16D_with_raw_quaternions'
                and q.get('policy_context_frames') == 1
                and set(q.get('native_state_sha256', {})) == set(STATE_DIMS), 'query_native_16D_contract_differs')
        require(q.get('raw_observation_sha256') == json_sha(dict(state=q['native_state_sha256'], images=q['image_sha256'])),
                'query_raw_observation_composite_hash_differs')
        hashes = [q[k] for k in ("state_sha256", "instruction_sha256", "action_sha256")]
        hashes += list(q["image_sha256"].values()) + list(q['native_state_sha256'].values())
        hashes += [q.get('evaluator_state_history_sha256')]
        require(all(isinstance(v, str) and HEX64.fullmatch(v) for v in hashes), "malformed_query_digest")
        normalized.append(dict(step=q["step"], state_hash=q["state_sha256"],
                               raw_observation_hash=q['raw_observation_sha256'],
                               instruction_hash=q["instruction_sha256"], action_hash=q["action_sha256"]))
    return raw, normalized


def _rng_schema(state):
    require(state.get('protocol') == PROTOCOL and type(state.get('seed')) is int
            and type(state.get('requests_since_reset')) is int and state['requests_since_reset'] >= 0,
            'malformed_gr00t_rng_protocol_seed_count')
    for key in ('server_instance', 'connection_id'):
        require(isinstance(state.get(key), str) and state[key], 'missing_rng_connection_identity')
    cuda = state.get('torch_cuda_rng_sha256')
    require(isinstance(cuda, list) and len(cuda) == 1, 'GR00T_must_have_exactly_one_CUDA_rng')
    digests = [state.get(k) for k in ('model_config_sha256', 'python_rng_sha256', 'numpy_rng_sha256', 'torch_cpu_rng_sha256')] + cuda
    require(all(isinstance(v, str) and HEX64.fullmatch(v) for v in digests), 'malformed_rng_digest')


def _episode_owner(episode, config, config_sha, job, output):
    owner = read(episode / 'owner.json')
    require(owner == read(output / 'processes' / (job['id'] + '.json')), 'episode_dispatch_owner_differs')
    require(type(owner.get('pid')) is int and owner['pid'] > 0
            and type(owner.get('process_start_ticks')) is int and owner['process_start_ticks'] > 0
            and owner.get('cwd') == config['root'], 'malformed_episode_process_identity')
    command = owner.get('command')
    require(isinstance(command, list) and command[:5] == [config['python'], '-u', '-m', NAMESPACE + '.runner', 'episode'],
            'episode_owner_executable_or_module_differs')
    for flag, expected in (('--config', str(output / 'config.json')), ('--config-sha', config_sha),
                           ('--index', str(job['assignment_index']))):
        require(command.count(flag) == 1 and command[command.index(flag) + 1] == expected, 'episode_owner_argument_differs:' + flag)
    require(command.count('--model-index') == 1, 'episode_owner_model_index_missing')
    index = int(command[command.index('--model-index') + 1])
    claim = read(output / 'claims' / (job['id'] + '.json'))
    require(claim.get('job') == job and claim.get('command') == command and claim.get('config_sha256') == config_sha
            and claim.get('model_index') == index, 'episode_claim_binding_differs')
    return index


def _model_identity(config, model, actual):
    require(config.get('service_namespace') == SERVICE_NAMESPACE, 'service_namespace_must_remain_explicit_v1')
    require(model.get('policy_id') == POLICY_ID and model.get('schema') == 'originx_gr00t_service_profile_v1'
            and model.get('ready') is True, 'wrong_GR00T_endpoint_profile')
    require(sha(model['server_manifest']) == model['server_manifest_sha256'], 'server_manifest_changed')
    saved = read(model['server_manifest'])
    require(all(model.get(k) == v for k, v in saved.items()), 'configuration_service_profile_differs')
    require(read(Path(model['server_manifest']).parent / 'owner.json') == model['owner'], 'model_owner_record_differs')
    owner = model['owner']; command = owner.get('command', [])
    base = Path(config['root']) / SERVICE_NAMESPACE
    require(owner.get('namespace') == SERVICE_NAMESPACE and owner.get('policy_id') == POLICY_ID
            and owner.get('gpu_uuid') == GPU_UUID and owner.get('cwd') == str(base)
            and type(owner.get('pid')) is int and type(owner.get('process_start_ticks')) is int
            and command and command[0] == str(base / 'env/bin/python') and command.count('-m') == 1
            and command[command.index('-m') + 1] == SERVICE_NAMESPACE + '.server', 'model_owner_identity_or_module_differs')
    expected = model['hello']
    require(all(actual.get(k) == v for k, v in expected.items()), 'server_hello_changed')
    require(isinstance(actual.get('connection_id'), str) and actual['connection_id'], 'missing_episode_connection')
    require(actual.get('protocol') == PROTOCOL and actual.get('policy_id') == POLICY_ID
            and actual.get('adapters_loaded') is False and actual.get('denoising_steps') == 4
            and actual.get('replan_steps') == actual.get('action_horizon') == 16
            and actual.get('observation_frames') == 1 and actual.get('camera_count') == 3
            and actual.get('action_dim') == 12 and actual.get('model_action_dim') == 32
            and actual.get('serial_forward') is True and actual.get('isolated_rng_stream') is True
            and actual.get('rng_isolation') == 'python_numpy_torch_cpu_all_cuda_per_connection'
            and actual.get('inference_optimization') == 'none' and actual.get('gpu_uuid') == GPU_UUID
            and actual.get('cuda_visible_device_count') == 1
            and actual.get('model_path') == config['checkpoint']
            and actual.get('model_revision') == MODEL_REVISION and actual.get('official_source_commit') == SOURCE_COMMIT
            and actual.get('model_assets_sha256') == CHECKPOINT_PINS
            and actual.get('model_config_sha256') == CHECKPOINT_PINS['config.json'], 'GR00T_policy_checkpoint_contract_invalid')
    identity = dict(model['identity_manifest'])
    require(set(identity.get('service_sources_sha256', {})) == set(SERVICE_SOURCE_FILES),
            'service_source_inventory_must_be_all_seven_v1_modules')
    require(all(config.get('source_sha256', {}).get(str(base / name)) == digest
                for name, digest in identity['service_sources_sha256'].items()),
            'service_source_identity_not_bound_to_config')
    for key in ('server_instance', 'torch_version', 'cuda_visible_device_count'):
        require(identity.pop(key) == actual[key], 'identity_manifest_runtime_fields_differ')
    digest = identity.pop('asset_source_identity_sha256')
    require(text_sha(json.dumps(identity, sort_keys=True)) == digest == actual['asset_source_identity_sha256'],
            'asset_source_identity_digest_differs')
    compact = {k: v for k, v in identity.items() if k not in ('official_source_sha256', 'service_sources_sha256')}
    require(all(actual.get(k) == v for k, v in compact.items()), 'compact_identity_differs_from_complete_manifest')


def _keepalive(episode, assistance, raw, tau):
    rows = jsonl(episode / "policy-keepalive.jsonl")
    require(len(rows) >= 2 and rows[0]["phase"] == "enter" and rows[-1]["phase"] == "exit", "missing_keepalive_enter_exit")
    require(all(x["phase"] == "periodic" for x in rows[1:-1]), "unexpected_keepalive_phase")
    baseline = None
    for i, row in enumerate(rows):
        require(row.get('schema') == 'originx_gr00t_same_socket_keepalive_v1'
                and row["sequence"] == i and row["status"] == "ok" and row["same_thread"] is True
                and row["control"] == "rng_state" and row["reconnect"] is False and row["reset"] is False,
                "invalid_keepalive_record")
        response = row["response"]
        require(text_sha(json.dumps(response, sort_keys=True)) == row["response_sha256"], "keepalive_response_hash_differs")
        state = {k: response[k] for k in STATE_FIELDS}
        require(state == row["state"], "keepalive_normalization_differs")
        _rng_schema(state)
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
    mapping = read(output / 'episodes' / job['id'] / 'request_mapping.json')
    require(mapping.get('schema') == 'gr00t_confirmatory_request_mapping_v1'
            and mapping.get('request_id') == mapping.get('local_request_id') == job['request_id']
            and all(mapping.get(k) == job[k] for k in ('case_id', 'policy_id', 'arm')), 'private_request_mapping_differs')
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
                import numpy as np
                key = CAMERAS[CAMERA_NAMES.index(camera['name'])]
                require(array_sha(np.asarray(image)[None]) == assistance['trigger_observation']['image_sha256'][key],
                        'exported_PNG_pixels_differ_from_actual_trigger_RGB')
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
    require(job['arm'] in ('C', 'V') and job['policy_id'] == POLICY_ID
            and a.get("schema") == "originx_gr00t_assistance_v1" and a.get("policy_id") == job["policy_id"]
            and a.get("case_id") == job["case_id"] and a.get("arm") == job["arm"]
            and a.get("trigger_step") == tau and a.get("policy_queries") == len(traces), "assistance_identity_or_query_count_differs")
    trigger = [q for q in traces if q["step"] == tau]
    result = dict(triggered=False, delivered=False, fallback=False, evidence_valid=True)
    if not trigger:
        require(raw["steps"] <= tau and a.get("trigger_reached") is False and a.get("requested") is False
                and a.get("applied") is False and a.get("fallback") is False and a.get("status") == "not_reached",
                "early_terminal_assistance_claim_invalid")
        require(not (episode / "intervention_receipt.json").exists(), "early_terminal_has_receipt")
        require(not (output / 'requests' / job['request_id']).exists(), 'early_terminal_exported_assistant_request')
        require(all(q['instruction_sha256'] == text_sha(original) and q.get('at_trigger') is False for q in traces),
                'early_terminal_instruction_or_trigger_changed')
        return result
    require(len(trigger) == 1 and a.get("trigger_reached") is True and a.get("step") == tau, "trigger_record_differs")
    trigger = trigger[0]
    expected_observation = {k: v for k, v in trigger.items() if k not in ('at_trigger', 'instruction_sha256', 'action_sha256')}
    require(a["trigger_observation"] == expected_observation, "trigger_observation_record_differs")
    require(all(q.get('at_trigger') is (q['step'] == tau) for q in traces), 'trigger_flags_differ')
    _keepalive(episode, a, raw, tau)
    receipt = read(episode / "intervention_receipt.json")
    require(receipt == a["application_receipt"] and receipt.get("schema") == "originx_gr00t_intervention_receipt_v1"
            and receipt.get("arm") == job["arm"] and receipt.get("policy_id") == job["policy_id"]
            and receipt.get("case_id") == job["case_id"] and receipt.get("step") == tau
            and receipt.get("trigger_query_completed") is True and receipt.get("action_sha256") == trigger["action_sha256"],
            "application_receipt_invalid")
    result["triggered"] = True
    expected_instruction = original
    if job["arm"] == "C":
        require(a.get("applied") is False and a.get("fallback") is False and a.get("requested") is False
                and a.get("application_kind") == "identity_sham" and a.get("status") == "unchanged", "sham_not_unchanged")
        require(not (output / 'requests' / job['request_id']).exists(), 'control_exported_assistant_request')
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
    require(receipt.get('request_id') == (job['request_id'] if a.get('requested') else None)
            and receipt.get('request_sha256') == a.get('request_sha256')
            and receipt.get('response_sha256') == a.get('response_sha256'), 'application_request_response_binding_differs')
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
        require(raw.get('schema') == 'originx_gr00t_episode_v1' and raw.get('namespace') == NAMESPACE,
                'wrong_episode_schema_or_namespace')
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
        require(read(episode / 'initial_replay.json') == raw['initial_replay'], 'initial_replay_file_differs')
        guard_path = episode / "readback-guard" / "summary.json"
        guard = read(guard_path)
        require(sha(guard_path) == raw["guard_report_sha256"] and guard.get("status") == "closed"
                and guard.get("failure") is None and set(guard["gl_error_counts"]) <= {"after_draw:1281"}
                and guard["raw_frames_checked"] > 0 and guard["sentinel_checks"] > 0, "readback_guard_invalid")
        require(tuple(e["event"] for e in raw["events"]) == EVENT_ORDER
                and [e["sequence"] for e in raw["events"]] == list(range(len(EVENT_ORDER))), "activation_order_differs")
        index = raw["endpoint_index"]
        require(type(index) is int and 0 <= index < len(config["models"]), "invalid_endpoint_index")
        require(_episode_owner(episode, config, config_sha, job, output) == index, 'owner_endpoint_index_differs')
        model = config["models"][index]
        actual = raw['server_identity']
        _model_identity(config, model, actual)
        ack, after = raw["rng_ack"], raw["rng_after"]
        queries = (steps + 15) // 16
        require(ack["seed"] == after["seed"] == job["policy_seed"] and ack["requests_since_reset"] == 0
                and after["requests_since_reset"] == queries, "policy_rng_seed_or_query_count_differs")
        for state in (ack, after):
            _rng_schema(state)
            require(all(state.get(k) == actual.get(k) for k in ("server_instance", "connection_id", 'protocol', 'model_config_sha256')),
                    "rng_connection_changed")
        require(raw.get('native_tap_steps') == steps and raw.get('policy_context_frames') == 1, 'native_tap_or_policy_context_differs')
        fingerprint, original = _initial(episode, raw)
        traces, normalized_trace = _traces(episode, steps, original)
        import numpy as np
        with np.load(episode / 'initial_native_observation.npz', allow_pickle=False) as arrays:
            first = traces[0]
            require(first['state_sha256'] == array_sha(np.concatenate([arrays[name].astype(np.float32) for name in STATE_DIMS]))
                    and first['native_state_sha256'] == raw['initial_scene']['native_state_sha256']
                    and all(first['image_sha256'][name] == array_sha(arrays[name][None]) for name in CAMERAS),
                    'first_query_does_not_match_exact_native_initial_observation')
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
    return dict(schema="originx_gr00t_confirmatory_cli_usage_v1", unique_observed_cli_invocations=len(calls),
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
    require(config.get("schema") == "originx_gr00t_confirmatory_config_v1"
            and config.get('namespace') == NAMESPACE and config.get('service_namespace') == SERVICE_NAMESPACE
            and config.get('policy_id') == POLICY_ID, "not_new_GR00T_confirmatory_config")
    output = Path(config["output"]).resolve()
    root = Path(config["root"]).resolve()
    require(output.parent == root / "results" and output.name in OUTPUT_NAMES.values(), "refuse_historical_or_foreign_output")
    require(config_path.parent == output, "config_not_in_its_own_output")
    destination = output / "analysis" if output_directory is None else Path(output_directory).resolve()
    require(destination.parent == output and destination.name.startswith('analysis'), 'analysis_must_stay_in_own_GR00T_output')
    require(not destination.exists(), 'preserve_existing_aggregate_use_new_analysis_directory')
    config_sha = sha(config_path)
    manifest_path = Path(config["manifest"])
    require(manifest_path.resolve().parent == output and sha(manifest_path) == config["manifest_sha256"], "manifest_authority_changed")
    manifest = read(manifest_path)
    validate_manifest(manifest)
    require(type(config.get("development")) is bool and config["development"] == manifest["development"], "development_mode_differs")
    require(output.name == OUTPUT_NAMES[config['development']], 'development_namespace_differs')
    reference_path = Path(config['reference_manifest'])
    require(sha(reference_path) == config['reference_manifest_sha256'] == manifest.get('source_manifest_sha256')
            and str(reference_path) == manifest.get('source_manifest_path'), 'fresh_source_roster_authority_differs')
    projected = make_manifest(read(reference_path), config['development'])
    require({k: v for k, v in manifest.items() if k not in ('source_manifest_path', 'source_manifest_sha256')} == projected,
            'manifest_is_not_exact_outcome_blind_CV_projection')
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
    freeze_path = Path(config['source_freeze'])
    require(sha(freeze_path) == config['source_freeze_sha256'], 'independent_source_freeze_changed')
    freeze = read(freeze_path)
    require(freeze.get('schema') == 'originx_gr00t_source_freeze_v1' and freeze.get('namespace') == NAMESPACE,
            'wrong_independent_freeze_schema')
    require(freeze.get('broker_source_sha256') == config.get('broker_source_sha256'), 'broker_freeze_digest_differs')
    for name, expected in freeze['source_sha256'].items():
        require(Path(name).name == name, 'unsafe_frozen_source_name')
        path = freeze_path.parent / name
        require(config['source_sha256'].get(str(path)) == expected, 'frozen_module_not_in_config_source_inventory')
    analysis_path = root / 'originx_confirmatory_20261009/analysis.py'
    analyze = load_analysis(analysis_path)
    source_checks.append(dict(path=str(analysis_path), expected_sha256=ANALYSIS_SHA, actual_sha256=sha(analysis_path), valid=True,
                              scope='unchanged pure statistics only; no main raw-normalizer reuse'))
    # The complete public checkpoint and official implementation are verified
    # once per aggregation, never loaded into a model or copied into another policy.
    seen_assets = {}
    for model in config['models']:
        _model_identity(config, model, dict(model['hello'], connection_id='aggregation_identity_only'))
        identity = model['identity_manifest']
        for name, expected in CHECKPOINT_PINS.items():
            seen_assets[str(Path(config['checkpoint']) / name)] = expected
        for name, expected in identity['official_source_sha256'].items():
            path = root / SERVICE_NAMESPACE / 'source' / name
            require(path.resolve().is_relative_to(root / SERVICE_NAMESPACE / 'source'), 'official_source_inventory_escaped')
            seen_assets[str(path)] = expected
        for name, expected in identity['service_sources_sha256'].items():
            require(Path(name).name == name and config['source_sha256'].get(str(root / SERVICE_NAMESPACE / name)) == expected,
                    'service_source_identity_not_bound_to_config')
    for name, expected in sorted(seen_assets.items()):
        check = dict(path=name, expected_sha256=expected, valid=False, scope='checkpoint_and_official_implementation')
        try:
            check['actual_sha256'] = sha(name); check['valid'] = check['actual_sha256'] == expected
        except OSError as error:
            check['error'] = str(error)
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
    report.update(schema='originx_gr00t_confirmation_analysis_v1', config_sha256=config_sha, manifest_sha256=config["manifest_sha256"],
                  development=manifest["development"], scored_confirmation=not manifest["development"],
                  authority_valid=authority_ok, raw_evidence_valid=sum(x["valid"] for x in audits),
                  raw_evidence_invalid_reasons=dict(Counter(x["reason"] for x in audits if not x["valid"])),
                  cli_usage={k: v for k, v in usage.items() if k != "calls"},
                  policy_family='GR00T_N1_5', native_state_dimensions=16, policy_context_frames=1,
                  resource_budget_metadata={key: config[key] for key in ('budget', 'shared_budget_evidence', 'shared_budget_evidence_sha256') if key in config},
                  cli_usage_scope='This GR00T study only; prior-study budget debits are metadata and are not added to observed usage or latency.',
                  development_interpretation='Infrastructure validation only; reused disclosed development IDs; not policy effectiveness.' if manifest['development'] else None,
                  trace_evidence_scope='Saved initial raw arrays and exported trigger RGB bytes are verified. Later state/action hashes are records from the source-pinned producer, not archived full action arrays.',
                  inference_note="Full fixed denominator and regressions retained. Conditional task-block CIs; McNemar is unadjusted sensitivity only; no superiority claim.")
    write(destination / "authority.json", dict(schema="originx_gr00t_aggregation_authority_v1", config_path=str(config_path),
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
