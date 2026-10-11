"""Strict audit of a CLI turn that completed after Windows connection reset.

Pure validation only: no subprocess, network, publication, marker deletion, or
CLI retry. Raw intermediate errors remain in the receipt and raw transcript.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re

MODEL = 'gpt-6-astra'
RECOVERY_SCHEMA = 'originx_cli_completed_transport_recovery_v1'
RECONNECT = re.compile(r'^Reconnecting\.\.\. ([1-5])/5 \(stream disconnected before completion: IO error: .+\(os error 10054\)\)$')


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def parse_events(raw):
    events = []
    for line in raw.decode('utf-8', 'strict').splitlines():
        if line.strip():
            row = json.loads(line); require(isinstance(row, dict), 'CLI event must be an object'); events.append(row)
    require(bool(events), 'Empty CLI transcript')
    return events


def recovered_transport_trace(events):
    """Recognize only recovered IO10054 events before one final completed turn."""
    require(isinstance(events, list) and events, 'No transcript events')
    terminal = [i for i, event in enumerate(events) if event.get('type') == 'turn.completed']
    require(len(terminal) == 1 and terminal[0] == len(events)-1, 'Require exactly one terminal completed turn at EOF')
    require(sum(e.get('type') == 'turn.started' for e in events) == 1, 'Multiple or missing turns')
    errors = []; messages = []; models = set()
    allowed = {'thread.started', 'turn.started', 'item.started', 'item.updated', 'item.completed', 'error', 'turn.completed'}
    started = next(i for i,e in enumerate(events) if e.get('type') == 'turn.started')
    for index, event in enumerate(events):
        require(event.get('type') in allowed, 'Unexpected/fatal/tool event: ' + str(event.get('type')))
        if event.get('type') == 'error':
            require(set(event) == {'type', 'message'} and isinstance(event['message'], str)
                    and RECONNECT.fullmatch(event['message']) and started < index < terminal[0],
                    'Error is not a narrowly recognized recovered connection reset')
            errors.append(dict(index=index, event=copy.deepcopy(event)))
        item = event.get('item')
        if item is not None:
            require(isinstance(item, dict) and item.get('type') in ('reasoning', 'agent_message'), 'Forbidden tool/item')
            if event.get('type') == 'item.completed' and item['type'] == 'agent_message':
                require(isinstance(item.get('text'), str), 'Agent output missing')
                messages.append(json.loads(item['text']))
        for candidate in (event, event.get('session', {}), event.get('turn', {})):
            if isinstance(candidate, dict) and isinstance(candidate.get('model'), str):
                models.add(candidate['model'])
    require(errors, 'No recovered transport error to audit')
    require(not models or models == {MODEL}, 'Wrong model in transcript')
    require(len(messages) == 1, 'Require one completed structured agent answer')
    usage = events[-1].get('usage', {})
    require(all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('input_tokens','output_tokens')), 'Complete token counters required')
    normalized = {k:usage.get(k) for k in ('input_tokens','cached_input_tokens','output_tokens')}
    reasoning = usage.get('reasoning_tokens', usage.get('reasoning_output_tokens'))
    if reasoning is None and isinstance(usage.get('output_tokens_details'), dict):
        reasoning = usage['output_tokens_details'].get('reasoning_tokens')
    normalized['reasoning_tokens'] = reasoning
    require(all(v is None or type(v) is int and v >= 0 for v in normalized.values()), 'Invalid usage counter')
    return dict(recovered_errors=errors, usage_totals=normalized, final_answer=messages[0])


def inspect_with_recovered_transport(raw, strict_inspector):
    """New broker only; final returncode/output checks still run afterwards."""
    inspected = strict_inspector(raw)
    if not inspected.get('event_errors'):
        return inspected
    try:
        review = recovered_transport_trace(parse_events(raw))
        require(not inspected.get('forbidden_items') and inspected.get('turn_completed') is True,
                'Incomplete or forbidden CLI trace')
        require(inspected.get('usage_totals') == review['usage_totals'], 'Usage normalization differs')
    except (ValueError, TypeError, KeyError, UnicodeError):
        return inspected
    result = copy.deepcopy(inspected)
    result['recovered_event_errors'] = result['event_errors']
    result['event_errors'] = []
    result['recovered_transport_candidate'] = True
    return result


def receipt_core(receipt):
    return {k:v for k,v in receipt.items() if k not in ('recovery_audit_file','recovery_audit_sha256')}


def validate_output_contract(value, ids, tokens):
    """Same structured response contract as the frozen broker; no imports."""
    require(isinstance(value, dict) and set(value) == {'requests'} and isinstance(value['requests'], list)
            and len(value['requests']) == len(ids), 'Structured answer shape/count differs')
    answer = {}
    for row in value['requests']:
        require(isinstance(row, dict) and set(row) == {'request_id','request_token','subgoal_instruction','rationale'}, 'Answer fields differ')
        name = row['request_id']; require(name in ids and name not in answer and row['request_token'] == tokens[name], 'Answer identity differs')
        advice, rationale = row['subgoal_instruction'], row['rationale']
        require(isinstance(advice, str) and 1 <= len(advice.strip()) <= 1500
                and isinstance(rationale, str) and len(rationale) <= 1500, 'Invalid advice or rationale')
        require(not any(ord(x) < 32 and x not in '\n\t' for x in advice+rationale), 'Unsupported output control character')
        answer[name] = dict(row, subgoal_instruction=advice.strip())
    return answer


def validate_completed_recovery(receipt, raw_stdout, raw_answer, validate_output):
    """Validate bytes and receipt from a finished new broker invocation."""
    require(receipt.get('schema') == 'astra_cli_receipt_v1' and receipt.get('cli_executed') is True,
            'Not an executed CLI receipt')
    require(type(receipt.get('returncode')) is int and receipt['returncode'] == 0
            and receipt.get('timed_out') is False and receipt.get('process_exception') is None
            and receipt.get('finished_at') and receipt.get('turn_completed') is True,
            'CLI did not terminate successfully')
    require(receipt.get('model') == receipt.get('model_requested') == MODEL
            and receipt.get('reasoning_effort') == 'high' and receipt.get('status') == 'ok'
            and receipt.get('output_validated') is True and receipt.get('no_tool_calls') is True
            and receipt.get('forbidden_items') == [] and receipt.get('event_errors') == [], 'Unvalidated/wrong-model receipt')
    require(receipt.get('models_reported') in ([], [MODEL]), 'Receipt model mismatch')
    command = receipt.get('command', [])
    require(command.count('--model') == 1 and command[command.index('--model')+1] == MODEL
            and [x for x in command if x.startswith('model_reasoning_effort=')] == ['model_reasoning_effort=high'], 'Command model/effort mismatch')
    require(hashlib.sha256(raw_stdout).hexdigest() == receipt.get('stdout_sha256')
            and hashlib.sha256(raw_answer).hexdigest() == receipt.get('raw_answer_sha256'), 'Saved output hashes differ')
    review = recovered_transport_trace(parse_events(raw_stdout))
    expected_errors = [json.dumps(e['event'], ensure_ascii=False)[:4000] for e in review['recovered_errors']]
    require(receipt.get('recovered_event_errors') == expected_errors
            and receipt.get('recovered_transport_candidate') is True, 'Original recovered events not preserved')
    require(receipt.get('usage_totals') == review['usage_totals'], 'Receipt token counters differ')
    answer = json.loads(raw_answer)
    require(answer == review['final_answer'], 'Final file not equal to terminal agent answer')
    bindings = receipt.get('requests', []); ids = [r['request_id'] for r in bindings]
    require(1 <= len(ids) <= 8 and len(set(ids)) == len(ids) and receipt.get('request_ids') == ids, 'Invalid request binding set')
    tokens = {r['request_id']:r['request_token'] for r in bindings}
    parsed = validate_output(answer, ids, tokens)
    return dict(schema=RECOVERY_SCHEMA, passed=True, rule='Only IO10054 reconnect events before a single successful completed turn',
        raw_stdout_sha256=receipt['stdout_sha256'], raw_answer_sha256=receipt['raw_answer_sha256'],
        receipt_core_sha256=canonical_sha(receipt_core(receipt)), request_ids=ids,
        recovered_errors=review['recovered_errors'], usage_totals=review['usage_totals'],
        completed_turn_count=1, cli_reinvoked=False, original_raw_errors_preserved=True), parsed


def verify_published_recovery(receipt, proof, raw_stdout, raw_answer, validate_output):
    expected, parsed = validate_completed_recovery(receipt, raw_stdout, raw_answer, validate_output)
    require(proof == expected, 'Published recovery audit differs from independent revalidation')
    return parsed
