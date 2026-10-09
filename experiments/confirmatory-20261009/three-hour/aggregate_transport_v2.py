"""Opt-in audit adapter for completed transport recovery in new batch v2 only.

The frozen aggregator stays unchanged. Original/v1 raw errors remain invalid;
only a published v2 proof revalidated against unchanged raw bytes is accepted.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

OUTPUT_NAME = 'originx-confirmatory-multigpu-20261009-v2'
RECOVERY_SHA = 'ff19d688761fd3c6f000c9d9b0510c8237f968991f8fa6654a0458ba7cd6708b'


def require(value, message):
    if not value:
        raise ValueError(message)


def recovery_module():
    path = Path(__file__).with_name('recover_cli_transport_v1.py')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == RECOVERY_SHA, 'Recovery audit source differs')
    spec = importlib.util.spec_from_file_location('originx_aggregate_recovery_helper_v1', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def install(aggregate_module):
    """Keep aggregate/normalize_episode/collect_usage API; return same facade."""
    require(not getattr(aggregate_module, '_originx_transport_adapter_installed', False), 'Recovery adapter already installed')
    helper = recovery_module(); original = aggregate_module._generated_evidence

    def audited_generated(output, job, assistance, original_instruction, tau, all_jobs):
        output = Path(output)
        # No alteration to the original or failed v1 study, even if a proof is copied there.
        if output.name != OUTPUT_NAME:
            return original(output, job, assistance, original_instruction, tau, all_jobs)
        folder = output/'requests'/job['request_id']; receipt_path = folder/'cli_receipt.json'
        receipt = aggregate_module.read(receipt_path)
        if not receipt.get('recovery_audit_file'):
            return original(output, job, assistance, original_instruction, tau, all_jobs)
        require(receipt['recovery_audit_file'] == 'transport_recovery_audit.json', 'Unexpected recovery proof path')
        proof_path = folder/receipt['recovery_audit_file']
        require(not proof_path.is_symlink() and aggregate_module.sha(proof_path) == receipt.get('recovery_audit_sha256'), 'Recovery proof hash differs')
        require(receipt.get('stdout_file') == 'cli_stdout.jsonl' and receipt.get('raw_answer_file') == 'cli_final_output.json', 'Unexpected audit artifact paths')
        stdout = (folder/'cli_stdout.jsonl').read_bytes(); answer = (folder/'cli_final_output.json').read_bytes()
        helper.verify_published_recovery(receipt, aggregate_module.read(proof_path), stdout, answer, helper.validate_output_contract)
        events = helper.parse_events(stdout); expected_events_sha = helper.canonical_sha(events)
        strict_inspect = aggregate_module._inspect_events

        def exact_inspect(candidate):
            result = strict_inspect(candidate)
            require(helper.canonical_sha(candidate) == expected_events_sha, 'Unexpected nested transcript during recovery validation')
            helper.recovered_transport_trace(candidate)
            require(set(result['problems']) <= {'cli_trace_error'}, 'Recovery cannot suppress model/tool/protocol errors')
            result = dict(result); result['recovered_transport_problems_preserved'] = list(result['problems']); result['problems'] = []
            return result

        aggregate_module._inspect_events = exact_inspect
        try:
            # All original request, image, instruction, model, output-schema,
            # usage, no-tool and final application checks still run unchanged.
            return original(output, job, assistance, original_instruction, tau, all_jobs)
        finally:
            aggregate_module._inspect_events = strict_inspect

    aggregate_module._generated_evidence = audited_generated
    aggregate_module._originx_transport_adapter_installed = True
    aggregate_module._originx_transport_adapter_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return aggregate_module
