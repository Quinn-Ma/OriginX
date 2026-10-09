"""Join sealed original/continuation evidence on the user-amended 50-case cohort.

No retry, outcome selection, endpoint invocation, or overwrite is performed.
Unchanged original normalization and paired analysis validate real evidence.
All pre-amendment CLI cost, including outside-cohort work, remains in the ledger.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
CONTINUATION_SHA = '142b0ac2bdccd4edffc7515134227d2961d9be7796734e52f62601fcf0b18968'
OLD_NAME = 'originx-confirmatory-20261009-v1'
NEW_NAME = 'originx-confirmatory-multigpu-20261009-v1'
FINAL_NAME = 'originx-three-hour-amendment-20261009-v1'
USAGE_KEYS = ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_tokens')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def load_continuation(root=ROOT):
    path = Path(root) / 'continue_multigpu_v1.py'
    require(sha(path) == CONTINUATION_SHA, 'Continuation analysis helper changed')
    spec = importlib.util.spec_from_file_location('reduced_continuation_verified', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def claimed(output, config_sha, jobs):
    answer = {}
    for path in sorted((Path(output) / 'claims').glob('*.json')):
        require(not path.is_symlink(), 'Claim symlink forbidden')
        row = read(path)
        require(path.stem in jobs and row.get('job') == jobs[path.stem]
                and row.get('config_sha256') == config_sha, 'Claim authority differs: ' + str(path))
        answer[path.stem] = row
    occupied = {p.name for p in (Path(output) / 'episodes').iterdir()} if (Path(output) / 'episodes').exists() else set()
    occupied.update(p.stem for p in (Path(output) / 'errors').glob('*.json'))
    occupied.update(p.stem for p in (Path(output) / 'processes').glob('*.json'))
    require(occupied <= set(answer), 'Unclaimed episode/process/error in batch')
    return answer


def merge_records(manifest, selection, old_records, new_records, old_claims, new_claims):
    """The claim provenance fixes precedence even if another result looks better."""
    jobs = {j['id']: j for j in manifest['jobs']}
    require(not set(old_claims).intersection(new_claims), 'Duplicate attempted assignment across batches')
    require(set(old_claims) == set(selection['excluded_claimed_ids']), 'Sealed old claim set changed')
    require(set(new_claims) <= set(selection['selected_ids']), 'New assignment outside fixed continuation selection')
    tables = []
    for records in (old_records, new_records):
        table = {r['episode_id']: r for r in records}
        require(len(table) == len(records) == len(jobs) and set(table) == set(jobs), 'Normalized records must cover every original assignment once')
        for key, row in table.items():
            job = jobs[key]
            require(all(row.get(k) == job.get(v) for k, v in
                [('case_id', 'case_id'), ('task_name', 'task'), ('policy_id', 'policy_id'),
                 ('arm', 'arm'), ('seed', 'env_seed'), ('horizon', 'horizon')]), 'Normalized identity differs')
            require(isinstance(row.get('trace'), list), 'Expected embedded normalized trace; external path cannot be silently rebased')
        tables.append(table)
    old, new = tables; target = set(selection['target_case_ids']); combined = []; provenance = []
    for job in manifest['jobs']:
        key = job['id']; source = 'old' if key in old_claims else 'new'
        row = copy.deepcopy(old[key] if source == 'old' else new[key])
        # Any observed artifact without the corresponding claim is inadmissible.
        if row.get('record_present'):
            require(key in (old_claims if source == 'old' else new_claims), 'Present result without authoritative claim')
        combined.append(row)
        provenance.append(dict(episode_id=key, source_batch=source, claimed=key in old_claims or key in new_claims,
                               in_reduced_cohort=job['case_id'] in target))
    reduced = [r for r in combined if r['case_id'] in target]
    retained = [copy.deepcopy(old[key]) for key in old_claims if jobs[key]['case_id'] not in target]
    require(len(reduced) == len(target) * 7, 'Reduced cohort does not contain all seven arms')
    return combined, reduced, retained, provenance


def merge_usage(usages):
    """Deduplicate actual invocations, never request count or repeated copies."""
    calls = {}; invalid = []; interrupted = []; complete = True
    for index, usage in enumerate(usages):
        complete = complete and usage.get('broker_inventory_provided') is True and usage.get('inventory_complete') is True
        invalid.extend(dict(batch=index, **item) for item in usage.get('invalid_receipts', []))
        interrupted.extend(usage.get('interrupted_cli_starts_without_receipt', []))
        for call in usage.get('calls', []):
            key = call.get('invocation_key'); require(isinstance(key, str) and key, 'Invocation key missing')
            if key in calls:
                prior = calls[key]
                require(all(prior.get(k) == call.get(k) for k in ('usage', 'latency_seconds', 'status', 'request_ids')),
                        'Duplicate invocation receipts disagree')
                prior['source_paths'] = sorted(set(prior.get('source_paths', []) + call.get('source_paths', [])))
                prior['source_batches'] = sorted(set(prior['source_batches'] + [index]))
            else:
                calls[key] = dict(copy.deepcopy(call), source_batches=[index])
    totals = {k: None for k in USAGE_KEYS}; unknown = {k: 0 for k in USAGE_KEYS}
    latency_sum = 0.; latency_unknown = 0
    for call in calls.values():
        for key in USAGE_KEYS:
            value = call.get('usage', {}).get(key)
            if type(value) is int and value >= 0:
                totals[key] = (totals[key] or 0) + value
            else:
                unknown[key] += 1
        latency = call.get('latency_seconds')
        if type(latency) in (int, float) and math.isfinite(latency) and latency >= 0:
            latency_sum += latency
        else:
            latency_unknown += 1
    complete = complete and not invalid and not interrupted
    return dict(schema='originx_amendment_cli_usage_v1', unique_observed_cli_invocations=len(calls),
        known_usage_totals=totals, observed_calls_missing_counters=unknown,
        known_invocation_latency_sum_seconds=latency_sum, observed_calls_missing_latency=latency_unknown,
        inventory_complete=complete,
        complete_input_output_token_accounting=complete and not unknown['input_tokens'] and not unknown['output_tokens'],
        invalid_receipts=invalid, interrupted_cli_starts_without_receipt=sorted(set(interrupted)),
        reasoning_tokens_included_in_output_not_added_again=True, cost_usd=None,
        scope='All original and continuation CLI usage, including work outside the amended 50-case cohort',
        no_inferred_zero_counters=True, calls=sorted(calls.values(), key=lambda x: x['invocation_key']))


def verify_terminal_batch(output, config, continuation):
    finished = read(output / 'campaign-finished.json'); completion = read(output / 'completion.json')
    require(finished.get('development') is False and finished.get('phase') in ('completed', 'failed', 'batch_terminated'), 'Batch not terminal')
    require(completion.get('all_children_drained') is True, 'Workers not fully drained')
    paths = set(output.glob('*.owner.json')) | {output/'campaign.owner.json', output/'rollout.owner.json', output/'launch.json'}
    paths.update((output/'processes').glob('*.json')); paths.update((output/'episodes').glob('*/owner.json'))
    for path in paths:
        require(continuation.owner_inactive(read(path)), 'Recorded batch owner remains alive: ' + str(path))
    for model in config['models']:
        require(continuation.owner_inactive(model['owner']), 'Recorded model remains alive; cleanup must finish first')
    return {str(p): sha(p) for p in paths | {output/'campaign-finished.json', output/'completion.json'}}


def verify_sealed_broker(terminal_path, terminal_sha, state, output, config_sha, manifest_sha, continuation):
    continuation.checked_file(terminal_path, terminal_sha, ROOT)
    # Strict existing terminal audit proves no outstanding owned CLI. Actual
    # token completeness is reported separately by original collect_usage.
    return continuation.verify_broker_terminal(terminal_path, state, output, config_sha, manifest_sha)


def verified_normalization(directory, config_path, manifest, claim_ids, aggregate):
    """Recheck sources and existing raw normalization without rewriting it."""
    config = read(config_path); cs = sha(config_path)
    authority = read(directory/'authority.json'); report = read(directory/'report.json')
    require(authority.get('config_sha256') == cs and report.get('config_sha256') == cs
            and authority.get('manifest_sha256') == config['manifest_sha256']
            and report.get('authority_valid') is True, 'Analysis authority differs')
    require(bool(config.get('source_sha256')), 'Empty source authority')
    for path, digest in config['source_sha256'].items():
        require(sha(path) == digest, 'Frozen source changed: ' + path)
    records = read(directory/'normalized_records.json')
    by_id = {r['episode_id']: r for r in records}; jobs = {j['request_id']: j for j in manifest['jobs']}
    require(len(by_id) == len(records) == len(manifest['jobs']), 'Incomplete normalized record inventory')
    for job in manifest['jobs']:
        if job['id'] in claim_ids:
            actual, _ = aggregate.normalize_episode(config, cs, job, Path(config['output']), jobs, True)
            require(actual == by_id[job['id']], 'Raw evidence no longer matches sealed normalization: ' + job['id'])
    return records


def run(args):
    sys.path.insert(0, str(ROOT))
    from originx_confirmatory_20261009 import aggregate, analysis
    continuation = load_continuation()
    old = ROOT/'results'/OLD_NAME; new = ROOT/'results'/NEW_NAME
    destination = ROOT/'results'/FINAL_NAME/'analysis-reduced'
    require(not destination.exists(), 'Reduced analysis already exists; never overwrite')
    oc, nc = read(old/'config.json'), read(new/'config.json')
    require(Path(oc['output']) == old and Path(nc['output']) == new, 'Foreign configuration output')
    require(Path(oc['manifest']).read_bytes() == Path(nc['manifest']).read_bytes()
            and sha(oc['manifest']) == oc['manifest_sha256'] == nc['manifest_sha256'], 'Original full manifests differ')
    manifest = read(oc['manifest']); jobs = {j['id']: j for j in manifest['jobs']}
    selection_ref = nc['continuation_selection']
    continuation.checked_file(selection_ref['path'], selection_ref['sha256'], ROOT)
    selection = read(selection_ref['path']); continuation.validate_selection(selection, manifest)
    expected_cohort = continuation.reduced_cohort(manifest)
    require(selection['reduced_cohort'] == expected_cohort and nc['fixed_reduced_cohort'] == expected_cohort
            and set(selection['target_case_ids']) == set(expected_cohort['case_ids']), 'Cohort is not the fixed minimum-seed task-balanced amendment')
    require(selection['original_config_sha256'] == sha(old/'config.json')
            and selection['original_manifest_sha256'] == oc['manifest_sha256'], 'Continuation old authority differs')
    pins = {}; pins.update(verify_terminal_batch(old, oc, continuation)); pins.update(verify_terminal_batch(new, nc, continuation))
    old_claims = claimed(old, sha(old/'config.json'), jobs); new_claims = claimed(new, sha(new/'config.json'), jobs)
    for path, digest in selection['original_claims_sha256'].items():
        require(sha(path) == digest, 'Original claim changed after selection')
    broker_proofs = []
    for name, output, config in [('old', old, oc), ('new', new, nc)]:
        terminal = getattr(args, name+'_broker_terminal'); state = getattr(args, name+'_broker_state')
        proof = verify_sealed_broker(terminal, getattr(args, name+'_broker_terminal_sha256'), state, output,
                                    sha(output/'config.json'), config['manifest_sha256'], continuation)
        broker_proofs.append(proof)
    old_analysis = old/'analysis-final'; new_analysis = new/'analysis-final'
    require(old_analysis.exists(), 'Original final analysis must be preserved before joining')
    if not new_analysis.exists():
        aggregate.aggregate(new/'config.json', broker_state=args.new_broker_state, output_directory=new_analysis,
                            bootstrap_samples=args.bootstrap_samples)
    old_records = verified_normalization(old_analysis, old/'config.json', manifest, old_claims, aggregate)
    new_records = verified_normalization(new_analysis, new/'config.json', manifest, new_claims, aggregate)
    combined, reduced, retained, provenance = merge_records(manifest, selection, old_records, new_records, old_claims, new_claims)
    normalized_manifest = read(old_analysis/'normalized_manifest.json')
    require(normalized_manifest == read(new_analysis/'normalized_manifest.json'), 'Normalized assignment authority differs')
    target = set(expected_cohort['case_ids']); reduced_manifest = [r for r in normalized_manifest if r['case_id'] in target]
    report = analysis.analyze(reduced_manifest, [r for r in reduced if r['record_present']], N=50,
                              bootstrap_samples=args.bootstrap_samples)
    usage_rows = []
    for output, config, broker_state in [(old, oc, args.old_broker_state), (new, nc, args.new_broker_state)]:
        usage_rows.append(aggregate.collect_usage(output, {j['request_id']: j for j in manifest['jobs']},
                          sha(output/'config.json'), config['manifest_sha256'], broker_state))
    usage = merge_usage(usage_rows)
    report.update(schema='originx_user_amended_50case_analysis_v1', scored_confirmation=False,
        original_2500_case_confirmation_complete=False, user_reduced_cohort=expected_cohort,
        original_registered_unique_cases=2500, original_registered_arm_outcomes=17500,
        amended_unique_cases=50, amended_arm_outcomes=350, authority_valid=True,
        historical_1496_of_2500_and_56_of_1004_unchanged=True,
        outside_cohort_prior_arm_records_retained=len(retained),
        outside_cohort_evidence_excluded_from_amended_estimand_not_deleted=True,
        normalized_amended_valid_arm_records=sum(r['valid'] for r in reduced),
        amended_claimed_arm_outcomes=len(set(old_claims).union(new_claims).intersection(r['episode_id'] for r in reduced)),
        cli_usage={k:v for k,v in usage.items() if k!='calls'},
        interpretation='Post-start user schedule amendment; task-balanced 50-case experiment, not completion of the original 2500-case confirmation or a new benchmark score',
        no_official_rank_claim=True, generated_unix=time.time())
    for directory in (old_analysis, new_analysis):
        for path in directory.glob('*.json'):
            pins[str(path)] = sha(path)
    for path in (old/'config.json', new/'config.json', Path(selection_ref['path']), Path(__file__).resolve(), ROOT/'continue_multigpu_v1.py'):
        pins[str(path)] = sha(path)
    destination.mkdir(parents=True)
    write_new(destination/'normalized_manifest.json', reduced_manifest)
    write_new(destination/'normalized_records.json', reduced)
    write_new(destination/'all_original_assignment_records.json', combined)
    write_new(destination/'outside_cohort_prior_records.json', retained)
    write_new(destination/'record_provenance.json', provenance)
    write_new(destination/'cli_usage.json', usage)
    write_new(destination/'authority.json', dict(schema='originx_amendment_join_authority_v1', passed=True,
              evidence_sha256=pins, broker_terminal_proofs=broker_proofs,
              original_claim_count=len(old_claims), continuation_claim_count=len(new_claims),
              claims_disjoint=True, fixed_cohort_count=50, full_manifest_identical=True,
              every_claimed_raw_record_renormalized=True, no_models_or_cli_invoked=True))
    write_new(destination/'report.json', report)
    return dict(output=str(destination), N=50, planned_arm_outcomes=350,
                valid_arm_records=report['normalized_amended_valid_arm_records'],
                cli_invocations=usage['unique_observed_cli_invocations'], complete_input_output_token_accounting=usage['complete_input_output_token_accounting'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('old', 'new'):
        p.add_argument('--'+name+'-broker-state', type=Path, required=True)
        p.add_argument('--'+name+'-broker-terminal', type=Path, required=True)
        p.add_argument('--'+name+'-broker-terminal-sha256', required=True)
    p.add_argument('--bootstrap-samples', type=int, default=2000)
    print(json.dumps(run(p.parse_args())))


if __name__ == '__main__':
    main()
