"""Offline reduced-cohort provenance, denominator and CLI accounting tests."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('amended_aggregate_test', HERE/'aggregate_reduced_v1.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
spec = importlib.util.spec_from_file_location('amended_continue_test', HERE/'continue_multigpu_v1.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
spec = importlib.util.spec_from_file_location('amended_analysis_test', HERE/'runtime/analysis.py')
analysis = importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)


@pytest.fixture
def cohort_fixture():
    cases = [dict(case_id=f'{task}-{seed}', task=task, env_seed=seed, horizon=64)
             for task in ('a', 'b') for seed in (10, 11)]
    jobs = []
    for case in cases:
        for policy, arms in [('B', 'CRGLV'), ('base', 'CV')]:
            for arm in arms:
                eid = f"{case['case_id']}--{policy}--{arm}"
                jobs.append(dict(case, id=eid, episode_id=eid, request_id='r-'+eid, policy_id=policy, arm=arm))
    manifest = dict(cases=cases, case_count=4, jobs=jobs)
    selected = c.reduced_cohort(manifest, expected_tasks=2)
    records = [dict(case_id=j['case_id'], task_name=j['task'], policy_id=j['policy_id'], arm=j['arm'],
                    seed=j['env_seed'], horizon=j['horizon'], episode_id=j['id'], query_interval=16,
                    record_present=False, valid=False, success=None, steps=None, terminal_reason='missing_record',
                    initial_fingerprint={}, trace=[], assistance={}) for j in jobs]
    return manifest, selected, records


def test_old_missing_record_claim_still_prevents_replacement(cohort_fixture):
    manifest, cohort, old = cohort_fixture; new = copy.deepcopy(old)
    job = manifest['jobs'][0]; claims = {job['id']: {'job': job, 'config_sha256': 'old'}}
    selection = c.build_selection(manifest, claims, 'old', cohort)
    combined, reduced, retained, provenance = a.merge_records(manifest, selection, old, new, claims, {})
    assert combined[0]['record_present'] is False and combined[0]['success'] is None
    assert provenance[0]['source_batch'] == 'old'
    assert len(reduced) == 14 and retained == []
    with pytest.raises(RuntimeError, match='Duplicate attempted'):
        a.merge_records(manifest, selection, old, new, claims, claims)


def test_outside_cohort_negative_record_retained(cohort_fixture):
    manifest, cohort, old = cohort_fixture; new = copy.deepcopy(old)
    index = next(i for i, j in enumerate(manifest['jobs']) if j['env_seed'] == 11)
    job = manifest['jobs'][index]; claims = {job['id']: {'job': job, 'config_sha256': 'old'}}
    old[index].update(record_present=True, valid=False, success=False, adapter_error='infra')
    selection = c.build_selection(manifest, claims, 'old', cohort)
    combined, reduced, retained, _ = a.merge_records(manifest, selection, old, new, claims, {})
    assert retained == [old[index]] and len(reduced) == 14
    assert old[index] not in reduced and old[index] in combined


def test_new_claim_outside_fixed_cohort_rejected(cohort_fixture):
    manifest, cohort, old = cohort_fixture; selection = c.build_selection(manifest, {}, 'old', cohort)
    wrong = next(j for j in manifest['jobs'] if j['env_seed'] == 11)
    with pytest.raises(RuntimeError, match='outside fixed'):
        a.merge_records(manifest, selection, old, old, {}, {wrong['id']: {}})


def test_normalized_duplicates_or_changed_identity_rejected(cohort_fixture):
    manifest, cohort, old = cohort_fixture; selection = c.build_selection(manifest, {}, 'old', cohort)
    bad = copy.deepcopy(old); bad[0]['seed'] = 99
    with pytest.raises(RuntimeError, match='identity differs'):
        a.merge_records(manifest, selection, bad, old, {}, {})
    with pytest.raises(RuntimeError, match='every original'):
        a.merge_records(manifest, selection, old+[old[0]], old, {}, {})


def test_analyzer_uses_reduced_N_and_keeps_missing_unknown(cohort_fixture):
    manifest, cohort, old = cohort_fixture; selection = c.build_selection(manifest, {}, 'old', cohort)
    _, records, _, _ = a.merge_records(manifest, selection, old, old, {}, {})
    rows = [{k:r[k] for k in ('case_id','task_name','policy_id','arm','seed','horizon','episode_id','query_interval')} for r in records]
    report = analysis.analyze(rows, [], N=2, bootstrap_samples=2)
    assert report['N_per_policy'] == 2 and report['planned_arm_outcomes'] == 14
    assert report['present_arm_records'] == 0
    for policy in report['policies'].values():
        for arm in policy['arms'].values():
            assert arm['successes'] == 0 and arm['unknown'] == 2


def usage(call=None, complete=True):
    return dict(broker_inventory_provided=True, inventory_complete=complete,
                invalid_receipts=[], interrupted_cli_starts_without_receipt=[], calls=[] if call is None else [call])


def call(key='same'):
    return dict(invocation_key=key, usage=dict(input_tokens=100, cached_input_tokens=20, output_tokens=30, reasoning_tokens=10),
                latency_seconds=5., status='error', request_ids=['outside-cohort-request'],
                source_paths=['old/batches/receipt.json'], receipt_sha256='receipt')


def test_cost_union_deduplicates_invocation_copies_and_includes_outside_cohort_failure():
    first = call(); second = copy.deepcopy(first); second['source_paths'] = ['new/published/receipt.json']
    result = a.merge_usage([usage(first), usage(second)])
    assert result['unique_observed_cli_invocations'] == 1
    assert result['known_usage_totals']['input_tokens'] == 100
    assert result['known_usage_totals']['output_tokens'] == 30
    assert result['known_usage_totals']['reasoning_tokens'] == 10
    assert result['known_invocation_latency_sum_seconds'] == 5
    assert len(result['calls'][0]['source_paths']) == 2
    assert result['calls'][0]['status'] == 'error'


def test_incomplete_cost_never_infers_zero_or_drops_interrupted():
    row = call(); row['usage']['input_tokens'] = None; row['latency_seconds'] = None
    partial = usage(row); partial['interrupted_cli_starts_without_receipt'] = ['started-but-missing']
    result = a.merge_usage([partial])
    assert result['known_usage_totals']['input_tokens'] is None
    assert result['observed_calls_missing_counters']['input_tokens'] == 1
    assert not result['inventory_complete'] and not result['complete_input_output_token_accounting']
    assert result['observed_calls_missing_latency'] == 1
    assert result['interrupted_cli_starts_without_receipt'] == ['started-but-missing']
    assert a.merge_usage([usage()])['known_usage_totals']['output_tokens'] is None


def test_duplicate_cli_conflicting_token_counters_rejected():
    row = call(); changed = copy.deepcopy(row); changed['usage']['output_tokens'] = 31
    with pytest.raises(RuntimeError, match='disagree'):
        a.merge_usage([usage(row), usage(changed)])
