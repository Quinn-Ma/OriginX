"""CPU-only partition, evidence gates and dispatch ownership regression tests."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('continuation_under_test', HERE / 'continue_multigpu_v1.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
spec2 = importlib.util.spec_from_file_location('frozen_protocol_test', HERE / 'runtime/protocol.py')
p = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(p)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def manifest():
    reference = dict(tasks=[dict(task=f'Task{i:02d}', stratum='x', horizon=100) for i in range(50)])
    return p.make_manifest(reference)


def claim(job, cs='frozen'):
    return dict(job=copy.deepcopy(job), config_sha256=cs)


def test_cohort_all_tasks_minimum_seed_independent_of_order(manifest):
    cohort = c.reduced_cohort(manifest)
    assert len(cohort['case_ids']) == len(set(cohort['case_ids'])) == 50
    assert all(name.endswith('-00') for name in cohort['case_ids'])
    reordered = copy.deepcopy(manifest); reordered['cases'].reverse()
    assert c.reduced_cohort(reordered) == cohort
    assert cohort['user_amendment_after_formal_start'] is True
    assert cohort['outcome_selection'] is False


def test_claimed_missing_error_and_success_all_excluded(manifest):
    cohort = c.reduced_cohort(manifest)
    target = [j for j in manifest['jobs'] if j['case_id'] in cohort['case_ids']]
    claims = {j['id']: claim(j) for j in target[:3]}
    # No results are supplied: a claim alone is an irrevocable exclusion.
    sel = c.build_selection(manifest, claims, 'frozen', cohort)
    assert sel['selected_count'] == 347 and sel['planned_arm_outcomes'] == 350
    assert set(sel['selected_ids']).isdisjoint(claims)
    assert set(sel['inherited_target_claimed_ids']) == set(claims)
    assert len(sel['out_of_scope_never_claimed_ids']) == 17150
    assert len(c.validate_selection(sel, manifest)) == 347


def test_outside_cohort_claims_preserved_separately(manifest):
    cohort = c.reduced_cohort(manifest)
    job = next(j for j in manifest['jobs'] if j['case_id'] not in cohort['case_ids'])
    sel = c.build_selection(manifest, {job['id']: claim(job)}, 'frozen', cohort)
    assert sel['selected_count'] == 350
    assert sel['inherited_outside_cohort_claimed_ids'] == [job['id']]
    assert job['id'] not in sel['out_of_scope_never_claimed_ids']
    c.validate_selection(sel, manifest)


def test_foreign_or_changed_claim_blocks_selection(manifest):
    job = manifest['jobs'][0]
    with pytest.raises(RuntimeError, match='authority'):
        c.build_selection(manifest, {job['id']: claim(job, 'changed')}, 'frozen')
    with pytest.raises(RuntimeError, match='Foreign'):
        c.build_selection(manifest, {'foreign': claim(job)}, 'frozen')


def test_selected_inherited_overlap_or_omitted_job_rejected(manifest):
    sel = c.build_selection(manifest, {}, 'frozen', c.reduced_cohort(manifest))
    bad = copy.deepcopy(sel); bad['excluded_claimed_ids'] = bad['selected_ids'][:1]; bad['excluded_claimed_count'] = 1
    with pytest.raises(RuntimeError, match='partition'):
        c.validate_selection(bad, manifest)
    bad = copy.deepcopy(sel)
    removed = bad['selected_ids'].pop(); bad['selected_indexes'].pop(); bad['selected_count'] -= 1
    bad['out_of_scope_never_claimed_ids'].append(removed)
    with pytest.raises(RuntimeError, match='omits'):
        c.validate_selection(bad, manifest)


def terminal_fixture(tmp_path, manifest):
    old = tmp_path / 'results' / c.OLD_NAME
    config = dict(output=str(old), development=False, manifest=str(old / 'manifest.json'), models=[],
                  protocol={'native': True}, budget=dict(c.CAPS, max_batch=8, quota_resets_allowed=0, paid_topup_allowed=False),
                  duration_seconds=c.ORIGINAL_DURATION, source_sha256={})
    put(old / 'manifest.json', manifest); config['manifest_sha256'] = c.sha(old / 'manifest.json')
    put(old / 'config.json', config); cs = c.sha(old / 'config.json')
    owner = dict(pid=101, process_start_ticks=111, command=['sim', 'run'], cwd=str(tmp_path))
    for name in ('campaign.owner.json', 'rollout.owner.json', 'launch.json'):
        put(old / name, owner)
    put(old / 'campaign-finished.json', dict(phase='failed', development=False, failed=True))
    put(old / 'completion.json', dict(schema='originx_confirmatory_completion_v1', all_children_drained=True))
    put(old / 'cleanup.json', dict(all_owned_services_stopped=True, still_active_owned_pids=[]))
    broker = old / 'broker_inventory_final'
    identity = dict(remote_output=str(old), config_sha256=cs, manifest_sha256=config['manifest_sha256'],
                    model='gpt-6-astra', reasoning_effort='high')
    put(broker / 'identity.json', identity); put(broker / 'broker_status.json', dict(cli_calls=0))
    terminal = broker / 'terminal-inventory.json'
    put(terminal, dict(owner_inactive=True, cli_scoped_processes_absent=True, identity=identity,
                       files_sha256={x.name: c.sha(x) for x in broker.iterdir()}))
    return old, broker, terminal, config, cs


def test_prior_gate_preserves_claim_without_result(tmp_path, manifest):
    old, broker, terminal, _, cs = terminal_fixture(tmp_path, manifest)
    job = next(j for j in manifest['jobs'] if j['seed_index'] == 0)
    put(old / 'claims' / (job['id'] + '.json'), claim(job, cs))
    _, _, sel, gate = c.prior_gate(tmp_path, terminal, broker, inactive=lambda _: True)
    assert gate['all_prior_owners_inactive'] and sel['selected_count'] == 349
    assert job['id'] in sel['excluded_claimed_ids']


def test_prior_gate_blocks_active_parent_and_unclaimed_orphan(tmp_path, manifest):
    old, broker, terminal, _, _ = terminal_fixture(tmp_path, manifest)
    with pytest.raises(RuntimeError, match='owner still lives'):
        c.prior_gate(tmp_path, terminal, broker, inactive=lambda _: False)
    (old / 'episodes' / manifest['jobs'][0]['id']).mkdir(parents=True)
    with pytest.raises(RuntimeError, match='Unclaimed prior'):
        c.prior_gate(tmp_path, terminal, broker, inactive=lambda _: True)


def test_prior_cleanup_and_cli_proof_required(tmp_path, manifest):
    old, broker, terminal, config, cs = terminal_fixture(tmp_path, manifest)
    put(old / 'cleanup.json', dict(all_owned_services_stopped=False, still_active_owned_pids=[100]))
    with pytest.raises(RuntimeError, match='cleanup'):
        c.prior_gate(tmp_path, terminal, broker, inactive=lambda _: True)
    term = c.read(terminal); term['owner_inactive'] = False; put(terminal, term)
    with pytest.raises(RuntimeError, match='terminal proof'):
        c.verify_broker_terminal(terminal, broker, old, cs, config['manifest_sha256'])


def test_broker_inventory_extra_file_blocks_resume(tmp_path, manifest):
    old, broker, terminal, config, cs = terminal_fixture(tmp_path, manifest)
    put(broker / 'late_cli_start.json', {'started': True})
    with pytest.raises(RuntimeError, match='unsealed'):
        c.verify_broker_terminal(terminal, broker, old, cs, config['manifest_sha256'])


def test_debit_incomplete_or_exhausted_rejected():
    good = dict(schema='originx_main_continuation_debit_v1', prior_usage=dict(cli_calls=2, input_tokens=500, output_tokens=100))
    assert c.validate_debit(good, dict(cli_starts=2))['cli_calls'] == 2
    bad = copy.deepcopy(good); bad['prior_usage']['input_tokens'] = None
    with pytest.raises(RuntimeError, match='unknown'):
        c.validate_debit(bad, dict(cli_starts=2))
    with pytest.raises(RuntimeError, match='every prior'):
        c.validate_debit(good, dict(cli_starts=3))
    bad = copy.deepcopy(good); bad['prior_usage']['output_tokens'] = 2000000
    with pytest.raises(RuntimeError, match='exhausted'):
        c.validate_debit(bad, dict(cli_starts=2))


def test_clone_keeps_science_and_budget_but_records_amendment(tmp_path, manifest):
    olddir, _, _, old, _ = terminal_fixture(tmp_path, manifest)
    old.update(response_wait_seconds=1200, intervention_fraction=.5, inference={'primary': ['B:V-C']})
    selection = c.build_selection(manifest, {}, 'frozen', c.reduced_cohort(manifest))
    dest = tmp_path / 'results' / c.NEW_NAME
    selpath = dest / 'continuation-selection.json'; put(selpath, selection)
    broker = tmp_path / 'broker_new.py'; broker.write_text('frozen broker adapter')
    debit = tmp_path / 'debit.json'; put(debit, {})
    models = [dict(gpu_index=3), dict(gpu_index=6)]
    new = c.clone_config(old, dest, tmp_path/'ready.json', models, selpath, broker, debit, [broker], 5000)
    for key in ('protocol', 'budget', 'response_wait_seconds', 'intervention_fraction', 'inference', 'manifest_sha256'):
        assert new[key] == old[key]
    assert new['duration_seconds'] == 5000 and new['fixed_reduced_cohort']['case_count'] == 50
    assert new['source_sha256'] != old['source_sha256']
    with pytest.raises(RuntimeError, match='5400'):
        c.clone_config(old, dest, tmp_path/'ready.json', models, selpath, broker, debit, [broker], 5401)


def test_batch_scope_never_claims_whole_study_complete(manifest):
    sel = c.build_selection(manifest, {}, 'frozen', c.reduced_cohort(manifest))
    done = {i: {'status': 'completed'} for i in sel['selected_indexes']}
    record = c.completion_record(sel, done, [], [])
    assert record['complete'] and record['batch_complete'] and not record['whole_study_complete']
    progress = c.progress_record(manifest, sel, done, [], [], [], 100)
    assert progress['planned_unique_cases'] == 50 and progress['planned_arm_outcomes'] == 350
    assert progress['originally_registered_unique_cases'] == 2500
    assert progress['completed_case_ids_in_this_batch_only'] == 50
    assert progress['combined_completed_case_ids'] is None


def test_result_binding_and_exit_status_cannot_be_forged(tmp_path):
    job = {'id': 'a'}; result = tmp_path / 'result.json'
    put(result, dict(job=job, config_sha256='correct', status='completed', success=False))
    assert c.verified_result(result, job, 'correct', 0)['success'] is False
    with pytest.raises(RuntimeError, match='authority'):
        c.verified_result(result, job, 'different', 0)
    with pytest.raises(RuntimeError, match='Exit/status'):
        c.verified_result(result, job, 'correct', 1)


def test_scheduler_drains_dispatch_without_signalling_live_child(tmp_path, manifest, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    selection = c.build_selection(manifest, {}, 'frozen', c.reduced_cohort(manifest))
    out = tmp_path / 'results' / c.NEW_NAME; out.mkdir(parents=True)
    put(out / 'manifest.json', manifest)
    config = dict(output=str(out), manifest=str(out / 'manifest.json'), duration_seconds=100,
                  continuation_selection={'sha256': 'sealed'}, python='fake-python',
                  models=[dict(policy_id='B', slots=1, owner={}), dict(policy_id='base', slots=1, owner={})])
    path = out / 'config.json'; put(path, config)
    owner = dict(pid=42, process_start_ticks=1, command=['fake'], cwd=str(tmp_path))
    runtime = SimpleNamespace(identity=lambda: owner, active=lambda _: True, environment=lambda _: {})
    stop = []; launched = []; signals = []

    class FakeChild:
        pid = 42
        returncode = 0
        def poll(self): return self.returncode

    class FakeTracked:
        def __init__(self, child, command, cwd): self.child = child
        def await_identity(self, timeout): return owner
        def close(self): pass
        def send(self, sig): signals.append(sig)
        def drain(self, **kwargs): signals.append('drain')
        def receipt(self): return {'reaped': True}

    def spawn(command, **kwargs):
        index = int(command[command.index('--index')+1]); job = manifest['jobs'][index]
        assert (out / 'claims' / (job['id']+'.json')).exists()
        launched.append(job['id'])
        put(out / 'episodes' / job['id'] / 'result.json',
            dict(job=job, config_sha256=c.sha(path), status='completed', success=False))
        # A normal scheduler SIGTERM appends this exact dispatch-drain reason.
        stop.append('requested_drain')
        return FakeChild()

    monkeypatch.setattr(c.subprocess, 'Popen', spawn)
    assert c.run_selected(path, config, selection, runtime, FakeTracked, stop) == 1
    assert len(launched) == 1 and signals == []
    result = c.read(out / 'completion.json')
    assert result['all_children_drained'] and result['finished'] == 1 and not result['whole_study_complete']
    assert c.read(out / 'progress.json')['pending'] == 349
