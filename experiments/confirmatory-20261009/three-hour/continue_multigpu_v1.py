"""Explicit continuation of NEVER-CLAIMED assignments, outside frozen sources.

prepare requires terminated prior owners, preserved cleanup, an uploaded sealed
broker inventory, a budget debit, and independently admitted services. campaign
uses the unchanged original episode/preflight implementation and full manifest.
No model is launched, stopped, or imported here. SIGTERM drains dispatch only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
OLD_NAME = 'originx-confirmatory-20261009-v1'
NEW_NAME = 'originx-confirmatory-multigpu-20261009-v1'
RUNTIME = 'originx_confirmatory_20261009'
TRACKER = 'start_full_admitted_v2.py'
TRACKER_SHA = '157dcdde03e64455a6efd7d6595741259afb34350754e41d1d7eaa81b5a7c58a'
ORIGINAL_DURATION = 432000
MAX_DURATION = 5400
CAPS = dict(max_cli_batches=1500, max_input_tokens=25000000, max_output_tokens=2000000)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())


def status(path, value):
    path = Path(path); temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    write_new(temp, value); temp.replace(path)


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked_file(path, expected, root):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()),
            'Evidence must be a regular path inside project: ' + str(path))
    require(path.is_file() and sha(path) == expected, 'Evidence SHA differs: ' + str(path))
    return path


def owner_inactive(owner):
    """Same-birth argv drift still means alive; never treat it as safe reuse."""
    require(type(owner.get('pid')) is int and type(owner.get('process_start_ticks')) is int
            and isinstance(owner.get('command'), list) and bool(owner.get('cwd')), 'Incomplete Linux owner')
    try:
        fields = (Path('/proc') / str(owner['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return True
    return fields[0] in ('Z', 'X') or int(fields[19]) != owner['process_start_ticks']


def reduced_cohort(manifest, expected_tasks=50):
    """One minimum-environment-seed case per task, without outcome access."""
    tasks = {c['task'] for c in manifest['cases']}
    require(len(tasks) == expected_tasks, 'Reduced cohort requires all original tasks')
    chosen = []
    for task in sorted(tasks):
        rows = [c for c in manifest['cases'] if c['task'] == task]
        require(all(type(c.get('env_seed')) is int for c in rows), 'Missing environment seed')
        minimum = min(c['env_seed'] for c in rows)
        winners = [c for c in rows if c['env_seed'] == minimum]
        require(len(winners) == 1, 'Minimum environment seed is ambiguous')
        chosen.append(winners[0]['case_id'])
    return dict(schema='originx_user_reduced_cohort_v1', case_ids=chosen, case_count=len(chosen),
                rule='minimum environment seed per task across all 50 original tasks; no outcome inspection',
                outcome_selection=False, user_amendment_after_formal_start=True,
                original_registration_preserved=True, original_case_count=manifest['case_count'],
                user_request='Reduce experiment size and complete delivery within three hours',
                amendment_utc='2026-10-09T20:46:19Z', delivery_deadline_utc='2026-10-09T23:46:19Z')


def build_selection(manifest, claims, config_sha, cohort=None):
    """Select by prior claims alone; success/error labels are never inspected."""
    jobs = manifest['jobs']; by_id = {j['id']: j for j in jobs}
    require(len(by_id) == len(jobs), 'Repeated original assignment ID')
    require(len({j['request_id'] for j in jobs}) == len(jobs), 'Repeated request ID')
    seen = set()
    for name, claim in claims.items():
        require(name in by_id and name not in seen, 'Foreign or duplicate prior claim: ' + name)
        require(claim.get('job') == by_id[name] and claim.get('config_sha256') == config_sha,
                'Prior claim authority differs: ' + name)
        seen.add(name)
    target_cases = set(cohort['case_ids']) if cohort is not None else {c['case_id'] for c in manifest['cases']}
    require(target_cases <= {c['case_id'] for c in manifest['cases']}, 'Foreign reduced-cohort case')
    target_ids = {j['id'] for j in jobs if j['case_id'] in target_cases}
    indexes = [i for i, j in enumerate(jobs) if j['id'] not in seen and j['id'] in target_ids]
    outside = [j['id'] for j in jobs if j['id'] not in seen and j['id'] not in target_ids]
    return dict(schema='originx_multigpu_continuation_selection_v1',
                planned_unique_cases=len(target_cases), planned_arm_outcomes=len(target_ids),
                originally_registered_unique_cases=manifest['case_count'], originally_registered_arm_outcomes=len(jobs),
                target_case_ids=sorted(target_cases), reduced_cohort=cohort,
                selected_indexes=indexes, selected_ids=[jobs[i]['id'] for i in indexes],
                excluded_claimed_ids=[j['id'] for j in jobs if j['id'] in seen],
                inherited_target_claimed_ids=[j['id'] for j in jobs if j['id'] in seen and j['id'] in target_ids],
                inherited_outside_cohort_claimed_ids=[j['id'] for j in jobs if j['id'] in seen and j['id'] not in target_ids],
                out_of_scope_never_claimed_ids=outside, outside_cohort_evidence_preserved=True,
                selected_count=len(indexes), excluded_claimed_count=len(seen),
                outcome_selection=False, rule='fixed task-balanced reduced cohort minus every prior claim, including missing/error/unknown results',
                no_executed_assignments_repeated=True, no_seed_replacements=True)


def verify_broker_terminal(path, state, expected_output, config_sha, manifest_sha):
    terminal = read(path); state = Path(state)
    require(not state.is_symlink(), 'Broker inventory symlink forbidden')
    if terminal.get('schema') == 'originx_confirmatory_broker_terminal_audit_v1':
        require(terminal.get('broker_owner_inactive') is True
                and terminal.get('all_owned_cli_processes_absent') is True
                and terminal.get('all_cli_wait_returns_terminal') is True,
                'Prior broker/CLI remains active or unverified')
        identity = terminal['broker_identity']
    else:
        require(terminal.get('owner_inactive') is True and terminal.get('cli_scoped_processes_absent') is True,
                'Finalizer broker terminal proof missing')
        identity = terminal['identity']
    require(identity.get('remote_output') == str(expected_output) and identity.get('config_sha256') == config_sha
            and identity.get('manifest_sha256') == manifest_sha
            and identity.get('model') == 'gpt-6-astra' and identity.get('reasoning_effort') == 'high',
            'Prior broker belongs to different configuration/model')
    files = terminal['files_sha256']; require(bool(files), 'Empty broker inventory')
    for name, expected in files.items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts, 'Broker inventory path escape')
        checked_file(state / p, expected, state)
    actual = {p.relative_to(state).as_posix(): sha(p) for p in state.rglob('*')
              if p.is_file() and p.resolve() != Path(path).resolve() and p.name != 'broker.lock'}
    require(actual == files, 'Broker inventory incomplete or contains unsealed files')
    starts = sorted((state / 'batches').glob('*/cli_started.json'))
    receipts = sorted((state / 'batches').glob('*/receipt.json'))
    require({p.parent for p in starts} == {p.parent for p in receipts}, 'Unfinished or orphan CLI receipt')
    for p in receipts:
        receipt = read(p)
        require(receipt.get('schema') == 'astra_cli_receipt_v1' and receipt.get('cli_executed') is True
                and type(receipt.get('returncode')) is int and receipt.get('finished_at')
                and receipt.get('process_exception') is None, 'Prior CLI lacks terminal wait receipt')
        require(sha(p.parent / 'stdout.jsonl') == receipt.get('stdout_sha256')
                and sha(p.parent / 'batch.json') == receipt.get('batch_manifest_sha256'), 'CLI raw hash differs')
    if not starts:
        bs = read(state / 'broker_status.json')
        require(type(bs.get('cli_calls')) is int and bs['cli_calls'] == 0, 'No CLI requires explicit zero accounting')
    return dict(terminal=binding(path), state=str(state), cli_starts=len(starts), identity=identity)


def prior_gate(root, broker_terminal, broker_state, inactive=owner_inactive):
    old = Path(root) / 'results' / OLD_NAME
    config = read(old / 'config.json'); cs = sha(old / 'config.json')
    require(config.get('output') == str(old) and config.get('development') is False, 'Wrong prior formal configuration')
    manifest_path = Path(config['manifest'])
    require(manifest_path == old / 'manifest.json' and sha(manifest_path) == config['manifest_sha256'], 'Old manifest changed')
    finished = read(old / 'campaign-finished.json'); complete = read(old / 'completion.json'); cleanup = read(old / 'cleanup.json')
    require(finished.get('phase') in ('failed', 'completed') and finished.get('development') is False,
            'Old campaign has no terminal receipt')
    require(complete.get('schema') == 'originx_confirmatory_completion_v1' and complete.get('all_children_drained') is True,
            'Old workers not drained')
    require(cleanup.get('all_owned_services_stopped') is True and cleanup.get('still_active_owned_pids') == [],
            'Prior model cleanup not verified')
    paths = {old / name for name in ('campaign.owner.json', 'rollout.owner.json', 'launch.json')}
    paths.update(old.glob('*.owner.json')); paths.update((old / 'processes').glob('*.json'))
    paths.update((old / 'episodes').glob('*/owner.json'))
    checks = []
    for path in sorted(paths):
        owner = read(path); require(inactive(owner), 'Prior owner still lives: ' + str(path))
        checks.append(dict(owner_file=binding(path), inactive=True))
    for model in config['models']:
        require(inactive(model['owner']), 'Prior model still alive')
        checks.append(dict(model_service_id=model['service_id'], owner=model['owner'], inactive=True))
    broker = verify_broker_terminal(broker_terminal, broker_state, old, cs, config['manifest_sha256'])
    claims = {}; claim_files = {}
    for path in sorted((old / 'claims').glob('*.json')):
        require(not path.is_symlink(), 'Prior claim symlink forbidden')
        claims[path.stem] = read(path); claim_files[str(path)] = sha(path)
    # An episode/CLI/error without a claim is unsafe to schedule; never infer it was unexecuted.
    occupied = {p.name for p in (old / 'episodes').iterdir()} if (old / 'episodes').exists() else set()
    occupied.update(p.stem for p in (old / 'errors').glob('*.json'))
    occupied.update(p.stem for p in (old / 'processes').glob('*.json'))
    require(occupied <= set(claims), 'Unclaimed prior episode/process/error; continuation requires audit')
    manifest = read(manifest_path); selection = build_selection(manifest, claims, cs, reduced_cohort(manifest))
    selection.update(original_output=str(old), original_config_sha256=cs,
                     original_manifest_sha256=config['manifest_sha256'], original_claims_sha256=claim_files)
    pins = {str(p): sha(p) for p in paths | {old / 'config.json', old / 'manifest.json',
                                          old / 'campaign-finished.json', old / 'completion.json', old / 'cleanup.json'}}
    pins.update(claim_files)
    return config, manifest, selection, dict(passed=True, all_prior_owners_inactive=True,
        owners=checks, broker=broker, prior_files_sha256=pins)


def validate_debit(value, broker_info):
    require(value.get('schema') == 'originx_main_continuation_debit_v1', 'Wrong continuation debit schema')
    usage = value.get('prior_usage', {})
    for name, capname in [('cli_calls', 'max_cli_batches'), ('input_tokens', 'max_input_tokens'), ('output_tokens', 'max_output_tokens')]:
        require(type(usage.get(name)) is int and 0 <= usage[name] < CAPS[capname], 'Prior budget unknown or exhausted: ' + name)
    require(usage['cli_calls'] == broker_info['cli_starts'], 'Debit must account for every prior main CLI exactly once')
    return usage


def clone_config(old, output, ready, models, selection_path, broker_source, debit, source_paths, duration_seconds):
    new = copy.deepcopy(old)
    require(all(old['budget'].get(k) == v for k, v in CAPS.items()) and old['budget'].get('max_batch') == 8
            and old['budget'].get('quota_resets_allowed') == 0 and old['budget'].get('paid_topup_allowed') is False,
            'Frozen budget differs')
    require(old.get('duration_seconds') == ORIGINAL_DURATION, 'Original batch duration differs')
    require(type(duration_seconds) is int and 1 <= duration_seconds <= MAX_DURATION, 'Continuation duration must be <=5400 seconds')
    new.update(output=str(output), manifest=str(output / 'manifest.json'), models=copy.deepcopy(models),
               services_ready=str(ready), broker_source_sha256=sha(broker_source),
               continuation_selection=binding(selection_path), continuation_debit=binding(debit), duration_seconds=duration_seconds,
               fixed_reduced_cohort=read(selection_path)['reduced_cohort'])
    new['source_sha256'].update({str(p): sha(p) for p in source_paths})
    new['infrastructure_continuation'] = dict(schema='originx_multigpu_infrastructure_change_v1',
        per_episode_scientific_protocol_unchanged=True, original_output=old['output'],
        original_config_sha256=sha(Path(old['output']) / 'config.json'),
        original_manifest_bytes_preserved=True, selection_rule='never previously claimed',
        batch_scope='remaining unclaimed assignments in user-amended fixed 50-case cohort',
        model_cleanup_owner='external service supervisor; never this scheduler',
        gpu_ids=sorted({m['gpu_index'] for m in models}))
    return new


def load_runtime(root=ROOT):
    sys.path.insert(0, str(root))
    from originx_confirmatory_20261009 import runner, protocol, services
    return runner, protocol, services


def tracked_type(root=ROOT):
    path = checked_file(Path(root) / TRACKER, TRACKER_SHA, root)
    spec = importlib.util.spec_from_file_location('originx_continuation_spawn_v1', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.TrackedChild


def prepare(args):
    root = ROOT; output = root / 'results' / NEW_NAME
    require(not output.exists(), 'Continuation namespace already exists; inspect, never overwrite/restart')
    files = {}
    for key in ('services_ready', 'service_admission', 'broker_terminal', 'broker_source', 'debit'):
        files[key] = checked_file(getattr(args, key), getattr(args, key + '_sha256'), root)
    checked_file(root / TRACKER, TRACKER_SHA, root)
    prior, manifest, selection, gate = prior_gate(root, files['broker_terminal'], args.broker_state or files['broker_terminal'].parent)
    require(selection['selected_count'] > 0, 'No unclaimed work; nothing may be restarted')
    validate_debit(read(files['debit']), gate['broker'])
    runtime, protocol, services = load_runtime(root)
    runtime.check_sources(prior); protocol.validate_manifest(manifest)
    ready = read(files['services_ready']); admission = read(files['service_admission'])
    require(ready.get('ready') is True and isinstance(ready.get('models'), list), 'New services not ready')
    require(admission.get('schema') == 'originx_multigpu_service_admission_v1' and admission.get('passed') is True
            and admission.get('services_ready_sha256') == sha(files['services_ready'])
            and admission.get('same_socket_action_rng_parity_passed') is True
            and isinstance(admission.get('hold_seconds'), (int, float)) and admission['hold_seconds'] >= 360,
            'New hardware/services require exact action/RNG same-socket admission')
    models = ready['models']
    require(models and {m['policy_id'] for m in models} == {'B', 'base'}, 'Both policies required')
    require(len({m['port'] for m in models}) == len(models) and len({m['service_id'] for m in models}) == len(models), 'Repeated service identity')
    for m in models:
        require(m['gpu_index'] in (3, 6) and m['host'] == '127.0.0.1' and m['slots'] == 6, 'Unadmitted service placement/capacity')
        require(runtime.active(m['owner']) and sha(m['server_manifest']) == m['server_manifest_sha256'], 'New service owner/manifest changed')
        services.validate_policy_hello(m['policy_id'], m['hello'], read(services.parity_for(m['policy_id'])[0]))
    # All prerequisite checks precede creating the immutable continuation namespace.
    for path, digest in gate['prior_files_sha256'].items():
        require(sha(path) == digest, 'Prior evidence changed during preparation')
    output.mkdir()
    with (output / 'manifest.json').open('xb') as f:
        f.write(Path(prior['manifest']).read_bytes()); f.flush(); os.fsync(f.fileno())
    write_new(output / 'continuation-selection.json', selection)
    write_new(output / 'prior-terminal-admission.json', gate)
    sources = [Path(__file__).resolve(), root / TRACKER, files['broker_source']]
    config = clone_config(prior, output, files['services_ready'], models, output / 'continuation-selection.json',
                          files['broker_source'], files['debit'], sources, args.duration_seconds)
    config['infrastructure_continuation']['service_admission'] = binding(files['service_admission'])
    require(sha(output / 'manifest.json') == prior['manifest_sha256'], 'Manifest byte preservation failed')
    runtime.check_sources(config)
    write_new(output / 'config.json', config)
    result = dict(schema='originx_multigpu_preparation_v1', passed=True, config=binding(output / 'config.json'),
        selection=binding(output / 'continuation-selection.json'), original_manifest_sha256=prior['manifest_sha256'],
        selected_arm_outcomes=selection['selected_count'], excluded_claimed_arm_outcomes=selection['excluded_claimed_count'],
        source_sha256={str(p): sha(p) for p in sources}, prerequisites={k: binding(p) for k, p in files.items()},
        scientifically_new_cases=0, unix=time.time())
    write_new(output / 'preparation.json', result)
    return result


def validate_selection(selection, manifest):
    indexes = selection['selected_indexes']; ids = selection['selected_ids']; excluded = selection['excluded_claimed_ids']
    outside = selection['out_of_scope_never_claimed_ids']
    require(selection.get('schema') == 'originx_multigpu_continuation_selection_v1'
            and selection.get('outcome_selection') is False and selection.get('no_executed_assignments_repeated') is True,
            'Unadmitted selection schema')
    require(all(type(i) is int and 0 <= i < len(manifest['jobs']) for i in indexes)
            and indexes == sorted(set(indexes)), 'Selection indexes duplicate, unordered or out of range')
    require(ids == [manifest['jobs'][i]['id'] for i in indexes]
            and len(ids) == selection['selected_count'] and len(excluded) == selection['excluded_claimed_count'], 'Selection identity/count mismatch')
    require(not set(ids).intersection(excluded) and not (set(ids) | set(excluded)).intersection(outside)
            and len(set(excluded)) == len(excluded) and len(set(outside)) == len(outside)
            and set(ids) | set(excluded) | set(outside) == {j['id'] for j in manifest['jobs']},
            'Selection is not a disjoint complete partition')
    expected = [j['id'] for j in manifest['jobs'] if j['case_id'] in set(selection['target_case_ids']) and j['id'] not in excluded]
    require(ids == expected, 'Reduced cohort selection omits or adds work')
    return list(indexes)


def progress_record(manifest, selection, done, live, pending, stop, elapsed):
    counts = Counter(manifest['jobs'][i]['case_id'] for i in done)
    return dict(schema='originx_confirmatory_progress_v1', batch_scope='continuation only; does not summarize prior outcomes',
        planned_unique_cases=selection['planned_unique_cases'], planned_arm_outcomes=selection['planned_arm_outcomes'],
        originally_registered_unique_cases=manifest['case_count'], originally_registered_arm_outcomes=len(manifest['jobs']),
        selected_arm_outcomes=selection['selected_count'], inherited_claimed_arm_outcomes=selection['excluded_claimed_count'],
        finished_arm_outcomes=len(done), active=len(live), pending=len(pending),
        completed_case_ids_in_this_batch_only=sum(v == 7 for v in counts.values()),
        combined_completed_case_ids=None, technical_unknown=sum(x.get('status') != 'completed' for x in done.values()),
        stop_reasons=list(stop), elapsed_seconds=elapsed, development=False, score_emitted=False, unix=time.time())


def completion_record(selection, done, live, stop):
    complete = len(done) == selection['selected_count'] and not live and not stop and all(x.get('status') == 'completed' for x in done.values())
    return dict(schema='originx_confirmatory_completion_v1', development=False,
        batch_scope='selected never-claimed assignments only', planned=selection['selected_count'], finished=len(done),
        complete=complete, batch_complete=complete, whole_study_complete=False,
        all_children_drained=not live, stop_reasons=list(stop), unix=time.time())


def verified_result(path, job, config_sha, returncode):
    result = read(path)
    require(result.get('job') == job and result.get('config_sha256') == config_sha, 'Result authority differs')
    require((returncode == 0) == (result.get('status') == 'completed'), 'Exit/status differs')
    return result


def run_selected(config_path, config, selection, runtime, TrackedChild, stop):
    output = Path(config['output']); cs = sha(config_path); manifest = read(config['manifest']); jobs = manifest['jobs']
    pending = validate_selection(selection, manifest); live = []; done = {}; started = time.monotonic()
    duration = config['duration_seconds']; require(type(duration) is int and 1 <= duration <= MAX_DURATION, 'Continuation duration exceeds cap')
    deadline = started + duration; term = kill = False; first_error = False
    write_new(output / 'launch.json', dict(runtime.identity(), config_sha256=cs, unix=time.time(), duration_seconds=duration,
               continuation_selection_sha256=config['continuation_selection']['sha256']))

    def progress():
        status(output / 'progress.json', progress_record(manifest, selection, done, live, pending, stop, time.monotonic()-started))

    def collect(item):
        nonlocal first_error
        rc = item['child'].poll()
        if rc is None:
            return False
        item['tracked'].close(); item['log'].close(); job = jobs[item['index']]
        try:
            result = verified_result(output / 'episodes' / job['id'] / 'result.json', job, cs, rc)
        except BaseException:
            result = dict(job=job, status='infrastructure_unknown', success=None, returncode=rc, traceback=traceback.format_exc())
            write_new(output / 'errors' / (job['id']+'.json'), result)
        done[item['index']] = result; live.remove(item)
        write_new(output / 'spawn-receipts' / (job['id']+'.json'), item['tracked'].receipt())
        if result['status'] != 'completed':
            if not first_error:
                first_error = True; write_new(output / 'first-error.json', result)
            if not stop:
                stop.append('infrastructure_failure')
        return True

    try:
        last = 0
        while live or (pending and not stop):
            now = time.monotonic()
            if now >= deadline-min(900, duration/10) and not stop:
                stop.append('deadline_stop_dispatch')
            if (output / 'drain.request.json').exists() and not stop:
                stop.append('broker_or_explicit_drain')
            if now >= deadline and not term:
                stop.append('deadline_workers_terminate')
                for item in live:
                    item['tracked'].send(signal.SIGTERM)
                term = True
            if now >= deadline+15 and not kill:
                for item in live:
                    item['tracked'].send(signal.SIGKILL)
                kill = True
            used = Counter(x['model_index'] for x in live)
            for mi, model in enumerate(config['models']):
                while used[mi] < model['slots'] and pending and not stop:
                    index = next((i for i in pending if jobs[i]['policy_id'] == model['policy_id']), None)
                    if index is None:
                        break
                    require(runtime.active(model['owner']), 'Model owner exited')
                    job = jobs[index]
                    command = [config['python'], '-u', str(ROOT / RUNTIME / 'runner.py'), 'episode',
                               '--config', str(config_path), '--config-sha', cs, '--index', str(index), '--model-index', str(mi)]
                    write_new(output / 'claims' / (job['id']+'.json'), dict(job=job, command=command, config_sha256=cs,
                              model_index=mi, continuation_selection_sha256=config['continuation_selection']['sha256']))
                    lp = output / 'logs' / (job['id']+'.log'); lp.parent.mkdir(exist_ok=True); log = lp.open('x')
                    try:
                        child = subprocess.Popen(command, cwd=ROOT, env=runtime.environment(config), stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close(); raise
                    tracked = TrackedChild(child, command, ROOT)
                    item = dict(index=index, model_index=mi, child=child, tracked=tracked, log=log)
                    live.append(item); pending.remove(index); used[mi] += 1
                    # Immediate parent-owned registration precedes the bounded /proc handshake.
                    owner = tracked.await_identity(timeout=10)
                    write_new(output / 'processes' / (job['id']+'.json'), owner)
            for item in list(live):
                collect(item)
            if now-last >= 30:
                progress(); last = now
            if live:
                time.sleep(.5)
    except BaseException:
        stop.append('supervisor_exception')
        write_new(output / 'supervisor-error.json', dict(traceback=traceback.format_exc(), unix=time.time()))
        for item in list(live):
            try:
                item['tracked'].drain(grace=15, kill_timeout=10); collect(item)
            except BaseException:
                write_new(output / 'drain-errors' / (jobs[item['index']]['id']+'.json'),
                          dict(error=traceback.format_exc(), spawn=item['tracked'].receipt()))
    finally:
        progress()
        completion = completion_record(selection, done, live, stop)
        write_new(output / 'completion.json', completion)
    return 0 if completion['batch_complete'] else 1


def campaign(args):
    import fcntl
    config_path = checked_file(args.config, args.config_sha256, ROOT)
    config = read(config_path); output = ROOT / 'results' / NEW_NAME
    require(config_path == output / 'config.json' and config['output'] == str(output), 'Wrong continuation output')
    lock = (output / 'campaign.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (output / 'campaign.owner.json').exists() and not (output / 'launch.json').exists(), 'Existing continuation cannot restart')
    runtime, protocol, _ = load_runtime(); runtime.check_sources(config)
    require(sha(config['manifest']) == config['manifest_sha256'], 'Full manifest changed')
    protocol.validate_manifest(read(config['manifest']))
    for key in ('continuation_selection', 'continuation_debit'):
        checked_file(config[key]['path'], config[key]['sha256'], ROOT)
    selection = read(config['continuation_selection']['path']); validate_selection(selection, read(config['manifest']))
    prep = read(output / 'preparation.json')
    require(prep['config']['sha256'] == sha(config_path), 'Preparation config binding differs')
    TrackedChild = tracked_type(); stop = []; started = time.monotonic(); phase = 'preflight'; child = None; rc = 1
    write_new(output / 'campaign.owner.json', dict(runtime.identity(), development=False, unix=time.time()))
    signal.signal(signal.SIGTERM, lambda *_: stop.append('requested_drain'))
    signal.signal(signal.SIGINT, lambda *_: stop.append('requested_drain'))

    def state(name, **extra):
        nonlocal phase
        phase = name
        status(output / 'campaign-status.json', dict(phase=name, development=False, batch_scope=True,
               elapsed_seconds=time.monotonic()-started, unix=time.time(), **extra))

    try:
        state('preflight')
        command = [config['python'], '-u', str(ROOT / RUNTIME / 'runner.py'), 'preflight', '--config', str(config_path)]
        with (output / 'preflight.log').open('x') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=runtime.environment(config), stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child = TrackedChild(proc, command, ROOT)
            write_new(output / 'preflight.owner.json', child.await_identity(timeout=10))
            require(proc.wait(timeout=300) == 0, 'Frozen preflight failed; preserve its log')
            child.close(); write_new(output / 'preflight-spawn.json', child.receipt()); child = None
        require(read(output / 'preflight.json')['config_sha256'] == sha(config_path), 'Preflight changed binding')
        state('waiting_broker'); deadline = time.monotonic()+1800
        while not (output / 'broker-ready.json').exists():
            require(not stop, 'Requested drain before broker readiness')
            require(time.monotonic() < deadline, 'Broker readiness timeout'); time.sleep(3)
        ready = read(output / 'broker-ready.json')
        require(ready.get('config_sha256') == sha(config_path) and ready.get('manifest_sha256') == config['manifest_sha256']
                and ready.get('model') == 'gpt-6-astra' and ready.get('reasoning_effort') == 'high', 'Broker binding differs')
        state('rollout')
        write_new(output / 'rollout.owner.json', runtime.identity())
        rc = run_selected(config_path, config, selection, runtime, TrackedChild, stop)
        state('completed' if rc == 0 else 'batch_terminated', whole_study_complete=False)
    except BaseException:
        state('failed', error=traceback.format_exc())
        write_new(output / 'campaign-error.json', dict(phase=phase, error=traceback.format_exc(), unix=time.time()))
    finally:
        if child is not None:
            child.drain(); write_new(output / 'preflight-spawn-final.json', child.receipt())
        write_new(output / 'campaign-finished.json', dict(development=False, phase=phase, failed=rc != 0,
            batch_scope='selected assignments only', whole_study_complete=False, model_cleanup_external=True,
            unix=time.time(), elapsed_seconds=time.monotonic()-started))
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='mode', required=True)
    q = sub.add_parser('prepare')
    for name in ('services-ready', 'service-admission', 'broker-terminal', 'broker-source', 'debit'):
        q.add_argument('--'+name, type=Path, required=True); q.add_argument('--'+name+'-sha256', required=True)
    q.add_argument('--broker-state', type=Path)
    q.add_argument('--duration-seconds', type=int, default=MAX_DURATION)
    q = sub.add_parser('campaign'); q.add_argument('--config', type=Path, required=True); q.add_argument('--config-sha256', required=True)
    args = p.parse_args()
    if args.mode == 'prepare':
        print(json.dumps(prepare(args))); return 0
    return campaign(args)


if __name__ == '__main__':
    raise SystemExit(main())
