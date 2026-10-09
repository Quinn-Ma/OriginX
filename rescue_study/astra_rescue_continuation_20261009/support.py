"""Whole-case exclusion ledger and narrow native-reset dispatch exception."""
import ast
import hashlib
import json
from pathlib import Path
import time

ORIGINAL_RUNNER_SHA = '1eaa370fe16d57a2b93eab7216cf011ea1b135c53fb2f367d822afc1bc0c9e4f'
ORIGINAL_ASSISTANCE_SHA = '82fcada288b85d20e23d4bf9e2256f0a6cd30e2b550c0c3190cf29741ab01162'
PRIOR_CONFIG_SHA = 'cda706cfda40c8f73dfd66cc2533a41fbe219ae9149505c02c8f57664893d0aa'
GATE_MESSAGE = 'Natural reset did not reconstruct the original initial fingerprints'
OUTPUT_NAME = 'astra-native-reset-continuation-20261009-v2'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def episode_equivalence(original, current):
    def body(path, normalize=False):
        tree = ast.parse(Path(path).read_text(encoding='utf8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'episode')
        replacements = 0
        for node in ast.walk(fn):
            if isinstance(node, ast.ImportFrom) and node.module == 'astra_rescue_continuation_20261009.assistance':
                require(normalize, 'Original episode unexpectedly imports continuation')
                node.module = 'astra_rescue_20261009.assistance'
                replacements += 1
        return ast.dump(fn, include_attributes=False), replacements
    left, _ = body(original)
    right, replacements = body(current, True)
    require(left == right and replacements == 1,
            'Episode semantics changed beyond the declared assistance import')
    return dict(exact_after_one_import_normalization=True,
                normalized_episode_ast_sha256=hashlib.sha256(left.encode()).hexdigest())


def terminal_ledger(prior, fixed, api):
    prior = Path(prior).resolve(strict=True)
    require(prior == api.ROOT/'results/astra-native-reset-full-20261009-v1', 'Unexpected prior run')
    require(sha(prior/'config.json') == PRIOR_CONFIG_SHA, 'Prior config identity differs')
    cfg = read(prior/'config.json'); manifest = read(cfg['manifest'])
    require(sha(cfg['manifest']) == cfg['manifest_sha256'], 'Prior manifest changed')
    completion = read(prior/'completion.json')
    finished = read(prior/'campaign-finished.json')
    require(completion.get('all_children_drained') is True, 'Prior children have not drained')
    require(finished.get('schema') == 'astra_full_campaign_finished_v1', 'Prior campaign terminal receipt missing')
    owner = read(prior/'campaign-owner.json')
    require(finished.get('owner') == owner and not api.active(owner), 'Prior campaign remains active or identity differs')
    require(not api.active(read(prior/'launch.json')), 'Prior episode supervisor remains active')
    jobs = {j['id']: j for j in manifest['jobs']}
    cases = {}; artifacts = {}; claims = []
    for path in sorted((prior/'claims').glob('*.json')):
        claim = read(path); job = claim['job']; key = job['id']
        require(job == jobs[key] and claim['config_sha256'] == PRIOR_CONFIG_SHA, 'Prior claim authority differs')
        require(path.stem == key, 'Prior claim filename differs')
        result_path = prior/'episodes'/key/'result.json'; row = read(result_path)
        require(row['job'] == job and row['config_sha256'] == PRIOR_CONFIG_SHA, 'Prior result authority differs')
        require(row['status'] in ('completed', 'infrastructure_unknown'), 'Prior result not terminal')
        for owner_path in (prior/'episodes'/key/'owner.json', prior/'processes'/(key+'.json')):
            require(not api.active(read(owner_path)), 'A prior claimed worker remains active')
            artifacts[str(owner_path)] = sha(owner_path)
        arms = cases.setdefault(job['original_id'], set())
        require(job['rescue_arm'] not in arms, 'Duplicate original arm')
        arms.add(job['rescue_arm']); claims.append(dict(job=job, claim_path=str(path), result_path=str(result_path)))
        for artifact in (path, result_path): artifacts[str(artifact)] = sha(artifact)
    require(len(claims) == 72 and len(cases) == 36 and all(v == {'control', 'astra'} for v in cases.values()),
            'Expected 36 whole prior cases/72 immutable claims; review any change')
    all_jobs = fixed['jobs']
    require(len(all_jobs) == 1004 and len({j['id'] for j in all_jobs}) == 1004, 'Fixed denominator differs')
    require(set(cases) <= {j['id'] for j in all_jobs}, 'Prior case outside fixed failure set')
    selected = [j for j in all_jobs if j['id'] not in cases]
    require(len(selected) == 968, 'Continuation must select exactly 968 never-attempted cases')
    for name in ('config.json', 'manifest.json', 'completion.json', 'campaign-finished.json', 'campaign-owner.json', 'launch.json'):
        artifacts[str(prior/name)] = sha(prior/name)
    return dict(schema='astra_continuation_exclusion_ledger_v2', prior_output=str(prior),
                fixed_denominator=1004, excluded_case_ids=[j['id'] for j in all_jobs if j['id'] in cases],
                selected_case_ids=[j['id'] for j in selected], selected_assignment_indices=[j['assignment_index'] for j in selected],
                prior_claims=claims, artifacts_sha256=artifacts, selection_uses_success=False,
                quota_retry_included=False, unix=time.time())


def prepare_continuation(args, api, inherited_prepare):
    require(args.output.resolve() == api.ROOT/'results'/OUTPUT_NAME, 'Use the fixed new v2 output')
    require(args.response_wait_seconds == 1200 and args.arms == ['control', 'astra'], 'Original response budget/paired arms required')
    require(not (args.output/'config.json').exists() and not (args.output/'launch.json').exists(), 'Fresh output required')
    old = api.ROOT/'astra_rescue_20261009'
    require(sha(old/'runner.py') == ORIGINAL_RUNNER_SHA and sha(old/'assistance.py') == ORIGINAL_ASSISTANCE_SHA,
            'Frozen original runner/assistance changed')
    equiv = episode_equivalence(old/'runner.py', api.HERE/'runner.py')
    ledger = terminal_ledger(args.prior_output, read(args.failure_manifest), api)
    indices = ledger['selected_assignment_indices']
    require(args.indices is None or args.indices == indices, 'Supplied indices differ from unattempted whole-case selection')
    args.indices = indices
    args.output.mkdir(parents=True, exist_ok=True)
    api.write(args.output/'continuation-exclusion-ledger.json', ledger, exclusive=True)
    inherited_prepare(args)
    path = args.output/'config.json'; cfg = read(path)
    # The just-created config has never been dispatched. Complete its versioned
    # source/protocol metadata before preflight or broker admission can succeed.
    for file in sorted(api.HERE.glob('*.py')):
        cfg['source_sha256'][str(file)] = sha(file)
    for name in ('runner.py', 'assistance.py', '__init__.py'):
        cfg['source_sha256'][str(old/name)] = sha(old/name)
    cfg['continuation'] = dict(schema='astra_native_reset_continuation_v2', prior_output=str(args.prior_output.resolve()),
        prior_config_sha256=PRIOR_CONFIG_SHA, exclusion_ledger=str(args.output/'continuation-exclusion-ledger.json'),
        exclusion_ledger_sha256=sha(args.output/'continuation-exclusion-ledger.json'),
        selected_cases=968, excluded_cases=36, selection_uses_success=False, quota_retry_included=False,
        inherited_episode_source_sha256=ORIGINAL_RUNNER_SHA, inherited_assistance_source_sha256=ORIGINAL_ASSISTANCE_SHA,
        episode_equivalence=equiv, no_reset_patch=True, no_seed_replacements=True, no_reconnect=True)
    cfg['protocol'].update(policy_wait_keepalive_seconds=120, policy_wait_keepalive_command='rng_state',
        policy_wait_keepalive_same_socket=True, response_wait_seconds=1200,
        expected_initial_deviation_does_not_stop_dispatch=True,
        protocol_diff_from_v1=['Same-thread same-socket read-only RNG keepalive on wait entry/every120s/exit',
                               'Only validated original-initial-state mismatch may continue dispatch',
                               'Whole-case exclusion of all36 v1 claimed cases; no automatic technical retry'])
    api.write(path, cfg)
    api.write(args.output/'protocol-diff.json', dict(continuation=cfg['continuation'],protocol=cfg['protocol']), exclusive=True)
    print(json.dumps(dict(prepared=True, config=str(path), config_sha256=sha(path), selected_cases=968,
                          excluded_cases=36, planned_episodes=1936)), flush=True)


def validate_ledger(config):
    c = config['continuation']
    require(c['schema'] == 'astra_native_reset_continuation_v2', 'Missing continuation contract')
    require(sha(c['exclusion_ledger']) == c['exclusion_ledger_sha256'], 'Exclusion ledger changed')
    ledger = read(c['exclusion_ledger'])
    for path, digest in ledger['artifacts_sha256'].items():
        require(sha(path) == digest, 'Prior immutable authority changed: ' + path)
    m = read(config['manifest'])
    require([j['id'] for j in m['selected_original_jobs']] == ledger['selected_case_ids'], 'Selection differs from exclusion ledger')
    require(not set(ledger['selected_case_ids']) & set(ledger['excluded_case_ids']), 'Prior case replay attempted')
    require(len(m['jobs']) == 1936, 'Unexpected paired episode count')
    return ledger


def preflight_continuation(args, config, api, inherited_preflight):
    validate_ledger(config)
    episode_equivalence(api.ROOT/'astra_rescue_20261009/runner.py', api.HERE/'runner.py')
    require(config['response_wait_seconds'] == 1200 and config['protocol']['policy_wait_keepalive_seconds'] == 120,
            'Continuation response/keepalive budget changed')
    inherited_preflight(args, config)
    path = Path(config['output'])/'preflight.json'; result = read(path)
    result.update(continuation_schema=config['continuation']['schema'], exclusion_ledger_sha256=config['continuation']['exclusion_ledger_sha256'])
    api.write(path, result)


def admit_run(args, config, api):
    require(not args.resume, 'Continuation never automatically retries claimed episodes')
    validate_ledger(config)
    out = Path(config['output'])
    require(not (out/'drain.request.json').exists(), 'Drain requested before dispatch')
    ready = read(out/'broker-ready.json')
    require(ready['model'] == 'gpt-6-astra' and ready['reasoning_effort'] == 'high'
            and ready['config_sha256'] == sha(args.config) and ready['remote_output'] == str(out), 'Broker readiness differs')
    require(not list((out/'claims').glob('*.json')), 'Existing claim forbids new launch')


def expected_initial_deviation(row, output, config_sha):
    """Narrow allow-list: original capture gate, healthy close, no policy work."""
    try:
        output = Path(output); job = row['job']; out = output/'episodes'/job['id']
        if row.get('status') != 'infrastructure_unknown' or row.get('config_sha256') != config_sha or row.get('cleanup'):
            return False
        replay = row.get('initial_replay')
        if not isinstance(replay, dict) or replay.get('exact_match') is not False:
            return False
        if replay.get('natural_reset') is not True or replay.get('state_restored') is not False or not replay.get('differences'):
            return False
        if replay != read(out/'initial_replay.json') or row['initial_scene'] != read(out/'initial_scene.json'):
            return False
        config = read(output/'config.json'); manifest = read(config['manifest'])
        if sha(output/'config.json') != config_sha or sha(config['manifest']) != config['manifest_sha256']:
            return False
        expected_job = next(j for j in manifest['jobs'] if j['id'] == job['id'])
        reference = manifest['reference_records'][job['original_id']]
        if expected_job != job or sha(reference['result_path']) != reference['result_sha256']:
            return False
        scene = row['initial_scene']; expected = reference['initial_scene']
        differences = {key:dict(expected=expected.get(key),actual=scene.get(key))
                       for key in set(expected)|set(scene) if expected.get(key) != scene.get(key)}
        if differences != replay['differences'] or replay['original_result_sha256'] != reference['result_sha256']:
            return False
        if [e['event'] for e in row['events']] != ['reset_requested', 'reset_completed']:
            return False
        a = row['assistance']
        if a.get('requested') is not False or a.get('applied') is not False or a.get('policy_queries') != 0:
            return False
        if a.get('status') not in ('control', 'not_reached') or a.get('keepalive'):
            return False
        if any((out/name).exists() for name in ('query_trace.jsonl', 'rpc.jsonl', 'policy-keepalive.jsonl')):
            return False
        if (output/'requests'/job['id']).exists():
            return False
        trace = row.get('traceback', '')
        if not trace.rstrip().endswith('RuntimeError: ' + GATE_MESSAGE) or 'in capture' not in trace:
            return False
        if 'During handling of the above exception' in trace or 'direct cause' in trace:
            return False
        guard = read(out/'readback-guard/summary.json')
        if not (guard['status'] == 'closed' and guard['failure'] is None
                and set(guard['gl_error_counts']) <= {'after_draw:1281'}
                and guard['raw_frames_checked'] > 0 and guard['sentinel_checks'] > 0):
            return False
        return True
    except (KeyError, ValueError, TypeError, OSError, StopIteration):
        return False


def infrastructure_stop(row, output, config_sha):
    if row.get('status') != 'completed':
        return not expected_initial_deviation(row, output, config_sha)
    return row.get('assistance', {}).get('status') in ('broker_error', 'timeout', 'invalid_response', 'keepalive_error', 'waiting')
