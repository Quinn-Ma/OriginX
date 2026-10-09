"""Independent strict admission gate; never starts a rollout or assistant call.

Run --snapshot-broker-terminal on the Windows host after the development broker
has exited, then upload its complete durable state and this receipt. On Linux,
--check reads the frozen development evidence and prints an approval candidate;
--approve performs the same checks and exclusively creates admission-review.json.
No frozen runtime file is changed. Negative policy outcomes do not fail admission;
missing evidence, absent generated intervention coverage or invalid replay does.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

# Admission is read-only apart from its explicitly requested audit/approval file.
# Importing the frozen verifier must not create bytecode under the source tree.
sys.dont_write_bytecode = True


ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
DEV_NAME = 'originx-confirmatory-development-20261009-v2'
FULL_NAME = 'originx-confirmatory-20261009-v1'
SERVICE_NAME = 'originx-confirmatory-services-20261009-v2'
RUNTIME_NAME = 'originx_confirmatory_20261009'
POLICY_ARMS = {'B': {'C', 'R', 'G', 'L', 'V'}, 'base': {'C', 'V'}}


class AdmissionError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise AdmissionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or symbolic-link evidence: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def utc():
    return datetime.now(timezone.utc).isoformat()


def exclusive_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def linux_owner_stopped(owner):
    for key in ('pid', 'process_start_ticks', 'command', 'cwd'):
        require(key in owner, 'Owner lacks ' + key)
    require(type(owner['pid']) is int and type(owner['process_start_ticks']) is int
            and isinstance(owner['command'], list) and bool(owner['command']) and isinstance(owner['cwd'], str),
            'Malformed Linux owner identity')
    proc = Path('/proc') / str(owner['pid'])
    try:
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X') or int(fields[19]) != owner['process_start_ticks']:
            return True
        actual_command = (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        require(actual_command == owner['command'] and str((proc / 'cwd').resolve()) == owner['cwd'],
                'Live PID/start identity has changed argv/cwd; do not infer it ended')
        return False
    except (FileNotFoundError, ProcessLookupError):
        return True


def live_cli_for_batches(batch_paths):
    """Read only Codex processes and report only exact owned batch-path matches."""
    if os.name == 'nt':
        script = r'''
$paths = @([Console]::In.ReadToEnd() | ConvertFrom-Json)
$matches = @()
Get-CimInstance Win32_Process -Filter "Name='codex.exe'" -ErrorAction Stop | ForEach-Object {
  $command = [string]$_.CommandLine
  $normalized = $command.Replace('\','/').ToLowerInvariant()
  $hits = @($paths | Where-Object { $normalized.Contains(([string]$_).Replace('\','/').ToLowerInvariant()) })
  if ($hits.Count -gt 0) { $matches += [pscustomobject]@{pid=$_.ProcessId;creation=[string]$_.CreationDate;batch_paths=$hits} }
}
@{matches=$matches;process_filter="Name='codex.exe'"} | ConvertTo-Json -Depth 5 -Compress
'''
        proc = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                              input=json.dumps(batch_paths).encode('utf-8'), capture_output=True, timeout=30)
        require(proc.returncode == 0, 'Cannot establish local CLI process termination: ' + proc.stderr.decode('utf-8', 'replace')[-1000:])
        return json.loads(proc.stdout)['matches']
    matches = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / 'cmdline').read_bytes().decode().split('\0')
            if not command or Path(command[0]).name not in ('codex', 'codex.exe'):
                continue
            hits = [path for path in batch_paths if any(path in arg for arg in command)]
            if hits:
                matches.append(dict(pid=int(proc.name), batch_paths=hits))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return matches


def inventory_files(directory, excluded=()):
    directory = Path(directory).resolve(); excluded = {Path(p).resolve() for p in excluded}
    result = {}
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(), 'Broker-state symlink forbidden: ' + str(path))
        if path.is_file() and path.resolve() not in excluded:
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def snapshot_broker_terminal(state, broker_source, output, *, owner_active=None, process_scan=None):
    state, broker_source, output = Path(state).resolve(), Path(broker_source).resolve(), Path(output).resolve()
    require(not output.exists(), 'Terminal audit already exists; preserve it')
    owner_bytes = (state / 'owner.json').read_bytes(); owner = read(state / 'owner.json')
    status = read(state / 'broker_status.json'); identity = read(state / 'identity.json')
    require(status.get('state') in ('finished', 'stopped'), 'Broker has no terminal status')
    require(identity.get('model') == 'gpt-6-astra' and identity.get('reasoning_effort') == 'high'
            and str(identity.get('remote_output', '')).replace('\\', '/').endswith('/results/' + DEV_NAME), 'Wrong broker campaign/model')
    if owner_active is None:
        module = load_module(broker_source, 'admission_local_frozen_broker')
        owner_active = module.local_owner_active
    require(not owner_active(owner), 'Development broker owner remains active')
    starts = sorted((state / 'batches').glob('*/cli_started.json'))
    calls = []
    for started in starts:
        batch = started.parent; receipt_path = batch / 'receipt.json'; receipt = read(receipt_path)
        require(receipt.get('schema') == 'astra_cli_receipt_v1' and receipt.get('cli_executed') is True
                and type(receipt.get('returncode')) is int and receipt.get('finished_at')
                and receipt.get('process_exception') is None, 'CLI lacks a verified wait-return terminal receipt: ' + str(batch))
        require(sha(batch / 'stdout.jsonl') == receipt.get('stdout_sha256'), 'CLI stdout hash differs')
        require(receipt.get('batch_manifest_sha256') == sha(batch / 'batch.json'), 'CLI batch manifest changed')
        calls.append(dict(batch_id=batch.name, receipt_sha256=sha(receipt_path),
                          returncode=receipt['returncode'], finished_at=receipt['finished_at'],
                          evidence='frozen broker communicate/wait returned; receipt+stdout hash verified'))
    require(len(list((state / 'batches').glob('*/receipt.json'))) == len(starts), 'Receipt/start inventory differs')
    process_scan = live_cli_for_batches if process_scan is None else process_scan
    batch_paths = [str(path.parent) for path in starts]
    require(not process_scan(batch_paths), 'A Codex CLI still references an owned development batch')
    # The platform file lock is not durable experiment evidence and is not
    # exported by the upload helper. Every other state file is mandatory.
    inventory = inventory_files(state, excluded=[output, state / 'broker.lock'])
    require((state / 'owner.json').read_bytes() == owner_bytes and not owner_active(owner), 'Broker owner changed during terminal audit')
    receipt = dict(
        schema='originx_confirmatory_broker_terminal_audit_v1', checked_at=utc(),
        broker_state_original_path=str(state), broker_identity=identity, broker_source_sha256=sha(broker_source),
        broker_owner=owner, broker_owner_inactive=True, broker_owner_sha256=sha(state / 'owner.json'),
        cli_starts=len(starts), verified_cli_wait_returns=calls, scoped_cli_process_matches=[],
        all_cli_wait_returns_terminal=True, all_owned_cli_processes_absent=True,
        process_check='Only Codex processes referencing exact durable batch paths; no per-CLI PID/cwd field is invented',
        per_cli_pid_start_cwd_available=False, files_sha256=inventory,
        excluded_non_evidence_files=['broker.lock'],
        audit_tool_sha256=sha(__file__), no_cli_invoked_by_audit=True,
    )
    exclusive_write(output, receipt)
    return receipt


def verify_raw(config, config_sha, manifest, output, broker_state):
    runtime = Path(config['source_freeze']).parent
    sys.path.insert(0, str(runtime))
    try:
        adapter = load_module(runtime / 'aggregate.py', 'admission_frozen_raw_aggregate')
        adapter.validate_manifest(manifest)
        all_jobs = {job['request_id']: job for job in manifest['jobs']}
        normalized, audits = [], []
        for job in manifest['jobs']:
            record, audit = adapter.normalize_episode(config, config_sha, job, output, all_jobs, True)
            normalized.append(record); audits.append(audit)
        require(all(row['valid'] for row in audits), 'Raw evidence replay failed: ' + repr([r for r in audits if not r['valid']]))
        usage = adapter.collect_usage(output, all_jobs, config_sha, config['manifest_sha256'], broker_state)
        rows = [dict(case_id=j['case_id'], task_name=j['task'], policy_id=j['policy_id'], arm=j['arm'],
                     seed=j['env_seed'], horizon=j['horizon'], episode_id=j['id'], query_interval=16) for j in manifest['jobs']]
        paired = adapter.analyze(rows, normalized, N=2, bootstrap_samples=2000)
        pair_checks = {policy: {name: {key: value[key] for key in
                       ('categories', 'unresolved_comparisons', 'valid_paired_2x2')}
                       for name, value in data['comparisons'].items()} for policy, data in paired['policies'].items()}
        return normalized, audits, usage, pair_checks
    finally:
        sys.path.remove(str(runtime))


def check_development(root, broker_state, terminal_audit, *, analysis_directory=None,
                      owner_stopped=linux_owner_stopped, raw_verifier=verify_raw):
    root = Path(root).resolve(); output = root / 'results' / DEV_NAME
    runtime = root / RUNTIME_NAME; services = root / 'results' / SERVICE_NAME
    broker_state, terminal_audit = Path(broker_state).resolve(), Path(terminal_audit).resolve()
    analysis = output / 'analysis-final' if analysis_directory is None else Path(analysis_directory).resolve()
    require(analysis.parent == output, 'Final analysis must stay directly under development output')
    require(broker_state.is_relative_to(root) and terminal_audit.is_relative_to(root), 'Uploaded broker evidence must stay under project root')
    full = root / 'results' / FULL_NAME
    require(not (full / 'launch.json').exists() and not any((full / 'episodes').glob('*/result.json'))
            and not any((full / 'claims').glob('*.json')), 'Confirmation has already dispatched work; admission is not prospective')
    bindings = {}
    def evidence(path):
        path = Path(path)
        require(path.resolve().is_relative_to(root), 'Evidence outside approved project')
        value = read(path); bindings[str(path)] = sha(path); return value
    config = evidence(output / 'config.json'); config_sha = bindings[str(output / 'config.json')]
    require(config.get('schema') == 'originx_confirmatory_config_v1' and config.get('development') is True
            and Path(config.get('output', '')).resolve() == output, 'Not the frozen development-v2 configuration')
    require(Path(config['manifest']) == output / 'manifest.json', 'Development manifest path differs')
    manifest = evidence(output / 'manifest.json')
    require(sha(output / 'manifest.json') == config['manifest_sha256'] and manifest.get('schema') == 'originx_confirmatory_manifest_v1'
            and manifest.get('development') is True and manifest.get('case_count') == 2
            and manifest.get('seed_selection_uses_outcomes') is False, 'Manifest authority/cohort differs')
    jobs = manifest['jobs']; ids = {j['id'] for j in jobs}
    require(len(jobs) == len(ids) == 14 and len({j['request_id'] for j in jobs}) == 14, 'Exactly 14 unique development arm assignments required')
    cases = {j['case_id'] for j in jobs}; combinations = {(p, a) for p, arms in POLICY_ARMS.items() for a in arms}
    require(len(cases) == 2 and all({(j['policy_id'], j['arm']) for j in jobs if j['case_id'] == case} == combinations for case in cases), 'Development seven-arm coverage differs')
    freeze_path = runtime / 'source-freeze.ready.json'
    require(Path(config['source_freeze']) == freeze_path and sha(freeze_path) == config['source_freeze_sha256'], 'Source-freeze identity changed')
    freeze = evidence(freeze_path)
    require(freeze.get('schema') == 'originx_confirmatory_source_freeze_v1' and freeze.get('confirmed_outcomes_seen') is False
            and freeze.get('broker_source_sha256') == config['broker_source_sha256'], 'Source-freeze provenance differs')
    for name, expected in freeze['source_sha256'].items():
        require(Path(name).name == name and sha(runtime / name) == expected, 'Frozen executable source differs: ' + name)
        bindings[str(runtime / name)] = expected
    for name, expected in config['source_sha256'].items():
        require(Path(name).resolve().is_relative_to(root) and sha(name) == expected, 'Pinned configuration source differs: ' + name)
        bindings[name] = expected
    preflight = evidence(output / 'preflight.json')
    require(preflight.get('passed') is True and preflight.get('config_sha256') == config_sha
            and preflight.get('manifest_sha256') == config['manifest_sha256'], 'Development preflight binding differs')
    probe_dir = services / 'probes' / 'development-v2'
    probe_config = evidence(probe_dir / 'config.json'); probe = evidence(probe_dir / 'result.json')
    require(probe_config.get('source_sha256') == freeze['source_sha256']['probe.py']
            and probe_config.get('hold_seconds', 0) >= 360 and probe_config.get('interval_seconds', 999) <= 120,
            'Probe source/wait authority differs')
    require(probe.get('schema') == 'originx_confirmatory_socket_probe_v1' and probe.get('passed') is True
            and probe.get('models') == 2 and probe.get('clients') == 12 and probe.get('full_forwards') == 36
            and probe.get('no_reconnect') is True and probe.get('one_reset_per_stream') is True, 'Require passed 12-client/36-forward development probe')
    require(len(probe.get('records', [])) == 12
            and len({row['path'] for row in probe['records']}) == 12, 'Probe record inventory incomplete or duplicated')
    for row in probe['records']:
        path = Path(row['path']); require(path.parent == probe_dir, 'Probe record escaped fixed directory')
        record = evidence(path)
        require(record.get('passed') is True and record.get('hold_seconds_actual', 0) >= 360
                and record.get('resets') == 1 and record.get('reconnects') == 0
                and len(record.get('queries', [])) == 3 and all(q.get('exact') is True for q in record['queries']), 'Probe stream failed or incomplete')
    completion = evidence(output / 'completion.json')
    require(completion.get('schema') == 'originx_confirmatory_completion_v1' and completion.get('complete') is True
            and completion.get('planned') == completion.get('finished') == 14 and completion.get('all_children_drained') is True,
            'Development rollout is not completely drained with all 14 outcomes')
    finished = evidence(output / 'campaign-finished.json')
    require(finished.get('development') is True and finished.get('failed') is False and finished.get('phase') == 'completed', 'Development campaign did not complete')
    ready = evidence(output / 'broker-ready.json')
    require(ready.get('config_sha256') == config_sha and ready.get('manifest_sha256') == config['manifest_sha256']
            and ready.get('model') == 'gpt-6-astra' and ready.get('reasoning_effort') == 'high'
            and ready.get('broker_source_sha256') == config['broker_source_sha256'], 'Broker source/readiness not bound to frozen configuration')
    terminal = evidence(terminal_audit)
    require(terminal.get('schema') == 'originx_confirmatory_broker_terminal_audit_v1'
            and terminal.get('broker_owner_inactive') is True and terminal.get('all_cli_wait_returns_terminal') is True
            and terminal.get('all_owned_cli_processes_absent') is True and terminal.get('scoped_cli_process_matches') == []
            and terminal.get('no_cli_invoked_by_audit') is True, 'Missing strict local broker/CLI terminal audit')
    require(terminal.get('broker_source_sha256') == config['broker_source_sha256'], 'Terminal audit used a different broker source')
    inventory = terminal['files_sha256']
    require(bool(inventory), 'Empty uploaded broker inventory')
    for name, expected in inventory.items():
        path = broker_state / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and path.resolve().is_relative_to(broker_state)
                and not path.is_symlink() and sha(path) == expected, 'Uploaded broker evidence changed: ' + name)
        bindings[str(path)] = expected
    require(terminal.get('excluded_non_evidence_files') == ['broker.lock'], 'Unexpected terminal inventory exclusions')
    actual = inventory_files(broker_state, excluded=[terminal_audit, broker_state / 'broker.lock'])
    require(actual == inventory, 'Broker-state upload is partial or includes unaudited files')
    identity = evidence(broker_state / 'identity.json'); owner = evidence(broker_state / 'owner.json')
    require(identity == terminal.get('broker_identity') and owner == terminal.get('broker_owner')
            and sha(broker_state / 'owner.json') == terminal.get('broker_owner_sha256'), 'Local terminal owner/identity receipt differs')
    require(identity.get('config_sha256') == config_sha and identity.get('manifest_sha256') == config['manifest_sha256']
            and identity.get('remote_output') == str(output), 'Broker inventory belongs to another run')
    owner_paths = [output / (name + '.owner.json') for name in ('campaign', 'socket-probe', 'prepare', 'preflight', 'rollout', 'aggregate')]
    owner_paths.append(probe_dir / 'owner.json')
    for job in jobs:
        out = output / 'episodes' / job['id']
        raw = evidence(out / 'result.json')
        require(raw.get('status') == 'completed' and raw.get('job') == job and raw.get('config_sha256') == config_sha, 'Development episode is incomplete/misbound')
        owner_paths.extend([out / 'owner.json', output / 'processes' / (job['id'] + '.json')])
    owner_checks = []
    for path in owner_paths:
        owner_record = evidence(path)
        require(owner_stopped(owner_record), 'Owned development process still alive: ' + str(path))
        owner_checks.append(dict(path=str(path), pid=owner_record['pid'], process_start_ticks=owner_record['process_start_ticks'], stopped=True))
    report = evidence(analysis / 'report.json')
    authority = evidence(analysis / 'authority.json')
    normalized = evidence(analysis / 'normalized_records.json')
    usage = evidence(analysis / 'cli_usage.json')
    require(report.get('development') is True and report.get('scored_confirmation') is False and report.get('authority_valid') is True
            and report.get('raw_evidence_valid') == 14 and report.get('planned_arm_outcomes') == report.get('present_arm_records') == 14
            and report.get('config_sha256') == config_sha and report.get('manifest_sha256') == config['manifest_sha256'], 'Final aggregation is stale, incomplete or invalid')
    require(authority.get('config_sha256') == config_sha and authority.get('manifest_sha256') == config['manifest_sha256'], 'Aggregation authority differs')
    verified, audits, recomputed_usage, pairs = raw_verifier(config, config_sha, manifest, output, broker_state)
    require(normalized == verified and authority.get('episode_checks') == audits, 'Saved normalized evidence differs from fresh raw verification')
    require(usage == recomputed_usage and report.get('cli_usage') == {k:v for k,v in usage.items() if k != 'calls'}, 'Final aggregate does not include the current complete broker inventory')
    require(usage.get('broker_inventory_provided') is True and usage.get('inventory_complete') is True
            and usage.get('complete_input_output_token_accounting') is True
            and usage.get('interrupted_cli_starts_without_receipt') == [] and usage.get('invalid_receipts') == [], 'Development CLI accounting remains incomplete')
    require(terminal['cli_starts'] == len(terminal['verified_cli_wait_returns']) == usage['unique_observed_cli_invocations'], 'Terminal CLI count disagrees with deduplicated final usage')
    applied = Counter()
    for row in verified:
        require(row.get('valid') is True, 'Raw-invalid normalized episode')
        assistance = row.get('assistance', {})
        if row['arm'] in ('L', 'V') and assistance.get('delivered') is True:
            require(assistance.get('evidence_valid') is True and assistance.get('triggered') is True,
                    'Generated application is not linked to actual trigger/CLI evidence')
            applied[row['arm']] += 1
            applied[row['policy_id'] + ':' + row['arm']] += 1
    require(applied['L'] >= 1 and applied['V'] >= 1, 'Development must include at least one verified linked L and V application; E=0 is not waived')
    require(set(pairs) == {'B', 'base'}, 'Recomputed policy comparison inventory incomplete')
    for policy, comparisons in pairs.items():
        require(comparisons, 'No recomputed pair comparisons')
        for name, comparison in comparisons.items():
            require(comparison.get('unresolved_comparisons') == 0, 'Development contains initial/prefix/unknown pair deviations')
            actual = report['policies'][policy]['comparisons'][name]
            require(all(actual.get(key) == value for key, value in comparison.items()), 'Saved paired report differs from fresh recomputation')
    # Bind every raw artifact used by replay, including traces, images, routing
    # receipts and generated text, rather than only the aggregate documents.
    for directory in ('episodes', 'requests'):
        for name, digest in inventory_files(output / directory).items():
            bindings[str(output / directory / name)] = digest
    for path, expected in bindings.items():
        require(sha(path) == expected, 'Evidence changed while checking admission: ' + path)
    return dict(
        schema='originx_confirmatory_admission_review_v1', checked_at=utc(), passed=True,
        no_confirmatory_outcomes_seen=True, development_output=str(output), config_sha256=config_sha,
        final_analysis_directory=str(analysis),
        manifest_sha256=config['manifest_sha256'], source_freeze_sha256=config['source_freeze_sha256'],
        broker_source_sha256=config['broker_source_sha256'], planned_development_outcomes=14,
        raw_verified_outcomes=14, verified_generated_applications=dict(applied),
        final_cli_invocations=usage['unique_observed_cli_invocations'], complete_cli_usage=True,
        owned_development_processes=owner_checks, cli_termination_evidence=terminal['process_check'],
        per_cli_pid_start_cwd_available=terminal.get('per_cli_pid_start_cwd_available', False),
        bindings_sha256=bindings, admission_tool_sha256=sha(__file__),
        negative_policy_outcomes_allowed=True, rollout_started_by_tool=False,
    )


def approve(candidate, root):
    require(candidate.get('passed') is True and candidate.get('no_confirmatory_outcomes_seen') is True, 'Cannot approve failed candidate')
    for path, expected in candidate['bindings_sha256'].items():
        require(sha(path) == expected, 'Candidate evidence changed before approval')
    path = Path(root).resolve() / 'results' / DEV_NAME / 'admission-review.json'
    exclusive_write(path, candidate)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--approve', action='store_true')
    action.add_argument('--snapshot-broker-terminal', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--broker-state', type=Path, required=True)
    parser.add_argument('--broker-terminal', type=Path, required=True)
    parser.add_argument('--broker-source', type=Path)
    parser.add_argument('--analysis-directory', type=Path)
    args = parser.parse_args()
    try:
        if args.snapshot_broker_terminal:
            require(args.broker_source is not None, '--broker-source is required for local terminal audit')
            result = snapshot_broker_terminal(args.broker_state, args.broker_source, args.broker_terminal)
            print(json.dumps(dict(passed=True, audit=str(args.broker_terminal), cli_starts=result['cli_starts'])))
        else:
            result = check_development(args.root, args.broker_state, args.broker_terminal,
                                       analysis_directory=args.analysis_directory)
            if args.approve:
                result['approval_path'] = str(approve(result, args.root))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps(dict(passed=False, error_type=type(error).__name__, error=str(error)), ensure_ascii=False, indent=2))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
