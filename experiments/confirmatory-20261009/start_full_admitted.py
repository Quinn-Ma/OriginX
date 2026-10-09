"""External, explicitly invoked full-study admission/launch wrapper.

Default action --check is read-only. --launch is the sole mutating entry point:
it revalidates the already approved development evidence, extends the existing
GPU6-only namespace from 1 B + 1 base to 4 B + 2 base, then starts the unchanged
432000-second campaign supervisor. It does not invoke Astra or start a broker.
Use the local Windows entry point so the actual local broker source is hashed
immediately before the SSH request. This file must first be copied unchanged to
the project root; it is deliberately outside the frozen runtime directory.
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
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
NAMESPACE = 'originx_confirmatory_20261009'
DEV = 'originx-confirmatory-development-20261009-v2'
FULL = 'originx-confirmatory-20261009-v1'
SERVICES = 'originx-confirmatory-services-20261009-v2'
BROKER_SHA = '706d85e48a2655e05cb857a2fbf98b1f8a9c1714e81a373a5e60b7eb1211cd99'
ADMIT_SHA = 'a9cc287845d188be23f2bf26cbb94b82d4915be5134669cd5415537b9a4fa170'
GPU_UUID = 'GPU-35278907-aae0-4ccc-e0fc-c25cb3a8bd17'
DURATION = 432000


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or symbolic-link evidence: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_no_full(root):
    output = Path(root) / 'results' / FULL
    require(not output.is_symlink(), 'Full output must not be symbolic link')
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            'Full output already contains an owner, configuration or other work; no restart')


def validate_extension(previous, current):
    for ready in (previous, current):
        require(ready.get('schema') == 'originx_confirmatory_services_v1'
                and ready.get('namespace') == NAMESPACE and ready.get('ready') is True
                and ready.get('gpu_index') == 6 and ready.get('gpu_uuid') == GPU_UUID,
                'Service namespace or GPU differs')
        require(ready.get('profiles') == ready.get('models'), 'Service profile inventories differ')
        require(ready.get('model_count') == len(ready['models']) and ready.get('clients') == 6 * len(ready['models']),
                'Service capacity/count differs')
        for model in ready['models']:
            require(model.get('gpu_index') == 6 and model.get('gpu_uuid') == GPU_UUID
                    and model.get('slots') == 6 and model.get('namespace') == NAMESPACE,
                    'A profile escaped GPU6-only capacity')
    old = {x['service_id']: x for x in previous['models']}
    new = {x['service_id']: x for x in current['models']}
    require(len(old) == len(previous['models']) == 2 and set(old) == {'B-gpu6-r0', 'base-gpu6-r0'},
            'Development must retain its exact original two profiles')
    require(len(new) == len(current['models']) == 6 and set(new) ==
            {f'B-gpu6-r{i}' for i in range(4)} | {f'base-gpu6-r{i}' for i in range(2)},
            'Expansion must be exactly 4 B + 2 base')
    require(all(new[name] == profile for name, profile in old.items()), 'Expansion replaced an original model/profile')
    require(Counter(m['policy_id'] for m in new.values()) == {'B': 4, 'base': 2}, 'Wrong policy allocation')
    require(len({m['port'] for m in new.values()}) == 6, 'Duplicated service port')


def validate_bindings(approval, root, *, previous_ready=None, current_ready=None):
    root = Path(root).resolve()
    ready_path = root / 'results' / SERVICES / 'services-ready.json'
    require(approval.get('schema') == 'originx_confirmatory_admission_review_v1'
            and approval.get('passed') is True and approval.get('no_confirmatory_outcomes_seen') is True
            and approval.get('raw_verified_outcomes') == 14 and approval.get('complete_cli_usage') is True,
            'Missing strict full development approval')
    require(approval.get('admission_tool_sha256') == ADMIT_SHA
            and approval.get('broker_source_sha256') == BROKER_SHA, 'Approved tool/broker version differs')
    require(sha(root / 'admit_full.py') == ADMIT_SHA, 'Installed admission tool changed')
    require(bool(approval.get('bindings_sha256')), 'Approval has no evidence bindings')
    for name, expected in approval['bindings_sha256'].items():
        path = Path(name)
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), 'Bound evidence escapes project')
        if current_ready is not None and path == ready_path:
            require(previous_ready is not None, 'Unreviewed service-ready mutation')
            validate_extension(previous_ready, current_ready)
            continue
        require(sha(path) == expected, 'Approved immutable evidence changed: ' + str(path))
    config = read(root / 'results' / DEV / 'config.json')
    require(sha(root / 'results' / DEV / 'config.json') == approval['config_sha256']
            and config['manifest_sha256'] == approval['manifest_sha256']
            and config['source_freeze_sha256'] == approval['source_freeze_sha256'], 'Approved configuration pins differ')
    require(config['broker_source_sha256'] == BROKER_SHA, 'Configuration expects another broker')
    return config


def precheck(root, broker_check, self_sha, *, owner_active, owner_stopped):
    root = Path(root).resolve(); output = root / 'results' / DEV
    require(broker_check.get('broker_source_sha256') == BROKER_SHA
            and broker_check.get('launcher_source_sha256') == self_sha
            and broker_check.get('performed_on_local_host') is True, 'Local broker-source launch check missing')
    ensure_no_full(root)
    require(not (root / 'results' / SERVICES / 'full-launch-admission-v1').exists(),
            'Prior external full-launch attempt exists; preserve it and review instead of restarting')
    approval_path = output / 'admission-review.json'; approval = read(approval_path)
    config = validate_bindings(approval, root)
    ready_path = root / 'results' / SERVICES / 'services-ready.json'; ready = read(ready_path)
    require(len(ready.get('models', [])) == 2 and ready['models'] == config['models'],
            'Current original services differ from admitted development profiles')
    require(not (ready_path.parent / 'draining.json').exists()
            and not (ready_path.parent / 'lease-expired.json').exists(), 'Services are draining/lease expired')
    lease = read(ready_path.parent / 'lease.owner.json')
    require(owner_active(lease), 'Existing bounded service lease is not alive')
    require(all(owner_active(model['owner']) for model in ready['models']), 'Original development model owner no longer alive')
    for check in approval['owned_development_processes']:
        require(owner_stopped(read(check['path'])), 'Development process revived or still live')
    return dict(schema='originx_confirmatory_full_start_precheck_v1', passed=True,
                approval_sha256=sha(approval_path), approval=approval,
                original_services_ready_sha256=sha(ready_path), original_services=ready,
                local_broker_source_check=broker_check, launcher_source_sha256=self_sha,
                duration_seconds=DURATION, requested_allocation={'B': 4, 'base': 2},
                gpu_index=6, gpu_uuid=GPU_UUID, no_rollout_started=True)


def remote(action, broker_check, expected_self_sha):
    require(os.name == 'posix' and Path(__file__).resolve() == ROOT / 'start_full_admitted.py',
            'Remote execution is restricted to the independent project-root launcher')
    require(sha(__file__) == expected_self_sha, 'Remote launcher differs from local reviewed source')
    require(sha(ROOT / 'admit_full.py') == ADMIT_SHA, 'Installed admission tool changed')
    # Verify pinned executable bytes before importing any frozen runtime module.
    validate_bindings(read(ROOT / 'results' / DEV / 'admission-review.json'), ROOT)
    sys.path.insert(0, str(ROOT))
    admission = load(ROOT / 'admit_full.py', 'frozen_admission_for_launch')
    # The frozen services module does not load a model at import.
    from originx_confirmatory_20261009 import services
    require(services.SERVICE_OUTPUT == ROOT / 'results' / SERVICES and services.GPU_INDEX == 6
            and services.GPU_UUID == GPU_UUID, 'Frozen service namespace/GPU differs')
    checked = precheck(ROOT, broker_check, expected_self_sha, owner_active=services.active,
                       owner_stopped=admission.linux_owner_stopped)
    if action == 'check':
        return {k: v for k, v in checked.items() if k not in ('approval', 'original_services')}
    require(action == 'launch', 'Unknown remote action')
    import fcntl
    location = ROOT / 'results' / SERVICES / 'full-launch-admission-v1'
    # A prior attempt, including a failed one, requires explicit review instead
    # of silently expanding/relaunching the same scored experiment.
    location.mkdir(exist_ok=False)
    lock = (location / 'lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_new(location / 'launcher.owner.json', services.process_identity(os.getpid()))
    write_new(location / 'pre-expansion.json', checked)
    campaign_owner = None; expanded = False
    try:
        # Frozen service launch owns its own mutex, validates all model/PID/GPU
        # identities and refuses foreign GPU users before each new allocation.
        expanded = True
        current = services.launch(4, 2)
        validate_extension(checked['original_services'], current)
        require(all(services.active(m['owner']) for m in current['models']), 'Expanded services not all live')
        validate_bindings(checked['approval'], ROOT, previous_ready=checked['original_services'], current_ready=current)
        require(sha(ROOT / 'results' / DEV / 'admission-review.json') == checked['approval_sha256'], 'Approval changed during expansion')
        require(sha(__file__) == expected_self_sha, 'Launcher changed during expansion')
        ensure_no_full(ROOT)
        extension = dict(schema='originx_confirmatory_service_extension_v1', passed=True,
                         old_services_ready_sha256=checked['original_services_ready_sha256'],
                         new_services_ready_sha256=sha(services.SERVICE_OUTPUT / 'services-ready.json'),
                         original_profiles_exactly_preserved=True, models=current['models'],
                         documented_mutable_binding=str(services.SERVICE_OUTPUT / 'services-ready.json'),
                         all_other_approved_bindings_unchanged=True)
        write_new(location / 'service-extension.json', extension)
        command = [str(ROOT / 'envs/sim/bin/python'), '-B', '-u', '-m', NAMESPACE + '.campaign',
                   '--duration-seconds', str(DURATION)]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                           PYTHONPATH=str(ROOT) + ':' + str(ROOT / 'remote_processor_candidate_v1/deps'),
                           CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (location / 'campaign.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        campaign_owner = services.process_identity(child.pid)
        require(campaign_owner['command'] == command and campaign_owner['cwd'] == str(ROOT), 'New campaign process identity differs')
        write_new(location / 'campaign-launch.owner.json', campaign_owner)
        deadline = time.monotonic() + 15
        owner_path = ROOT / 'results' / FULL / 'campaign.owner.json'
        while not owner_path.exists():
            require(child.poll() is None, 'Campaign exited before recording its own owner')
            require(time.monotonic() < deadline, 'Campaign owner publication timed out; preserve launcher identity')
            time.sleep(.1)
        owner = read(owner_path)
        require(all(owner.get(k) == campaign_owner[k] for k in ('pid', 'process_start_ticks', 'command', 'cwd'))
                and services.active(owner), 'Campaign owner does not match launched process')
        result = dict(schema='originx_confirmatory_full_start_v1', started=True, campaign_owner=owner,
                      duration_seconds=DURATION, approval_sha256=checked['approval_sha256'],
                      extension_sha256=sha(location / 'service-extension.json'), launcher_source_sha256=expected_self_sha,
                      broker_source_sha256=BROKER_SHA, broker_started_by_tool=False,
                      model_count=6, gpu_index=6, gpu_uuid=GPU_UUID,
                      next_phase='Campaign performs full socket probe/preflight and waits for separately started fixed broker')
        write_new(location / 'started.json', result)
        return result
    except BaseException:
        write_new(location / 'error.json', dict(error=traceback.format_exc(), expansion_attempted=expanded,
                   campaign_owner=campaign_owner, no_automatic_restart=True))
        if expanded and campaign_owner is None:
            try:
                write_new(location / 'cleanup.json', services.cleanup())
            except BaseException:
                write_new(location / 'cleanup-error.json', dict(error=traceback.format_exc(),
                          existing_namespace_lease_retained=True))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--launch', action='store_true')
    action.add_argument('--remote-check', action='store_true', help=argparse.SUPPRESS)
    action.add_argument('--remote-launch', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--broker-source', type=Path, default=Path(__file__).with_name('broker.py'))
    parser.add_argument('--expected-self-sha', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.remote_check or args.remote_launch:
            result = remote('launch' if args.remote_launch else 'check', json.load(sys.stdin), args.expected_self_sha)
            print(json.dumps(result, indent=2)); return 0
        require(os.name == 'nt', 'Use the local Windows entry point for a fresh actual broker-source check')
        digest = sha(__file__)
        require(sha(args.broker_source) == BROKER_SHA, 'Actual local frozen broker source differs; launch refused')
        receipt = dict(performed_on_local_host=True, broker_source_sha256=BROKER_SHA,
                       local_broker_path=str(args.broker_source.resolve()), launcher_source_sha256=digest,
                       checked_at=datetime.now(timezone.utc).isoformat(), no_cli_invoked=True)
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'hkust-cluster',
                   (ROOT / 'envs/sim/bin/python').as_posix(), '-B', (ROOT / 'start_full_admitted.py').as_posix(),
                   '--remote-launch' if args.launch else '--remote-check', '--expected-self-sha', digest]
        completed = subprocess.run(command, input=json.dumps(receipt).encode('utf-8'), timeout=4000 if args.launch else 120)
        return completed.returncode
    except Exception as error:
        print(json.dumps(dict(passed=False, error_type=type(error).__name__, error=str(error))))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
