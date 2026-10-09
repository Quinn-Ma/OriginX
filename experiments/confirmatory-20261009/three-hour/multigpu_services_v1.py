"""External GPU3/GPU6 service guardian; no policy, rollout, or Astra changes.

Each explicitly requested idle GPU receives exactly four B and two base models.
Original source files are hash-verified and never edited. Only this interpreter's
service layout/profile adapter varies GPU identity and ports. --check is read-only.
--run holds parent-owned Popen/pidfd identities through service retirement. At the
selected dispatch deadline draining.json blocks new clients; active sockets are
never forcibly disconnected, including at the maximum lease deadline.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
NAMESPACE = 'originx_confirmatory_20261009'
OUTPUT_NAME = 'originx-confirmatory-multigpu-services-20261009-v1'
GPU_UUIDS = {3: 'GPU-d0f25b35-5423-72d4-9f45-047dfe7ed87c',
             6: 'GPU-35278907-aae0-4ccc-e0fc-c25cb3a8bd17'}
PORTS = {3: 28000, 6: 28010}
SERVICES_SHA = 'db3d4ab562d609a6c6884c5ddd4cf6aab035ea05fd6f7be27100ce04e5a6c867'
PROBE_SHA = 'b1db00fde2e0644895d96bd7e1059961a6e24e3f51d8de2864b92d2b4933f8d6'
TRACKER_SHA = '157dcdde03e64455a6efd7d6595741259afb34350754e41d1d7eaa81b5a7c58a'
MAX_LEASE = 435600
DRAIN_RESERVE = 1800
OWNER_KEYS = ('pid', 'process_start_ticks', 'command', 'cwd')


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def load_pinned(path, name, digest):
    require(sha(path) == digest, 'Frozen helper/source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def group_output(gpu):
    require(type(gpu) is int and gpu in GPU_UUIDS, 'Only explicitly admitted GPU3/GPU6 are allowed')
    return ROOT/'results'/OUTPUT_NAME/f'gpu{gpu}'


def layout(gpu, b_models=4, base_models=2):
    require((b_models, base_models) in ((4, 2), (3, 3)), 'Unsupported six-model layout')
    output = group_output(gpu); rows = []
    for policy, count in [('B', b_models), ('base', base_models)]:
        for replica in range(count):
            name = f'{policy}-gpu{gpu}-r{replica}'
            directory = output/'services'/name
            rows.append(dict(policy_id=policy, replica=replica, service_id=name,
                port=PORTS[gpu]+(replica if policy == 'B' else 5-replica), slots=6,
                gpu_index=gpu, gpu_uuid=GPU_UUIDS[gpu], directory=str(directory),
                manifest=str(directory/'servers.json')))
    return rows


def service_command(services, spec):
    prefix = [str(ROOT/'envs/training/bin/python'), '-u']
    if spec['policy_id'] == 'B':
        return prefix+['-m', 'continuous_eval2000_v3.server', '--mode', 'serve',
            '--binding', str(services.B_BINDING), '--binding-sha256', services.B_BINDING_SHA,
            '--parity', str(services.B_PARITY), '--parity-sha256', services.B_PARITY_SHA,
            '--port', str(spec['port']), '--max-clients', '6', '--server-manifest', spec['manifest']]
    require(spec['policy_id'] == 'base', 'Unrecognized policy')
    return prefix+[str(ROOT/'multigpu_services_v1.py'), '--_serve-base', '--gpu', str(spec['gpu_index']),
        '--port', str(spec['port']), '--server-manifest', spec['manifest'], '--max-clients', '6']


def services_adapter(gpu):
    """Private module instance: adapt infrastructure identities, no disk edits."""
    output = group_output(gpu)
    s = load_pinned(ROOT/NAMESPACE/'services.py', f'_originx_services_gpu{gpu}', SERVICES_SHA)
    s.GPU_INDEX = gpu; s.GPU_UUID = GPU_UUIDS[gpu]; s.SERVICE_OUTPUT = output
    s.layout = lambda b_models=4, base_models=2, output_arg=None: layout(gpu, b_models, base_models)
    s.service_command = lambda spec: service_command(s, spec)
    original_profile = s.profile_from_manifest
    def truthful_profile(spec, owner, manifest, manifest_sha256, parity):
        profile = original_profile(spec, owner, manifest, manifest_sha256, parity)
        # The frozen function has a literal gpu_index=6; all other identity
        # checks still execute against the actually pinned GPU UUID/owner.
        profile.update(gpu_index=gpu, gpu_uuid=GPU_UUIDS[gpu],
            profile_source_sha256=sha(__file__), frozen_profile_source_sha256=SERVICES_SHA,
            infrastructure_adapter='multigpu_services_v1',
            service_group=str(output), service_guard_owner=str(output/'owner.json'))
        return profile
    s.profile_from_manifest = truthful_profile
    return s


def gpu_snapshot(services, gpu):
    rows = [v.strip() for v in services.smi('-i', str(gpu),
        '--query-gpu=uuid,memory.total,memory.used,memory.free,utilization.gpu,ecc.errors.uncorrected.volatile.total',
        '--format=csv,noheader,nounits').split(',')]
    require(len(rows) == 6 and rows[0] == GPU_UUIDS[gpu], 'GPU index/UUID changed')
    require(rows[5] == '0', 'GPU ECC state is not verified zero')
    result = dict(gpu_index=gpu, gpu_uuid=rows[0], total_mib=int(rows[1]), used_mib=int(rows[2]),
        free_mib=int(rows[3]), utilization_percent=int(rows[4]), ecc_uncorrected=0, occupants=[])
    raw = services.smi('-i', str(gpu), '--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits')
    for line in raw.splitlines():
        if not line.strip(): continue
        pid, uuid, executable, used = [v.strip() for v in line.split(',', 3)]
        if uuid == GPU_UUIDS[gpu]:
            result['occupants'].append(dict(pid=int(pid), gpu_uuid=uuid, executable=executable, used_mib=used))
    return result


def admit_gpu(services, gpu, snapshot, owners, remaining_models):
    require(snapshot.get('gpu_index') == gpu and snapshot.get('gpu_uuid') == GPU_UUIDS[gpu]
        and snapshot.get('ecc_uncorrected') == 0, 'GPU identity/ECC mismatch')
    by_pid = {owner['pid']: owner for owner in owners}
    specs = {spec['service_id']: spec for spec in layout(gpu)}
    require(len(by_pid) == len(owners), 'Duplicate owner PID')
    for owner in owners:
        require(owner.get('service_id') in specs, 'Unrecognized owned service')
        services.validate_owner(owner, specs[owner['service_id']])
    for row in snapshot['occupants']:
        require(row.get('gpu_uuid') == GPU_UUIDS[gpu] and row['pid'] in by_pid,
            'Foreign GPU occupant; no launch or termination: ' + str(row))
    required = services.memory_required_mib(remaining_models)
    require(snapshot['free_mib'] >= required, f'GPU{gpu} needs {required} MiB free, has {snapshot["free_mib"]}')
    return snapshot


def verify_sources():
    pins = {str(ROOT/NAMESPACE/'services.py'): SERVICES_SHA, str(ROOT/NAMESPACE/'probe.py'): PROBE_SHA,
            str(ROOT/'start_full_admitted_v2.py'): TRACKER_SHA}
    for path, digest in pins.items(): require(sha(path) == digest, 'Frozen source changed: '+path)
    return pins


def check(gpu):
    require(os.name == 'posix', 'Linux only')
    require(Path(__file__).resolve() == ROOT/'multigpu_services_v1.py', 'Use project-root external helper')
    require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'Linux pidfd support required')
    output = group_output(gpu)
    require(not output.exists() and not output.is_symlink(), 'Group already exists; preserve evidence, no restart')
    source_pins = verify_sources(); services = services_adapter(gpu)
    parities = services.verify_references()
    snapshot = admit_gpu(services, gpu, gpu_snapshot(services, gpu), [], 6)
    for spec in layout(gpu):
        with socket.socket() as sock: sock.bind(('127.0.0.1', spec['port']))
    return dict(read_only=True, gpu=snapshot, source_sha256=source_pins,
        helper_sha256=sha(__file__), layout=layout(gpu), models=6, clients=36,
        policy_parity_sha256={policy:services.parity_for(policy)[1] for policy in parities},
        no_rollouts_no_training_no_Astra=True)


def owner_alive(services, owner):
    try: actual = services.process_identity(owner['pid'])
    except (FileNotFoundError, ProcessLookupError): return False
    require(all(actual[k] == owner[k] for k in OWNER_KEYS), 'Live owner identity drift; preserve process')
    return True


def sockets_empty(services, records):
    blocked = []
    for spec, owner, tracked in records:
        if tracked.child.poll() is not None: continue
        require(owner_alive(services, owner), 'Owned service not found while unreaped')
        connections = services.tcp_connections(owner['pid'], spec['port'])
        if connections: blocked.append(dict(service_id=spec['service_id'], connections=connections))
    return blocked


def validate_release(request, ready_sha):
    require(request.get('schema') == 'originx_multigpu_service_release_v1'
        and request.get('services_ready_sha256') == ready_sha,
        'Release request must bind exact group services-ready.json')
    require(request.get('all_assigned_workers_stopped') is True,
        'Release requires explicit controller worker-drain attestation')


def mark_draining(output, reason):
    path = output/'draining.json'
    if not path.exists(): write_new(path, dict(reason=reason, unix=time.time(), no_new_clients=True))


class Guardian:
    def __init__(self, gpu, lease_seconds):
        self.gpu = gpu; self.services = services_adapter(gpu); self.output = group_output(gpu)
        self.tracker = load_pinned(ROOT/'start_full_admitted_v2.py', '_originx_tracked_child_v2', TRACKER_SHA)
        self.children = []; self.records = []; self.interrupted = False
        self.deadline = time.monotonic()+lease_seconds
        self.dispatch_deadline = self.deadline-min(DRAIN_RESERVE, lease_seconds/2)

    def interrupt(self, *_): self.interrupted = True

    def check_startup(self):
        require(not self.interrupted, 'Guardian interrupted during startup')
        require(time.monotonic() < self.dispatch_deadline, 'Service startup lease exhausted')

    def spawn(self, label, command, environment):
        directory = self.output/'spawns'/label; directory.mkdir(parents=True, exist_ok=False)
        with (directory/'stdout.log').open('xb') as stdout, (directory/'stderr.log').open('xb') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True)
            tracked = self.tracker.TrackedChild(child, command, ROOT)
            self.children.append((label, tracked))  # register before any fallible identity read/write
            try:
                write_new(directory/'pending-owner.json', dict(pid=child.pid, parent_pid=os.getpid(),
                    requested_command=command, requested_cwd=str(ROOT), identity_bound=False))
                tracked.await_identity(timeout=10)
            finally: write_new(directory/'handshake.json', tracked.receipt())
        return tracked

    def start_model(self, spec, parities):
        self.check_startup(); s = self.services
        admit_gpu(s, self.gpu, gpu_snapshot(s, self.gpu), [r[1] for r in self.records], 6-len(self.records))
        directory = Path(spec['directory']); directory.mkdir(parents=True, exist_ok=False)
        with socket.socket() as sock: sock.bind(('127.0.0.1', spec['port']))
        tracked = self.spawn(spec['service_id'], s.service_command(spec), s.service_environment())
        owner = dict(tracked.owner, namespace=NAMESPACE, policy_id=spec['policy_id'],
            service_id=spec['service_id'], port=spec['port'], slots=6, gpu_index=self.gpu,
            gpu_uuid=GPU_UUIDS[self.gpu], started_unix=time.time())
        self.records.append((spec, owner, tracked)); s.validate_owner(owner, spec)
        write_new(directory/'owner.json', owner)
        deadline = min(self.dispatch_deadline, time.monotonic()+900)
        while time.monotonic() < deadline:
            self.check_startup(); require(tracked.child.poll() is None, 'Model exited before readiness')
            s.validate_owner(owner, spec)
            try:
                if spec['policy_id'] == 'base' and not Path(spec['manifest']).exists():
                    hello = s.live_hello(spec['port']); s.validate_policy_hello('base', hello, parities['base'])
                    server = {k:owner[k] for k in OWNER_KEYS}; server.update(port=spec['port'], gpu=GPU_UUIDS[self.gpu])
                    write_new(spec['manifest'], dict(ready=True, repo=str(ROOT), model=str(s.BASE),
                        source_sha256=hello['multiplex_sources_sha256'], hello=hello, servers=[server], max_clients=6,
                        parity_report=str(s.BASE_PARITY), parity_sha256=s.BASE_PARITY_SHA,
                        namespace=NAMESPACE, policy_id='base', created_unix=time.time()))
                else: read(spec['manifest'])
                profile = s.validate_service(spec, owner, parities)
            except (FileNotFoundError, json.JSONDecodeError, ConnectionError, socket.timeout) as error:
                time.sleep(.25); continue
            write_new(directory/'profile.json', profile)
            return profile
        raise TimeoutError('Model readiness timed out; no blind restart')

    def run_probe(self):
        command = [str(ROOT/'envs/training/bin/python'), '-u', str(ROOT/'multigpu_services_v1.py'),
            '--_probe', '--gpu', str(self.gpu)]
        env = self.services.service_environment(); env['CUDA_VISIBLE_DEVICES'] = ''
        child = self.spawn('probe', command, env)
        deadline = min(self.dispatch_deadline, time.monotonic()+1200)
        while child.child.poll() is None and time.monotonic() < deadline:
            self.check_startup(); time.sleep(.25)
        require(child.child.poll() == 0, 'Probe failed or timed out; preserve raw receipts')
        child.close(); result = read(self.output/'probes/confirmatory-v1/result.json')
        require(result.get('passed') is True and result.get('models') == 6 and result.get('clients') == 36
            and result.get('full_forwards') == 108 and result.get('required_hold_seconds', 0) >= 360
            and result.get('no_reconnect') is True and result.get('one_reset_per_stream') is True,
            'Exact six-stream action/RNG keepalive admission incomplete')
        return result

    def await_release(self, ready_sha):
        while True:
            release = self.output/'release-request.json'
            if release.exists():
                validate_release(read(release), ready_sha)
                mark_draining(self.output, 'exact controller release requested'); return
            if self.interrupted:
                mark_draining(self.output, 'guardian interrupted; preserve active clients'); return
            for _, owner, tracked in self.records:
                require(tracked.child.poll() is None and owner_alive(self.services, owner), 'Owned model exited or identity changed')
            if time.monotonic() >= self.dispatch_deadline:
                mark_draining(self.output, 'selected service dispatch lease reached'); return
            time.sleep(1)

    def cleanup(self):
        """Stop only held direct children; never signal an in-use model."""
        errors = []; blocked = []; model_pids = {tracked.pid for _, _, tracked in self.records}
        for label, tracked in reversed(self.children):
            if tracked.pid in model_pids: continue
            try: tracked.drain()
            except BaseException as error: errors.append(dict(label=label, error=repr(error)))
        mark_draining(self.output, 'group shutdown requested')
        while True:
            try: blocked = sockets_empty(self.services, self.records)
            except BaseException as error:
                errors.append(dict(stage='ownership/socket validation', error=repr(error))); break
            if not blocked or time.monotonic() >= self.deadline: break
            time.sleep(1)
        # Preflight every model before any signal; changed owners/sockets fail
        # closed. This intentionally preserves a model if clients outlive lease.
        if not blocked and not errors:
            time.sleep(2)
            try:
                require(not sockets_empty(self.services, self.records), 'Client connected during quiet drain')
                for spec, owner, tracked in reversed(self.records):
                    if tracked.child.poll() is None:
                        require(owner_alive(self.services, owner), 'Owner disappeared during cleanup')
                        require(not self.services.tcp_connections(owner['pid'], spec['port']), 'Active model preserved')
                    tracked.drain(grace=30, kill_timeout=10)
            except BaseException as error: errors.append(dict(stage='exact model drain', error=repr(error)))
        result = dict(schema='originx_multigpu_service_cleanup_v1', gpu_index=self.gpu,
            children=[dict(label=label, **tracked.receipt()) for label, tracked in self.children],
            blocked_connections=blocked, errors=errors, all_children_reaped=all(t.reaped for _, t in self.children),
            active_clients_never_forcibly_disconnected=True, foreign_processes_untouched=True)
        write_new(self.output/'cleanup.json', result); return result


def run(gpu, lease_seconds=MAX_LEASE):
    require(1200 <= lease_seconds <= MAX_LEASE, 'Lease must be 1200..435600 seconds')
    preflight = check(gpu); output = group_output(gpu)
    output.mkdir(parents=True, exist_ok=False); guardian = Guardian(gpu, lease_seconds)
    failure = None; probe = None; cleanup = None
    write_new(output/'owner.json', dict(guardian.services.process_identity(os.getpid()), gpu_index=gpu,
        gpu_uuid=GPU_UUIDS[gpu], helper_sha256=sha(__file__), lease_seconds=lease_seconds,
        models_max=6, clients_max=36, retained_parent_pidfds=True))
    write_new(output/'preflight.json', preflight)
    handlers = {sig:signal.signal(sig, guardian.interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        parities = guardian.services.verify_references()
        profiles = [guardian.start_model(spec, parities) for spec in layout(gpu)]
        admit_gpu(guardian.services, gpu, gpu_snapshot(guardian.services, gpu), [r[1] for r in guardian.records], 0)
        ready = dict(schema='originx_confirmatory_services_v1', namespace=NAMESPACE, ready=True,
            gpu_index=gpu, gpu_uuid=GPU_UUIDS[gpu], models=profiles, profiles=profiles, model_count=6, clients=36,
            infrastructure_adapter='multigpu_services_v1', helper_sha256=sha(__file__),
            starts_rollouts=False, starts_Astra=False, admission='await group ready.json action/RNG parity')
        write_new(output/'services-ready.json', ready); ready_sha = sha(output/'services-ready.json')
        probe = guardian.run_probe()
        write_new(output/'ready.json', dict(schema='originx_multigpu_service_group_admission_v1', passed=True,
            gpu_index=gpu, gpu_uuid=GPU_UUIDS[gpu], services_ready_path=str(output/'services-ready.json'),
            services_ready_sha256=ready_sha, parity_path=str(output/'probes/confirmatory-v1/result.json'),
            parity_sha256=sha(output/'probes/confirmatory-v1/result.json'), hold_seconds=360,
            same_socket_action_rng_parity_passed=True, models=6, clients=36, full_forwards=108,
            helper_sha256=sha(__file__), frozen_sources_sha256=verify_sources(), no_rollouts_no_Astra=True))
        guardian.await_release(ready_sha)
    except BaseException:
        failure = traceback.format_exc(); write_new(output/'first-error.json', dict(error=failure, unix=time.time()))
    finally:
        try: cleanup = guardian.cleanup()
        except BaseException:
            cleanup_error = traceback.format_exc(); write_new(output/'cleanup-error.json', dict(error=cleanup_error))
            if failure is None: failure = cleanup_error
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        write_new(output/'finished.json', dict(failed=failure is not None, parity_passed=bool(probe and probe.get('passed')),
            all_children_reaped=bool(cleanup and cleanup['all_children_reaped']), no_rollouts_no_Astra=True))
    return 0 if not failure and cleanup and cleanup['all_children_reaped'] else 1


def probe_entry(gpu):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Probe must be CPU-only')
    s = services_adapter(gpu); old = sys.modules.get('services'); sys.modules['services'] = s
    try: probe = load_pinned(ROOT/NAMESPACE/'probe.py', '_originx_multigpu_probe', PROBE_SHA)
    finally:
        if old is None: sys.modules.pop('services', None)
        else: sys.modules['services'] = old
    probe.services = s
    return probe.run_probe(group_output(gpu)/'services-ready.json', group_output(gpu)/'probes/confirmatory-v1',
        hold_seconds=360, interval_seconds=120)


def serve_base(gpu, port, manifest, max_clients):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU_UUIDS[gpu], 'Base child sees wrong GPU UUID')
    require(max_clients == 6, 'Base service capacity changed')
    spec = next((s for s in layout(gpu) if s['policy_id'] == 'base' and s['port'] == port), None)
    require(spec is not None and str(Path(manifest).resolve()) == spec['manifest'], 'Base manifest/port outside fixed group')
    s = services_adapter(gpu); s.enable_reference_imports()
    sys.argv = ['multiplex_inference.server', '--xr1-repo', str(ROOT/'code/Xiaomi-Robotics-1'),
        '--model', str(s.BASE), '--port', str(port), '--max-clients', '6']
    runpy.run_module('multiplex_inference.server', run_name='__main__')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check', action='store_true'); actions.add_argument('--run', action='store_true')
    actions.add_argument('--_probe', action='store_true'); actions.add_argument('--_serve-base', action='store_true')
    parser.add_argument('--gpu', type=int, choices=tuple(GPU_UUIDS), required=True)
    parser.add_argument('--lease-seconds', type=int, default=MAX_LEASE)
    parser.add_argument('--port', type=int); parser.add_argument('--server-manifest'); parser.add_argument('--max-clients', type=int)
    args = parser.parse_args(argv)
    if args._serve_base: serve_base(args.gpu, args.port, args.server_manifest, args.max_clients); return 0
    if args._probe: print(json.dumps(probe_entry(args.gpu), indent=2)); return 0
    if args.check: print(json.dumps(check(args.gpu), indent=2)); return 0
    return run(args.gpu, args.lease_seconds)


if __name__ == '__main__': sys.exit(main())
