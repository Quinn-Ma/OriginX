"""GPU-6-only, ownership-pinned B/base services for the confirmatory namespace.

No model is imported or started at module import.  launch is an explicit action;
status is read-only; cleanup refuses changed owners or active TCP connections.
The historical direct/socket gates admit identities, not new rollout validity.
Use make_client(profile, ...) for development probes before formal evaluation.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
NAMESPACE = 'originx_confirmatory_20261009'
SERVICE_OUTPUT = ROOT / 'results/originx-confirmatory-services-20261009-v2'
CAMPAIGN_OUTPUT = ROOT / 'results/originx-confirmatory-20261009-v1'
BASE = Path('/ephemeral/qinzhen/ckpt/robocasa/Xiaomi-Robotics-1-RoboCasa365')
GPU_INDEX = 6
GPU_UUID = 'GPU-35278907-aae0-4ccc-e0fc-c25cb3a8bd17'
SLOTS = 6
MAX_MODELS = 6
MODEL_BUDGET_MIB = 12 * 1024
RESERVE_MIB = 4 * 1024
MIN_NEW_FREE_MIB = 16 * 1024
B_BINDING = ROOT / 'results/continued-ab2000-v1/B-endpoint.json'
B_BINDING_SHA = 'c3138b49dcf51c0514f129073870701b5cd59cb88df384ac5a4b9ac24a77c601'
B_PARITY = ROOT / 'results/continued-ab2000-v1/parity/B.json'
B_PARITY_SHA = '4c3726cf8871061e7f29a68a613ceba3b981d34d0bfcc9fd51dc2e9ee56df813'
BASE_PARITY = ROOT / 'results/lora-dev150-final-v1/base-parity.json'
BASE_PARITY_SHA = '4ec949be31784ee75589b67a7419246660800cd6b80966d20654ecb0314975a8'
BRANCH_SHA = '41988b92391953687b39a0bc5a35bce962fbfcd08fbf6e550108430850d9944f'
A_BINDING_SHA = 'fb47bbc98c4671f9afbed6d91eae0c3947badecf574070b5848df7686695d890'
A_ADAPTER_SHA = '5892389863751f2245f83a6d3b96bf5725a052aa5b023230b093d8d8ece70ac4'
EXPECTED_TORCH = '2.8.0+cu128'
OWNER_KEYS = ('pid', 'process_start_ticks', 'command', 'cwd')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def replace_status(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    write_new(temp, value)
    temp.replace(path)


def allowed_output(output):
    path = Path(output)
    require(path == SERVICE_OUTPUT, 'Only the fixed confirmatory services output is permitted')
    require(not path.is_symlink(), 'Service output must not be a symlink')
    if os.name == 'posix':
        require(path.resolve() == SERVICE_OUTPUT, 'Service output resolves outside fixed namespace')
    return path


def layout(b_models=4, base_models=2, output=SERVICE_OUTPUT):
    """Stable ports permit 1+1 -> 4+2 expansion; also supports 3+3."""
    require(type(b_models) is int and type(base_models) is int, 'Integer service counts required')
    require(1 <= b_models <= 4 and 1 <= base_models <= 3 and b_models + base_models <= MAX_MODELS,
            'Require 1..4 B, 1..3 base, and at most six total models')
    output = allowed_output(output)
    rows = []
    for policy, count in [('B', b_models), ('base', base_models)]:
        for replica in range(count):
            name = f'{policy}-gpu6-r{replica}'
            port = 27800 + replica if policy == 'B' else 27805 - replica
            directory = output / 'services' / name
            rows.append(dict(policy_id=policy, replica=replica, service_id=name,
                             port=port, slots=SLOTS, gpu_index=GPU_INDEX, gpu_uuid=GPU_UUID,
                             directory=str(directory), manifest=str(directory/'servers.json')))
    require(len({r['port'] for r in rows}) == len(rows), 'Service port collision')
    return rows


def service_command(spec):
    prefix = [str(ROOT/'envs/training/bin/python'), '-u', '-m']
    if spec['policy_id'] == 'B':
        return prefix + ['continuous_eval2000_v3.server', '--mode', 'serve',
                         '--binding', str(B_BINDING), '--binding-sha256', B_BINDING_SHA,
                         '--parity', str(B_PARITY), '--parity-sha256', B_PARITY_SHA,
                         '--port', str(spec['port']), '--max-clients', str(SLOTS),
                         '--server-manifest', spec['manifest']]
    require(spec['policy_id'] == 'base', 'Unknown policy id')
    # The wrapper invokes the original base module in a new interpreter.  The
    # namespace and manifest are explicit in argv even though that module has
    # no --server-manifest option.  It never imports the adapter/branch loader.
    return prefix + [NAMESPACE+'.services', '_serve-base', '--port', str(spec['port']),
                     '--server-manifest', spec['manifest'], '--max-clients', str(SLOTS)]


def process_identity(pid):
    p = Path('/proc') / str(pid)
    fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
    if fields[0] in ('Z', 'X'):
        raise ProcessLookupError('Process exited')
    return dict(pid=int(pid), process_start_ticks=int(fields[19]),
                command=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                cwd=str((p/'cwd').resolve(strict=True)))


def owner_matches(owner, actual):
    return all(owner.get(k) == actual.get(k) for k in OWNER_KEYS)


def active(owner):
    try:
        return owner_matches(owner, process_identity(owner['pid']))
    except (OSError, ProcessLookupError, KeyError):
        return False


def validate_owner(owner, spec, actual=None):
    require(owner.get('namespace') == NAMESPACE and owner.get('gpu_uuid') == GPU_UUID
            and owner.get('gpu_index') == GPU_INDEX, 'Owner namespace/GPU differs')
    require(owner.get('policy_id') == spec['policy_id'] and owner.get('service_id') == spec['service_id']
            and owner.get('port') == spec['port'] and owner.get('slots') == SLOTS,
            'Owner policy/service/port/capacity differs')
    require(owner.get('command') == service_command(spec) and owner.get('cwd') == str(ROOT),
            'Owner argv/cwd differs; preserving process')
    require(type(owner.get('pid')) is int and owner['pid'] > 0
            and type(owner.get('process_start_ticks')) is int and owner['process_start_ticks'] > 0,
            'Incomplete owner identity')
    if actual is None:
        actual = process_identity(owner['pid'])
    require(owner_matches(owner, actual), 'PID reused or process identity changed; preserving it')
    return owner


def memory_required_mib(remaining_models):
    require(type(remaining_models) is int and 0 <= remaining_models <= MAX_MODELS,
            'Invalid remaining-model count')
    return max(MIN_NEW_FREE_MIB, remaining_models*MODEL_BUDGET_MIB + RESERVE_MIB) if remaining_models else RESERVE_MIB


def memory_plan(b_models, base_models):
    count = len(layout(b_models, base_models))
    return dict(models=count, clients=count*SLOTS, b_models=b_models, base_models=base_models,
                historical_base_parity_peak_cuda_gib=9.750996589660645,
                per_model_budget_mib=MODEL_BUDGET_MIB, reserve_mib=RESERVE_MIB,
                initial_required_free_mib=memory_required_mib(count),
                inherited_new_service_floor_mib=MIN_NEW_FREE_MIB,
                estimate_note='12 GiB/model is a conservative planning allowance above the observed 9.751 GiB base peak, not a measured B peak. Six models require 76 GiB free initially; each launch rechecks actual free memory and foreign occupancy. No GPU capacity is reserved by this estimate.')


def smi(*args):
    p = subprocess.run(['nvidia-smi', *args], capture_output=True, text=True, timeout=20)
    require(p.returncode == 0, 'nvidia-smi failed; ownership unknown: '+p.stderr[-300:])
    return p.stdout.strip()


def gpu_snapshot():
    row = [x.strip() for x in smi('-i', '6', '--query-gpu=uuid,memory.total,memory.used,memory.free,utilization.gpu,ecc.errors.uncorrected.volatile.total',
                                 '--format=csv,noheader,nounits').split(',')]
    require(len(row) == 6 and row[0] == GPU_UUID, 'GPU 6 UUID changed')
    require(row[5] == '0', 'GPU ECC state is not verified zero')
    result = dict(gpu_index=6, gpu_uuid=row[0], total_mib=int(row[1]), used_mib=int(row[2]),
                  free_mib=int(row[3]), utilization_percent=int(row[4]), ecc_uncorrected=0, occupants=[])
    for line in smi('-i', '6', '--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory',
                    '--format=csv,noheader,nounits').splitlines():
        pid, gpu, executable, used = [x.strip() for x in line.split(',', 3)]
        if gpu == GPU_UUID:
            result['occupants'].append(dict(pid=int(pid), gpu_uuid=gpu, executable=executable, used_mib=used))
    return result


def admit_gpu(snapshot, owners, specs, remaining_models, identity_reader=process_identity):
    require(snapshot.get('gpu_index') == 6 and snapshot.get('gpu_uuid') == GPU_UUID
            and snapshot.get('ecc_uncorrected') == 0, 'GPU identity/ECC mismatch')
    by_pid = {o['pid']: o for o in owners}
    by_service = {s['service_id']: s for s in specs}
    for proc in snapshot['occupants']:
        owner = by_pid.get(proc['pid'])
        require(owner is not None and owner.get('service_id') in by_service,
                f"Foreign GPU6 occupant PID {proc['pid']}; no launch/termination attempted")
        validate_owner(owner, by_service[owner['service_id']], identity_reader(owner['pid']))
    require(snapshot['free_mib'] >= memory_required_mib(remaining_models),
            f"GPU6 free {snapshot['free_mib']} MiB below required {memory_required_mib(remaining_models)} MiB")
    return snapshot


def enable_reference_imports():
    for p in (ROOT, ROOT/'remote_processor_candidate_v1/deps'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def parity_for(policy):
    if policy == 'B':
        return B_PARITY, B_PARITY_SHA
    require(policy == 'base', 'Unknown policy id')
    return BASE_PARITY, BASE_PARITY_SHA


def verify_references():
    """Read-only full model/source pins; no CUDA/model import."""
    require(sha(B_BINDING) == B_BINDING_SHA, 'B binding changed')
    require(sha(B_PARITY) == B_PARITY_SHA and sha(BASE_PARITY) == BASE_PARITY_SHA,
            'Pinned parity report changed')
    parities = {'B': read(B_PARITY), 'base': read(BASE_PARITY)}
    common_assets = parities['base']['model']['model_assets_sha256']
    require(common_assets == parities['B']['model']['model_assets_sha256'], 'Base checkpoint family identity differs')
    for name, digest in common_assets.items():
        require(Path(name).name == name and sha(BASE/name) == digest, 'Base model asset changed: '+name)
    for policy, parity in parities.items():
        require(parity.get('passed') is True and len(parity.get('trace', [])) == 6
                and parity.get('ambient_rng_unchanged') is True
                and all(x['full_action_exact'] and x['all_rng_exact'] for x in parity['trace']),
                'Direct/socket parity is incomplete: '+policy)
        for name, digest in parity['source_sha256'].items():
            require(Path(name).name == name and sha(ROOT/'multiplex_inference'/name) == digest,
                    'Multiplex source changed: '+name)
        require(sha(ROOT/'code/Xiaomi-Robotics-1/deploy/server.py') == parity['model']['official_server_sha256'],
                'Official base loader changed')
    enable_reference_imports()
    from continuous_eval2000_v3.loader import read_binding
    read_binding(B_BINDING, B_BINDING_SHA)
    return parities


def validate_policy_hello(policy, hello, parity):
    """Reject a valid transport identity serving the wrong policy/checkpoint."""
    require(policy in ('B', 'base'), 'Unknown policy id')
    for key in ('model_path', 'model_config_sha256', 'model_assets_sha256', 'official_server_sha256'):
        require(hello.get(key) == parity['model'].get(key), 'Wrong model identity: '+key)
    require(hello.get('model_path') == str(BASE), 'Unexpected base-model path')
    require(hello.get('multiplex_sources_sha256') == parity['source_sha256'], 'Wrong transport sources')
    require(hello.get('model_config_runtime_sha256') == parity['trace'][0]['actual_rng']['model_config_runtime_sha256'],
            'Runtime configuration differs')
    require(hello.get('torch_version') == EXPECTED_TORCH and hello.get('cuda_visible_device_count') == 1,
            'Unadmitted Torch/CUDA runtime')
    require(hello.get('protocol') == 'robocasa-multiplex-rng-v1' and hello.get('exclusive_connection') is False
            and hello.get('shared_model') is True and hello.get('isolated_rng_stream') is True
            and hello.get('serial_forward') is True and hello.get('full_request_only') is True
            and hello.get('inference_optimization') == 'none'
            and hello.get('rng_isolation') == 'python_numpy_torch_cpu_all_cuda_per_connection'
            and hello.get('model_execution_thread_name') == 'xr1-serial-forward'
            and bool(hello.get('server_instance')), 'Wrong runtime/stream protocol')
    if policy == 'base':
        require(not any(key in hello for key in ('stage_serving_identity', 'lora_serving_identity', 'loader_kind')),
                'Base profile received adapter/branch identity')
    else:
        stage, lora = hello.get('stage_serving_identity', {}), hello.get('lora_serving_identity', {})
        require(stage == parity['model'].get('stage_serving_identity') and lora == parity['model'].get('lora_serving_identity'),
                'B or parent-A identity differs from parity')
        require(stage.get('arm') == 'B' and stage.get('binding_sha256') == B_BINDING_SHA
                and stage.get('branch_file_sha256') == BRANCH_SHA
                and lora.get('binding_sha256') == A_BINDING_SHA
                and lora.get('adapter_file_sha256') == A_ADAPTER_SHA
                and hello.get('loader_kind') == 'completed_frozen_A_plus_online_stage_branch_v2',
                'Wrong B branch or frozen parent A')
    return hello


def live_hello(port):
    enable_reference_imports()
    from multiplex_inference.client import WireClient
    client = WireClient(port, timeout=15)
    try:
        return client.hello
    finally:
        client.close()


def tcp_connections(pid, port, proc_root=Path('/proc')):
    """Count owned non-LISTEN sockets, including CLOSE_WAIT/in-flight connects."""
    p = Path(proc_root)/str(pid)
    inodes = set()
    for fd in (p/'fd').iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith('socket:['):
            inodes.add(target[8:-1])
    rows = []
    for table in ('tcp', 'tcp6'):
        for line in (p/'net'/table).read_text().splitlines()[1:]:
            cols = line.split()
            if len(cols) >= 10 and cols[9] in inodes and int(cols[1].rsplit(':',1)[1],16) == port and cols[3] != '0A':
                rows.append(dict(inode=cols[9], state_hex=cols[3], local=cols[1], remote=cols[2]))
    return rows


def assert_cleanup_safe(owner, spec, actual, connections):
    validate_owner(owner, spec, actual)
    require(not connections, 'Active or closing client connections; service preserved')
    return True


def profile_from_manifest(spec, owner, manifest, manifest_sha256, parity):
    validate_policy_hello(spec['policy_id'], manifest['hello'], parity)
    require(manifest.get('ready') is True and manifest.get('max_clients') == SLOTS
            and len(manifest.get('servers', [])) == 1, 'Manifest capacity/shape differs')
    s = manifest['servers'][0]
    require(all(s.get(k) == owner[k] for k in ('pid','process_start_ticks','command'))
            and s.get('port') == spec['port'] and s.get('gpu') == GPU_UUID,
            'Server manifest does not match exact owner')
    parity_path, parity_sha = parity_for(spec['policy_id'])
    require(manifest.get('parity_sha256') == parity_sha, 'Wrong manifest parity pin')
    if spec['policy_id'] == 'B':
        require(manifest.get('stage_binding_sha256') == B_BINDING_SHA, 'Wrong manifest B binding')
    return dict(schema='originx_confirmatory_service_profile_v1', namespace=NAMESPACE,
                policy_id=spec['policy_id'], service_id=spec['service_id'], port=spec['port'],
                host='127.0.0.1', slots=SLOTS, gpu_index=6, gpu_uuid=GPU_UUID,
                owner=owner, hello=manifest['hello'], manifest_path=spec['manifest'],
                manifest_sha256=manifest_sha256, server_manifest=spec['manifest'],
                server_manifest_sha256=manifest_sha256, parity_path=str(parity_path), parity_sha256=parity_sha,
                model_path=str(BASE), xr1_repo=str(ROOT/'code/Xiaomi-Robotics-1'),
                binding_path=str(B_BINDING) if spec['policy_id']=='B' else None,
                binding_sha256=B_BINDING_SHA if spec['policy_id']=='B' else None,
                model_assets_sha256=manifest['hello']['model_assets_sha256'],
                multiplex_sources_sha256=manifest['hello']['multiplex_sources_sha256'],
                official_server_sha256=manifest['hello']['official_server_sha256'],
                profile_source_sha256=sha(Path(__file__)),
                admission_scope='pinned historical six-query full-action/RNG direct-socket gate; new development rollout preflight still required')


def validate_service(spec, owner, parities, *, probe_live=True):
    validate_owner(owner, spec)
    manifest = read(spec['manifest'])
    enable_reference_imports()
    if spec['policy_id'] == 'B':
        from continuous_eval2000_v3.client import validate_stage_endpoint
        validate_stage_endpoint(spec['manifest'], B_PARITY, BASE, binding_path=B_BINDING, binding_sha256=B_BINDING_SHA)
    else:
        from local_eval.remote_client import validate_endpoint
        validate_endpoint(spec['manifest'], BASE_PARITY, BASE)
    profile = profile_from_manifest(spec, owner, manifest, sha(spec['manifest']), parities[spec['policy_id']])
    if probe_live:
        hello = live_hello(spec['port'])
        validate_policy_hello(spec['policy_id'], hello, parities[spec['policy_id']])
        from local_eval.remote_client import check_hello
        check_hello(hello, manifest['hello'])
    validate_owner(owner, spec)
    return profile


def make_client(profile, *, timeout=1200, telemetry_path=None):
    """Factory for a fresh one-episode connection; caller invokes reset(seed)."""
    require(profile.get('namespace') == NAMESPACE and profile.get('schema') == 'originx_confirmatory_service_profile_v1',
            'Foreign/untyped service profile')
    policy = profile['policy_id']
    legal={row['service_id']:row for counts in [(4,2),(3,3)] for row in layout(*counts)}
    require(profile.get('service_id') in legal, 'Unknown profile service id')
    spec=legal[profile['service_id']]
    require(profile.get('port')==spec['port'] and profile.get('policy_id')==spec['policy_id']
            and profile.get('slots')==SLOTS and profile.get('manifest_path')==spec['manifest'],
            'Profile service layout differs')
    validate_owner(profile['owner'],spec)
    require(profile.get('gpu_index') == 6 and profile.get('gpu_uuid') == GPU_UUID, 'Profile GPU differs')
    parity_path, parity_sha = parity_for(policy)
    require(profile.get('parity_path') == str(parity_path) and profile.get('parity_sha256') == parity_sha
            and sha(parity_path) == parity_sha, 'Profile parity changed')
    require(profile.get('model_path') == str(BASE) and profile.get('xr1_repo') == str(ROOT/'code/Xiaomi-Robotics-1'),
            'Profile model/processor differs')
    validate_policy_hello(policy, profile['hello'], read(parity_path))
    require(sha(profile['manifest_path']) == profile['manifest_sha256'], 'Profile server manifest changed')
    require(not (SERVICE_OUTPUT/'draining.json').exists(), 'Services are draining; no new client admitted')
    enable_reference_imports()
    if policy == 'B':
        from continuous_eval2000_v3.client import StageEvalClient as Client
    else:
        from local_eval.remote_client import RemoteEvalClient as Client
    client = Client(profile['xr1_repo'], BASE, profile['port'], profile['hello'], timeout=timeout, telemetry_path=telemetry_path)
    try:
        validate_policy_hello(policy, client.wire.hello, read(parity_path))
    except BaseException:
        client.close()
        raise
    return client


def parity_command(policy, report_path, *, port=0):
    """Return (do not run) a fresh engineering-gate command for real fixtures."""
    report = Path(report_path).resolve()
    require(report.parent == SERVICE_OUTPUT/'fresh-parity' and not report.exists(), 'Need new namespace parity report')
    require(port == 0, 'Parity uses an OS-assigned localhost port')
    p = read(BASE_PARITY)
    fixtures = [Path(x['path']) for x in p['fixtures']]
    require(len(fixtures) == 2, 'Two real request fixtures required')
    for path, pin in zip(fixtures, p['fixtures']):
        require(sha(path)==pin['sha256'] and sha(path.with_suffix('.npz'))==pin['npz_sha256'], 'Parity fixture changed')
    prefix=[str(ROOT/'envs/training/bin/python'),'-u','-m']
    if policy == 'base':
        args=prefix+['multiplex_inference.parity','--xr1-repo',str(ROOT/'code/Xiaomi-Robotics-1'),
                     '--model',str(BASE),'--minimum-free-gib','16']
    else:
        require(policy == 'B','Unknown policy')
        args=prefix+['continuous_eval2000_v3.server','--mode','parity','--binding',str(B_BINDING),
                     '--binding-sha256',B_BINDING_SHA]
    return args+['--request-a',str(fixtures[0]),'--request-b',str(fixtures[1]),'--port','0','--report',str(report)]


@contextlib.contextmanager
def service_lock(output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output/'services.lock').open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another launcher/cleanup owns this namespace') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def all_recorded(output):
    result=[]
    for directory in sorted((output/'services').glob('*')):
        require(directory.is_dir() and not directory.is_symlink(), 'Unrecognized service artifact')
        require((directory/'owner.json').is_file(), 'Partial service directory; preserve evidence, no blind restart: '+str(directory))
        result.append((directory,read(directory/'owner.json')))
    return result


def service_environment():
    env=dict(os.environ)
    env.update(PYTHONPATH=str(ROOT)+':'+str(ROOT/'remote_processor_candidate_v1/deps'),
               CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=GPU_UUID,
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env


def launch(b_models=4, base_models=2, *, output=SERVICE_OUTPUT, timeout=600):
    output=allowed_output(output);specs=layout(b_models,base_models,output)
    by_name={s['service_id']:s for s in specs}
    with service_lock(output):
        require(not (output/'draining.json').exists(), 'Cleanup already requested; preserve namespace and review')
        parities=verify_references();owners=[];profiles={}
        for directory,owner in all_recorded(output):
            require(directory.name in by_name, 'Existing owner is outside requested layout; no rearrangement')
            spec=by_name[directory.name]
            profiles[directory.name]=validate_service(spec,owner,parities)
            owners.append(owner)
        require(len({o['pid'] for o in owners})==len(owners),'Duplicate PID owners')
        remaining=len(specs)-len(profiles)
        initial=admit_gpu(gpu_snapshot(),owners,specs,remaining)
        replace_status(output/'launch-status.json',dict(state='launching',namespace=NAMESPACE,
                       layout=specs,memory_plan=memory_plan(b_models,base_models),initial_gpu=initial,unix=time.time()))
        for spec in specs:
            if spec['service_id'] in profiles:
                continue
            admit_gpu(gpu_snapshot(),owners,specs,remaining)
            with socket.socket() as port_probe:
                port_probe.bind(('127.0.0.1',spec['port']))
            directory=Path(spec['directory']);directory.mkdir(parents=True,exist_ok=False)
            cmd=service_command(spec)
            with (directory/'server.log').open('xb') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=service_environment(),stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            owner=process_identity(child.pid)
            owner.update(namespace=NAMESPACE,policy_id=spec['policy_id'],service_id=spec['service_id'],
                         port=spec['port'],slots=SLOTS,gpu_index=6,gpu_uuid=GPU_UUID,started_unix=time.time())
            validate_owner(owner,spec);write_new(directory/'owner.json',owner);owners.append(owner)
            deadline=time.monotonic()+timeout
            while True:
                require(child.poll() is None and active(owner),'Own service exited; preserve owner/log, no restart')
                if time.monotonic()>=deadline:
                    raise TimeoutError('Model readiness timed out; preserve partial service and inspect before explicit cleanup')
                try:
                    if spec['policy_id']=='base' and not Path(spec['manifest']).exists():
                        hello=live_hello(spec['port'])
                        validate_policy_hello('base',hello,parities['base'])
                        server={k:owner[k] for k in OWNER_KEYS};server.update(port=spec['port'],gpu=GPU_UUID)
                        write_new(spec['manifest'],dict(ready=True,repo=str(ROOT),model=str(BASE),
                                  source_sha256=hello['multiplex_sources_sha256'],hello=hello,servers=[server],
                                  max_clients=SLOTS,parity_report=str(BASE_PARITY),parity_sha256=BASE_PARITY_SHA,
                                  namespace=NAMESPACE,policy_id='base',created_unix=time.time()))
                    else:
                        read(spec['manifest'])
                except (FileNotFoundError,json.JSONDecodeError,ConnectionError,socket.timeout,OSError) as exc:
                    # Only readiness transport/file incompleteness is retried.
                    # Identity/policy RuntimeError is terminal and preserved.
                    if isinstance(exc,OSError) and getattr(exc,'errno',None) not in (None,2,61,111,10061):
                        raise
                    time.sleep(.25);continue
                profiles[spec['service_id']]=validate_service(spec,owner,parities)
                write_new(directory/'profile.json',profiles[spec['service_id']])
                remaining-=1
                print(json.dumps(dict(event='ready',policy_id=spec['policy_id'],pid=owner['pid'],port=spec['port'])),flush=True)
                break
        admit_gpu(gpu_snapshot(),owners,specs,0)
        ordered=[profiles[s['service_id']] for s in specs]
        for spec,profile in zip(specs,ordered):
            validate_service(spec,profile['owner'],parities)
        receipt=dict(schema='originx_confirmatory_services_v1',namespace=NAMESPACE,ready=True,
                     models=ordered,model_count=len(ordered),clients=len(ordered)*SLOTS,gpu_index=6,gpu_uuid=GPU_UUID,
                     profiles=ordered,memory_plan=memory_plan(b_models,base_models),created_unix=time.time(),
                     rollout_admission='pending new development preflight; ready means identity-admitted services only')
        replace_status(output/'services-ready.json',receipt)
        replace_status(output/'launch-status.json',dict(state='ready',namespace=NAMESPACE,models=len(ordered),unix=time.time()))
        return receipt


def status(output=SERVICE_OUTPUT):
    output=allowed_output(output)
    report=dict(namespace=NAMESPACE,gpu=gpu_snapshot(),services=[],draining=(output/'draining.json').exists())
    for directory,owner in all_recorded(output):
        row=dict(service_id=directory.name,owner=owner,active=active(owner))
        if row['active']:
            row['connections']=tcp_connections(owner['pid'],owner['port'])
        report['services'].append(row)
    return report


def cleanup(output=SERVICE_OUTPUT, *, quiet_seconds=2, wait_seconds=30):
    output=allowed_output(output)
    with service_lock(output):
        records=all_recorded(output)
        # Reconstruct the legal spec independently of owner argv.
        legal={s['service_id']:s for counts in [(4,2),(3,3)] for s in layout(*counts)}
        living=[];inactive=[]
        for directory,owner in records:
            require(directory.name in legal,'Unknown service directory; refusing cleanup')
            spec=legal[directory.name]
            try:
                actual=process_identity(owner['pid'])
            except (FileNotFoundError,ProcessLookupError):
                inactive.append(owner['pid']);continue
            assert_cleanup_safe(owner,spec,actual,tcp_connections(owner['pid'],spec['port']))
            living.append((spec,owner))
        if not (output/'draining.json').exists():
            write_new(output/'draining.json',dict(namespace=NAMESPACE,requested_unix=time.time(),
                      reason='Explicit cleanup blocks new clients through make_client; no forced disconnects'))
        time.sleep(quiet_seconds)
        # Preflight all, then check each again immediately before its signal.
        for spec,owner in living:
            assert_cleanup_safe(owner,spec,process_identity(owner['pid']),tcp_connections(owner['pid'],spec['port']))
        signalled=[]
        for spec,owner in living:
            assert_cleanup_safe(owner,spec,process_identity(owner['pid']),tcp_connections(owner['pid'],spec['port']))
            os.kill(owner['pid'],signal.SIGTERM);signalled.append(owner['pid'])
        deadline=time.monotonic()+wait_seconds
        while any(active(o) for _,o in living) and time.monotonic()<deadline:
            time.sleep(.2)
        remaining=[o['pid'] for _,o in living if active(o)]
        receipt=dict(namespace=NAMESPACE,signalled_owned_pids=signalled,previously_inactive_pids=inactive,
                     still_active_owned_pids=remaining,all_owned_services_stopped=not remaining,
                     forced_kill=False,evidence_preserved=True,unix=time.time())
        replace_status(output/'cleanup-status.json',receipt)
        require(not remaining,'Some owned services did not stop after SIGTERM; no SIGKILL issued')
        return receipt


def serve_base(port, manifest, max_clients):
    """Clean-process trampoline only; never load or inject adaptation weights."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID,'Base child must see GPU6 UUID only')
    spec=next((s for s in layout(3,3) if s['policy_id']=='base' and s['port']==port),None)
    require(spec is not None and str(Path(manifest).resolve())==spec['manifest'] and max_clients==SLOTS,
            'Base child manifest/port/capacity outside namespace')
    enable_reference_imports()
    sys.argv=['multiplex_inference.server','--xr1-repo',str(ROOT/'code/Xiaomi-Robotics-1'),
              '--model',str(BASE),'--port',str(port),'--max-clients',str(max_clients)]
    runpy.run_module('multiplex_inference.server',run_name='__main__')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('launch');p.add_argument('--b-models',type=int,default=4);p.add_argument('--base-models',type=int,default=2)
    p.add_argument('--timeout',type=float,default=600)
    subs.add_parser('status');subs.add_parser('cleanup')
    p=subs.add_parser('probe-plan');p.add_argument('--policy-id',choices=['B','base'],required=True)
    p.add_argument('--report',type=Path,required=True)
    p=subs.add_parser('_serve-base');p.add_argument('--port',type=int,required=True)
    p.add_argument('--server-manifest',required=True);p.add_argument('--max-clients',type=int,required=True)
    args=parser.parse_args(argv)
    if args.command=='_serve-base':
        serve_base(args.port,args.server_manifest,args.max_clients);return
    if args.command=='launch':
        require(args.timeout>0,'Positive readiness timeout required')
        result=launch(args.b_models,args.base_models,timeout=args.timeout)
    elif args.command=='cleanup':result=cleanup()
    elif args.command=='status':result=status()
    else:result=dict(command=parity_command(args.policy_id,args.report),environment_gpu=GPU_UUID,
                     starts_process=False,requires_idle_owned_gpu_admission=True,
                     note='Plan only. Root must owner-pin and admit GPU before executing; this command does not run parity.')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
