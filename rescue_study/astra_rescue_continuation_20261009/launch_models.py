"""Launch/reuse only owned frozen B2000 services on idle GPU3 and GPU6."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

R = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
BASE = '/ephemeral/qinzhen/ckpt/robocasa/Xiaomi-Robotics-1-RoboCasa365'
GPU = {3: 'GPU-d0f25b35-5423-72d4-9f45-047dfe7ed87c',
       6: 'GPU-35278907-aae0-4ccc-e0fc-c25cb3a8bd17'}
BINDING_SHA = 'c3138b49dcf51c0514f129073870701b5cd59cb88df384ac5a4b9ac24a77c601'
PARITY_SHA = '4c3726cf8871061e7f29a68a613ceba3b981d34d0bfcc9fd51dc2e9ee56df813'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def identity(pid):
    path = Path('/proc') / str(pid)
    stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    if stat[0] in ('Z', 'X'):
        raise ProcessLookupError('Process has exited')
    return dict(pid=pid, process_start_ticks=int(stat[19]),
                command=(path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                cwd=str((path / 'cwd').resolve()))


def active(owner):
    try:
        return identity(owner['pid']) == {key: owner[key] for key in
                                         ('pid', 'process_start_ticks', 'command', 'cwd')}
    except (FileNotFoundError, ProcessLookupError):
        return False


def smi(*args):
    # The only process this launcher terminates is its own timed-out nvidia-smi.
    process = subprocess.Popen(['nvidia-smi', *args], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        out, err = process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError('nvidia-smi timed out; GPU ownership is unknown; no further model launch')
    require(process.returncode == 0, 'nvidia-smi failed: ' + err[-500:])
    return out.strip()


def audit(gpu, known, *, new_launch=True):
    row = [value.strip() for value in smi(
        '-i', str(gpu), '--query-gpu=uuid,memory.free,utilization.gpu,ecc.errors.uncorrected.volatile.total',
        '--format=csv,noheader,nounits').split(',')]
    require(len(row) == 4 and row[0] == GPU[gpu] and int(row[3]) == 0,
            'GPU identity/ECC mismatch: ' + repr(row))
    processes = smi('-i', str(gpu), '--query-compute-apps=pid,gpu_uuid,process_name',
                    '--format=csv,noheader,nounits')
    for line in processes.splitlines():
        pid_text, uuid, _ = line.split(',', 2)
        if uuid.strip() != GPU[gpu]:
            continue
        owner = known.get(int(pid_text))
        require(owner is not None and owner.get('gpu') == GPU[gpu] and active(owner),
                'Foreign or changed GPU process; preserving it: ' + line)
    if new_launch:
        require(int(row[1]) >= 16384, 'Insufficient GPU memory for a new service: ' + repr(row))
    return row


def command(binding, parity, manifest, gpu, slot):
    port = 18800 + list(GPU).index(gpu) * 10 + slot
    return [str(R / 'envs/training/bin/python'), '-u', '-m', 'continuous_eval2000_v3.server',
            '--mode', 'serve', '--binding', str(binding), '--binding-sha256', BINDING_SHA,
            '--parity', str(parity), '--parity-sha256', PARITY_SHA, '--port', str(port),
            '--max-clients', '6', '--server-manifest', str(manifest)], port


def reference_validator(binding, parity):
    require(sha(binding) == BINDING_SHA, 'Frozen B2000 binding changed')
    require(sha(parity) == PARITY_SHA, 'Frozen B2000 parity changed')
    for path in (R, R / 'remote_processor_candidate_v1/deps'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from continuous_eval2000_v3.loader import read_binding
    from continuous_eval2000_v3.client import validate_stage_endpoint
    # Checks frozen source, branch, parent adapter and completion before GPU launch.
    read_binding(binding, BINDING_SHA)
    return validate_stage_endpoint


def wait_manifest(manifest, owner, deadline):
    while True:
        require(active(owner), 'Service exited or identity changed while awaiting manifest: ' + str(manifest))
        try:
            return read(manifest)
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise TimeoutError('Readiness manifest incomplete; directory/owner/log retained for review: '
                                   + str(manifest))
            time.sleep(.2)


def validate_service(directory, gpu, slot, binding, parity, validator, *, owner=None, deadline=None):
    manifest = directory / 'servers.json'
    if owner is None:
        try:
            owner = read(directory / 'owner.json')
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise partial_error(directory) from error
    cmd, port = command(binding, parity, manifest, gpu, slot)
    require(active(owner), 'Recorded service is inactive or changed; preserve evidence and use a new '
            'astra-rescue-services-* output after review: ' + str(directory))
    require(owner.get('gpu') == GPU[gpu] and owner['command'] == cmd and owner['cwd'] == str(R),
            'Recorded service command/cwd/GPU differs; preserving it: ' + str(directory))
    require(sha(binding) == BINDING_SHA and sha(parity) == PARITY_SHA, 'Frozen reference changed')
    wait_manifest(manifest, owner, time.monotonic() + 300 if deadline is None else deadline)
    info = validator(manifest, parity, BASE, binding_path=str(binding), binding_sha256=BINDING_SHA)
    require(info.get('ready') is True and len(info.get('servers', [])) == 1,
            'Service manifest is not ready: ' + str(manifest))
    server = info['servers'][0]
    require(all(server.get(key) == owner[key] for key in ('pid', 'process_start_ticks', 'command'))
            and server.get('gpu') == GPU[gpu] and server.get('port') == port,
            'Service manifest owner/GPU/port differs: ' + str(manifest))
    require(info.get('stage_binding_sha256') == BINDING_SHA and info.get('parity_sha256') == PARITY_SHA
            and info.get('max_clients') == 6,
            'Service manifest binding/parity/capacity differs: ' + str(manifest))
    require(active(owner), 'Service exited during admission: ' + str(directory))
    return owner


def partial_error(directory):
    return RuntimeError('Partial service directory retained for review; no restart or deletion was attempted. '
                        'Inspect owner/log and use a new astra-rescue-services-* output if appropriate: '
                        + str(directory))


def main(argv=None):
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slots-per-gpu', type=int, choices=range(1, 7), default=1,
                        help='Model services per GPU; each service accepts six clients. Set runner parallel separately.')
    args = parser.parse_args(argv)
    directory = args.output.resolve()
    require(directory.parent == R / 'results' and directory.name.startswith('astra-rescue-services-'),
            'Unexpected service output')
    directory.mkdir(parents=True, exist_ok=True)
    # The persistent file holds evidence only. Kernel flock lifetime is the lock.
    with (directory / 'launch.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another launcher owns this service output; no launch attempted') from error
        try:
            lock.seek(0)
            lock.truncate()
            json.dump(identity(os.getpid()), lock)
            lock.flush()
            binding = R / 'results/continued-ab2000-v1/B-endpoint.json'
            parity = R / 'results/continued-ab2000-v1/parity/B.json'
            validator = reference_validator(binding, parity)
            own = {}
            locations = {f'B-gpu{gpu}-r{slot}': (gpu, slot) for gpu in GPU for slot in range(6)}
            for service in (directory / 'services').glob('*'):
                require(service.name in locations and service.is_dir(),
                        'Unknown service artifact; preserve and inspect: ' + str(service))
                if not (service / 'owner.json').is_file() or not (service / 'servers.json').is_file():
                    raise partial_error(service)
                gpu, slot = locations[service.name]
                owner = validate_service(service, gpu, slot, binding, parity, validator)
                require(owner['pid'] not in own, 'Duplicate service PID; preserve and inspect: ' + str(service))
                own[owner['pid']] = owner
            manifests = []
            for gpu in GPU:
                audit(gpu, own, new_launch=False)
                for slot in range(args.slots_per_gpu):
                    service = directory / 'services' / f'B-gpu{gpu}-r{slot}'
                    manifest = service / 'servers.json'
                    if service.exists():
                        if not manifest.is_file() or not (service / 'owner.json').is_file():
                            raise partial_error(service)
                        validate_service(service, gpu, slot, binding, parity, validator)
                        manifests.append(str(manifest))
                        continue
                    audit(gpu, own)
                    cmd, port = command(binding, parity, manifest, gpu, slot)
                    # A port refusal leaves no new service directory behind.
                    with socket.socket() as probe:
                        probe.bind(('127.0.0.1', port))
                    service.mkdir(parents=True, exist_ok=False)
                    env = os.environ.copy()
                    env.update(PYTHONPATH=str(R) + ':' + str(R / 'remote_processor_candidate_v1/deps'),
                               CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=GPU[gpu],
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1',
                               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                               CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    with (service / 'server.log').open('xb') as log:
                        process = subprocess.Popen(cmd, cwd=R, env=env, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    owner = identity(process.pid)
                    owner.update(gpu=GPU[gpu], started_unix=time.time())
                    write(service / 'owner.json', owner)
                    own[process.pid] = owner
                    deadline = time.monotonic() + 300
                    while True:
                        require(process.poll() is None and active(owner),
                                'Own model exited or identity changed; owner/log retained: ' + str(service))
                        try:
                            # The frozen server publishes using open(x), so JSON may be briefly incomplete.
                            read(manifest)
                        except (FileNotFoundError, json.JSONDecodeError):
                            if time.monotonic() >= deadline:
                                raise TimeoutError('Own model readiness timeout; partial directory/owner/log retained '
                                                   'for review: ' + str(service))
                            time.sleep(.2)
                            continue
                        validate_service(service, gpu, slot, binding, parity, validator, owner=owner, deadline=deadline)
                        break
                    manifests.append(str(manifest))
                    print(json.dumps(dict(event='ready', gpu=gpu, slot=slot, pid=process.pid, port=port)), flush=True)
            # Recheck all selected owners and GPU occupancy before publishing readiness.
            for gpu in GPU:
                audit(gpu, own, new_launch=False)
                for slot in range(args.slots_per_gpu):
                    validate_service(directory / 'services' / f'B-gpu{gpu}-r{slot}',
                                     gpu, slot, binding, parity, validator)
            snapshot = dict(ready=True, models=len(manifests), manifests=manifests, gpus=GPU,
                            binding_sha256=BINDING_SHA, parity_sha256=PARITY_SHA, unix=time.time())
            tmp = directory / 'services-ready.tmp'
            tmp.write_text(json.dumps(snapshot, indent=2) + '\n')
            tmp.replace(directory / 'services-ready.json')
            print(json.dumps(snapshot), flush=True)
            return snapshot
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == '__main__':
    main()
