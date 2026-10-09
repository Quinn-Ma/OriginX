"""Admit any new GPU3/6 B2000 service list with exact six-client parity.

Run only after the root launcher has admitted and started its owned services.
This script never starts/stops model servers, simulators, or evaluation episodes.
It opens six inference clients per listed service and makes one frozen reference
request on each, using the original campaign warmup implementation unchanged.

Example (from the remote project root):
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD:$PWD/remote_processor_candidate_v1/deps \
  envs/training/bin/python -m astra_rescue_20261009.pilot_warmup \
    --services-ready results/astra-rescue-services-<run>/services-ready.json \
    --output results/astra-rescue-services-<run>/pilot-warmup.json
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
BASE = '/ephemeral/qinzhen/ckpt/robocasa/Xiaomi-Robotics-1-RoboCasa365'
REFERENCE = ROOT / 'results/continued-ab2000-v1'
BINDING_SHA = 'c3138b49dcf51c0514f129073870701b5cd59cb88df384ac5a4b9ac24a77c601'
PARITY_SHA = '4c3726cf8871061e7f29a68a613ceba3b981d34d0bfcc9fd51dc2e9ee56df813'
RESOURCE_SHA = '7a7eaa4ffc176ad840776cdfd381ec13301947ad2d242f9426e97f83d2c2af72'
CORE_SHA = '8deb49c506b6bf3c3c795694caf215096f6463813283aa7ff42211ab0878d7ea'
GPU_UUIDS = {'GPU-d0f25b35-5423-72d4-9f45-047dfe7ed87c',
             'GPU-35278907-aae0-4ccc-e0fc-c25cb3a8bd17'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def exact_owner(owner):
    p = Path('/proc') / str(owner['pid'])
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    require(fields[0] not in ('Z', 'X'), 'Owned service is no longer active')
    actual = dict(pid=owner['pid'], process_start_ticks=int(fields[19]),
                  command=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                  cwd=str((p/'cwd').resolve()))
    require(actual == {key: owner[key] for key in actual}, 'Service PID/start/command/cwd changed')
    return actual


def run(args):
    started = time.monotonic()
    ready_path = args.services_ready.resolve(strict=True)
    require(ROOT/'results' in ready_path.parents, 'Service list must be under this project results')
    output = (args.output or ready_path.parent/'pilot-warmup.json').resolve()
    require(ROOT/'results' in output.parents, 'Warmup evidence must be under this project results')
    require(output != ready_path and not output.exists(), 'Use a fresh independent warmup evidence path')
    record = dict(schema='astra_B2000_six_client_warmup_v1', passed=False,
                  services_ready=str(ready_path), services_ready_sha256=sha(ready_path),
                  script_sha256=sha(__file__), unix=time.time(), no_servers_started_or_stopped=True,
                  no_simulator_or_episode_started=True)
    try:
        # Client-side processor fixtures and RNG audit stay on the CPU.
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                          OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        sys.path[:0] = [str(ROOT), str(ROOT/'remote_processor_candidate_v1/deps')]
        resources_path = ROOT/'five_hour_campaign_v1/recovery_b600_v2/resources.py'
        core_path = resources_path.with_name('core.py')
        require(sha(resources_path) == RESOURCE_SHA and sha(core_path) == CORE_SHA,
                'Original campaign warmup helper source changed')
        require(sha(REFERENCE/'B-endpoint.json') == BINDING_SHA and
                sha(REFERENCE/'parity/B.json') == PARITY_SHA, 'Frozen B2000 reference changed')
        from five_hour_campaign_v1.recovery_b600_v2.resources import warmup
        from continuous_eval2000_v3.client import validate_stage_endpoint
        from continuous_eval2000_v3.loader import check_stage_hello
        from local_eval.remote_client import check_hello
        from multiplex_inference.client import WireClient
        import torch

        ready = read(ready_path)
        manifests = [Path(value).resolve(strict=True) for value in ready['manifests']]
        require(ready.get('ready') is True and manifests and len(manifests) == len(set(manifests)),
                'Need a ready, nonempty, unique service list')
        require(ready.get('models', len(manifests)) == len(manifests), 'Ready model count differs')
        parity = read(REFERENCE/'parity/B.json')
        require(parity.get('passed') is True and parity.get('stage_path_checked_on_every_forward') is True,
                'Frozen stage/direct parity did not pass')
        fixtures = [dict(path=path) for path in parity['fixture_sha256']]
        require(len(fixtures) == 2, 'The original warmup uses two distinct processor fixtures')
        for item in fixtures:
            require(sha(item['path']) == parity['fixture_sha256'][item['path']], 'Frozen processor fixture changed')
        items, evidence, owners, ports = [], [], [], set()
        for i, path in enumerate(manifests):
            require(ROOT/'results' in path.parents, 'Manifest must be under this project results')
            manifest_sha = sha(path)
            service = validate_stage_endpoint(path, REFERENCE/'parity/B.json', BASE,
                                             binding_path=str(REFERENCE/'B-endpoint.json'), binding_sha256=BINDING_SHA)
            server = service['servers'][0]
            owner_path = path.parent/'owner.json'
            owner = read(owner_path)
            require(owner.get('gpu') in GPU_UUIDS and server.get('gpu') == owner['gpu'],
                    'Only this launch GPU3/6 services are allowed')
            require(all(server.get(key) == owner[key] for key in ('pid','process_start_ticks','command')),
                    'Manifest and launch owner differ')
            require(service.get('max_clients') == 6 and server['port'] not in ports,
                    'Need six-client capacity and unique ports')
            exact_owner(owner)
            # Check live endpoint identity before the original helper opens its
            # six simultaneous clients. This temporary hello connection closes
            # first and issues no model request or RNG reset.
            client = WireClient(server['port'], timeout=180)
            try:
                check_hello(client.hello, service['hello'])
                check_stage_hello(client.hello, service['hello'])
                live_connection = client.hello['connection_id']
            finally:
                client.close()
            key = f'B-{i}'
            items.append(dict(key=key, arm='B', port=server['port']))
            owners.append(owner)
            ports.add(server['port'])
            evidence.append(dict(key=key, server_manifest=str(path), server_manifest_sha256=manifest_sha,
                                 owner_path=str(owner_path), owner_sha256=sha(owner_path), owner=owner,
                                 port=server['port'], live_hello_connection_id=live_connection))
        record.update(models=len(items), expected_concurrent_clients=6*len(items), services=evidence,
                      binding_sha256=BINDING_SHA, parity_sha256=PARITY_SHA,
                      warmup_resource_sha256=RESOURCE_SHA, warmup_core_sha256=CORE_SHA,
                      fixture_sha256=parity['fixture_sha256'])
        result = warmup(dict(fixtures=fixtures), items, {'B': parity})
        require(result['passed'] is True and result['clients_per_model'] == 6
                and result['concurrent_clients'] == 6*len(items), 'Concurrent client total differs')
        require(Counter(row['key'] for row in result['rows']) == {item['key']: 6 for item in items},
                'Not every listed service completed six clients')
        require(all(row['full_action_exact'] is True and row['all_rng_exact'] is True for row in result['rows']),
                'At least one client differed from the frozen direct action/RNG reference')
        require(len({row['connection_id'] for row in result['rows']}) == 6*len(items),
                'Repeated logical-worker connection identity')
        require(sha(ready_path) == record['services_ready_sha256'], 'Service readiness list changed during warmup')
        for owner, item in zip(owners, evidence):
            exact_owner(owner)
            require(sha(item['server_manifest']) == item['server_manifest_sha256']
                    and sha(item['owner_path']) == item['owner_sha256'], 'Service binding changed during warmup')
        require(not torch.cuda.is_initialized(), 'Warmup client unexpectedly initialized CUDA')
        record.update(result, passed=True, client_cuda_initialized=False)
    except BaseException:
        record['error'] = traceback.format_exc()
        raise
    finally:
        record['wall_seconds'] = time.monotonic()-started
        write_exclusive(output, record)
        print(json.dumps(dict(passed=record['passed'], output=str(output), models=record.get('models'),
                              concurrent_clients=record.get('concurrent_clients'),
                              wall_seconds=record['wall_seconds'])), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--services-ready', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
