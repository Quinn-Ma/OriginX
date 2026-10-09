"""Real same-socket >300s keepalive and frozen fixture/action/RNG verification.

No simulator episodes or model services are started. Each admitted service gets
one socket, one initial seed, and the frozen three-request parity sequence.
Wait after request one for 360 seconds, reading RNG at entry/120/240/exit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from astra_rescue_20261009 import pilot_warmup as authority
from .assistance import AssistanceClient, normalized_state, INTERVAL

RNG_KEYS = ('python_rng_sha256','numpy_rng_sha256','torch_cpu_rng_sha256','torch_cuda_rng_sha256')


def run(args):
    a = authority; started = time.monotonic(); output = args.output.resolve()
    a.require(not output.exists() and not output.with_suffix('.clients').exists(), 'Fresh probe evidence path required')
    a.require(args.duration_seconds >= 360, 'Probe must cross the300s server idle limit with margin')
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path[:0] = [str(a.ROOT), str(a.ROOT/'remote_processor_candidate_v1/deps')]
    record = dict(schema='astra_same_socket_keepalive_probe_v2', passed=False,
                  duration_seconds=args.duration_seconds, interval_seconds=INTERVAL,
                  no_simulator_or_episode_started=True, no_servers_started_or_stopped=True,
                  one_initial_seed_per_connection=True, no_reconnect=True,
                  script_sha256=a.sha(__file__), assistance_sha256=a.sha(Path(__file__).with_name('assistance.py')),
                  services_ready=str(args.services_ready.resolve()), services_ready_sha256=a.sha(args.services_ready),
                  unix=time.time())
    try:
        import torch
        from multiplex_inference.client import WireClient
        from multiplex_inference.parity import load_request_fixture, tensor_hash
        from continuous_eval2000_v3.client import validate_stage_endpoint
        from continuous_eval2000_v3.loader import check_stage_hello
        from local_eval.remote_client import check_hello
        a.require(a.sha(a.REFERENCE/'B-endpoint.json') == a.BINDING_SHA
                  and a.sha(a.REFERENCE/'parity/B.json') == a.PARITY_SHA, 'Frozen B2000 reference changed')
        parity = a.read(a.REFERENCE/'parity/B.json')
        fixture_paths = list(parity['fixture_sha256'])
        for file, digest in parity['fixture_sha256'].items(): a.require(a.sha(file) == digest, 'Fixture changed')
        for file, digest in parity['fixture_npz_sha256'].items(): a.require(a.sha(file) == digest, 'Fixture tensor changed')
        requests = [load_request_fixture(path, torch) for path in fixture_paths]
        reference = sorted((r for r in parity['trace'] if r['client'] == 'a'), key=lambda r:r['sequence_index'])
        a.require([r['sequence_index'] for r in reference] == [0,1,2], 'Frozen sequence unavailable')
        ready = a.read(args.services_ready)
        paths = [Path(p) for p in ready['manifests']]
        a.require(ready.get('ready') is True and ready['models'] == len(paths), 'Services not ready')
        if args.service_index is not None:
            a.require(0 <= args.service_index < len(paths), 'Bad service index')
            paths = [paths[args.service_index]]
        admitted = []
        for path in paths:
            a.require(a.ROOT/'results' in path.resolve().parents, 'Manifest outside project results')
            service = validate_stage_endpoint(path, a.REFERENCE/'parity/B.json', a.BASE,
                        binding_path=str(a.REFERENCE/'B-endpoint.json'), binding_sha256=a.BINDING_SHA)
            owner = a.read(path.parent/'owner.json'); a.exact_owner(owner)
            a.require(owner['gpu'] in a.GPU_UUIDS and service['servers'][0]['gpu'] == owner['gpu'], 'GPU identity differs')
            a.require(all(service['servers'][0][k] == owner[k] for k in ('pid','process_start_ticks','command')), 'Owner differs')
            admitted.append(dict(manifest=service, path=path, sha256=a.sha(path), owner=owner))

        def one(item):
            service = item['manifest']; port = service['servers'][0]['port']
            folder = output.with_suffix('.clients')/str(port); folder.mkdir(parents=True, exist_ok=False)
            client = WireClient(port, timeout=180)
            rows = []; receipt = dict(port=port, server_manifest=str(item['path']), server_manifest_sha256=item['sha256'],
                                     owner=item['owner'], passed=False)
            try:
                check_hello(client.hello, service['hello']); check_stage_hello(client.hello, service['hello'])
                receipt['connection_id'] = client.hello['connection_id']
                ack = client.reset(940001)  # Exactly one initialization, before any fixture forward.
                a.require(ack['requests_since_reset'] == 0, 'Reset acknowledgement differs')
                lazy = SimpleNamespace(policy=SimpleNamespace(wire=client), seed=940001, infer_calls=0)
                helper = AssistanceClient(lazy, request_root=folder/'unused-queue', episode_output=folder,
                    request_id='probe', case_id='transport-probe', arm='astra', horizon=450, wait_seconds=1200)
                for i, ref in enumerate(reference):
                    action = client.request(requests[ref['input_index']]); lazy.infer_calls += 1
                    state = normalized_state(client.control('rng_state'))
                    a.require(state['requests_since_reset'] == i+1 and state['seed'] == 940001, 'Counter/seed differs')
                    action_exact = tensor_hash(action, torch) == ref['reference_sha256']
                    rng_exact = all(state[k] == ref['reference_rng'][k] for k in RNG_KEYS)
                    a.require(action_exact and rng_exact, 'Frozen full-action or RNG parity differs')
                    rows.append(dict(sequence_index=i, input_index=ref['input_index'], full_action_exact=action_exact,
                                     all_rng_exact=rng_exact, requests_since_reset=state['requests_since_reset']))
                    if i == 0:
                        helper._keepalive('enter'); begin = time.monotonic(); next_ping = begin + INTERVAL
                        while time.monotonic()-begin < args.duration_seconds:
                            if time.monotonic() >= next_ping:
                                helper._keepalive('periodic'); next_ping = time.monotonic()+INTERVAL
                            time.sleep(min(.5, max(.001, args.duration_seconds-(time.monotonic()-begin))))
                        helper._keepalive('exit')
                        receipt['observed_wait_seconds'] = time.monotonic()-begin
                        receipt['keepalive'] = helper.record['keepalive']
                        a.require(receipt['observed_wait_seconds'] > 300 and helper._keepalive_count >= 4,
                                  'Did not cross idle boundary with periodic same-socket reads')
                a.exact_owner(item['owner'])
                a.require(a.sha(item['path']) == item['sha256'], 'Manifest changed during probe')
                receipt.update(passed=True, rows=rows, one_reset_only=True)
                return receipt
            except BaseException:
                receipt.update(error=traceback.format_exc(), rows=rows)
                raise
            finally:
                client.close(); a.write_exclusive(folder/'receipt.json', receipt)
        with ThreadPoolExecutor(max_workers=len(admitted)) as pool:
            record['services'] = list(pool.map(one, admitted))
        a.require(a.sha(args.services_ready) == record['services_ready_sha256'], 'Ready manifest changed')
        a.require(not torch.cuda.is_initialized(), 'Client unexpectedly initialized CUDA')
        record.update(passed=True, models=len(admitted), client_cuda_initialized=False)
    except BaseException:
        record['error'] = traceback.format_exc()
        raise
    finally:
        record['wall_seconds'] = time.monotonic()-started
        a.write_exclusive(output, record)
        print(json.dumps(dict(passed=record['passed'], output=str(output), models=record.get('models'),
                              wall_seconds=record['wall_seconds'])), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--services-ready', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--service-index', type=int)
    p.add_argument('--duration-seconds', type=float, default=360)
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
