"""Six-client real-action/RNG replay and 360 s same-socket keepalive admission.

This does not run environment episodes or inspect the confirmatory outcomes.
It replays the two pinned engineering fixtures and seed 940001 from each
policy's own six-query direct/socket parity report. All six streams perform
the first two reference forwards, hold their same sockets for at least 360 s
with read-only RNG keepalives, then verify the third real forward. No reset
or reconnect is permitted between these forwards.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import threading
import time

try:
    from . import services
except ImportError:
    import services

RNG_KEYS = ('python_rng_sha256','numpy_rng_sha256','torch_cpu_rng_sha256','torch_cuda_rng_sha256')
STREAM_KEYS = ('connection_id','server_instance','seed','requests_since_reset',
               'model_path','model_config_sha256','model_config_runtime_sha256') + RNG_KEYS


def reference_rows(parity):
    rows=sorted([r for r in parity['trace'] if r.get('client')=='a'],key=lambda r:r['sequence_index'])
    services.require(parity.get('passed') is True and parity.get('seeds',{}).get('a')==940001,
                     'Pinned reference must be passed seed-940001 stream a')
    services.require(len(rows)==3 and [r['sequence_index'] for r in rows]==[0,1,2]
                     and [r['input_index'] for r in rows]==[0,1,0]
                     and all(r['full_action_exact'] and r['all_rng_exact'] and r['action_sha256']==r['reference_sha256'] for r in rows),
                     'Reference trace is not the complete exact three-forward sequence')
    return rows


def validate_action_rng(reference, action_sha256, rng, expected_count):
    services.require(action_sha256==reference['reference_sha256'],'Real action differs from direct reference')
    services.require(rng.get('seed')==940001 and rng.get('requests_since_reset')==expected_count,
                     'Stream reset/replay count changed')
    for key,value in reference['reference_rng'].items():
        services.require(rng.get(key)==value,'Policy RNG differs from direct reference: '+key)
    return True


def validate_keepalive(before, after):
    services.require(all(key in before and key in after for key in STREAM_KEYS), 'Incomplete keepalive identity')
    services.require(all(before[key]==after[key] for key in STREAM_KEYS),
                     'Keepalive changed socket/instance/RNG/request count; no reconnect or reset is allowed')
    return True


def fixture_paths(policy, parity):
    # B names the same two fixture paths in insertion order; the base report
    # carries the original descriptor order explicitly. Verify both kinds.
    if policy=='base':
        pins=parity['fixtures']
        paths=[Path(p['path']) for p in pins]
        for path,pin in zip(paths,pins):
            services.require(services.sha(path)==pin['sha256'] and services.sha(path.with_suffix('.npz'))==pin['npz_sha256'],
                             'Base probe fixture changed')
    else:
        paths=[Path(p) for p in parity['fixture_sha256']]
        for path in paths:
            services.require(services.sha(path)==parity['fixture_sha256'][str(path)]
                             and services.sha(path.with_suffix('.npz'))==parity['fixture_npz_sha256'][str(path.with_suffix('.npz'))],
                             'B probe fixture changed')
    services.require(len(paths)==2,'Exactly two engineering fixtures are required')
    return paths


def one_stream(profile, client_index, requests, refs, parity, barrier, stop_event,
               output, *, hold_seconds, interval_seconds):
    from multiplex_inference.client import WireClient
    from multiplex_inference.parity import tensor_hash
    from local_eval.remote_client import check_hello
    import torch

    record=dict(policy_id=profile['policy_id'],service_id=profile['service_id'],port=profile['port'],
                client_index=client_index,reference_seed=940001,reference_client='a',
                same_socket=True,reconnects=0,resets=0,queries=[],keepalives=[],passed=False)
    client=None
    try:
        client=WireClient(profile['port'],timeout=180)
        check_hello(client.hello,profile['hello'])
        services.validate_policy_hello(profile['policy_id'],client.hello,parity)
        record['connection_id']=client.hello['connection_id']
        record['server_instance']=client.hello['server_instance']
        record['reset_ack']=client.reset(940001);record['resets']=1
        barrier.wait(timeout=90)  # all six sockets on this endpoint stay open
        before=None
        for q in (0,1):
            if stop_event.is_set():raise RuntimeError('Peer probe failed; no further inference')
            begin=time.monotonic();action=client.request(requests[refs[q]['input_index']])
            rng=client.control('rng_state');digest=tensor_hash(action,torch)
            validate_action_rng(refs[q],digest,rng,q+1)
            record['queries'].append(dict(sequence_index=q,action_sha256=digest,reference_sha256=refs[q]['reference_sha256'],
                                          rng=rng,elapsed_seconds=time.monotonic()-begin,exact=True))
            before=rng
        barrier.wait(timeout=180)
        begin_hold=time.monotonic();deadline=begin_hold+hold_seconds
        while time.monotonic()<deadline:
            if stop_event.wait(min(interval_seconds,max(0,deadline-time.monotonic()))):
                raise RuntimeError('Peer probe failed during keepalive hold')
            after=client.control('rng_state')
            validate_keepalive(before,after)
            services.validate_policy_hello(profile['policy_id'],after,parity)
            record['keepalives'].append(dict(elapsed_seconds=time.monotonic()-begin_hold,unchanged=True,rng=after))
        record['hold_seconds_actual']=time.monotonic()-begin_hold
        services.require(record['hold_seconds_actual']>=hold_seconds,'Keepalive duration truncated')
        q=2;begin=time.monotonic();action=client.request(requests[refs[q]['input_index']])
        rng=client.control('rng_state');digest=tensor_hash(action,torch)
        validate_action_rng(refs[q],digest,rng,3)
        record['queries'].append(dict(sequence_index=2,action_sha256=digest,reference_sha256=refs[2]['reference_sha256'],
                                      rng=rng,elapsed_seconds=time.monotonic()-begin,exact=True,after_hold=True))
        services.require(rng['connection_id']==record['connection_id'],'Final forward used a different socket')
        record['passed']=True
    except BaseException as exc:
        stop_event.set()
        try:barrier.abort()
        except Exception:pass
        record.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        if client is not None:
            client.close()
        services.write_new(output/f"{profile['service_id']}-client{client_index}.json",record)
    return record


def run_probe(services_manifest, output, *, hold_seconds=360, interval_seconds=120):
    path=Path(services_manifest).resolve();output=Path(output).resolve()
    services.require(path==services.SERVICE_OUTPUT/'services-ready.json','Probe must use current namespace services manifest')
    services.require(output.parent==services.SERVICE_OUTPUT/'probes' and not output.exists(),
                     'Probe requires a fresh namespace probe directory; no overwrites')
    services.require(hold_seconds>=360 and 0<interval_seconds<=120,'Require >=360 s hold and <=120 s keepalive')
    services.require(not (services.SERVICE_OUTPUT/'draining.json').exists(),'Service cleanup is in progress')
    ready=services.read(path);profiles=ready['models']
    services.require(ready.get('namespace')==services.NAMESPACE and ready.get('ready') is True
                     and 2<=len(profiles)<=6 and {p['policy_id'] for p in profiles}=={'B','base'},
                     'Both policy profiles must be present, with <=6 model services')
    services.require(len({p['service_id'] for p in profiles})==len(profiles),'Duplicate service profiles')
    # This process owns only CPU tensors. GPU requests run solely in the pinned
    # model processes; no CUDA model or extra adapter is loaded by this probe.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    services.enable_reference_imports()
    from multiplex_inference.parity import load_request_fixture
    import torch
    services.require(not torch.cuda.is_initialized(),'Probe process must not initialize CUDA')
    parities={policy:services.read(services.parity_for(policy)[0]) for policy in ('B','base')}
    legal={s['service_id']:s for counts in [(4,2),(3,3)] for s in services.layout(*counts)}
    requests={};references={}
    for policy,parity in parities.items():
        pp,digest=services.parity_for(policy)
        services.require(services.sha(pp)==digest,'Policy parity pin changed')
        references[policy]=reference_rows(parity)
        requests[policy]=[load_request_fixture(p,torch) for p in fixture_paths(policy,parity)]
    for profile in profiles:
        services.require(profile['service_id'] in legal and profile['slots']==6,'Unsupported probe layout')
        verified=services.validate_service(legal[profile['service_id']],profile['owner'],parities,probe_live=False)
        services.require(verified['manifest_sha256']==profile['manifest_sha256'],'Ready profile manifest differs')
        services.require(not services.tcp_connections(profile['owner']['pid'],profile['port']),
                         'Endpoint has active clients; probe must not compete with rollouts')
    output.mkdir(parents=True,exist_ok=False)
    services.write_new(output/'owner.json',dict(services.process_identity(os.getpid()),namespace=services.NAMESPACE,
                       purpose='six-client engineering replay and keepalive probe',starts_models=False))
    services.write_new(output/'config.json',dict(services_manifest=str(path),services_manifest_sha256=services.sha(path),
                       source_sha256=services.sha(Path(__file__)),hold_seconds=hold_seconds,interval_seconds=interval_seconds,
                       clients_per_endpoint=6,models=len(profiles),seed=940001,sequence=[0,1,0],
                       no_environment_rollouts=True,no_confirmatory_outcomes_read=True))
    stop=threading.Event();jobs=[]
    begin=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6*len(profiles),thread_name_prefix='same-socket-probe') as pool:
        for profile in profiles:
            barrier=threading.Barrier(6)
            for idx in range(6):
                jobs.append(pool.submit(one_stream,profile,idx,requests[profile['policy_id']],references[profile['policy_id']],
                                        parities[profile['policy_id']],barrier,stop,output,
                                        hold_seconds=hold_seconds,interval_seconds=interval_seconds))
        records=[job.result() for job in jobs]
    passed=all(r['passed'] for r in records)
    for profile in profiles:
        group=[r for r in records if r['service_id']==profile['service_id']]
        passed=passed and len(group)==6 and len({r.get('connection_id') for r in group})==6
    services.require(not torch.cuda.is_initialized(),'Probe unexpectedly initialized CUDA')
    result=dict(schema='originx_confirmatory_socket_probe_v1',namespace=services.NAMESPACE,passed=passed,
                models=len(profiles),clients=len(records),full_forwards=sum(len(r['queries']) for r in records),
                required_hold_seconds=hold_seconds,elapsed_seconds=time.monotonic()-begin,
                no_reconnect=True,one_reset_per_stream=True,no_environment_rollouts=True,
                scope='Exact action/RNG replay of historical engineering fixtures, six concurrent connections per endpoint, third action verified after unchanged same-socket hold; not an environment-result claim',
                records=[dict(service_id=r['service_id'],client_index=r['client_index'],passed=r['passed'],
                              error=r.get('error'),path=str(output/f"{r['service_id']}-client{r['client_index']}.json")) for r in records])
    services.write_new(output/'result.json',result)
    services.require(passed,'Engineering socket/keepalive probe failed; all records preserved')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--services',type=Path,default=services.SERVICE_OUTPUT/'services-ready.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hold-seconds',type=float,default=360)
    parser.add_argument('--interval-seconds',type=float,default=120)
    args=parser.parse_args(argv)
    print(json.dumps(run_probe(args.services,args.output,hold_seconds=args.hold_seconds,interval_seconds=args.interval_seconds),indent=2))


if __name__=='__main__':main()
