"""GR00T direct reference and six-stream same-socket numerical parity admission.

fixtures and socket phases are CPU-only. direct explicitly loads one GR00T GPU
model, and therefore must be scheduled only after the parent's GPU admission.
No phase creates environments, reads confirmation results, trains, or calls Astra.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
import numpy as np
from .adapter import CAMERAS,STATE_DIMS,LANGUAGE,PolicyAdapter,flatten_actions,build_observation,PROTOCOL
from .core import POLICY_ID,RngBank
from .client import GR00TClient,load_profile
from . import server

RNG_KEYS=('python_rng_sha256','numpy_rng_sha256','torch_cpu_rng_sha256','torch_cuda_rng_sha256')
STATE_KEYS=('protocol','server_instance','connection_id','model_config_sha256','model_config_runtime_sha256','seed','requests_since_reset')+RNG_KEYS
SEEDS=tuple(940001+i for i in range(6))
SEQUENCE=(0,1,0)

def action_hash(action):
    a=np.ascontiguousarray(action,dtype=np.float32)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()

def validate_keepalive(before,after):
    server.require(all(k in before and k in after for k in STATE_KEYS),'Incomplete RNG/identity receipt')
    server.require(all(before[k]==after[k] for k in STATE_KEYS),'Same-socket identity/RNG changed during wait')

def validate_result(reference,actions,rng,count,seed):
    server.require(action_hash(actions)==reference['action_sha256'],'Full decoded action differs from direct reference')
    server.require(rng['seed']==seed and rng['requests_since_reset']==count,'Seed/query count differs')
    server.require(all(rng.get(k)==reference['rng'].get(k) for k in RNG_KEYS),'RNG differs from direct reference')

def new_directory(path):
    path=Path(path)
    server.require(path.resolve().is_relative_to(server.BASE) and not path.exists(),'Fresh owned namespace output required')
    path.mkdir(parents=True);return path

def fixtures(output):
    """Synthetic engineering tensors, openly labelled; not robot scene evidence."""
    out=new_directory(output);records=[]
    for i in range(2):
        native={k:np.zeros(d,np.float32) for k,d in STATE_DIMS.items()}
        native['state.end_effector_rotation_relative'][-1]=1
        native['state.base_rotation'][-1]=1
        native['state.base_position'][:]=[1.0+i*.1,-2.0,.7]
        native['state.end_effector_position_relative'][:]=[.3,.05,.5+i*.02]
        native['state.gripper_qpos'][:]=[.03,-.03]
        yy,xx=np.indices((256,256))
        for cam,key in enumerate(CAMERAS):
            native[key]=np.stack([(xx+i*17+cam*31)%256,(yy+cam*53)%256,(xx+yy+i*29)%256],axis=-1).astype(np.uint8)
        instruction=('Open the drawer.' if i==0 else 'Close the toaster oven door.')
        build_observation(native,instruction)
        path=out/f'fixture-{i}.npz';np.savez(path,**native)
        records.append(dict(path=str(path),sha256=server.sha(path),instruction=instruction))
    receipt=dict(schema='originx_gr00t_parity_fixtures_v1',synthetic=True,
        purpose='Infrastructure numeric/RNG parity only; not a physical scene or task evaluation',fixtures=records)
    server.write_new(out/'fixtures.json',receipt);return receipt

def load_fixtures(path,digest):
    path=Path(path);server.require(server.sha(path)==digest,'Fixture manifest differs')
    data=server.read(path);server.require(data['schema']=='originx_gr00t_parity_fixtures_v1' and len(data['fixtures'])==2,'Expected two pinned fixtures')
    result=[]
    for row in data['fixtures']:
        p=Path(row['path']);server.require(p.resolve().is_relative_to(server.BASE) and server.sha(p)==row['sha256'],'Fixture NPZ differs/outside namespace')
        with np.load(p,allow_pickle=False) as f:native={k:f[k] for k in f.files}
        required=set(CAMERAS)|set(STATE_DIMS)
        server.require(set(native) in (required,required|{LANGUAGE}),'Unexpected fixture fields')
        if LANGUAGE in native:
            language=np.asarray(native.pop(LANGUAGE))
            server.require(language.size==1 and language.dtype.kind in ('U','S') and
                           str(language.reshape(-1)[0])==row['instruction'],'Fixture instruction differs from observation')
        build_observation(native,row['instruction']);result.append((native,row['instruction']))
    return result

def direct_trace(policy,torch,inputs):
    bank=RngBank(torch);ambient=bank.capture();adapter=PolicyAdapter(policy);streams=[]
    try:
        for index,seed in enumerate(SEEDS):
            bank.seed(seed);rows=[]
            for q,fixture in enumerate(SEQUENCE):
                start=time.monotonic();actions=adapter.infer_native(*inputs[fixture]);state=bank.capture()
                rows.append(dict(sequence_index=q,fixture_index=fixture,action_sha256=action_hash(actions),
                    full_actions=actions.tolist(),rng=bank.hashes(state),elapsed_seconds=time.monotonic()-start))
            streams.append(dict(client_index=index,seed=seed,trace=rows))
    finally:bank.restore(ambient)
    server.require(bank.hashes(bank.capture())==bank.hashes(ambient),'Direct fixture altered ambient RNG')
    return streams

def direct(args):
    out=new_directory(args.output);server.configure_environment()
    identity,stamps=server.verify_assets();owners=server.coexist_owners(args.coexist_services,args.coexist_gr00t)
    admission=server.admit_gpu(server.gpu_snapshot(),owners)
    server.write_new(out/'owner.json',dict(server.process_identity(os.getpid()),namespace=server.BASE.name,
        policy_id=POLICY_ID,gpu_uuid=server.GPU_UUID,purpose='direct parity reference'))
    server.write_new(out/'gpu-admission.json',admission)
    import torch
    inputs=load_fixtures(args.fixtures,args.fixtures_sha256)
    policy=server.load_policy();streams=direct_trace(policy,torch,inputs)
    server.require(all((Path(n).stat().st_size,Path(n).stat().st_mtime_ns)==stamp for n,stamp in stamps.items()),'Assets changed during reference')
    receipt=dict(schema='originx_gr00t_direct_parity_v1',passed=True,identity=identity,
        model_config_runtime_sha256=hashlib.sha256(json.dumps(policy.model.config.to_dict(),sort_keys=True,default=str).encode()).hexdigest(),
        fixtures=str(args.fixtures),fixtures_sha256=args.fixtures_sha256,seeds=list(SEEDS),sequence=list(SEQUENCE),
        stream_count=6,streams=streams,ambient_rng_unchanged=True,model_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        no_environment_rollouts=True,no_confirmation_outcomes_read=True)
    server.write_new(out/'direct.json',receipt)
    return receipt

def one_stream(profile,index,inputs,reference,barrier,stop,out,hold=360,interval=120):
    client=None;record=dict(client_index=index,seed=SEEDS[index],queries=[],keepalives=[],passed=False,resets=0,reconnects=0)
    try:
        client=GR00TClient(profile,timeout=1200)
        record['connection_id']=client.hello['connection_id'];record['reset_ack']=client.reset(SEEDS[index]);record['resets']=1
        barrier.wait(120)
        for q in (0,1):
            if stop.is_set():raise RuntimeError('Peer failed')
            start=time.monotonic();actions=flatten_actions(client.get_action(build_observation(*inputs[SEQUENCE[q]])))
            rng=client.control('rng_state');validate_result(reference[q],actions,rng,q+1,SEEDS[index])
            record['queries'].append(dict(sequence_index=q,exact=True,action_sha256=action_hash(actions),rng=rng,elapsed_seconds=time.monotonic()-start))
        barrier.wait(1200);before=client.control('rng_state');begin=time.monotonic()
        while time.monotonic()-begin<hold:
            if stop.wait(min(interval,max(0,hold-(time.monotonic()-begin)))):raise RuntimeError('Peer failed during hold')
            after=client.control('rng_state');validate_keepalive(before,after)
            record['keepalives'].append(dict(elapsed_seconds=time.monotonic()-begin,rng=after,unchanged=True))
        record['hold_seconds_actual']=time.monotonic()-begin
        actions=flatten_actions(client.get_action(build_observation(*inputs[0])))
        rng=client.control('rng_state');validate_result(reference[2],actions,rng,3,SEEDS[index])
        record['queries'].append(dict(sequence_index=2,exact=True,action_sha256=action_hash(actions),rng=rng,after_hold=True))
        record['passed']=True
    except BaseException:
        record['error']=traceback.format_exc();stop.set()
        try:barrier.abort()
        except Exception:pass
    finally:
        if client:client.close()
        server.write_new(out/f'client-{index}.json',record)
    return record

def socket_probe(args):
    server.require(args.hold_seconds>=360 and 0<args.interval_seconds<=120,'Require >=360 sec same socket, <=120 sec interval')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    profile=load_profile(args.profile,args.profile_sha256)
    server.require(profile['slots']==6,'Parity requires six simultaneous slots')
    server.require(server.sha(args.reference)==args.reference_sha256,'Direct reference hash differs')
    reference=server.read(args.reference)
    server.require(reference['schema']=='originx_gr00t_direct_parity_v1' and reference['passed'] and reference['ambient_rng_unchanged'],'Unverified direct reference')
    server.require(reference['identity']['asset_source_identity_sha256']==profile['hello']['asset_source_identity_sha256'],'Direct/server asset-source identity differs')
    server.require(reference['model_config_runtime_sha256']==profile['hello']['model_config_runtime_sha256'],'Direct/server loaded configuration differs')
    server.require(reference['seeds']==list(SEEDS) and reference['sequence']==list(SEQUENCE),'Direct reference stream plan differs')
    active=server.run_read(['ss','-Htn','state','established',f"( sport = :{profile['port']} )"])
    server.require(not active,'Do not compete with an active rollout')
    out=new_directory(args.output);inputs=load_fixtures(reference['fixtures'],reference['fixtures_sha256'])
    server.write_new(out/'owner.json',dict(server.process_identity(os.getpid()),namespace=server.BASE.name,purpose='CPU socket parity',starts_models=False))
    stop=threading.Event();barrier=threading.Barrier(6)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs=[pool.submit(one_stream,profile,i,inputs,reference['streams'][i]['trace'],barrier,stop,out,
                          args.hold_seconds,args.interval_seconds) for i in range(6)]
        rows=[j.result() for j in jobs]
    passed=all(r['passed'] for r in rows) and len({r.get('connection_id') for r in rows})==6
    receipt=dict(schema='originx_gr00t_socket_parity_v1',passed=passed,profile=str(args.profile),profile_sha256=args.profile_sha256,
        reference=str(args.reference),reference_sha256=args.reference_sha256,clients=6,hold_seconds=args.hold_seconds,
        interval_seconds=args.interval_seconds,no_reconnect=True,one_reset=True,
        records=[dict(client_index=r['client_index'],passed=r['passed'],error=r.get('error')) for r in rows])
    server.write_new(out/'result.json',receipt);server.require(passed,'Parity failed; all receipts retained')
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fixtures');f.add_argument('--output',type=Path,required=True);f.add_argument('--synthetic',action='store_true',required=True)
    d=sub.add_parser('direct');d.add_argument('--output',type=Path,required=True);d.add_argument('--fixtures',type=Path,required=True);d.add_argument('--fixtures-sha256',required=True)
    d.add_argument('--coexist-services',nargs=2,action='append',default=[]);d.add_argument('--coexist-gr00t',nargs=2,action='append',default=[])
    s=sub.add_parser('socket');s.add_argument('--output',type=Path,required=True);s.add_argument('--profile',type=Path,required=True);s.add_argument('--profile-sha256',required=True)
    s.add_argument('--reference',type=Path,required=True);s.add_argument('--reference-sha256',required=True)
    s.add_argument('--hold-seconds',type=float,default=360);s.add_argument('--interval-seconds',type=float,default=120)
    a=p.parse_args()
    if a.command=='fixtures':fixtures(a.output)
    elif a.command=='direct':direct(a)
    else:socket_probe(a)

if __name__=='__main__':main()
