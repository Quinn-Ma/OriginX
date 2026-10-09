"""Explicit GPU6-only GR00T server. Importing this module does not load a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from .adapter import PROTOCOL
from .core import POLICY_ID, SerialExecutor
from . import wire

ROOT=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
BASE=ROOT/'originx_gr00t_confirmation_20261009'
GPU_UUID='GPU-35278907-aae0-4ccc-e0fc-c25cb3a8bd17'
SOURCE_COMMIT='9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10'
MODEL_REVISION='c484448aba1a9b60a04c9b0ca117241518ea69f3'
COEXIST_READY=ROOT/'results/originx-confirmatory-services-20261009-v2/services-ready.json'
PINS={
 'config.json':(1706,'6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526'),
 'model.safetensors.index.json':(104606,'bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746'),
 'experiment_cfg/metadata.json':(14140,'8be0fc606c9220356bad497feefc4ae4daaa05670acccc31caecaa7ae5590b69'),
 'model-00001-of-00002.safetensors':(4999367032,'08f1891947973e2e5ec2422201cd90261806f77f2634f9ec0477c27aa5a4fe42'),
 'model-00002-of-00002.safetensors':(2586705312,'deb9c9cf40cd8983a7779af85341f6344db15f65bf3307073a0e3c3085450435'),
}

def require(condition,message):
    if not condition:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def process_identity(pid):
    p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(')',1)[1].split()
    if f[0] in ('Z','X'):raise ProcessLookupError('Owner exited')
    return dict(pid=int(pid),process_start_ticks=int(f[19]),
        command=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),cwd=str((p/'cwd').resolve(strict=True)))
def owner_matches(owner,actual):
    return all(owner.get(k)==actual.get(k) for k in ('pid','process_start_ticks','command','cwd'))
def run_read(command):
    return subprocess.check_output(command,text=True,timeout=30).strip()
def gpu_snapshot():
    r=run_read(['nvidia-smi','-i','6','--query-gpu=uuid,memory.free,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits']).split(',')
    result=dict(gpu_uuid=r[0].strip(),free_mib=int(r[1]),ecc=int(r[2]),occupants=[])
    for line in run_read(['nvidia-smi','-i','6','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits']).splitlines():
        if line.strip():
            pid,gpu=map(str.strip,line.split(','));require(gpu==GPU_UUID,'Wrong device occupant');result['occupants'].append(int(pid))
    return result

def validate_main_owner(owner,actual,manifest):
    require(owner.get('namespace')=='originx_confirmatory_20261009' and owner.get('gpu_uuid')==GPU_UUID and owner.get('gpu_index')==6,'Wrong coexist namespace/GPU')
    require(owner.get('policy_id') in ('B','base') and owner.get('cwd')==str(ROOT),'Wrong coexist policy/cwd')
    command=owner.get('command',[])
    module='continuous_eval2000_v3.server' if owner['policy_id']=='B' else 'originx_confirmatory_20261009.services'
    require(command[:4]==[str(ROOT/'envs/training/bin/python'),'-u','-m',module],'Wrong coexist command')
    require('--server-manifest' in command and command[command.index('--server-manifest')+1]==manifest,'Coexist manifest not bound in argv')
    require(owner_matches(owner,actual),'Coexist PID reused or argv/cwd changed')

def validate_gr00t_owner(owner,actual):
    require(owner.get('namespace')==BASE.name and owner.get('policy_id')==POLICY_ID and owner.get('gpu_uuid')==GPU_UUID,'Wrong GR00T owner')
    require(owner.get('cwd')==str(BASE) and owner_matches(owner,actual),'GR00T owner identity changed')
    command=owner['command']
    require(command[0]==str(BASE/'env/bin/python') and '-m' in command and
            command[command.index('-m')+1] in (BASE.name+'.server',BASE.name+'.parity_probe'),'Wrong GR00T executable/module')

def coexist_owners(main_receipts,gr00t_receipts,identity_reader=process_identity):
    owners=[]
    for path,digest in main_receipts:
        path=Path(path);require(path==COEXIST_READY and not path.is_symlink(),'Only explicitly allowed main service manifest')
        require(sha(path)==digest,'Coexist ready manifest hash mismatch')
        for row in read(path)['models']:
            owner=row['owner'];manifest=row['server_manifest']
            require(Path(manifest).is_relative_to(COEXIST_READY.parent/'services'),'Coexist service path outside scope')
            require(sha(manifest)==row['server_manifest_sha256'],'Coexist service manifest changed')
            validate_main_owner(owner,identity_reader(owner['pid']),manifest);owners.append(owner)
    for path,digest in gr00t_receipts:
        path=Path(path);require(path.is_relative_to(BASE) and sha(path)==digest,'GR00T coexist profile not pinned')
        row=read(path);owner=row['owner'];validate_gr00t_owner(owner,identity_reader(owner['pid']));owners.append(owner)
    require(len({o['pid'] for o in owners})==len(owners),'Duplicate coexist owner')
    return owners

def admit_gpu(snapshot,owners,identity_reader=process_identity,min_free_mib=16384):
    require(snapshot['gpu_uuid']==GPU_UUID and snapshot['ecc']==0,'GPU6 UUID/ECC mismatch')
    by_pid={o['pid']:o for o in owners}
    for pid in snapshot['occupants']:
        require(pid in by_pid,f'Unknown GPU6 occupant {pid}; no model load permitted')
        require(owner_matches(by_pid[pid],identity_reader(pid)),'Live coexist owner changed')
    require(snapshot['free_mib']>=min_free_mib,'GPU6 free-memory admission failed')
    return dict(snapshot,minimum_free_mib=min_free_mib,verified_coexist_pids=sorted(by_pid))

def verify_assets():
    checkpoint=BASE/'checkpoint-120000';source=BASE/'source'
    require(checkpoint.resolve()==checkpoint and source.resolve()==source,'Symlinked assets not allowed')
    stamps={}
    for name,(size,digest) in PINS.items():
        p=checkpoint/name;require(p.stat().st_size==size and sha(p)==digest,'Wrong complete checkpoint asset '+name)
        stamps[str(p)]=(p.stat().st_size,p.stat().st_mtime_ns)
    require(run_read(['git','-C',str(source),'rev-parse','HEAD'])==SOURCE_COMMIT,'Official source revision changed')
    require(not run_read(['git','-C',str(source),'diff','HEAD','--','gr00t','pyproject.toml']),'Official tracked policy source modified')
    names=run_read(['git','-C',str(source),'ls-files','gr00t','pyproject.toml','LICENSE']).splitlines()
    hashes={n:sha(source/n) for n in names}
    for n in names:stamps[str(source/n)]=((source/n).stat().st_size,(source/n).stat().st_mtime_ns)
    service_files=('__init__.py','adapter.py','wire.py','core.py','server.py','client.py','parity_probe.py')
    wrapper={}
    for name in service_files:
        p=Path(__file__).parent/name
        wrapper[name]=sha(p);stamps[str(p)]=(p.stat().st_size,p.stat().st_mtime_ns)
    identity=dict(protocol=PROTOCOL,policy_id=POLICY_ID,policy_family='GR00T_N1_5',model_path=str(checkpoint),
        model_revision=MODEL_REVISION,model_config_sha256=PINS['config.json'][1],
        model_assets_sha256={n:v[1] for n,v in PINS.items()},official_source_commit=SOURCE_COMMIT,
        official_source_sha256=hashes,service_sources_sha256=wrapper,adapters_loaded=False,
        action_dim=12,model_action_dim=32,action_horizon=16,replan_steps=16,denoising_steps=4,
        observation_frames=1,camera_count=3,crop_ratio=.95,resize=[224,224],compute_dtype='bfloat16',
        inference_optimization='none',gpu_uuid=GPU_UUID)
    # Compact identity digest binds the full source manifest without repeating 100 hashes per RPC.
    identity['asset_source_identity_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return identity,stamps

def load_policy():
    """Only called explicitly from the admitted model execution thread/direct probe."""
    import torch,flash_attn,transformers,gr00t
    require(Path(gr00t.__file__).resolve()==BASE/'source/gr00t/__init__.py','Official GR00T imported from wrong source')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID,'Exactly GPU6 UUID must be visible')
    require(torch.__version__=='2.5.1+cu124' and flash_attn.__version__=='2.7.1.post4' and transformers.__version__=='4.51.3','Runtime versions differ')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one admitted CUDA device required')
    from gr00t.model.policy import Gr00tPolicy
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    cfg=DATA_CONFIG_MAP['panda_omron']
    policy=Gr00tPolicy(model_path=str(BASE/'checkpoint-120000'),modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),embodiment_tag='new_embodiment',denoising_steps=4,device='cuda:0')
    require(not policy.model.training and policy.model.action_head.num_inference_timesteps==4,'Official eval/denoising configuration changed')
    require(policy.model.config.action_horizon==16,'Action horizon changed')
    return policy

def serve(executor,listener,max_clients,stop,ready=None,draining_path=None):
    slots=threading.BoundedSemaphore(max_clients);connections=set();lock=threading.Lock();threads=[]
    def worker(conn):
        cid=secrets.token_hex(16)
        try:
            conn.settimeout(300)
            while not stop.is_set():wire.send(conn,executor.submit(cid,wire.receive(conn)).result(timeout=1200))
        except (EOFError,ConnectionError,socket.timeout):pass
        except BaseException as e:print(json.dumps(dict(connection_id=cid,error=repr(e),closed_without_retry=True)),flush=True)
        finally:
            try:executor.submit(cid,close=True).result(60)
            except Exception:pass
            with lock:connections.discard(conn)
            conn.close();slots.release()
    listener.listen(max_clients);listener.settimeout(1)
    if ready:ready()
    try:
        while not stop.is_set():
            try:conn,_=listener.accept()
            except socket.timeout:continue
            if draining_path is not None and Path(draining_path).exists():conn.close();continue
            if not slots.acquire(False):conn.close();continue
            with lock:connections.add(conn)
            t=threading.Thread(target=worker,args=(conn,),daemon=True);threads.append(t);t.start()
    finally:
        listener.close()
        with lock:
            for conn in connections:
                try:conn.shutdown(socket.SHUT_RDWR)
                except OSError:pass
        for t in threads:t.join(120)

def cleanup_profile(path,digest):
    """Explicit cleanup utility; no model load, no broad PID patterns or SIGKILL."""
    path=Path(path)
    require(path.resolve().is_relative_to(BASE/'services') and not path.is_symlink() and sha(path)==digest,'Cleanup profile not pinned')
    profile=read(path);owner=profile['owner'];validate_gr00t_owner(owner,process_identity(owner['pid']))
    drain=path.parent/'draining.json'
    if not drain.exists():write_new(drain,dict(owner=owner,profile_sha256=digest))
    else:require(read(drain)['owner']==owner,'Drain owner differs')
    for _ in range(2):
        connections=run_read(['ss','-Htn',f"( sport = :{profile['port']} )"])
        require(not connections,'Active or closing service connections; cleanup preserves model')
        validate_gr00t_owner(owner,process_identity(owner['pid']));time.sleep(.5)
    require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'PID-fd cleanup requires Linux support')
    fd=os.pidfd_open(owner['pid'])
    try:
        validate_gr00t_owner(owner,process_identity(owner['pid']))
        signal.pidfd_send_signal(fd,signal.SIGTERM)
    finally:os.close(fd)
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        try:actual=process_identity(owner['pid'])
        except (OSError,ProcessLookupError):
            write_new(path.parent/'cleanup.json',dict(owner=owner,terminated=True,signal='SIGTERM',pidfd=True));return
        require(owner_matches(owner,actual),'PID identity changed during cleanup');time.sleep(.5)
    raise RuntimeError('Exact owned process did not exit; no SIGKILL attempted')

def runtime_identity(torch):
    # TorchVersion is a str subclass; normalize metadata before strict wire encoding.
    return dict(torch_version=str(torch.__version__),cuda_visible_device_count=1)

def configure_environment():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID,'Set CUDA_VISIBLE_DEVICES to GPU6 UUID explicitly')
    os.environ.update(USE_TF='0',USE_FLAX='0',NO_ALBUMENTATIONS_UPDATE='1',TOKENIZERS_PARALLELISM='false',
                      HF_HOME=str(BASE/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,required=True);p.add_argument('--max-clients',type=int,default=6)
    p.add_argument('--coexist-services',nargs=2,action='append',default=[],metavar=('PATH','SHA256'))
    p.add_argument('--coexist-gr00t',nargs=2,action='append',default=[],metavar=('PATH','SHA256'))
    a=p.parse_args();require(Path.cwd()==BASE,'Launch only from fixed namespace cwd')
    require(27900<=a.port<=27905 and 1<=a.max_clients<=6,'Dedicated port/client bound required')
    output=BASE/f'services/port-{a.port}';require(not output.exists(),'Existing owner/output; never replace a service')
    output.mkdir(parents=True)
    configure_environment()
    identity,stamps=verify_assets()
    owners=coexist_owners(a.coexist_services,a.coexist_gr00t)
    admission=admit_gpu(gpu_snapshot(),owners)
    write_new(output/'gpu-admission.json',admission)
    owner=dict(process_identity(os.getpid()),namespace=BASE.name,policy_id=POLICY_ID,gpu_uuid=GPU_UUID,port=a.port)
    write_new(output/'owner.json',owner)
    listener=socket.socket();listener.bind(('127.0.0.1',a.port))
    import torch
    identity.update(server_instance=secrets.token_hex(16),**runtime_identity(torch))
    compact={k:v for k,v in identity.items() if k not in ('official_source_sha256','service_sources_sha256')}
    wire.encode(compact)  # Fail metadata encoding before constructing a GPU model.
    executor=SerialExecutor(load_policy,torch,compact,capacity=2*a.max_clients)
    require(all((Path(n).stat().st_size,Path(n).stat().st_mtime_ns)==stamp for n,stamp in stamps.items()),'Source/assets changed while loading')
    stop=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
    profile=dict(schema='originx_gr00t_service_profile_v1',ready=True,owner=owner,port=a.port,slots=a.max_clients,
        hello=executor.identity,identity_manifest=identity,policy_id=POLICY_ID,gpu_uuid=GPU_UUID,
        admission=admission,server_manifest=str(output/'server.json'),model_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    try:serve(executor,listener,a.max_clients,stop,ready=lambda:write_new(output/'server.json',profile),draining_path=output/'draining.json')
    finally:
        executor.close();write_new(output/'terminated.json',dict(owner=owner,model_thread_drained=True))

if __name__=='__main__':main()
