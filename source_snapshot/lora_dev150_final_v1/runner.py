#!/usr/bin/env python3
"""Fixed450 base/L/A runner: private CPU OSMesa, sealed outcomes, persistent per-assignment claims."""
import argparse
import os
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
MANIFEST_SHA='8643051dab874f68ca8763242265ad3016788c4da62af3f57060004fe5144fe1'
ENTRY_SHA='18d70fade4c990210dc5b9c34285c8030eac142d9c332066c73335e1c0aaa28e'
MODEL_CODE_SHA='d2f90e88391ae06ce7beca27416544ad2cb623a2963b6c61ba9ddafa2fd42ec9'
ARMS=('base','L','A')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)
def save(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def design(manifest):
    require(sha(manifest)==MANIFEST_SHA,'Frozen450 schedule changed')
    m=read(manifest);jobs=m['jobs'];tasks={t['task']:t['horizon'] for t in m['tasks']}
    require(m['schema']=='lora_retention_dev450_v1' and len(tasks)==30,'Expected fixed30 tasks')
    require(len(jobs)==450 and Counter(j['arm'] for j in jobs)==dict.fromkeys(ARMS,150),'Exactly150 per arm required')
    require(m['dispatch_order']==list(range(450)),'Dispatch order changed')
    require(len({j['id'] for j in jobs})==450,'Duplicate assignment ID')
    require(all(j['assignment_index']==i and j['split']=='pretrain' and j['horizon']==tasks[j['task']]
        and j['env_seed']==j['policy_seed']==2040200000+i for i,j in enumerate(jobs)),'Assignment, seed or horizon changed')
    require(Counter((j['arm'],j['task']) for j in jobs)=={(a,t):5 for a in ARMS for t in tasks},'Five independent episodes/task/arm required')
    return m


def audit_seed_binding(item):
    require(isinstance(item,dict) and item.get('path') and item.get('sha256') and sha(item['path'])==item['sha256'],'Missing/changed seed collision audit')
    audit=read(item['path'])
    require(audit.get('passed') is True and audit.get('manifest_sha256')==MANIFEST_SHA
        and audit.get('environment_inference_seed_range')==[2040200000,2040200449]
        and audit.get('collisions')==[] and bool(audit.get('scope')),'Seed collision audit did not cover fixed450')

def chosen_indices(m,arms):
    require(arms and len(set(arms))==len(arms) and set(arms)<=set(ARMS),'Unknown/repeated arm filter')
    return [j['assignment_index'] for j in m['jobs'] if j['arm'] in arms]

def endpoint_for(job,b,m):
    ids=b['arm_endpoints'][job['arm']]
    ordinal=sum(j['arm']==job['arm'] for j in m['jobs'][:job['assignment_index']])
    return ids[ordinal%len(ids)]


def environment(binding):
    root=Path(binding['root']);lib=Path(binding['osmesa_library'])
    env=dict(os.environ,MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',CUDA_VISIBLE_DEVICES='',
        LIBGL_ALWAYS_SOFTWARE='true',GALLIUM_DRIVER='llvmpipe',MESA_DEBUG='1',LP_NUM_THREADS='4',
        LD_LIBRARY_PATH=str(lib.parent),PYTHONHASHSEED='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        NUMBA_CACHE_DIR=str(root/'cache/numba'),PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(root)+':'+str(root/'remote_processor_candidate_v1/deps'))
    for key in ('MUJOCO_EGL_DEVICE_ID','DISPLAY','XAUTHORITY','WAYLAND_DISPLAY','__GLX_VENDOR_LIBRARY_NAME'):
        env.pop(key,None)
    for key in ('MESA_NO_ERROR','MESA_GL_VERSION_OVERRIDE','MESA_GLSL_VERSION_OVERRIDE','MESA_EXTENSION_OVERRIDE','LP_NO_RAST','LP_PERF','GALLIUM_HUD'):
        require(not env.get(key),'Renderer-altering inherited variable '+key)
    return env

def bindings(path,arms,live=True):
    b=read(path)
    require(b.get('schema')=='lora_dev150_final_bindings_v1','Unknown binding schema')
    require(b.get('approved_for_evaluation') is True,'Actual root-reviewed evaluation binding required')
    require(b.get('runner_pins_sha256')==sha(HERE/'runner-source-pins.json'),'Runner inventory changed')
    require(set(b['arm_endpoints'])==set(ARMS),'Three arm endpoint mappings required')
    audit_seed_binding(b.get('seed_audit'))
    require(Path(b['cohort_root']).is_absolute(),'Explicit absolute cohort root required')
    selected=[]
    for arm in arms:
        ids=b['arm_endpoints'][arm]
        require(1<=len(ids)<=8 and len(set(ids))==len(ids),'Bind one to eight actual endpoints per selected arm')
        for key in ids:
            require(key in b['models'] and b['models'][key]['arm']==arm,'Endpoint arm differs')
            selected.append(key)
    require(len(set(selected))==len(selected),'Same endpoint assigned to multiple arms')
    for file,digest in read(HERE/'runner-source-pins.json').items():require(sha(HERE/file)==digest,'Runner source changed: '+file)
    for file,digest in b['source_sha256'].items():require(sha(file)==digest,'External source changed: '+file)
    require(sha(Path(b['repo'])/'eval_robocasa365/entry.py')==ENTRY_SHA,'Official entry changed')
    require({name:importlib.metadata.version(name) for name in b['packages']}==b['packages'],'Simulator package versions changed')
    sys.path[:0]=[str(HERE),str(Path(b['root'])/'remote_processor_candidate_v1/deps'),b['root']]
    from local_eval.remote_client import validate_endpoint
    import inspect
    actual=Path(inspect.getfile(validate_endpoint)).resolve()
    require(str(actual) in b['source_sha256'] and sha(actual)==b['source_sha256'][str(actual)],'Loaded RPC client is not pinned')
    ports=[]
    for key in selected:
        model=b['models'][key];arm=model['arm']
        for field in ('checkpoint_path','checkpoint_sha256','server_manifest','server_manifest_sha256','parity_report','parity_sha256','available_client_slots'):
            require(model.get(field) not in (None,''),'Unbound endpoint '+key+': '+field)
        require(type(model['available_client_slots']) is int and 1<=model['available_client_slots']<=8,'Invalid client reservation')
        require(sha(model['server_manifest'])==model['server_manifest_sha256'] and sha(model['parity_report'])==model['parity_sha256'],'Server/parity binding changed')
        if arm=='base':
            server=validate_endpoint(model['server_manifest'],model['parity_report'],model['checkpoint_path'])
            require(not server['hello'].get('lora_serving_identity'),'Base arm cannot use an adapter')
        else:
            from lora_dev150_final_v1.client import validate_adapter_endpoint
            require(model.get('endpoint_binding') and model.get('endpoint_binding_sha256'),'Trained endpoint1613/80000 binding required')
            server=validate_adapter_endpoint(model['server_manifest'],model['parity_report'],model['checkpoint_path'],
                binding_path=model['endpoint_binding'],binding_sha256=model['endpoint_binding_sha256'])
            require(server['hello']['lora_serving_identity']['arm']==arm and
                server['hello']['lora_serving_identity']['endpoint_updates']==1613 and
                server['hello']['lora_serving_identity']['endpoint_samples']==80000,'Wrong arm or nonterminal adapter')
        hello=server['hello'];owner=server['servers'][0];command=owner['command']
        require(hello['model_path']==str(Path(model['checkpoint_path']).resolve()) and canonical(hello['model_assets_sha256'])==model['checkpoint_sha256'],'Model asset/path identity differs')
        require(hello['model_assets_sha256']['modeling_mibot.py']==MODEL_CODE_SHA,'Changed official five-step forward')
        require(command.count('--max-clients')==1 and model['available_client_slots']<=int(command[command.index('--max-clients')+1]),'Capacity exceeds original server limit')
        if live:
            stat=Path('/proc')/str(owner['pid']);fields=(stat/'stat').read_text().rsplit(')',1)[1].split()
            require(fields[0]!='Z' and int(fields[19])==owner['process_start_ticks'] and
                (stat/'cmdline').read_bytes().decode().rstrip('\0').split('\0')==command and
                str((stat/'cwd').resolve())==str(Path(server['repo']).resolve()),'Owned model process identity changed')
        if arm=='base':
            require(model['checkpoint_sha256']=='21f3a9ddf040e8afd1e56268ee76263df61692bd1c568f76ae9a7f905e065816','Wrong released base')
        ports.append(owner['port'])
    require(len(set(ports))==len(ports),'Endpoints share a port')
    return b

def empty_connections(b,arms):
    # A read-only start check; every inference server keeps its original max8 cap.
    for arm in arms:
        for key in b['arm_endpoints'][arm]:
            port=read(b['models'][key]['server_manifest'])['servers'][0]['port']
            result=subprocess.run(['ss','-Htn','state','established',f'( sport = :{port} )'],capture_output=True,text=True,check=True)
            require(not result.stdout.strip(),'Endpoint already has active connections: '+str(port))

def dispatch_allowed(job,active,b,m,limit):
    key=endpoint_for(job,b,m)
    return len(active)<limit and sum(endpoint_for(j,b,m)==key for j in active)<b['models'][key]['available_client_slots']


def guard_complete(report):
    # Match the unchanged guard's one explicitly allowed material-draw error.
    return (report['status']=='closed' and report['failure'] is None
        and set(report['gl_error_counts'])<={'after_draw:1281'}
        and report['raw_frames_checked']>0 and report['sentinel_checks']>0)

def public_record(row,path):
    allowed=('id','assignment_index','arm','task','env_seed','policy_seed','status','bindings_sha256','manifest_sha256','endpoint_key','wall_seconds')
    return dict({k:row[k] for k in allowed if k in row},sealed_result_path=str(path),sealed_result_sha256=sha(path))

def summary(jobs,records):
    expected={j['id']:j for j in jobs};seen=set()
    for row in records:
        require(row['id'] in expected and row['id'] not in seen,'Duplicate/unplanned record');seen.add(row['id']);j=expected[row['id']]
        require(row['assignment_index']==j['assignment_index'] and row['arm']==j['arm'],'Record assignment mismatch')
        require(row['status'] in ('completed','infrastructure_unknown'),'Unknown result status')
    complete=len(records)==len(jobs) and all(r['status']=='completed' for r in records)
    return dict(complete=complete,planned=len(jobs),attempted=len(records),completed=sum(r['status']=='completed' for r in records),
        infrastructure_unknown=sum(r['status']=='infrastructure_unknown' for r in records),not_attempted=len(jobs)-len(records),
        per_arm={arm:dict(planned=sum(j['arm']==arm for j in jobs),completed=sum(r['arm']==arm and r['status']=='completed' for r in records)) for arm in ARMS},
        score_emitted=False,outcomes_sealed=True,no_replacement=True)


def episode(args, *, engineering_job=None):
    require(sha(args.bindings)==args.bindings_sha,'Launch bindings changed')
    if engineering_job is None:
        m=design(args.manifest);require(args.index is not None and 0<=args.index<450,'Assignment index outside manifest')
    else:
        require(args.index==0 and read(args.manifest).get('schema')=='lora_dev150_final_engineering_gate_v1','Explicit engineering job only')
        require(engineering_job in read(args.manifest)['jobs'],'Engineering job absent from pinned gate manifest')
        require(engineering_job['env_seed'] in (2040200900,2040200901,2040200902) and engineering_job['task']=='OpenOven' and engineering_job['horizon']==300,'Fixed independent engineering task/seed required')
        m={'jobs':[engineering_job]}

    job=m['jobs'][args.index];b=bindings(args.bindings,[job['arm']])
    claim=read(args.output/'claims'/(job['id']+'.json'))
    require(claim['job']==job and claim['bindings_sha256']==args.bindings_sha,'Episode lacks its unique assignment claim')
    expected_env=environment(b)
    for key in ('MUJOCO_GL','PYOPENGL_PLATFORM','CUDA_VISIBLE_DEVICES','LP_NUM_THREADS','LD_LIBRARY_PATH','PYTHONHASHSEED','NUMBA_CACHE_DIR'):
        require(os.environ.get(key)==expected_env[key],'Child environment differs: '+key)
    import dev_randomized_broad_alpha as legacy
    import dev_repeat_audit as audit
    from local_eval.remote_client import RemoteEvalClient,check_hello
    import readback_guard as guard_module
    import context_lifetime as context_module
    import candidate as render_module
    pins=read(HERE/'runner-source-pins.json')
    for mod in (legacy,audit,legacy.common,audit.reference_tools,guard_module,context_module,render_module):
        source=Path(mod.__file__).resolve()
        require(source.parent==HERE and sha(source)==pins[source.name],'Imported helper differs from frozen local copy')
    install_guard=guard_module.install_guard;install=context_module.install
    DecimatedGym=render_module.DecimatedGym;QueryCheckedClient=render_module.QueryCheckedClient
    import numpy as np
    import torch
    import gymnasium as gym
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    require(Path(robocasa.__file__).resolve().parent==Path(b['package_root']),'Wrong RoboCasa package')
    require(get_task_horizon(job['task'])==job['horizon'],'Official task horizon differs')
    entry=load(Path(b['repo'])/'eval_robocasa365/entry.py','_anchor_dev_official')
    out=args.output/'.sealed'/'episodes'/job['id'];out.mkdir(parents=True,exist_ok=False,mode=0o700)
    new_json(out/'episode-owner.json',identity())
    events=[];initial={};observers=[];client=guard=None;record=None
    def event(name,**fields):
        events.append(dict(event=name,sequence=len(events),monotonic_seconds=time.monotonic(),**fields))
        with (out/'events.jsonl').open('a') as stream:stream.write(json.dumps(events[-1])+'\n')
    def select_model():
        # First use of this episode's arm for model selection is after capture.
        arm=m['jobs'][args.index]['arm'];model=dict(b['models'][endpoint_for(job,b,m)],name=arm)
        model['server']=read(model['server_manifest']);return model
    def make_policy(model):
        s=model['server'];client_class=RemoteEvalClient
        if job['arm']!='base':
            from lora_dev150_final_v1.client import AdapterEvalClient
            client_class=AdapterEvalClient
        policy=client_class(b['repo'],model['checkpoint_path'],s['servers'][0]['port'],s['hello'],telemetry_path=out/'rpc.jsonl')
        policy.hello=policy.wire.hello
        return policy
    def check_identity(h,model):
        check_hello(h,model['server']['hello'])
        if job['arm']!='base':
            from lora_dev150_final_v1.server import check_lora_hello
            check_lora_hello(h,model['server']['hello'])
    lazy=legacy.LazyClient(select_model,make_policy,check_identity,job['policy_seed'],event)
    def capture(env,observation):
        guard.check();core=env.unwrapped.env
        initial.update(audit.capture_initial(out,observation,entry.observation_to_state(observation),core.sim))
        initial.update(layout_id=int(core.layout_id),style_id=int(core.style_id),xml_sha256=sha(out/'initial.xml'))
        save(out/'initial_scene.json',initial)
    class Observer(legacy.ResetObserver):
        steps=0
        def __getattr__(self,name):return getattr(self.env,name)
        def step(self,action):
            result=self.env.step(action);guard.check();self.steps+=1;return result
    class ObservedGym:
        @staticmethod
        def make(*a,**kw):
            require(not observers,'Only one natural-reset simulator per assignment')
            obs=Observer(gym.make(*a,**kw),lazy,capture,event);observers.append(obs);return obs
    cfg=entry.parse_args(['--model-path','unassigned-until-reset','--split','pretrain','--task-set','pretrain300',
        '--task-name',job['task'],'--num-trials','1','--seed',str(job['env_seed']),'--replan-steps','16',
        '--obs-history','4','--obs-interval','2','--crop-ratio','0.95'])
    entry.validate_args(cfg);reduced=DecimatedGym(ObservedGym,cfg);checked=QueryCheckedClient(lazy,reduced)
    started=time.monotonic()
    try:
        fix=install();guard=install_guard(out/'readback-guard',library_path=b['osmesa_library'])
        random.seed(job['env_seed']);np.random.seed(job['env_seed'])
        stats=entry.evaluate_task(job['task'],0,cfg,checked,reduced,get_task_horizon,convert_action,out/'official',episode_indices=[0],show_progress=False,write_task_stats=False)
        from robosuite.utils import binding_utils
        require(not getattr(binding_utils,'_osmesa_lifetime_failure',None),'Renderer context close failure')
        guard.check();after=lazy.policy.wire.control('rng_state');ep=stats['episodes'][0]
        require(tuple(e['event'] for e in events)==legacy.EVENT_ORDER and len(observers)==1 and observers[0].reset_calls==1,'Activation order/reset changed')
        require(len(stats['episodes'])==1 and ep['seed']==job['env_seed'] and ep['episode']==ep['global_episode_index']==0 and
            type(ep['success']) is bool and 1<=ep['steps']<=job['horizon'] and ep['steps']==observers[0].steps and stats['horizon']==job['horizon'] and
            (ep['success'] or ep['steps']==job['horizon']),'Official episode protocol changed or incomplete failed episode')
        require(after['requests_since_reset']==lazy.infer_calls==(ep['steps']+15)//16 and not torch.cuda.is_initialized(),'Policy query count or CPU-only execution changed')
        record=dict(status='completed',success=ep['success'],stats=stats,rng_ack=lazy.rng_ack,rng_after=after,
            server_identity=lazy.policy.hello,local_rng=lazy.local_rng,render_counts=dict(reduced.current.counts),context_lifetime=fix,guard_passed=True)
    except BaseException:
        record=dict(status='infrastructure_unknown',success=None,traceback=traceback.format_exc())
    finally:
        for name,obj in [('client',lazy),('guard',guard)]:
            if obj is not None:
                try:obj.close()
                except BaseException:
                    record=dict(status='infrastructure_unknown',success=None,cleanup=name,traceback=traceback.format_exc())
        if record['status']=='completed':
            try:
                report=read(out/'readback-guard/summary.json')
                require(guard_complete(report),'Guard close not complete/healthy')
                record['guard_close_proof']=dict(path=str(out/'readback-guard/summary.json'),
                    sha256=sha(out/'readback-guard/summary.json'),closed=True)
            except BaseException:
                record=dict(status='infrastructure_unknown',success=None,cleanup='guard_close_proof',traceback=traceback.format_exc())
        record.update(id=job['id'],assignment_index=job['assignment_index'],arm=job['arm'],task=job['task'],env_seed=job['env_seed'],policy_seed=job['policy_seed'],
            bindings_sha256=args.bindings_sha,manifest_sha256=sha(args.manifest),endpoint_key=endpoint_for(job,b,m),initial_fingerprints=initial,events=events,wall_seconds=time.monotonic()-started)
        save(out/'result.json',record)
    return 0 if record['status']=='completed' else 1

def identity():
    return dict(pid=os.getpid(),process_start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        command=Path('/proc/self/cmdline').read_bytes().decode().rstrip('\0').split('\0'),cwd=os.getcwd())

def new_json(path,value):
    path=Path(path)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def process_active(owner):
    path=Path('/proc')/str(owner['pid'])/'stat'
    try:fields=path.read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    # A matching PID/start is conservatively alive even if its argv changed.
    return fields[0]!='Z' and int(fields[19])==owner['process_start_ticks']

def reserve_arms(cohort,arms,owner,resume=False):
    import fcntl
    cohort.mkdir(parents=True,exist_ok=True);handles=[]
    try:
        for arm in sorted(arms):
            handle=(cohort/(arm+'.running.lock')).open('a+')
            handles.append(handle);fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for arm in arms:
            path=cohort/(arm+'.lease.json')
            if resume:
                previous=read(path)
                require(all(previous.get(k)==owner.get(k) for k in ('output','bindings_sha256','manifest_sha256','source_pins_sha256','arms','dispatch_order')),'Resume must reuse exactly the prior cohort/assignment authority')
            else:require(not path.exists(),'Persistent arm lease exists; use explicit same-output resume')
        if not resume:
            for arm in arms:new_json(cohort/(arm+'.lease.json'),dict(owner,arm=arm))
        return handles
    except BaseException:
        for handle in handles:handle.close()
        raise

def admit_public(row,job,path,binding_sha):
    require(all(row.get(k)==job[k] for k in ('id','assignment_index','arm','task','env_seed','policy_seed')),'Child assignment/seed mismatch')
    require(row.get('bindings_sha256')==binding_sha and row.get('manifest_sha256')==MANIFEST_SHA,'Child authority mismatch')
    require(row.get('status') in ('completed','infrastructure_unknown'),'Unknown child status')
    return public_record(row,path)

def resume_records(output,jobs,binding_sha):
    records=[];pending=[]
    known={j['id']:j for j in jobs}
    require(all(p.stem in known for p in (output/'claims').glob('*.json')),'Unplanned claim')
    previous=read(output/'records.json') if (output/'records.json').exists() else []
    require(len({r['id'] for r in previous})==len(previous),'Duplicate prior public record')
    old={r['id']:r for r in previous}
    require(set(old)<=set(known),'Unplanned prior record')
    for job in jobs:
        claim=output/'claims'/(job['id']+'.json')
        if not claim.exists():
            require(job['id'] not in old,'Result without claim');pending.append(job);continue
        data=read(claim);require(data['job']==job and data['bindings_sha256']==binding_sha,'Changed claim')
        proc=output/'processes'/(job['id']+'.json')
        if not proc.exists():proc=output/'.sealed/episodes'/job['id']/'episode-owner.json'
        require(proc.exists(),'Claim lacks process evidence; preserve unknown, no automatic retry')
        require(not process_active(read(proc)),'Claimed child still alive; wait without starting duplicate')
        result=output/'.sealed/episodes'/job['id']/'result.json'
        require(result.exists(),'Claim has no terminal result; preserve unknown, no automatic retry')
        row=admit_public(read(result),job,result,binding_sha)
        if job['id'] in old:require(old[job['id']]==row,'Previously recorded result bytes changed')
        records.append(row)
    require(all(r['status']=='completed' for r in records),'Unresolved infrastructure_unknown blocks automatic resume')
    return records,pending

def execute(args,b,m):
    require(1<=args.parallel<=48,'Parallel bound1..48')
    indices=chosen_indices(m,args.arms);jobs=[m['jobs'][i] for i in indices]
    require(args.parallel<=sum(b['models'][k]['available_client_slots'] for a in args.arms for k in b['arm_endpoints'][a]),'Parallel request exceeds reserved endpoint capacity')
    cohort=Path(b['cohort_root']).resolve();output=args.output.resolve()
    require(output.parent==cohort,'Output must be direct child of pinned cohort root')
    require(output.exists() if args.resume else not output.exists(),'Explicit resume of old output, otherwise new output required')
    binding_sha=sha(args.bindings);owner=dict(identity(),output=str(output),bindings_path=str(args.bindings.resolve()),bindings_sha256=binding_sha,
        manifest_sha256=MANIFEST_SHA,source_pins_sha256=sha(HERE/'runner-source-pins.json'),arms=args.arms,dispatch_order=indices)
    locks=reserve_arms(cohort,args.arms,owner,args.resume)
    try:
        if args.resume:
            original=read(output/'launch.json')
            require(all(original[k]==owner[k] for k in ('bindings_sha256','manifest_sha256','source_pins_sha256','arms','dispatch_order')),'Resume authority changed')
            records,pending=resume_records(output,jobs,binding_sha)
            stamp=str(time.time_ns());history=output/'history';history.mkdir(exist_ok=True)
            for name in ('records.json','summary.json','completion.json'):
                if (output/name).exists():new=history/(stamp+'-'+name);new.write_bytes((output/name).read_bytes())
        else:
            output.mkdir(exist_ok=False,mode=0o700)
            for sub in ('.sealed/logs','claims','processes','sessions'):(output/sub).mkdir(parents=True,mode=0o700)
            new_json(output/'launch.json',dict(owner,models=b['models'],source_sha256=read(HERE/'runner-source-pins.json')))
            records=[];pending=jobs[:]
        empty_connections(b,args.arms)
        new_json(output/'sessions'/(str(time.time_ns())+'.json'),dict(owner,resume=args.resume))
        active=[];next_index=0;stop=[]
        signal.signal(signal.SIGTERM,lambda *_:stop.append('parent_signal_natural_drain'))
        signal.signal(signal.SIGINT,lambda *_:stop.append('parent_signal_natural_drain'))
        def persist():
            ordered=sorted(records,key=lambda row:row['assignment_index']);save(output/'records.json',ordered);save(output/'summary.json',summary(jobs,ordered))
        try:
            while active or (next_index<len(pending) and not stop):
                if (output/'drain.request.json').exists() and not stop:stop.append('episode_boundary_drain_request')
                while not stop and next_index<len(pending) and dispatch_allowed(pending[next_index],[x['job'] for x in active],b,m,args.parallel):
                    job=pending[next_index]
                    command=[b['python'],'-u',str(HERE/'runner.py'),'episode','--bindings',str(args.bindings.resolve()),'--bindings-sha',binding_sha,
                        '--manifest',str(args.manifest.resolve()),'--output',str(output),'--index',str(job['assignment_index'])]
                    log=None
                    try:
                        require(sha(args.bindings)==binding_sha,'Bindings changed during dispatch')
                        new_json(output/'claims'/(job['id']+'.json'),dict(job=job,bindings_sha256=binding_sha,parent=owner,command=command))
                        log=(output/'.sealed/logs'/(job['id']+'.log')).open('x')
                        child=subprocess.Popen(command,env=environment(b),cwd=b['root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        active.append(dict(child=child,job=job,log=log));next_index+=1
                        ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
                        child_owner=dict(index=job['assignment_index'],id=job['id'],pid=child.pid,process_start_ticks=ticks,command=command,cwd=b['root'])
                        new_json(output/'processes'/(job['id']+'.json'),child_owner)
                        with (output/'dispatch.jsonl').open('a') as stream:stream.write(json.dumps(child_owner)+'\n');stream.flush();os.fsync(stream.fileno())
                    except BaseException:
                        # Spawn may already have succeeded. Keep child alive; a
                        # missing durable process record is never retryable.
                        if active and active[-1]['job']['id']==job['id']:raise
                        if log:log.close()
                        path=output/'.sealed'/(job['id']+'-dispatch-error.json');save(path,dict(job,status='infrastructure_unknown',traceback=traceback.format_exc(),bindings_sha256=binding_sha,manifest_sha256=MANIFEST_SHA))
                        records.append(public_record(read(path),path));next_index+=1;stop.append('dispatch_failed');persist();break
                for item in list(active):
                    child=item['child'];code=child.poll()
                    if code is None:continue
                    item['log'].close();job=item['job'];path=output/'.sealed/episodes'/job['id']/'result.json'
                    try:
                        require(path.exists(),'Child exited without result');row=read(path)
                        pub=admit_public(row,job,path,binding_sha)
                        require(not(code!=0 and row['status']=='completed'),'Exit contradicts completion')
                    except BaseException:
                        path=output/'.sealed'/(job['id']+'-supervisor-error.json');row=dict(job,status='infrastructure_unknown',traceback=traceback.format_exc(),returncode=code,bindings_sha256=binding_sha,manifest_sha256=MANIFEST_SHA);save(path,row);pub=public_record(row,path)
                    records.append(pub);active.remove(item)
                    if row['status']!='completed':stop.append('infrastructure_unknown')
                    persist()
                if active:time.sleep(.2)
        except BaseException:
            save(output/('supervisor-error-'+str(time.time_ns())+'.json'),dict(error=traceback.format_exc(),healthy_children_left_running=[{'pid':x['child'].pid,'index':x['job']['assignment_index']} for x in active if x['child'].poll() is None],no_restart=True))
            raise
        persist();save(output/'completion.json',dict(stop_reasons=stop,selected_assignments_complete=summary(jobs,records)['complete'],active_children=0,score_emitted=False,dispatch_order=indices))
        return 0 if summary(jobs,records)['complete'] else 1
    finally:
        for handle in locks:handle.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['preflight','execute','episode'])
    for key in ('bindings','manifest'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arms',nargs='+',choices=ARMS,default=['base']);p.add_argument('--output',type=Path)
    p.add_argument('--parallel',type=int,default=12);p.add_argument('--resume',action='store_true');p.add_argument('--index',type=int);p.add_argument('--bindings-sha')
    a=p.parse_args()
    if a.mode=='episode':return episode(a)
    m=design(a.manifest);indices=chosen_indices(m,a.arms);b=bindings(a.bindings,a.arms)
    if a.mode=='preflight':
        empty_connections(b,a.arms);require(1<=a.parallel<=48,'Parallel bound1..48')
        require(a.parallel<=sum(b['models'][k]['available_client_slots'] for arm in a.arms for k in b['arm_endpoints'][arm]),'Parallel exceeds endpoint reservation')
        print(json.dumps(dict(status='ready',episodes=len(indices),parallel=a.parallel,arms=a.arms,model_or_environment_started=False,score_emitted=False)));return 0
    require(a.output is not None,'Explicit output required')
    return execute(a,b,m)

if __name__=='__main__':raise SystemExit(main())
