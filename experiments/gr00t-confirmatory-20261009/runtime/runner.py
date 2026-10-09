"""Independent native GR00T C/V executor: prepare, preflight, episode and dispatch.

No service launcher, GPU model loader, or cleanup-by-name is
provided here. A separately admitted service and explicit frozen configuration
are mandatory. The original 2,500-case roster is projected without outcomes.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
from .processes import TrackedChild

ROOT=Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
HERE=Path(__file__).resolve().parent
NAMESPACE='originx_gr00t_confirmation_20261009_v2'
SERVICE_NAMESPACE='originx_gr00t_confirmation_20261009'
POLICY_ID='gr00t_n15_robocasa_multitask120000'
PROTOCOL='originx-gr00t-rng-v1'
OUTPUT_NAMES={True:'originx-gr00t-confirmatory-development-20261009-v2',
              False:'originx-gr00t-confirmatory-20261009-v1'}
ENTRY_SHA='18d70fade4c990210dc5b9c34285c8030eac142d9c332066c73335e1c0aaa28e'
COUNTER_SHA='77f992d01aa1ae5f21ed7f170d0aab9c303c32148f4b427651c04912be8ce68a'
EVENT_ORDER=('reset_requested','reset_completed','initial_fingerprints_recorded',
             'assignment_selected','server_connected','rng_ack','first_infer')
SCHEDULE_SEED=202610091910
DEVELOPMENT_START=1986200900
CASE_FIELDS=('case_id','task','task_name','stratum','seed','env_seed','policy_seed',
             'horizon','task_index','seed_index')


def require(value,message):
    if not value:raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x',encoding='utf-8') as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
    else:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(path)


def identity(pid=None):
    pid=os.getpid() if pid is None else pid;p=Path('/proc')/str(pid)
    fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,process_start_ticks=int(fields[19]),
        command=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),cwd=str((p/'cwd').resolve()))


def active(owner):
    try:
        p=Path('/proc')/str(owner['pid']);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        actual=identity(owner['pid'])
        return fields[0] not in ('Z','X') and all(actual[k]==owner[k]
                    for k in ('pid','process_start_ticks','command','cwd'))
    except (OSError,KeyError,ValueError):return False


def signal_owned(owner,sig):
    """Exact process identity only; no process-group or name-based signaling."""
    if not active(owner):return False
    require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Linux pidfd support required')
    fd=os.pidfd_open(owner['pid'])
    try:
        require(active(owner),'PID changed after pidfd opened')
        signal.pidfd_send_signal(fd,sig)
    finally:os.close(fd)
    return True


def live_tcp_connections(text):
    """TIME-WAIT no longer has a service socket; retain every other connection."""
    return [line for line in text.splitlines() if line.strip()
            and line.split()[0].upper() not in ('LISTEN','TIME-WAIT','CLOSED')]


def cleanup_service(model):
    """Drain one pinned service and signal only its still-identical Linux owner."""
    from . import server
    path=Path(model['server_manifest']);digest=model['server_manifest_sha256'];owner=model['owner']
    require(path.resolve()==path and path.resolve().is_relative_to(server.BASE/'services') and not path.is_symlink()
            and sha(path)==digest,'Cleanup profile not pinned')
    profile=read(path)
    require(profile.get('schema')=='originx_gr00t_service_profile_v1' and profile['owner']==owner
            and profile['port']==model['port'] and profile['policy_id']==POLICY_ID,'Cleanup authority differs')
    # Validate the static authority even when this exact process has exited.
    server.validate_gr00t_owner(owner,owner)
    receipt=dict(owner=owner,server_manifest=str(path),profile_sha256=digest,port=profile['port'],
                 schema='originx_gr00t_owned_service_cleanup_v1',foreign_processes_untouched=True)
    if not active(owner):return dict(receipt,already_not_owned_active=True,owned_process_still_active=False)
    server.validate_gr00t_owner(owner,identity(owner['pid']))
    drain=path.parent/'draining.json'
    if not drain.exists():write(drain,dict(owner=owner,profile_sha256=digest),True)
    else:require(read(drain).get('owner')==owner and read(drain).get('profile_sha256')==digest,'Drain owner differs')
    deadline=time.monotonic()+60;empty_checks=0;checks=[]
    while active(owner) and empty_checks<2:
        server.validate_gr00t_owner(owner,identity(owner['pid']))
        snapshot=subprocess.run(['ss','-Htn','state','all',f"( sport = :{profile['port']} )"],
                                check=True,capture_output=True,text=True,timeout=15).stdout
        connected=live_tcp_connections(snapshot)
        checks.append(dict(unix=time.time(),active_or_closing=connected))
        empty_checks=empty_checks+1 if not connected else 0
        require(time.monotonic()<deadline,'Service connections remain; exact service preserved')
        if empty_checks<2:time.sleep(.5)
    sent=signal_owned(owner,signal.SIGTERM)
    deadline=time.monotonic()+60
    while active(owner) and time.monotonic()<deadline:time.sleep(.5)
    require(not active(owner),'Exact owned service did not exit; no SIGKILL attempted')
    receipt.update(terminated=sent,already_not_owned_active=not sent,signal='SIGTERM' if sent else None,
                   pidfd=sent,connection_checks=checks,owned_process_still_active=False)
    retained=path.parent/'campaign-cleanup.json'
    if not retained.exists():write(retained,receipt,True)
    else:require(read(retained).get('owner')==owner,'Existing cleanup receipt belongs to another owner')
    return receipt


def cleanup_servers(config):
    evidence=[]
    for model in config['models']:
        item=dict(owner=model['owner'],server_manifest=model['server_manifest'],port=model['port'])
        try:item.update(cleanup_service(model))
        except BaseException:item.update(preserved=True,error=traceback.format_exc())
        item['owned_process_still_active']=active(model['owner']);evidence.append(item)
    return dict(schema='originx_gr00t_confirmatory_cleanup_v1',models=evidence,
        foreign_processes_untouched=True,all_owned_models_inactive=all(not e['owned_process_still_active'] for e in evidence))


def setup_path():
    sys.path[:0]=[str(ROOT/'official_b2500_v1'),str(ROOT),str(ROOT/'remote_processor_candidate_v1/deps')]


def environment(config):
    """Reuse the pinned CPU OSMesa environment only; no GR00T GPU dependency."""
    setup_path()
    from official_b2500_v1.runner import environment as native_environment
    return native_environment(config)


def check_sources(config):
    require(config.get('schema')=='originx_gr00t_confirmatory_config_v1','Wrong independent configuration')
    require(config.get('namespace')==NAMESPACE and config.get('policy_id')==POLICY_ID,'Wrong namespace/policy')
    require(config.get('service_namespace')==SERVICE_NAMESPACE,'Actual service namespace changed')
    require(type(config.get('development')) is bool and Path(config['output'])==ROOT/'results'/OUTPUT_NAMES[config['development']],
            'Configuration output outside independent namespace')
    require(bool(config.get('source_sha256')),'Empty source inventory')
    for name,digest in config['source_sha256'].items():require(sha(name)==digest,'Source changed: '+name)
    require(sha(Path(config['repo'])/'eval_robocasa365/entry.py')==ENTRY_SHA,'Official environment/action loop changed')
    require(sha(config['source_freeze'])==config['source_freeze_sha256'],'GR00T source freeze changed')
    prior=load_shared_budget(config.get('shared_budget_evidence'),config.get('shared_budget_evidence_sha256'),
                             development=config['development'])
    require(config.get('budget',{}).get('prior_usage')==prior,'Shared resource debit changed')


def load_shared_budget(path,digest,*,development):
    if path is None:
        require(development,'Formal GR00T must wait for complete prior campaigns and sealed shared-resource debit')
        require(digest is None,'Shared budget digest without a file')
        return dict(cli_calls=0,input_tokens=0,output_tokens=0)
    path=Path(path)
    require(path.is_absolute() and not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve()),
            'Shared budget receipt outside project or symlinked')
    require(sha(path)==digest,'Shared budget receipt hash changed')
    receipt=read(path)
    require(receipt.get('schema')=='originx_shared_budget_debit_v1'
        and receipt.get('all_prior_brokers_inactive') is True
        and receipt.get('complete_input_output_token_accounting') is True,
        'Prior campaign/broker usage is not complete and inactive')
    prior=receipt.get('prior_usage',{})
    require(set(prior)=={'cli_calls','input_tokens','output_tokens'}
        and all(type(v) is int and v>=0 for v in prior.values()),'Malformed shared usage counters')
    return dict(prior)


def native_reset_proof(config):
    from robocasa.models.fixtures.counter import Counter as CounterFixture
    import robocasa
    module=sys.modules[CounterFixture.__module__];source_file=Path(module.__file__).resolve()
    method=CounterFixture.get_reset_regions;source=inspect.getsource(method)
    require(Path(robocasa.__file__).resolve().parent==Path(config['package_root']),'Unexpected RoboCasa package')
    require(sha(source_file)==COUNTER_SHA,'Original Counter source changed')
    require(Path(method.__code__.co_filename).resolve()==source_file,'Counter reset dynamically patched')
    require('valid_geoms = list(set(valid_geoms))' in source and 'dict.fromkeys(valid_geoms)' not in source,
            'Native geometry ordering differs')
    require('continuous_branch_v3.reset_fix' not in sys.modules,'Development reset patch imported')
    return dict(native_reset=True,source_path=str(source_file),source_sha256=COUNTER_SHA,
        method_source_sha256=hashlib.sha256(source.encode()).hexdigest(),method_code_filename=method.__code__.co_filename,
        pythonhashseed=os.environ.get('PYTHONHASHSEED'),reset_patch_applied=False)


def make_manifest(reference,development=False):
    """Read only the sealed fresh roster; discard every source policy/arm job."""
    require(reference.get('schema')=='originx_confirmatory_manifest_v1' and reference.get('development') is False,
            'Project the sealed fresh full cohort, never a historical failure subset')
    tasks=reference['tasks'];original=reference['cases']
    require(len(tasks)==50 and len({t['task'] for t in tasks})==50,'Require all 50 registered tasks')
    require(len(original)==2500 and len({c['case_id'] for c in original})==2500
            and len({c['seed'] for c in original})==2500,'Require exactly 2500 unique fresh cases/seeds')
    roster={t['task']:t for t in tasks}
    require(Counter(c['task'] for c in original)==Counter({name:50 for name in roster}),'Require 50 seeds/task')
    for c in original:
        require(set(CASE_FIELDS)<=set(c),'Incomplete source case')
        require(c['task_name']==c['task'] and c['horizon']==roster[c['task']]['horizon'],'Source task/horizon differs')
        require(c['seed']==c['env_seed']==c['policy_seed'] and c['case_id'].startswith('fresh-'),'Source seed identity differs')
        require(c['seed']==1986100900+c['task_index']*50+c['seed_index'],'Source prospective seed schedule differs')
    if development:
        selected=sorted(tasks,key=lambda t:(t['horizon'],t['task']))[:2]
        cases=[dict(case_id=f'dev-{i:02d}-00',task=t['task'],task_name=t['task'],stratum=t['stratum'],
            seed=DEVELOPMENT_START+i*50,env_seed=DEVELOPMENT_START+i*50,policy_seed=DEVELOPMENT_START+i*50,
            horizon=t['horizon'],task_index=i,seed_index=0) for i,t in enumerate(selected)]
        require(not ({c['seed'] for c in cases}&{c['seed'] for c in original}),'Development overlaps scored seeds')
    else:
        selected=sorted(tasks,key=lambda t:t['task'])
        cases=[{k:c[k] for k in CASE_FIELDS} for c in original]
    rng=random.Random(SCHEDULE_SEED);blocks=[]
    for task_index in sorted({c['task_index'] for c in cases}):
        task_cases=[c for c in cases if c['task_index']==task_index]
        for group in sorted({c['seed_index']//8 for c in task_cases}):
            block=[c for c in task_cases if c['seed_index']//8==group]
            arms=['C','V'];rng.shuffle(arms);jobs=[]
            for arm in arms:
                for case in block:
                    eid=case['case_id']+'--gr00t--'+arm
                    request_id=hashlib.sha256((NAMESPACE+':'+eid).encode()).hexdigest()[:32]
                    jobs.append(dict(case,id=eid,episode_id=eid,policy_id=POLICY_ID,arm=arm,
                        request_id=request_id,batch_group=f'{task_index:02d}-{group:02d}-gr00t-{arm}'))
            blocks.append(jobs)
    rng.shuffle(blocks);jobs=[j for block in blocks for j in block]
    for i,j in enumerate(jobs):j['assignment_index']=i
    result=dict(schema='originx_gr00t_confirmatory_manifest_v1',namespace=NAMESPACE,policy_id=POLICY_ID,
        development=development,tasks=selected,cases=cases,jobs=jobs,case_count=len(cases),
        arm_outcome_count=len(jobs),policy_arms={POLICY_ID:['C','V']},query_interval=16,
        schedule_seed=SCHEDULE_SEED,source_case_projection_only=True,seed_selection_uses_outcomes=False,
        shared_fresh_case_ids_across_policies=not development,development_reuses_disclosed_case_ids=development,
        development_note='Public prior development IDs reused only for isolated infrastructure validation' if development else None,
        no_seed_replacements=True,copied_trajectories=False)
    validate_manifest(result);return result


def validate_manifest(manifest):
    require(manifest.get('schema')=='originx_gr00t_confirmatory_manifest_v1'
            and manifest.get('namespace')==NAMESPACE and manifest.get('policy_id')==POLICY_ID,'Wrong manifest schema/namespace')
    cases=manifest['cases'];jobs=manifest['jobs'];count=2 if manifest['development'] else 2500
    require(len(cases)==manifest['case_count']==count and len(jobs)==manifest['arm_outcome_count']==2*count,
            'Wrong fixed case/arm denominator')
    require(len({c['case_id'] for c in cases})==len({c['seed'] for c in cases})==count,'Duplicate case/seed')
    require(len({j['id'] for j in jobs})==len({j['request_id'] for j in jobs})==len(jobs),'Duplicate episode/request ID')
    case_map={c['case_id']:c for c in cases};seen={cid:set() for cid in case_map}
    for i,j in enumerate(jobs):
        require(j['case_id'] in case_map and all(j[k]==v for k,v in case_map[j['case_id']].items()),'Job/case mismatch')
        require(j['policy_id']==POLICY_ID and j['arm'] in ('C','V') and j['assignment_index']==i,'Job policy/arm/order differs')
        require(j['id']==j['episode_id']==j['case_id']+'--gr00t--'+j['arm'],'Episode ID differs')
        require(j['request_id']==hashlib.sha256((NAMESPACE+':'+j['id']).encode()).hexdigest()[:32], 'Opaque request ID differs')
        seen[j['case_id']].add(j['arm'])
    require(all(v=={'C','V'} for v in seen.values()),'Each case must have exactly C/V')
    require(manifest.get('no_seed_replacements') is True and manifest.get('seed_selection_uses_outcomes') is False,
            'Outcome-conditioned inclusion is forbidden')
    if not manifest['development']:
        require(len({c['task'] for c in cases})==50 and set(Counter(c['task'] for c in cases).values())=={50},'Incomplete task roster')
    return True


class TappedResetObserver:
    """Only tap the observation returned by each existing reset/step call."""
    def __init__(self,env,client,capture,event,tap,guard_check):
        self.env,self.client,self.capture,self.event=env,client,capture,event
        self.tap,self.guard_check=tap,guard_check
        self.reset_calls=0;self.reset_complete=False;self.fingerprints_recorded=False
        self.steps=0;self.terminated=False;self.truncated=False

    def __getattr__(self,name):return getattr(self.env,name)

    def reset(self,*args,**kwargs):
        require(self.reset_calls==0,'One natural reset per assignment; no retry')
        self.reset_calls+=1;self.event('reset_requested')
        observation,info=self.env.reset(*args,**kwargs)
        self.guard_check();self.tap.capture(observation,steps=0)
        self.reset_complete=True;self.event('reset_completed')
        self.capture(self.env,observation);self.fingerprints_recorded=True
        self.event('initial_fingerprints_recorded');self.client.activate(self)
        return observation,info

    def step(self,action):
        require(self.reset_complete,'Cannot step before the natural reset')
        result=self.env.step(action);self.guard_check();self.steps+=1
        self.tap.capture(result[0],steps=self.steps)
        self.terminated,self.truncated=bool(result[2]),bool(result[3])
        return result

    def close(self):return self.env.close()


def episode(args,config):
    setup_path();check_sources(config)
    require(sha(args.config)==args.config_sha,'Episode config changed')
    require(sha(config['manifest'])==config['manifest_sha256'],'Manifest changed')
    manifest=read(config['manifest']);validate_manifest(manifest)
    require(type(args.index) is int and 0<=args.index<len(manifest['jobs']),'Job index out of range')
    job=manifest['jobs'][args.index];out=Path(config['output'])/'episodes'/job['id']
    out.mkdir(parents=True,exist_ok=False);write(out/'owner.json',identity(),exclusive=True)
    started=time.monotonic();guard=lazy=assistance=None;events=[];initial={};replay=None;record=None
    try:
        import numpy as np
        import torch
        import gymnasium as gym
        import robocasa
        import dev_randomized_broad_alpha as legacy
        import dev_repeat_audit as audit
        from candidate import DecimatedGym,QueryCheckedClient
        from context_lifetime import install
        from readback_guard import install_guard
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        from robocasa.utils.env_utils import convert_action
        from official_b2500_v1.runner import load,guard_complete
        from .adapter import LatestObservation,EntryBridge,STATE_DIMS,CAMERAS
        from .assistance import AssistanceClient,array_sha
        from .client import GR00TClient,load_profile
        before=native_reset_proof(config)
        require(get_task_horizon(job['task'])==job['horizon'],'Official horizon changed')
        require(os.environ.get('PYTHONHASHSEED')=='0' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU/native environment required')
        entry=load(Path(config['repo'])/'eval_robocasa365/entry.py','originx_gr00t_native_entry')
        require(type(args.model_index) is int and 0<=args.model_index<len(config['models']),'Endpoint index out of range')
        model=config['models'][args.model_index]
        require(model['policy_id']==POLICY_ID and active(model['owner']),'GR00T policy owner unavailable')
        profile=load_profile(model['server_manifest'],model['server_manifest_sha256'])
        require(profile['hello']==model['hello'],'GR00T profile hello changed')
        tap=LatestObservation()
        def event(name,**fields):events.append(dict(event=name,sequence=len(events),monotonic=time.monotonic(),**fields))
        def make_policy(_):return GR00TClient(profile,timeout=1200,telemetry_path=out/'rpc.jsonl')
        def validate(hello,_):
            require(hello.get('protocol')==PROTOCOL,'Wrong policy wire protocol')
            require(all(hello.get(k)==v for k,v in model['hello'].items() if k!='connection_id'),'GR00T server identity changed')
        lazy=legacy.LazyClient(lambda:dict(name=POLICY_ID),make_policy,validate,job['policy_seed'],event,
            instrument=lambda policy:EntryBridge(policy,tap,entry.observation_to_state))
        def capture(env,observation):
            nonlocal replay
            guard.check();core=env.unwrapped.env
            initial.update(audit.capture_initial(out,observation,entry.observation_to_state(observation),core.sim))
            arrays={key:np.array(observation[key],copy=True) for key in (*CAMERAS,*STATE_DIMS)}
            with (out/'initial_native_observation.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
            initial.update(layout_id=int(core.layout_id),style_id=int(core.style_id),xml_sha256=sha(out/'initial.xml'),
                native_state_sha256={key:array_sha(arrays[key]) for key in STATE_DIMS},
                native_observation_file='initial_native_observation.npz',native_observation_sha256=sha(out/'initial_native_observation.npz'),
                native_state_representation='named_float32_16D_with_raw_quaternions')
            write(out/'initial_scene.json',initial)
            replay=dict(natural_reset=True,state_restored=False,reference='Within-GR00T C/V fresh-case pairing only')
            write(out/'initial_replay.json',replay)
        observers=[]
        class ObservedGym:
            @staticmethod
            def make(*a,**kw):
                require(not observers,'One native environment per assignment')
                observer=TappedResetObserver(gym.make(*a,**kw),lazy,capture,event,tap,lambda:guard.check())
                observers.append(observer);return observer
        # model-path is parsed but never constructs a Xiaomi client/model. The
        # supplied GR00T EntryBridge is the sole policy inference endpoint.
        cfg=entry.parse_args(['--model-path',config['checkpoint'],'--split','pretrain','--task-set','target50',
            '--task-name',job['task'],'--num-trials','1','--seed',str(job['env_seed']),
            '--replan-steps','16','--obs-history','4','--obs-interval','2','--crop-ratio','0.95'])
        entry.validate_args(cfg);reduced=DecimatedGym(ObservedGym,cfg)
        assistance=AssistanceClient(lazy,request_root=Path(config['output'])/'requests',episode_output=out,
            request_id=job['request_id'],case_id=job['case_id'],arm=job['arm'],policy_id=POLICY_ID,
            horizon=job['horizon'],wait_seconds=config['response_wait_seconds'],
            latest_observation=tap,pack_state=entry.observation_to_state)
        checked=QueryCheckedClient(assistance,reduced)
        lifetime=install();guard=install_guard(out/'readback-guard',library_path=config['osmesa_library'])
        random.seed(job['env_seed']);np.random.seed(job['env_seed'])
        stats=entry.evaluate_task(job['task'],0,cfg,checked,reduced,get_task_horizon,convert_action,
            out/'official',episode_indices=[0],show_progress=False,write_task_stats=False)
        after=native_reset_proof(config);require(before==after,'Reset implementation changed during episode')
        from robosuite.utils import binding_utils
        require(not getattr(binding_utils,'_osmesa_lifetime_failure',None),'Renderer context close failure')
        guard.check();rng=lazy.policy.wire.control('rng_state');ep=stats['episodes'][0]
        require(tuple(e['event'] for e in events)==EVENT_ORDER and len(observers)==1 and observers[0].reset_calls==1,
            'Activation order or reset count changed')
        require(ep['seed']==job['env_seed'] and ep['episode']==ep['global_episode_index']==0
            and type(ep['success']) is bool and 1<=ep['steps']<=job['horizon']
            and ep['steps']==observers[0].steps==tap.steps and stats['horizon']==job['horizon']
            and (ep['success'] or ep['steps']==job['horizon'] or observers[0].terminated or observers[0].truncated),
            'Incomplete official episode')
        require(rng['requests_since_reset']==lazy.infer_calls==lazy.forward.calls==(ep['steps']+15)//16
                and not torch.cuda.is_initialized(),'Query count/RNG/CPU execution differs')
        require(all(rng[k]==lazy.policy.hello[k] for k in ('connection_id','server_instance','protocol')),
                'Terminal policy connection changed')
        record=dict(status='completed',success=ep['success'],stats=stats,
            terminal_reason='success' if ep['success'] else 'terminated' if observers[0].terminated else 'truncated' if observers[0].truncated else 'horizon',
            steps=ep['steps'],native_reset_before=before,native_reset_after=after,rng_ack=lazy.rng_ack,rng_after=rng,
            server_identity=lazy.policy.hello,local_rng=lazy.local_rng,render_counts=reduced.current.counts,
            context_lifetime=lifetime,native_tap_steps=tap.steps,policy_context_frames=1)
    except BaseException:
        record=dict(status='infrastructure_unknown',success=None,traceback=traceback.format_exc())
    finally:
        for name,obj in [('client',lazy),('guard',guard)]:
            if obj is not None:
                try:obj.close()
                except BaseException:
                    record=dict(status='infrastructure_unknown',success=None,cleanup=name,previous_record=record,traceback=traceback.format_exc())
        if record and record['status']=='completed':
            try:
                require(guard_complete(read(out/'readback-guard/summary.json')),'Readback guard did not close healthy')
                record['guard_report_sha256']=sha(out/'readback-guard/summary.json')
            except BaseException:record=dict(status='infrastructure_unknown',success=None,previous_record=record,traceback=traceback.format_exc())
        if record is None:record=dict(status='infrastructure_unknown',success=None,error='No terminal record')
        if assistance is not None:
            try:record['assistance']=assistance.finish()
            except BaseException:record=dict(status='infrastructure_unknown',success=None,previous_record=record,traceback=traceback.format_exc())
        record.update(schema='originx_gr00t_episode_v1',namespace=NAMESPACE,initial_replay=replay,initial_scene=initial,
            job=job,config_sha256=args.config_sha,manifest_sha256=config['manifest_sha256'],
            endpoint_index=args.model_index,events=events,wall_seconds=time.monotonic()-started)
        write(out/'result.json',record,exclusive=True)
    return 0 if record['status']=='completed' else 1


def prepare(args):
    from .client import load_profile
    require(sha(args.reference_config)==args.reference_config_sha256,'Native reference config changed')
    require(sha(args.reference_manifest)==args.reference_manifest_sha256,'Fresh reference manifest changed')
    require(sha(args.source_freeze)==args.source_freeze_sha256,'Independent source freeze changed')
    reference=read(args.reference_config);freeze=read(args.source_freeze)
    require(freeze.get('schema')=='originx_gr00t_source_freeze_v1' and freeze.get('namespace')==NAMESPACE,
            'Root-reviewed independent source freeze required')
    pins=freeze.get('source_sha256',{})
    require(set(pins)=={p.name for p in HERE.glob('*.py')},'Freeze must cover every independent executable module')
    for name,digest in pins.items():require(Path(name).name==name and sha(HERE/name)==digest,'Frozen GR00T module changed')
    manifest=make_manifest(read(args.reference_manifest),args.development)
    prior_usage=load_shared_budget(args.shared_budget_evidence,args.shared_budget_evidence_sha256,
                                   development=args.development)
    manifest['source_manifest_path']=str(args.reference_manifest.resolve())
    manifest['source_manifest_sha256']=args.reference_manifest_sha256
    output=args.output.resolve()
    require(output==ROOT/'results'/OUTPUT_NAMES[args.development],'Require exact new GR00T output namespace')
    require(not output.exists(),'Fresh output required; no overwrite or in-place rerun')
    profile=load_profile(args.service_manifest,args.service_manifest_sha256)
    require(profile.get('ready') is True and profile.get('policy_id')==POLICY_ID,'Admitted GR00T service profile required')
    require(profile['hello']['protocol']==PROTOCOL and active(profile['owner']),'GR00T owner/hello unavailable')
    model=dict(profile,server_manifest=str(args.service_manifest.resolve()),server_manifest_sha256=args.service_manifest_sha256)
    sources=dict(reference['source_sha256']);sources.update({str(HERE/name):digest for name,digest in pins.items()})
    service_pins=profile['identity_manifest']['service_sources_sha256']
    service_names={'__init__.py','adapter.py','wire.py','core.py','server.py','client.py','parity_probe.py'}
    require(set(service_pins)==service_names,'Actual service source inventory differs')
    for name,digest in service_pins.items():
        actual=ROOT/SERVICE_NAMESPACE/name
        require(sha(actual)==digest and sha(HERE/name)==digest,'Copied service and actual executed service differ')
        sources[str(actual)]=digest
    config={k:reference[k] for k in ('root','repo','package_root','python','osmesa_library','packages')}
    config.update(schema='originx_gr00t_confirmatory_config_v1',namespace=NAMESPACE,service_namespace=SERVICE_NAMESPACE,policy_id=POLICY_ID,
        output=str(output),manifest=str(output/'manifest.json'),models=[model],source_sha256=sources,
        checkpoint=str(args.checkpoint.resolve()),development=args.development,response_wait_seconds=1200,
        intervention_fraction=.5,source_freeze=str(args.source_freeze.resolve()),source_freeze_sha256=args.source_freeze_sha256,
        reference_config=str(args.reference_config.resolve()),reference_config_sha256=args.reference_config_sha256,
        reference_manifest=str(args.reference_manifest.resolve()),reference_manifest_sha256=args.reference_manifest_sha256,
        shared_budget_evidence=(str(args.shared_budget_evidence.resolve()) if args.shared_budget_evidence is not None else None),
        shared_budget_evidence_sha256=args.shared_budget_evidence_sha256,
        duration_seconds=(args.duration_seconds if args.duration_seconds is not None else 7200 if args.development else 5*86400),
        broker_source_sha256=freeze.get('broker_source_sha256'),
        protocol=dict(native_reset=True,pythonhashseed=0,replan=16,policy_context_frames=1,
            evaluator_history=[4,2],policy_inputs='current public named 16D raw-quaternion state plus three RGB frames',
            native_tap='same reset/step return; no extra render/reset/action',gr00t_denoising_steps=4,
            image_transform='official GR00T transform; no client crop or extra flip',
            action='official decoded groups concatenated to float32 16x12, existing convert_action',
            arms={'C':'unchanged instruction sham','V':'Astra High once with three current RGB views'},
            trigger='16*ceil(H/32)',policy_wait_keepalive_seconds=120,response_fallback='same original instruction',
            latency_matched=False,action_budget_matched=True,seed_replacements=False,training=False,
            historical_results_unchanged=True,organizer_verified=False,automatic_submission=False),
        inference=dict(primary=['gr00t:V-C'],denominator=manifest['case_count'],unknowns='fixed-denominator identification bounds',
            statistical_superiority_claim=False),
        budget=dict(max_cli_batches=1500,max_input_tokens=25000000,max_output_tokens=2000000,
            quota_resets_allowed=0,paid_topup_allowed=False,max_batch=8,
            prior_usage=prior_usage,scope='combined main+GR00T, one final batch may overshoot'))
    require(type(config['duration_seconds']) is int and 1800<=config['duration_seconds']<=5*86400,'Invalid fixed campaign time cap')
    require(isinstance(config['broker_source_sha256'],str) and len(config['broker_source_sha256'])==64,'Frozen independent broker digest required')
    check_sources(config)
    output.mkdir(parents=True,exist_ok=False);write(output/'manifest.json',manifest,exclusive=True)
    config['manifest_sha256']=sha(output/'manifest.json');write(output/'config.json',config,exclusive=True)
    write(output/'seed-projection.json',dict(source_manifest_sha256=args.reference_manifest_sha256,
        development=args.development,case_count=manifest['case_count'],seeds=[c['seed'] for c in manifest['cases']],
        shared_fresh_cases_not_replaced=not args.development,disclosed_development_ids_reused=args.development,
        no_outcomes_read=True),exclusive=True)
    print(json.dumps(dict(config=str(output/'config.json'),unique_cases=manifest['case_count'],arms=len(manifest['jobs']))))


def preflight(args,config):
    import importlib.metadata
    setup_path();check_sources(config)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONHASHSEED')=='0','CPU/native environment required')
    require({n:importlib.metadata.version(n) for n in config['packages']}==config['packages'],'Simulator dependency versions changed')
    require(sha(config['manifest'])==config['manifest_sha256'],'Manifest changed')
    manifest=read(config['manifest']);validate_manifest(manifest)
    proof=native_reset_proof(config)
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    require(all(get_task_horizon(j['task'])==j['horizon'] for j in manifest['jobs']),'Official horizon differs')
    from .client import load_profile
    for model in config['models']:
        profile=load_profile(model['server_manifest'],model['server_manifest_sha256'])
        require(profile['owner']==model['owner'] and active(model['owner']) and profile['hello']==model['hello'],'Service profile/owner changed')
    write(Path(config['output'])/'preflight.json',dict(schema='originx_gr00t_preflight_v1',passed=True,
        native_reset=proof,manifest_sha256=config['manifest_sha256'],config_sha256=sha(args.config),
        starts_models=False,executes_rollouts=False,unix=time.time()),exclusive=True)


def validate_broker_ready(config,config_sha,ready):
    require(ready.get('schema')=='originx_gr00t_confirmatory_broker_ready_v1'
        and ready.get('config_sha256')==config_sha and ready.get('manifest_sha256')==config['manifest_sha256']
        and ready.get('broker_source_sha256')==config['broker_source_sha256']
        and ready.get('model')=='gpt-6-astra' and ready.get('reasoning_effort')=='high'
        and ready.get('remote_output')==config['output'],'Broker readiness authority differs')


def validate_run_admission(config,config_sha,admission):
    require(admission.get('schema')=='originx_gr00t_runtime_admission_v1' and admission.get('passed') is True
        and admission.get('config_sha256')==config_sha and admission.get('manifest_sha256')==config['manifest_sha256'],
        'Independent same-socket numerical admission required')
    require(admission.get('development') is config['development'],'Admission cohort differs')
    require(admission.get('profiles')==[dict(path=m['server_manifest'],sha256=m['server_manifest_sha256']) for m in config['models']],
        'Admission service profile differs')
    parity=admission.get('socket_parity',{})
    require(sha(parity['path'])==parity['sha256'],'Socket parity receipt changed')
    if not config['development']:
        gate=admission.get('development_review',{})
        require(sha(gate['path'])==gate['sha256'],'Development review changed')
        reviewed=read(gate['path'])
        require(reviewed.get('passed') is True and reviewed.get('no_confirmatory_outcomes_seen') is True,
            'Full cohort requires independent successful development review')


def run(args,config):
    """Own each fixed arm once, drain on fault, retain all incomplete evidence."""
    import fcntl
    setup_path();check_sources(config)
    output=Path(config['output']);config_sha=sha(args.config)
    require(config_sha==args.config_sha and sha(config['manifest'])==config['manifest_sha256'],'Run authority changed')
    manifest=read(config['manifest']);validate_manifest(manifest);jobs=manifest['jobs']
    pre=read(output/'preflight.json')
    require(pre.get('passed') is True and pre.get('config_sha256')==config_sha,'CPU preflight binding differs')
    validate_broker_ready(config,config_sha,read(output/'broker-ready.json'))
    validate_run_admission(config,config_sha,read(output/'runtime-admission.json'))
    lock=(output/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (output/'launch.json').exists(),'Prior launch exists; no automatic rerun')
    models=config['models']
    require(len(models)==1 and models[0]['policy_id']==POLICY_ID and models[0]['slots']==6,
            'Current independent admission fixes one model and six client slots')
    leases=[]
    for model in models:
        require(active(model['owner']),'Admitted GR00T owner unavailable')
        lease=(Path(model['server_manifest']).parent/'rollout.lock').open('a+')
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(lease)
        sockets=subprocess.run(['ss','-Htn','state','established',f"( sport = :{model['port']} )"],
            capture_output=True,text=True,check=True,timeout=10)
        require(not sockets.stdout.strip(),'Admitted policy already has active clients')
    write(output/'launch.json',dict(identity(),namespace=NAMESPACE,config_sha256=config_sha,
        manifest_sha256=config['manifest_sha256'],unix=time.time(),duration_seconds=config['duration_seconds']),True)
    pending=list(range(len(jobs)));done={};live=[];stop=[];started=time.monotonic()
    deadline=started+config['duration_seconds'];term=False;kill=False;first_error=None
    def stop_once(reason):
        if reason not in stop:stop.append(reason)
    def requested_drain(*_):
        nonlocal deadline
        stop_once('requested_drain');deadline=min(deadline,time.monotonic())
    signal.signal(signal.SIGTERM,requested_drain)
    signal.signal(signal.SIGINT,requested_drain)
    def progress():
        counts=Counter(x['job']['case_id'] for x in done.values())
        write(output/'progress.json',dict(schema='originx_gr00t_confirmatory_progress_v1',
            planned_unique_cases=manifest['case_count'],planned_arm_outcomes=len(jobs),finished_arm_outcomes=len(done),
            active=len(live),pending=len(pending),completed_case_ids=sum(counts[c['case_id']]==2 for c in manifest['cases']),
            technical_unknown=sum(x['status']!='completed' for x in done.values()),elapsed_seconds=time.monotonic()-started,
            stop_reasons=stop,unix=time.time(),development=config['development'],score_emitted=False))
    def consume(item):
        nonlocal first_error
        rc=item['child'].poll()
        if rc is None:return False
        item['handle'].close()
        write(output/'spawns'/(jobs[item['index']]['id']+'.json'),item['handle'].receipt())
        item['log'].close();job=jobs[item['index']];path=output/'episodes'/job['id']/'result.json'
        try:
            require(item['owner'] is not None,'Spawn never completed startup identity binding')
            result=read(path)
            require(result.get('schema')=='originx_gr00t_episode_v1' and result['job']==job
                and result['config_sha256']==config_sha and result['manifest_sha256']==config['manifest_sha256'],
                'Result authority differs')
            require((rc==0)==(result['status']=='completed'),'Child exit/status differs')
        except BaseException:
            result=dict(schema='originx_gr00t_episode_supervisor_unknown_v1',job=job,status='infrastructure_unknown',
                success=None,returncode=rc,traceback=traceback.format_exc())
            write(output/'errors'/(job['id']+'.json'),result,True)
        done[item['index']]=result;live.remove(item)
        if result['status']!='completed':
            stop_once('infrastructure_failure')
            if first_error is None:first_error=result;write(output/'first-error.json',result,True)
        return True
    try:
        last=0
        while live or (pending and not stop):
            now=time.monotonic()
            if now>=deadline-min(1800,config['duration_seconds']/10):stop_once('deadline_stop_dispatch')
            if (output/'drain.request.json').exists():stop_once('broker_or_explicit_drain')
            if now>=deadline and not term:
                stop_once('deadline_workers_terminate')
                for item in live:item['handle'].send(signal.SIGTERM)
                term=True
            if now>=deadline+15 and not kill:
                for item in live:item['handle'].send(signal.SIGKILL)
                kill=True
            used=Counter(x['model_index'] for x in live)
            for mi,model in enumerate(models):
                while pending and not stop and used[mi]<model['slots']:
                    require(active(model['owner']),'GR00T service owner exited/changed')
                    index=pending[0];job=jobs[index]
                    command=[config['python'],'-u','-m',NAMESPACE+'.runner','episode','--config',str(args.config.resolve()),
                        '--config-sha',config_sha,'--index',str(index),'--model-index',str(mi)]
                    write(output/'claims'/(job['id']+'.json'),dict(job=job,command=command,config_sha256=config_sha,
                        model_index=mi,supervisor=identity()),True)
                    path=output/'logs'/(job['id']+'.log');path.parent.mkdir(exist_ok=True);log=path.open('x')
                    child=subprocess.Popen(command,cwd=ROOT,env=environment(config),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    handle=TrackedChild(child,command,ROOT)
                    item=dict(index=index,model_index=mi,child=child,handle=handle,owner=None,log=log)
                    live.append(item);pending.pop(0);used[mi]+=1
                    # The Popen handle is now owned even while /proc argv is empty.
                    owner=handle.await_identity();item['owner']=owner
                    write(output/'spawns'/(job['id']+'.json'),handle.receipt())
                    write(output/'processes'/(job['id']+'.json'),owner,True)
            for item in list(live):consume(item)
            if now-last>=30:progress();last=now
            if live:time.sleep(.5)
    except BaseException:
        stop_once('supervisor_exception');write(output/'supervisor-error.json',dict(traceback=traceback.format_exc()),True)
        for item in live:
            try:item['handle'].send(signal.SIGTERM)
            except BaseException:
                write(output/'spawn-drain-errors'/(jobs[item['index']]['id']+'.signal.json'),dict(error=traceback.format_exc()),True)
        for item in list(live):
            try:
                item['handle'].drain();consume(item)
            except BaseException:
                stop_once('owned_child_drain_unconfirmed')
                write(output/'spawn-drain-errors'/(jobs[item['index']]['id']+'.wait.json'),dict(error=traceback.format_exc()),True)
    finally:
        progress()
        write(output/'completion.json',dict(schema='originx_gr00t_confirmatory_completion_v1',namespace=NAMESPACE,
            development=config['development'],planned=len(jobs),finished=len(done),
            complete=len(done)==len(jobs) and all(x['status']=='completed' for x in done.values()),
            all_children_drained=not live,unstarted_arm_outcomes=len(pending),stop_reasons=stop,
            remaining_owned_workers=[x['handle'].receipt() for x in live],unix=time.time()),True)
        for lease in leases:lease.close()
        lock.close()
    return 0 if len(done)==len(jobs) and not stop and not live else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','preflight','episode','run'])
    parser.add_argument('--config',type=Path);parser.add_argument('--config-sha')
    parser.add_argument('--index',type=int);parser.add_argument('--model-index',type=int,default=0)
    parser.add_argument('--output',type=Path);parser.add_argument('--development',action='store_true')
    parser.add_argument('--reference-config',type=Path);parser.add_argument('--reference-config-sha256')
    parser.add_argument('--reference-manifest',type=Path);parser.add_argument('--reference-manifest-sha256')
    parser.add_argument('--service-manifest',type=Path);parser.add_argument('--service-manifest-sha256')
    parser.add_argument('--source-freeze',type=Path);parser.add_argument('--source-freeze-sha256')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--duration-seconds',type=int)
    parser.add_argument('--shared-budget-evidence',type=Path)
    parser.add_argument('--shared-budget-evidence-sha256')
    args=parser.parse_args()
    if args.mode=='prepare':
        names=('output','reference_config','reference_config_sha256','reference_manifest','reference_manifest_sha256',
               'service_manifest','service_manifest_sha256','source_freeze','source_freeze_sha256','checkpoint')
        require(all(getattr(args,k) is not None for k in names),'prepare requires all explicit reference/service/source pins and checkpoint')
        prepare(args);return
    require(args.config is not None and args.config_sha is not None,'Explicit configuration SHA required')
    require(sha(args.config)==args.config_sha,'Config digest differs');config=read(args.config)
    if args.mode=='preflight':preflight(args,config)
    elif args.mode=='episode':sys.exit(episode(args,config))
    else:sys.exit(run(args,config))


if __name__=='__main__':main()
