"""Fresh full-cohort native-reset C/R/G/L/V study with separately loaded policies.

Historical results remain immutable; this namespace records every planned arm,
including early successes, clean assistance fallbacks, and technical failures.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
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

ROOT = Path('/ephemeral/qinzhen/robocasa-xr1-20261003')
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'official_b2500_v1'
REFERENCE = ROOT / 'results/continued-ab2000-v1'
BASE = '/ephemeral/qinzhen/ckpt/robocasa/Xiaomi-Robotics-1-RoboCasa365'
OLD_MANIFEST_SHA = '1cc808ccb54f116350017ac1a60f85225382920c13bab0e4cfa0a6ec2bce677a'
ENDPOINT_SHA = 'c3138b49dcf51c0514f129073870701b5cd59cb88df384ac5a4b9ac24a77c601'
ENTRY_SHA = '18d70fade4c990210dc5b9c34285c8030eac142d9c332066c73335e1c0aaa28e'
COUNTER_SHA = '77f992d01aa1ae5f21ed7f170d0aab9c303c32148f4b427651c04912be8ce68a'

def require(value, message):
    if not value:
        raise RuntimeError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
    else:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        tmp.replace(path)

def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, process_start_ticks=int(fields[19]),
                command=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                cwd=str((p/'cwd').resolve()))

def active(owner):
    try:
        p = Path('/proc') / str(owner['pid'])
        f = (p/'stat').read_text().rsplit(')', 1)[1].split()
        return (f[0] not in ('Z', 'X') and int(f[19]) == owner['process_start_ticks']
                and identity(owner['pid'])['command']==owner['command']
                and identity(owner['pid'])['cwd']==owner['cwd'])
    except (FileNotFoundError,ProcessLookupError):
        return False

def signal_owned(owner, sig):
    """Signal one exact PID/start identity; never search by process name."""
    if not active(owner):
        return False
    actual=identity(owner['pid'])
    require(actual['command']==owner['command'] and actual['cwd']==owner['cwd'],'Owned process argv/cwd changed; preserve it')
    fd=os.pidfd_open(owner['pid'])
    try:
        require(active(owner),'PID identity changed after pidfd open')
        signal.pidfd_send_signal(fd,sig)
    finally:
        os.close(fd)
    return True

def cleanup_servers(config):
    evidence=[]
    for model in config['models']:
        owner=dict(model['owner']); owner.setdefault('cwd', str(ROOT)); item=dict(port=model['port'],owner=owner)
        if active(owner):
            actual=identity(owner['pid'])
            require(actual['command']==owner['command'] and actual['cwd']==owner['cwd'],'Server identity changed; preserve it')
            connections=subprocess.run(['ss','-Htn','state','established',f"( sport = :{model['port']} )"],capture_output=True,text=True,check=True,timeout=10)
            if connections.stdout.strip():
                item['preserved']='active_client_connections'
            else:
                item['sigterm_sent']=signal_owned(owner,signal.SIGTERM)
        evidence.append(item)
    limit=time.monotonic()+30
    while time.monotonic()<limit and any(active(x['owner']) for x in evidence if x.get('sigterm_sent')):time.sleep(.25)
    for item in evidence:item['still_active']=active(item['owner'])
    write(Path(config['output'])/'cleanup.json',dict(models=evidence,unix=time.time(),foreign_processes_untouched=True))

def setup_path():
    sys.path[:0] = [str(OLD), str(ROOT), str(ROOT/'remote_processor_candidate_v1/deps')]

def environment(config):
    setup_path()
    from official_b2500_v1.runner import environment as frozen_environment
    return frozen_environment(config)

def check_sources(config):
    for file, digest in config['source_sha256'].items():
        require(sha(file) == digest, 'Source changed: ' + file)
    require(sha(REFERENCE/'B-endpoint.json') == ENDPOINT_SHA, 'Frozen B endpoint changed')

def native_reset_proof(config):
    from robocasa.models.fixtures.counter import Counter as CounterFixture
    import robocasa
    module = sys.modules[CounterFixture.__module__]
    source_file = Path(module.__file__).resolve()
    method = CounterFixture.get_reset_regions
    source = inspect.getsource(method)
    require(Path(robocasa.__file__).resolve().parent == Path(config['package_root']), 'Unexpected RoboCasa package')
    require(sha(source_file) == COUNTER_SHA, 'Original Counter source changed')
    require(Path(method.__code__.co_filename).resolve() == source_file, 'Counter reset method is dynamically patched')
    require('valid_geoms = list(set(valid_geoms))' in source, 'Native geometry ordering missing')
    require('dict.fromkeys(valid_geoms)' not in source, 'Development reset patch detected')
    require('continuous_branch_v3.reset_fix' not in sys.modules, 'Development reset module was imported')
    return dict(native_reset=True, source_path=str(source_file), source_sha256=COUNTER_SHA,
                method_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                method_code_filename=method.__code__.co_filename,
                pythonhashseed=os.environ.get('PYTHONHASHSEED'), reset_patch_applied=False)

def episode(args, config):
    setup_path()
    check_sources(config)
    require(sha(args.config) == args.config_sha, 'Episode config changed')
    require(sha(config['manifest']) == config['manifest_sha256'], 'Manifest changed')
    m = read(config['manifest']); job = m['jobs'][args.index]
    out = Path(config['output'])/'episodes'/job['id']
    out.mkdir(parents=True,exist_ok=False)
    write(out/'owner.json',identity(),exclusive=True)
    started = time.monotonic(); guard = lazy = None; events = []; initial = {}; record = None; assistance = None; replay = None
    try:
        import numpy as np
        import torch
        import gymnasium as gym
        import robocasa
        import dev_randomized_broad_alpha as legacy
        import dev_repeat_audit as audit
        from candidate import DecimatedGym, QueryCheckedClient
        from originx_confirmatory_20261009.assistance import AssistanceClient
        from context_lifetime import install
        from readback_guard import install_guard
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        from robocasa.utils.env_utils import convert_action
        from continuous_eval2000_v3.client import StageEvalClient
        from continuous_eval2000_v3.loader import check_stage_hello
        from local_eval.remote_client import check_hello
        from official_b2500_v1.runner import load, guard_complete
        before = native_reset_proof(config)
        require(get_task_horizon(job['task']) == job['horizon'], 'Official horizon changed')
        require(os.environ.get('PYTHONHASHSEED') == '0' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Unexpected process environment')
        entry = load(Path(config['repo'])/'eval_robocasa365/entry.py','native_official_entry')
        model = config['models'][args.model_index]
        require(model['policy_id'] == job['policy_id'], 'Wrong policy routing')
        require(sha(model['server_manifest']) == model['server_manifest_sha256'] and active(model['owner']), 'Model binding changed')
        def event(name, **fields):
            events.append(dict(event=name,sequence=len(events),monotonic=time.monotonic(),**fields))
        def make_policy(_):
            from local_eval.remote_client import RemoteEvalClient
            cls = StageEvalClient if job['policy_id'] == 'B' else RemoteEvalClient
            p = cls(config['repo'],BASE,model['port'],model['hello'],telemetry_path=out/'rpc.jsonl')
            p.hello = p.wire.hello
            return p
        def validate(hello, _):
            check_hello(hello,model['hello'])
            if job['policy_id'] == 'B': check_stage_hello(hello,model['hello'])
            else: require(not hello.get('stage_serving_identity') and not hello.get('lora_serving_identity'), 'Base has adaptations')
        lazy = legacy.LazyClient(lambda:dict(name=job['policy_id']),make_policy,validate,job['policy_seed'],event)
        def capture(env, observation):
            nonlocal replay
            guard.check(); core = env.unwrapped.env
            initial.update(audit.capture_initial(out,observation,entry.observation_to_state(observation),core.sim))
            initial.update(layout_id=int(core.layout_id),style_id=int(core.style_id),xml_sha256=sha(out/'initial.xml'))
            write(out/'initial_scene.json',initial)
            replay = dict(natural_reset=True,state_restored=False,reference='paired fresh arms; evaluated at aggregation')
            write(out/'initial_replay.json',replay)
        observers = []
        class Observer(legacy.ResetObserver):
            steps = 0
            terminated = False
            truncated = False
            def __getattr__(self,name): return getattr(self.env,name)
            def step(self,action):
                r = self.env.step(action); guard.check(); self.steps += 1
                self.terminated = bool(r[2]); self.truncated = bool(r[3])
                return r
        class ObservedGym:
            @staticmethod
            def make(*a,**kw):
                require(not observers,'One natural-reset environment per assignment')
                obj=Observer(gym.make(*a,**kw),lazy,capture,event); observers.append(obj); return obj
        cfg = entry.parse_args(['--model-path',BASE,'--split','pretrain','--task-set','target50',
                                '--task-name',job['task'],'--num-trials','1','--seed',str(job['env_seed']),
                                '--replan-steps','16','--obs-history','4','--obs-interval','2','--crop-ratio','0.95'])
        entry.validate_args(cfg)
        reduced = DecimatedGym(ObservedGym,cfg)
        assistance = AssistanceClient(lazy,request_root=Path(config['output'])/'requests',episode_output=out,
                                      request_id=job['request_id'],case_id=job['case_id'],arm=job['arm'],policy_id=job['policy_id'],
                                      horizon=job['horizon'],wait_seconds=config['response_wait_seconds'],
                                      intervention_fraction=config['intervention_fraction'])
        checked = QueryCheckedClient(assistance,reduced)
        lifetime = install(); guard = install_guard(out/'readback-guard',library_path=config['osmesa_library'])
        random.seed(job['env_seed']); np.random.seed(job['env_seed'])
        stats = entry.evaluate_task(job['task'],0,cfg,checked,reduced,get_task_horizon,convert_action,
                                    out/'official',episode_indices=[0],show_progress=False,write_task_stats=False)
        after = native_reset_proof(config); require(before == after,'Reset implementation changed during episode')
        from robosuite.utils import binding_utils
        require(not getattr(binding_utils,'_osmesa_lifetime_failure',None),'Renderer context close failure')
        guard.check(); rng = lazy.policy.wire.control('rng_state'); ep = stats['episodes'][0]
        require(tuple(e['event'] for e in events) == legacy.EVENT_ORDER and len(observers)==1 and observers[0].reset_calls==1,'Reset/policy activation order changed')
        require(ep['seed']==job['env_seed'] and ep['episode']==ep['global_episode_index']==0 and
                type(ep['success']) is bool and 1<=ep['steps']<=job['horizon'] and ep['steps']==observers[0].steps and
                stats['horizon']==job['horizon'] and (ep['success'] or ep['steps']==job['horizon'] or observers[0].terminated or observers[0].truncated),'Incomplete official episode')
        require(rng['requests_since_reset']==lazy.infer_calls==(ep['steps']+15)//16 and not torch.cuda.is_initialized(),'Policy query/RNG/CPU execution changed')
        record=dict(status='completed',success=ep['success'],stats=stats,
                    terminal_reason=('success' if ep['success'] else 'terminated' if observers[0].terminated else 'truncated' if observers[0].truncated else 'horizon'),
                    steps=ep['steps'],native_reset_before=before,native_reset_after=after,
                    rng_ack=lazy.rng_ack,rng_after=rng,server_identity=lazy.policy.hello,local_rng=lazy.local_rng,
                    render_counts=reduced.current.counts,context_lifetime=lifetime,initial_scene=initial)
    except BaseException:
        record=dict(status='infrastructure_unknown',success=None,traceback=traceback.format_exc())
    finally:
        for name,obj in [('client',lazy),('guard',guard)]:
            if obj is not None:
                try: obj.close()
                except BaseException: record=dict(status='infrastructure_unknown',success=None,cleanup=name,traceback=traceback.format_exc())
        if record['status']=='completed':
            try:
                require(guard_complete(read(out/'readback-guard/summary.json')),'Readback guard did not close healthy')
                record['guard_report_sha256']=sha(out/'readback-guard/summary.json')
            except BaseException: record=dict(status='infrastructure_unknown',success=None,traceback=traceback.format_exc())
        if assistance is not None:
            record['assistance'] = assistance.finish()
        record['initial_replay'] = replay
        record['initial_scene'] = initial
        record.update(job=job,config_sha256=args.config_sha,manifest_sha256=config['manifest_sha256'],
                      endpoint_index=args.model_index,events=events,wall_seconds=time.monotonic()-started)
        write(out/'result.json',record,exclusive=True)
    return 0 if record['status']=='completed' else 1


# New campaign logic follows.
def prepare(args):
    from originx_confirmatory_20261009.protocol import make_manifest,seed_audit
    freeze=HERE/'source-freeze.ready.json';wait_until=time.monotonic()+240
    while not freeze.exists():
        require(time.monotonic()<wait_until,'Root source/test freeze receipt not available')
        time.sleep(2)
    for name,digest in read(freeze)['source_sha256'].items():
        require(Path(name).name==name and sha(HERE/name)==digest,'Frozen executable source differs: '+name)
    output=args.output.resolve()
    require(output.parent==ROOT/'results' and output.name.startswith('originx-confirmatory-'), 'New namespace required')
    require(not (output/'config.json').exists(), 'Configuration already exists; no overwrite')
    reference=read(ROOT/'results/native-reset-b2500-20261009-v1/config.json')
    require(sha(reference['manifest'])==reference['manifest_sha256'],'Original task roster changed')
    original=read(ROOT/'results/native-reset-b2500-20261009-v1/manifest.json')
    manifest=make_manifest(original,args.development)
    ready=read(args.services_ready)
    require(ready.get('ready') is True,'Services not ready')
    models=ready['models']
    require(isinstance(models,list) and {m['policy_id'] for m in models}=={'B','base'},'Both separate policies required')
    for model in models:
        require(active(model['owner']) and sha(model['server_manifest'])==model['server_manifest_sha256'],'Live owner/manifest required')
    audit=seed_audit(ROOT,manifest)
    write(output/'seed-audit.json',audit,exclusive=True)
    write(output/'manifest.json',manifest,exclusive=True)
    sources=dict(reference['source_sha256'])
    sources.update({str(p):sha(p) for p in HERE.glob('*.py')})
    sources.update({str(ROOT/'astra_rescue_continuation_20261009/runner.py'):sha(ROOT/'astra_rescue_continuation_20261009/runner.py')})
    c={k:reference[k] for k in ('root','repo','package_root','python','osmesa_library','packages')}
    c.update(schema='originx_confirmatory_config_v1',output=str(output),manifest=str(output/'manifest.json'),
             manifest_sha256=sha(output/'manifest.json'),models=models,source_sha256=sources,
             development=args.development,response_wait_seconds=1200,intervention_fraction=.5,
             source_freeze=str(freeze),source_freeze_sha256=sha(freeze),broker_source_sha256=read(freeze)['broker_source_sha256'],
             duration_seconds=args.duration_seconds,services_ready=str(args.services_ready.resolve()),
             protocol=dict(reference['protocol'],study='prospective complete fresh seed cohort, intention to assist',
               policies='frozen B2000 and separately loaded unadapted original Xiaomi; same model family',
               arms={'C':'unchanged instruction sham','R':'task restatement','G':'fixed generic continuation',
                     'L':'Astra High without images','V':'Astra High with three current RGB views'},
               trigger='16*ceil(H/32)',response_fallback='same original instruction after clean assistance failure',
               policy_wait_keepalive_seconds=120,latency_matched=False,action_budget_matched=True,
               seed_replacements=False,training=False,successes_and_regressions_reported=True,
               historical_results_unchanged=True),
             inference=dict(primary=['B:V-C','B:V-L','base:V-C'],
               incomplete_primary='identification bounds on fixed 2500 denominator; no full-score point claim',
               uncertainty='fixed-seed task block bootstrap; conditional if missing',statistical_superiority_claim=False),
             budget=dict(max_cli_batches=1500,max_input_tokens=25000000,max_output_tokens=2000000,
                         quota_resets_allowed=0,paid_topup_allowed=False,max_batch=8,grouping='policy/arm/task/group of eight'))
    check_sources(c)
    write(output/'config.json',c,exclusive=True)
    print(json.dumps(dict(config=str(output/'config.json'),case_count=manifest['case_count'],episodes=len(manifest['jobs']),development=args.development)))

def preflight(args,config):
    import importlib.metadata
    from originx_confirmatory_20261009.protocol import validate_manifest
    setup_path();check_sources(config)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('PYTHONHASHSEED')=='0','CPU/native environment required')
    require({n:importlib.metadata.version(n) for n in config['packages']}==config['packages'],'Package versions changed')
    proof=native_reset_proof(config)
    require(sha(config['manifest'])==config['manifest_sha256'],'Manifest changed')
    manifest=read(config['manifest']);validate_manifest(manifest)
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    require(all(get_task_horizon(j['task'])==j['horizon'] for j in manifest['jobs']),'Horizon differs')
    for model in config['models']:
        require(active(model['owner']) and sha(model['server_manifest'])==model['server_manifest_sha256'],'Model owner changed')
    write(Path(config['output'])/'preflight.json',dict(passed=True,native_reset=proof,manifest_sha256=config['manifest_sha256'],config_sha256=sha(args.config),unix=time.time()),exclusive=True)

def run(args,config):
    import fcntl
    setup_path();check_sources(config)
    output=Path(config['output']);cs=sha(args.config);m=read(config['manifest']);jobs=m['jobs']
    require(sha(config['manifest'])==config['manifest_sha256'],'Manifest changed')
    require(read(output/'preflight.json')['config_sha256']==cs,'Preflight binding differs')
    require(read(output/'broker-ready.json')['config_sha256']==cs,'Broker not bound')
    lock=(output/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (output/'launch.json').exists(),'Existing launch requires audited continuation; no blind restart')
    write(output/'launch.json',dict(identity(),config_sha256=cs,unix=time.time(),duration_seconds=config['duration_seconds']),exclusive=True)
    pending=list(range(len(jobs)));done={};live=[];stop=[];models=config['models'];started=time.monotonic()
    deadline=started+config['duration_seconds'];term=False;kill=False
    signal.signal(signal.SIGTERM,lambda *_:stop.append('requested_drain'))
    signal.signal(signal.SIGINT,lambda *_:stop.append('requested_drain'))
    first_error=None
    def progress():
        case_counts=Counter(r.get('job',{}).get('case_id') for r in done.values())
        write(output/'progress.json',dict(schema='originx_confirmatory_progress_v1',planned_unique_cases=m['case_count'],
              planned_arm_outcomes=len(jobs),finished_arm_outcomes=len(done),active=len(live),pending=len(pending),
              completed_case_ids=sum(case_counts[c['case_id']]==7 for c in m['cases']),
              technical_unknown=sum(r.get('status')!='completed' for r in done.values()),elapsed_seconds=time.monotonic()-started,
              stop_reasons=stop,unix=time.time(),development=config['development'],score_emitted=False))
    try:
        last=0
        while live or (pending and not stop):
            now=time.monotonic()
            if now>=deadline-min(1800,config['duration_seconds']/10) and not stop:stop.append('deadline_stop_dispatch')
            if (output/'drain.request.json').exists() and not stop:stop.append('broker_or_explicit_drain')
            if now>=deadline and not term:
                stop.append('deadline_workers_terminate')
                for x in live:signal_owned(x['owner'],signal.SIGTERM)
                term=True
            if now>=deadline+15 and not kill:
                for x in live:signal_owned(x['owner'],signal.SIGKILL)
                kill=True
            used=Counter(x['model_index'] for x in live)
            for mi,model in enumerate(models):
                while used[mi]<model['slots'] and pending and not stop:
                    index=next((i for i in pending if jobs[i]['policy_id']==model['policy_id']),None)
                    if index is None:break
                    require(active(model['owner']),'Model owner exited')
                    j=jobs[index];command=[config['python'],'-u',str(HERE/'runner.py'),'episode','--config',str(args.config.resolve()),'--config-sha',cs,'--index',str(index),'--model-index',str(mi)]
                    write(output/'claims'/(j['id']+'.json'),dict(job=j,command=command,config_sha256=cs,model_index=mi),exclusive=True)
                    lp=output/'logs'/(j['id']+'.log');lp.parent.mkdir(exist_ok=True);log=lp.open('x')
                    child=subprocess.Popen(command,cwd=ROOT,env=environment(config),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    owner=identity(child.pid);owner.update(command=command,cwd=str(ROOT))
                    write(output/'processes'/(j['id']+'.json'),owner,exclusive=True)
                    live.append(dict(index=index,model_index=mi,child=child,owner=owner,log=log));pending.remove(index);used[mi]+=1
            for item in list(live):
                rc=item['child'].poll()
                if rc is None:continue
                item['log'].close();j=jobs[item['index']];path=output/'episodes'/j['id']/'result.json'
                try:
                    result=read(path)
                    require(result['job']==j and result['config_sha256']==cs,'Result authority differs')
                    require((rc==0)==(result['status']=='completed'),'Exit/status differs')
                except BaseException:
                    result=dict(job=j,status='infrastructure_unknown',success=None,returncode=rc,traceback=traceback.format_exc())
                    write(output/'errors'/(j['id']+'.json'),result,exclusive=True)
                done[item['index']]=result;live.remove(item)
                if result['status']!='completed':
                    if first_error is None:
                        first_error=result;write(output/'first-error.json',result,exclusive=True)
                    if not stop:stop.append('infrastructure_failure')
            if now-last>=30:progress();last=now
            if live:time.sleep(.5)
    except BaseException:
        stop.append('supervisor_exception')
        write(output/'supervisor-error.json',dict(traceback=traceback.format_exc()),exclusive=True)
        for item in live:signal_owned(item['owner'],signal.SIGTERM)
        end=time.monotonic()+15
        while time.monotonic()<end and any(active(x['owner']) for x in live):time.sleep(.2)
        for item in live:
            if active(item['owner']):signal_owned(item['owner'],signal.SIGKILL)
            item['child'].wait(timeout=10);item['log'].close()
        live=[]
    finally:
        progress()
        write(output/'completion.json',dict(schema='originx_confirmatory_completion_v1',development=config['development'],
              planned=len(jobs),finished=len(done),complete=len(done)==len(jobs) and all(r['status']=='completed' for r in done.values()),
              all_children_drained=not live,stop_reasons=stop,unix=time.time()),exclusive=True)
    return 0 if len(done)==len(jobs) and not stop else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','preflight','episode','run'])
    p.add_argument('--output',type=Path);p.add_argument('--services-ready',type=Path)
    p.add_argument('--development',action='store_true');p.add_argument('--duration-seconds',type=int,default=432000)
    p.add_argument('--config',type=Path);p.add_argument('--config-sha');p.add_argument('--index',type=int);p.add_argument('--model-index',type=int)
    args=p.parse_args()
    if args.mode=='prepare':return prepare(args)
    config=read(args.config)
    return dict(preflight=preflight,episode=episode,run=run)[args.mode](args,config)

if __name__=='__main__':raise SystemExit(main())
