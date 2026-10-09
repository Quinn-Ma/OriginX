"""Native RoboCasa reset; frozen B2000, five-step policy, official target50 horizons.

Reuses the already validated rendering/RPC helpers, never the development reset
patch. Historical runners, manifests, bindings and results are read-only inputs.
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
    return json.loads(Path(path).read_text())

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
        return f[0] not in ('Z', 'X') and int(f[19]) == owner['process_start_ticks']
    except FileNotFoundError:
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

def prepare(args):
    setup_path()
    require(sha(OLD/'manifest.json') == OLD_MANIFEST_SHA, 'Frozen task schedule changed')
    require(sha(REFERENCE/'B-endpoint.json') == ENDPOINT_SHA, 'Frozen B endpoint changed')
    old = read(REFERENCE/'B-bindings.json')
    output = args.output.resolve()
    require(output.parent == ROOT/'results', 'Use a new direct results child')
    require('native-reset' in output.name, 'Native evaluation must have an explicit separate name')
    m = read(OLD/'manifest.json')
    if args.smoke:
        tasks = [dict(task='CloseDrawer', stratum='development_smoke', horizon=450),
                 dict(task='PackDessert', stratum='development_smoke', horizon=600)]
        jobs = [dict(assignment_index=i, task=t['task'], horizon=t['horizon'],
                     env_seed=2090800000+i, policy_seed=2090800000+i, arm='B', split='pretrain')
                for i,t in enumerate(tasks)]
    else:
        tasks = m['tasks']
        jobs = [dict(j, env_seed=2090900000+i, policy_seed=2090900000+i)
                for i,j in enumerate(m['jobs'])]
    for i,j in enumerate(jobs):
        j['id'] = 'native-B-%04d-%s-%d' % (i,j['task'],j['env_seed'])
    manifest = dict(schema='native_reset_B2000_v1', tasks=tasks, jobs=jobs,
                    smoke=args.smoke, split='pretrain', no_seed_replacements=True,
                    seed_range=[jobs[0]['env_seed'],jobs[-1]['env_seed']])
    manifest_path = output/'manifest.json'
    write(manifest_path,manifest,exclusive=True)
    models = []
    from continuous_eval2000_v3.client import validate_stage_endpoint
    for path in args.servers:
        path = path.resolve()
        parity = REFERENCE/'parity/B.json'
        s = validate_stage_endpoint(path, parity, BASE, binding_path=str(REFERENCE/'B-endpoint.json'), binding_sha256=ENDPOINT_SHA)
        owner = s['servers'][0]
        require(active(owner), 'Server is not live')
        require(s['hello']['stage_serving_identity']['arm'] == 'B', 'Wrong frozen model arm')
        require(owner['gpu'] not in (2,'2'), 'Quarantined GPU2 is forbidden')
        models.append(dict(server_manifest=str(path),server_manifest_sha256=sha(path),
                           port=owner['port'],owner=owner,hello=s['hello'],slots=6,
                           parity_path=str(parity),parity_sha256=sha(parity)))
    require(models and len({m['port'] for m in models}) == len(models), 'Need unique admitted model servers')
    config = {k:old[k] for k in ('root','repo','package_root','python','osmesa_library','packages')}
    # Small sources are hashed; model assets are independently pinned by the
    # frozen endpoint, loader and exact parity manifest above.
    sources = dict(old['source_sha256'])
    sources.update({str(OLD/n):h for n,h in read(OLD/'runner-source-pins.json').items()})
    sources.update({str(p):sha(p) for p in HERE.glob('*.py')})
    sources[str(Path(config['repo'])/'eval_robocasa365/entry.py')] = ENTRY_SHA
    sources[str(Path(config['package_root'])/'models/fixtures/counter.py')] = COUNTER_SHA
    sources[str(REFERENCE/'B-endpoint.json')] = ENDPOINT_SHA
    config.update(schema='native_reset_B2000_config_v1',output=str(output),manifest=str(manifest_path),
                  manifest_sha256=sha(manifest_path),models=models,source_sha256=sources,
                  smoke=args.smoke,seed_range=manifest['seed_range'],
                  protocol=dict(reset='unmodified Counter.get_reset_regions',pythonhashseed=0,
                                split='pretrain',replan=16,history=4,interval=2,crop=0.95,Euler=5,
                                horizon='official 1.0.1 get_task_horizon',render='CPU OSMesa, query-pixel-preserving decimation',
                                organizer_verified=False,automatic_submission=False))
    check_sources(config)
    write(output/'config.json',config,exclusive=True)
    print(json.dumps(dict(config=str(output/'config.json'),models=len(models),episodes=len(jobs))))

def preflight(args, config):
    setup_path()
    check_sources(config)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Simulator must be CPU only')
    require(os.environ.get('PYTHONHASHSEED') == '0', 'Require reproducible native hash seed')
    proof = native_reset_proof(config)
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    m = read(config['manifest'])
    require(sha(config['manifest']) == config['manifest_sha256'], 'Manifest changed')
    require(all(get_task_horizon(t['task']) == t['horizon'] for t in m['tasks']), 'Official horizon mismatch')
    if not config['smoke']:
        require(set(TASK_SET_REGISTRY['target50']) == {t['task'] for t in m['tasks']}, 'Official target50 mismatch')
        require(len(m['jobs']) == 2500 and Counter(j['task'] for j in m['jobs']) == dict.fromkeys(TASK_SET_REGISTRY['target50'],50), 'Expected 50x50')
        require(Counter(t['stratum'] for t in m['tasks']) == {'atomic_seen':18,'composite_seen':16,'composite_unseen':16}, 'Strata mismatch')
    for model in config['models']:
        require(active(model['owner']) and sha(model['server_manifest']) == model['server_manifest_sha256'], 'Model owner/binding changed')
    write(Path(config['output'])/'preflight.json',dict(passed=True,native_reset=proof,episodes=len(m['jobs']),unix=time.time()))

def episode(args, config):
    setup_path()
    check_sources(config)
    require(sha(args.config) == args.config_sha, 'Episode config changed')
    require(sha(config['manifest']) == config['manifest_sha256'], 'Manifest changed')
    m = read(config['manifest']); job = m['jobs'][args.index]
    out = Path(config['output'])/'episodes'/job['id']
    out.mkdir(parents=True,exist_ok=False)
    write(out/'owner.json',identity(),exclusive=True)
    started = time.monotonic(); guard = lazy = None; events = []; initial = {}; record = None
    try:
        import numpy as np
        import torch
        import gymnasium as gym
        import robocasa
        import dev_randomized_broad_alpha as legacy
        import dev_repeat_audit as audit
        from candidate import DecimatedGym, QueryCheckedClient
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
        model = config['models'][args.index % len(config['models'])]
        require(sha(model['server_manifest']) == model['server_manifest_sha256'] and active(model['owner']), 'Model binding changed')
        def event(name, **fields):
            events.append(dict(event=name,sequence=len(events),monotonic=time.monotonic(),**fields))
        def make_policy(_):
            p = StageEvalClient(config['repo'],BASE,model['port'],model['hello'],telemetry_path=out/'rpc.jsonl')
            p.hello = p.wire.hello
            return p
        def validate(hello, _):
            check_hello(hello,model['hello']); check_stage_hello(hello,model['hello'])
        lazy = legacy.LazyClient(lambda:dict(name='B'),make_policy,validate,job['policy_seed'],event)
        def capture(env, observation):
            guard.check(); core = env.unwrapped.env
            initial.update(audit.capture_initial(out,observation,entry.observation_to_state(observation),core.sim))
            initial.update(layout_id=int(core.layout_id),style_id=int(core.style_id),xml_sha256=sha(out/'initial.xml'))
            write(out/'initial_scene.json',initial)
        observers = []
        class Observer(legacy.ResetObserver):
            steps = 0
            def __getattr__(self,name): return getattr(self.env,name)
            def step(self,action):
                r = self.env.step(action); guard.check(); self.steps += 1; return r
        class ObservedGym:
            @staticmethod
            def make(*a,**kw):
                require(not observers,'One natural-reset environment per assignment')
                obj=Observer(gym.make(*a,**kw),lazy,capture,event); observers.append(obj); return obj
        cfg = entry.parse_args(['--model-path',BASE,'--split','pretrain','--task-set','target50',
                                '--task-name',job['task'],'--num-trials','1','--seed',str(job['env_seed']),
                                '--replan-steps','16','--obs-history','4','--obs-interval','2','--crop-ratio','0.95'])
        entry.validate_args(cfg)
        reduced = DecimatedGym(ObservedGym,cfg); checked = QueryCheckedClient(lazy,reduced)
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
                stats['horizon']==job['horizon'] and (ep['success'] or ep['steps']==job['horizon']),'Incomplete official episode')
        require(rng['requests_since_reset']==lazy.infer_calls==(ep['steps']+15)//16 and not torch.cuda.is_initialized(),'Policy query/RNG/CPU execution changed')
        record=dict(status='completed',success=ep['success'],stats=stats,native_reset_before=before,native_reset_after=after,
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
        record.update(job=job,config_sha256=args.config_sha,manifest_sha256=config['manifest_sha256'],
                      endpoint_index=args.index%len(config['models']),events=events,wall_seconds=time.monotonic()-started)
        write(out/'result.json',record,exclusive=True)
    return 0 if record['status']=='completed' else 1

def run(args, config):
    import fcntl
    setup_path(); check_sources(config)
    output = Path(config['output']); m=read(config['manifest']); jobs=m['jobs']; cs=sha(args.config)
    require(sha(config['manifest'])==config['manifest_sha256'],'Manifest changed')
    require(read(output/'preflight.json')['passed'],'Run preflight first')
    lock=(output/'run.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args._run_lock=lock
    require(not (output/'launch.json').exists() or args.resume,'Prior launch exists; inspect before explicit resume')
    require(args.duration_seconds>=60,'At least 60 seconds duration required')
    launch=dict(identity(),config_sha256=cs,manifest_sha256=config['manifest_sha256'],unix=time.time(),duration_seconds=args.duration_seconds)
    if not args.resume: write(output/'launch.json',launch,exclusive=True)
    else: require(read(output/'launch.json')['config_sha256']==cs,'Resume config changed')
    models=config['models']; slots=min(args.parallel,len(models)*6)
    require(slots>0,'Positive concurrency required')
    pending=[]; records={}
    for i,j in enumerate(jobs):
        claim=output/'claims'/(j['id']+'.json'); result=output/'episodes'/j['id']/'result.json'
        if claim.exists():
            require(args.resume and result.exists(),'Claim without result is not automatically retryable')
            r=read(result); require(r['job']==j and r['config_sha256']==cs and r['status']=='completed','Unresolved or changed result blocks resume')
            require(not active(read(output/'episodes'/j['id']/'owner.json')),'Claimed episode remains active')
            records[i]=r
        else: pending.append(i)
    args._owns_run=True
    active_items=[]; stop=[]; started=time.monotonic(); hard_deadline=started+args.duration_seconds; term_sent=False; kill_sent=False
    signal.signal(signal.SIGTERM,lambda *_:stop.append('natural_drain_signal'))
    signal.signal(signal.SIGINT,lambda *_:stop.append('natural_drain_signal'))
    def progress():
        write(output/'progress.json',dict(planned=len(jobs),attempted=len(records),completed=sum(r['status']=='completed' for r in records.values()),
              infrastructure_unknown=sum(r['status']!='completed' for r in records.values()),active=len(active_items),pending=len(pending),
              elapsed_seconds=time.monotonic()-started,stop_reasons=stop,unix=time.time(),score_emitted=False))
    last=0
    while active_items or (pending and not stop):
        now=time.monotonic()
        if now>=hard_deadline-min(900,args.duration_seconds/10) and not stop:stop.append('deadline_stop_new_dispatch')
        if now>=hard_deadline and not term_sent:
            stop.append('hard_deadline_owned_workers_sigterm')
            for item in active_items:signal_owned(item['owner'],signal.SIGTERM)
            term_sent=True
        if now>=hard_deadline+10 and not kill_sent:
            for item in active_items:signal_owned(item['owner'],signal.SIGKILL)
            kill_sent=True
        if (output/'drain.request.json').exists() and not stop:stop.append('explicit_drain_request')
        used=Counter(i['index']%len(models) for i in active_items)
        while len(active_items)<slots and pending and not stop:
            index=next((i for i in pending if used[i%len(models)]<6),None)
            if index is None:break
            j=jobs[index]; command=[config['python'],'-u',str(HERE/'runner.py'),'episode','--config',str(args.config.resolve()),'--config-sha',cs,'--index',str(index)]
            write(output/'claims'/(j['id']+'.json'),dict(job=j,command=command,config_sha256=cs),exclusive=True)
            logpath=output/'logs'/(j['id']+'.log'); logpath.parent.mkdir(exist_ok=True); log=logpath.open('x')
            child=subprocess.Popen(command,cwd=ROOT,env=environment(config),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child_owner=identity(child.pid)
            child_owner.update(command=command,cwd=str(ROOT))
            write(output/'processes'/(j['id']+'.json'),child_owner,exclusive=True)
            active_items.append(dict(index=index,child=child,owner=child_owner,log=log)); pending.remove(index); used[index%len(models)]+=1
        for item in list(active_items):
            rc=item['child'].poll()
            if rc is None:continue
            item['log'].close(); index=item['index']; j=jobs[index]; path=output/'episodes'/j['id']/'result.json'
            try:
                r=read(path); require(r['job']==j and r['config_sha256']==cs,'Episode authority differs')
                require(r['status'] in ('completed','infrastructure_unknown') and (rc==0)==(r['status']=='completed'),'Episode exit/status differ')
            except BaseException:
                r=dict(job=j,status='infrastructure_unknown',success=None,traceback=traceback.format_exc(),returncode=rc)
                write(output/'errors'/(j['id']+'.json'),r,exclusive=True)
            records[index]=r; active_items.remove(item)
            if r['status']!='completed':stop.append('infrastructure_unknown')
        if time.monotonic()-last>5:progress();last=time.monotonic()
        if active_items:time.sleep(.25)
    progress()
    complete=len(records)==len(jobs) and all(r['status']=='completed' for r in records.values())
    write(output/'completion.json',dict(complete=complete,episodes=len(records),unix=time.time(),stop_reasons=stop,all_children_drained=True))
    if complete: analyze(config)
    else:
        write(output/'analysis/report.json',dict(schema='native_reset_B2000_report_v1',complete=False,smoke=config['smoke'],
              planned=len(jobs),completed=sum(r['status']=='completed' for r in records.values()),
              infrastructure_unknown=sum(r['status']!='completed' for r in records.values()),not_attempted=len(pending),
              score_emitted=False,stop_reasons=stop,config_sha256=cs,manifest_sha256=config['manifest_sha256']))
    if args.cleanup_servers:cleanup_servers(config)
    return 0 if complete else 1

def analyze(config):
    output=Path(config['output']); m=read(config['manifest']); cs=sha(output/'config.json')
    rows=[]; hashes={}
    for job in m['jobs']:
        p=output/'episodes'/job['id']/'result.json'; r=read(p)
        require(r['job']==job and r['config_sha256']==cs and r['status']=='completed' and type(r['success']) is bool,'Need all original completed assignments')
        require(r['native_reset_before']==r['native_reset_after'] and r['native_reset_before']['native_reset'],'Missing unpatched reset proof')
        hashes[str(p)]=sha(p);rows.append(r)
    tasks={}
    for t in m['tasks']:
        subset=[r for r in rows if r['job']['task']==t['task']]
        tasks[t['task']]=dict(t,episodes=len(subset),successes=sum(r['success'] for r in subset),success_rate=sum(r['success'] for r in subset)/len(subset))
    strata={s:dict(tasks=len(v),success_rate=sum(t['success_rate'] for t in v)/len(v)) for s in sorted({t['stratum'] for t in m['tasks']}) if (v:=[t for t in tasks.values() if t['stratum']==s])}
    report=dict(schema='native_reset_B2000_report_v1',complete=True,smoke=config['smoke'],episodes=len(rows),tasks=tasks,strata=strata,
                overall_task_mean=sum(t['success_rate'] for t in tasks.values())/len(tasks),successes=sum(r['success'] for r in rows),
                native_reset=True,protocol=config['protocol'],config_sha256=cs,manifest_sha256=config['manifest_sha256'],result_sha256=hashes,
                limits=['Local execution of the official task/split/horizon protocol; organizer acceptance and leaderboard submission have not occurred.',
                        'Frozen B2000 weights include prior adapters; no claim of training exclusively on the official baseline dataset.',
                        'CPU OSMesa and previously validated query-pixel-preserving rendering optimization are disclosed.'])
    write(output/'analysis/report.json',report)
    print(json.dumps({k:report[k] for k in ('complete','smoke','episodes','overall_task_mean','successes')}))

def run_guarded(args, config):
    """A supervisor exception also closes this launch's exact worker identities."""
    try:
        return run(args, config)
    except BaseException:
        if not getattr(args,'_owns_run',False):
            raise
        error=traceback.format_exc(); output=Path(config['output']); owners=[]
        # Only workers recorded by this independent runner, under this exact
        # config path. Never use a host-wide process-name match.
        for p in (output/'processes').glob('*.json'):
            owner=read(p); command=owner.get('command',[])
            if str(HERE/'runner.py') in command and str(args.config.resolve()) in command and active(owner):owners.append(owner)
        cleanup_errors=[]
        for owner in owners:
            try: signal_owned(owner,signal.SIGTERM)
            except BaseException: cleanup_errors.append(traceback.format_exc())
        deadline=time.monotonic()+10
        while time.monotonic()<deadline and any(active(o) for o in owners):time.sleep(.2)
        for owner in owners:
            if active(owner):
                try: signal_owned(owner,signal.SIGKILL)
                except BaseException: cleanup_errors.append(traceback.format_exc())
        write(output/'supervisor-error.json',dict(error=error,cleanup_errors=cleanup_errors,
              own_worker_pids=[o['pid'] for o in owners],unix=time.time()))
        write(output/'analysis/report.json',dict(schema='native_reset_B2000_report_v1',complete=False,
              smoke=config['smoke'],score_emitted=False,supervisor_error=error,
              limits=['Aborted infrastructure execution; missing episodes are not policy failures.']))
        if args.cleanup_servers:
            try: cleanup_servers(config)
            except BaseException: write(output/'cleanup-error.json',dict(error=traceback.format_exc(),unix=time.time()))
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','preflight','episode','run','analyze'])
    p.add_argument('--output',type=Path);p.add_argument('--servers',nargs='+',type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--config',type=Path);p.add_argument('--config-sha');p.add_argument('--index',type=int)
    p.add_argument('--parallel',type=int,default=108);p.add_argument('--resume',action='store_true')
    p.add_argument('--duration-seconds',type=int,default=13500);p.add_argument('--cleanup-servers',action='store_true')
    a=p.parse_args()
    if a.mode=='prepare':return prepare(a)
    c=read(a.config)
    if a.mode=='preflight':return preflight(a,c)
    if a.mode=='episode':return episode(a,c)
    if a.mode=='run':return run_guarded(a,c)
    if a.mode=='analyze':return analyze(c)

if __name__=='__main__':raise SystemExit(main())
