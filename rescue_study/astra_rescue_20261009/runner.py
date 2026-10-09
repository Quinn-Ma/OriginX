"""Fixed historical-failure native-reset control/Astra study.

Independent results, original seeds/horizons, frozen B2000, one observation-only
language intervention at the first query at/after half the official horizon.
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
    require(args.response_wait_seconds > 0, 'Positive bounded Astra response wait required')
    original = ROOT/'results/native-reset-b2500-20261009-v1'
    reference_config = read(original/'config.json')
    report = read(original/'analysis/report.json')
    fixed = read(args.failure_manifest)
    provenance = fixed['provenance']
    require(fixed['selected_failures'] == 1004 and len(fixed['jobs']) == 1004,
            'Need exactly the frozen original 1004-failure set')
    require(report['complete'] and report['episodes'] == 2500 and report['successes'] == 1496,
            'Original completed report differs')
    for filename, key in [('manifest.json','frozen_manifest_sha256'),
                          ('config.json','frozen_config_sha256'),
                          ('analysis/report.json','frozen_report_sha256')]:
        require(sha(original/filename) == provenance[key], 'Original reference artifact changed: '+filename)
    original_manifest = read(original/'manifest.json')
    original_jobs = {j['id']: j for j in original_manifest['jobs']}
    fixed_records = {r['id']:r for r in fixed['failure_records']}
    ordered_failures = [j for j in original_manifest['jobs']
                       if read(original/'episodes'/j['id']/'result.json')['success'] is False]
    require(fixed['jobs'] == ordered_failures, 'Failure subset/order changed')
    selected = fixed['jobs']
    if args.indices is not None:
        require(len(set(args.indices)) == len(args.indices), 'Duplicate pilot assignment')
        selected = [j for j in selected if j['assignment_index'] in args.indices]
        require(len(selected) == len(args.indices), 'Pilot index is not an original failure')
    require(selected and args.arms and len(set(args.arms)) == len(args.arms), 'Empty/duplicate study arms')
    reference_records = {}
    for job in selected:
        path = original/'episodes'/job['id']/'result.json'
        r = read(path)
        require(job == original_jobs[job['id']] and r['job'] == job and
                r['status'] == 'completed' and r['success'] is False and
                sha(path) == report['result_sha256'][str(path)] == fixed_records[job['id']]['source']['result_sha256'],
                'Original failed assignment/result differs')
        require(r['initial_scene'] == fixed_records[job['id']]['original_initial_scene'], 'Original initial scene differs')
        reference_records[job['id']] = dict(result_path=str(path), result_sha256=sha(path),
                                           initial_scene=r['initial_scene'], job=job)
    output = args.output.resolve()
    require(output.parent == ROOT/'results' and 'native-reset' in output.name and 'astra' in output.name,
            'Use a new direct results child with astra and native-reset in its name')
    require(output != original and not (output/'config.json').exists(), 'Independent new config required')
    models = []
    from continuous_eval2000_v3.client import validate_stage_endpoint
    allowed_gpus = {3,6,'3','6','GPU-d0f25b35-5423-72d4-9f45-047dfe7ed87c',
                    'GPU-35278907-aae0-4ccc-e0fc-c25cb3a8bd17'}
    parity = REFERENCE/'parity/B.json'
    for path in args.servers:
        path = path.resolve()
        srv = validate_stage_endpoint(path, parity, BASE, binding_path=str(REFERENCE/'B-endpoint.json'), binding_sha256=ENDPOINT_SHA)
        owner = srv['servers'][0]
        require(active(owner) and owner['gpu'] in allowed_gpus, 'Only owned active services on GPU3/6 are allowed')
        require(srv['hello']['stage_serving_identity']['arm'] == 'B', 'Wrong frozen model arm')
        models.append(dict(server_manifest=str(path),server_manifest_sha256=sha(path),port=owner['port'],
                           owner=owner,hello=srv['hello'],slots=6,parity_path=str(parity),parity_sha256=sha(parity)))
    require(models and len({m['port'] for m in models}) == len(models), 'Need unique admitted model servers')
    # Arm is distinct from model arm B, and each literal original job is preserved.
    jobs = [dict(j,id=j['id']+'--'+arm,original_id=j['id'],rescue_arm=arm)
            for j in selected for arm in args.arms]
    task_names = {j['task'] for j in selected}
    manifest = dict(schema='astra_native_reset_control_rescue_v1',
                    tasks=[t for t in original_manifest['tasks'] if t['task'] in task_names],
                    jobs=jobs,selected_original_jobs=selected,arms=args.arms,
                    original_failure_count=1004,full_failure_subset=len(selected)==1004,
                    smoke=False,no_seed_replacements=True,reference_records=reference_records)
    write(output/'manifest.json',manifest,exclusive=True)
    sources = dict(reference_config['source_sha256'])
    for name in ('runner.py','assistance.py','__init__.py'):
        sources[str(HERE/name)] = sha(HERE/name)
    c = {k:reference_config[k] for k in ('root','repo','package_root','python','osmesa_library','packages')}
    c.update(schema='astra_native_reset_control_rescue_config_v1',output=str(output),
             manifest=str(output/'manifest.json'),manifest_sha256=sha(output/'manifest.json'),
             models=models,source_sha256=sources,smoke=False,
             failure_manifest=str(args.failure_manifest.resolve()),failure_manifest_sha256=sha(args.failure_manifest),
             original_output=str(original),original_manifest_sha256=sha(original/'manifest.json'),
             original_config_sha256=sha(original/'config.json'),original_report_sha256=sha(original/'analysis/report.json'),
             response_wait_seconds=args.response_wait_seconds,intervention_fraction=0.5,
             protocol=dict(reference_config['protocol'],study='postselected fixed-failure conditional rescue',
                           intervention='one RGB-only Astra language plan at first fixed query >= half horizon',
                           control='frozen B2000 same-seed natural-reset replay',
                           no_stall_detector=True,no_oracle_assistance=True,full_original_horizon=True))
    check_sources(c)
    write(output/'config.json',c,exclusive=True)
    print(json.dumps(dict(config=str(output/'config.json'),models=len(models),episodes=len(jobs),
                          original_cases=len(selected),arms=args.arms)))


def preflight(args, config):
    import importlib.metadata
    setup_path(); check_sources(config)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('PYTHONHASHSEED') == '0',
            'CPU simulator with fixed native hash seed required')
    require({name:importlib.metadata.version(name) for name in config['packages']} == config['packages'],
            'Frozen simulator package versions differ')
    proof = native_reset_proof(config)
    require(sha(config['manifest']) == config['manifest_sha256'], 'Study manifest changed')
    require(sha(config['failure_manifest']) == config['failure_manifest_sha256'], 'Fixed failure artifact changed')
    original = Path(config['original_output'])
    for name, key in [('manifest.json','original_manifest_sha256'),('config.json','original_config_sha256'),
                      ('analysis/report.json','original_report_sha256')]:
        require(sha(original/name) == config[key], 'Historical reference changed')
    m = read(config['manifest'])
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    require(all(get_task_horizon(j['task']) == j['horizon'] for j in m['jobs']), 'Official horizon mismatch')
    require(len({j['id'] for j in m['jobs']}) == len(m['jobs']), 'Duplicate episode id')
    for j in m['jobs']:
        reference = m['reference_records'][j['original_id']]
        expected = dict(reference['job'],id=j['original_id']+'--'+j['rescue_arm'],
                        original_id=j['original_id'],rescue_arm=j['rescue_arm'])
        require(j == expected and j['rescue_arm'] in ('control','astra'), 'Original job/seed changed')
        require(sha(reference['result_path']) == reference['result_sha256'], 'Original episode result changed')
    for model in config['models']:
        require(active(model['owner']) and sha(model['server_manifest']) == model['server_manifest_sha256'],
                'Model owner/binding changed')
    write(Path(config['output'])/'preflight.json',dict(passed=True,native_reset=proof,
          episodes=len(m['jobs']),original_cases=len(m['selected_original_jobs']),unix=time.time()))


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
        from astra_rescue_20261009.assistance import AssistanceClient
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
            nonlocal replay
            guard.check(); core = env.unwrapped.env
            initial.update(audit.capture_initial(out,observation,entry.observation_to_state(observation),core.sim))
            initial.update(layout_id=int(core.layout_id),style_id=int(core.style_id),xml_sha256=sha(out/'initial.xml'))
            write(out/'initial_scene.json',initial)
            reference = m['reference_records'][job['original_id']]
            require(sha(reference['result_path']) == reference['result_sha256'], 'Original result changed')
            expected = reference['initial_scene']
            differences = {key:dict(expected=expected.get(key),actual=initial.get(key))
                           for key in set(expected)|set(initial) if expected.get(key) != initial.get(key)}
            replay = dict(exact_match=not differences,original_id=job['original_id'],
                          original_result_sha256=reference['result_sha256'],differences=differences,
                          natural_reset=True,state_restored=False)
            write(out/'initial_replay.json',replay)
            require(replay['exact_match'], 'Natural reset did not reconstruct the original initial fingerprints')
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
        reduced = DecimatedGym(ObservedGym,cfg)
        assistance = AssistanceClient(lazy,request_root=Path(config['output'])/'requests',episode_output=out,
                                      request_id=job['id'],case_id=job['original_id'],arm=job['rescue_arm'],
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
        if assistance is not None:
            record['assistance'] = assistance.finish()
        record['initial_replay'] = replay
        record['initial_scene'] = initial
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
    analyze(config,stop_reasons=stop)
    if args.cleanup_servers:cleanup_servers(config)
    return 0 if complete else 1

def _receipt_evidence(output, row, reference, traces):
    """Verify linked request, response and actual CLI event/output artifacts."""
    import re
    answer=dict(valid=False,reason='not_checked')
    try:
        job=row['job']; a=row['assistance']; folder=output/'requests'/job['id']
        require(a.get('applied') is True and a.get('status')=='applied', 'assistance_not_applied')
        request=read(folder/'request.json'); response=read(folder/'response.json')
        require(set(request)=={'schema','request_id','request_token','case_id','step','horizon','remaining_steps',
                               'original_instruction','cameras','permitted_inputs','oracle_inputs_included'},
                'unexpected_request_fields')
        request_sha=sha(folder/'request.json'); response_sha=sha(folder/'response.json')
        require(request_sha==a['request_sha256'] and response_sha==a['response_sha256'] and response==a['response'],
                'request_response_record_hash_mismatch')
        token=request.get('request_token')
        require(isinstance(token,str) and re.fullmatch('[0-9a-f]{32}',token), 'missing_request_token')
        require(request.get('schema')=='astra_observation_request_v1' and request.get('request_id')==job['id']
                and request.get('case_id')==job['original_id'] and request.get('horizon')==job['horizon']
                and request.get('step')==a['trigger_step'] and request.get('remaining_steps')==job['horizon']-a['trigger_step']
                and request.get('original_instruction')==reference['initial_scene']['instruction']
                and request.get('oracle_inputs_included') is False and a.get('request_token')==token,
                'request_identity_or_budget_mismatch')
        require(request.get('permitted_inputs')==['original_instruction','current_rgb','step_budget'], 'unexpected_assistance_inputs')
        cameras=request['cameras']
        require(len(cameras)==3 and [c['name'] for c in cameras]==['left','right','wrist'], 'camera_set_mismatch')
        for camera in cameras:
            require(camera['file']==camera['name']+'.png' and sha(folder/camera['file'])==camera['sha256'], 'camera_hash_mismatch')
        require(response.get('schema')=='astra_observation_response_v1' and response.get('request_id')==job['id']
                and response.get('request_sha256')==request_sha and response.get('request_token')==token
                and response.get('model')=='gpt-6-astra' and response.get('status')=='ok', 'response_binding_mismatch')
        require(response.get('cli_receipt_file')=='cli_receipt.json', 'missing_cli_receipt_file')
        receipt=read(folder/'cli_receipt.json')
        require(sha(folder/'cli_receipt.json')==response.get('cli_receipt_sha256'), 'cli_receipt_hash_mismatch')
        require(receipt.get('schema')=='astra_cli_receipt_v1' and receipt.get('cli_executed') is True
                and receipt.get('model')==receipt.get('model_requested')=='gpt-6-astra'
                and receipt.get('returncode')==0 and receipt.get('status')=='ok'
                and receipt.get('no_tool_calls') is True and receipt.get('output_validated') is True,
                'cli_receipt_execution_invalid')
        command=receipt.get('command',[])
        require(isinstance(command,list) and command.count('--model')==1
                and command[command.index('--model')+1]=='gpt-6-astra' and 'exec' in command
                and '--json' in command, 'cli_command_model_mismatch')
        executable=command[0].replace('\\','/').rsplit('/',1)[-1].lower()
        require(executable in ('codex','codex.exe','codex.cmd','codex.bat'), 'receipt_not_from_codex_executable')
        bindings=receipt.get('requests',[])
        require(sum(x.get('request_id')==job['id'] for x in bindings)==1, 'cli_receipt_request_missing_or_duplicate')
        bound=next(x for x in bindings if x.get('request_id')==job['id'])
        require(bound.get('request_sha256')==request_sha and bound.get('request_token')==token, 'cli_receipt_token_mismatch')
        require(receipt.get('stdout_file')=='cli_stdout.jsonl' and receipt.get('raw_answer_file')=='cli_final_output.json',
                'cli_artifact_names_mismatch')
        require(sha(folder/'cli_stdout.jsonl')==receipt.get('stdout_sha256')
                and sha(folder/'cli_final_output.json')==receipt.get('raw_answer_sha256'), 'cli_trace_or_output_hash_mismatch')
        final=read(folder/'cli_final_output.json')
        events=[json.loads(line) for line in (folder/'cli_stdout.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        messages=[]; completed=False; models=set()
        for event in events:
            require(isinstance(event,dict) and event.get('type') not in ('error','turn.failed'), 'cli_trace_error')
            for candidate in (event,event.get('session',{}),event.get('turn',{})):
                if isinstance(candidate,dict) and isinstance(candidate.get('model'),str):models.add(candidate['model'])
            item=event.get('item')
            if isinstance(item,dict):
                require(item.get('type') in ('agent_message','reasoning'), 'cli_trace_tool_call')
                if event.get('type')=='item.completed' and item.get('type')=='agent_message':
                    try:messages.append(json.loads(item['text']))
                    except (ValueError,KeyError):pass
            if event.get('type')=='turn.completed':
                usage=event.get('usage',{})
                completed=completed or (type(usage.get('input_tokens')) is int and usage['input_tokens']>0
                                        and type(usage.get('output_tokens')) is int and usage['output_tokens']>0)
        require(not models or models=={'gpt-6-astra'}, 'cli_trace_model_mismatch')
        require(completed and final in messages, 'cli_final_answer_not_in_completed_trace')
        items=final.get('requests',[])
        require(sum(x.get('request_id')==job['id'] for x in items)==1, 'cli_answer_request_missing_or_duplicate')
        item=next(x for x in items if x.get('request_id')==job['id'])
        require(item.get('request_token')==token and item.get('subgoal_instruction')==response.get('subgoal_instruction'),
                'cli_answer_token_or_instruction_mismatch')
        instruction=(request['original_instruction']+'\nImmediate next actions: '+response['subgoal_instruction'].strip()
                     +'\nThen complete the original task.')
        actual=[t for t in traces if t['step']==a['trigger_step']]
        require(a.get('policy_instruction')==instruction and len(actual)==1
                and actual[0]['instruction_sha256']==hashlib.sha256(instruction.encode('utf-8')).hexdigest(),
                'accepted_plan_not_observed_in_policy_trace')
        answer.update(valid=True,reason='verified',request_sha256=request_sha,response_sha256=response_sha,
                      cli_receipt_sha256=response['cli_receipt_sha256'],request_token=token,
                      trace_completed=True,model_requested='gpt-6-astra',models_reported=sorted(models),
                      model_evidence='Actual CLI command and completed JSON event trace, with matching output token; not a provider-signed attestation.')
    except Exception as error:
        answer['reason']=str(error)
    return answer


def _classify_case(pair, initial_deviation, prefix, evidence):
    """Mutually exclusive, conservative attribution; raw outcome is separate."""
    if initial_deviation:return 'initial_state_deviation','original_initial_fingerprints_differ'
    if set(pair)!={'control','astra'} or any(not r.get('_valid_completed') for r in pair.values()):
        return 'unknown','missing_invalid_or_infrastructure_episode'
    control,astra=pair['control'],pair['astra']
    a=astra.get('assistance',{})
    if not isinstance(a,dict) or a.get('status') not in ('applied','not_reached','broker_error','invalid_response','timeout','waiting'):
        return 'unknown','missing_or_invalid_assistance_record'
    if a.get('status') in ('broker_error','invalid_response','timeout','waiting'):
        return 'unknown','assistance_'+a.get('status','missing')
    if a.get('applied') is not True:
        return (('not_rescued','no_applied_intervention_natural_outcome') if a.get('status')=='not_reached'
                else ('unknown','inconsistent_assistance_record'))
    if prefix.get('exact') is not True:return 'unknown','preintervention_prefix_not_verified'
    if evidence.get('valid') is not True:return 'unknown','astra_invocation_evidence_not_verified'
    if control['success']:return 'not_rescued','control_already_succeeded'
    if not astra['success']:return 'not_rescued','assisted_episode_failed'
    return 'rescued','control_failed_astra_succeeded_with_verified_intervention'


def analyze(config, *, stop_reasons=None, supervisor_error=None):
    output=Path(config['output']); m=read(config['manifest']); cs=sha(output/'config.json')
    require(sha(config['manifest'])==config['manifest_sha256'], 'Study manifest changed during analysis')
    fixed=read(config['failure_manifest'])
    require(len(fixed['jobs'])==1004 and sha(config['failure_manifest'])==config['failure_manifest_sha256'],
            'Fixed 1004-case denominator artifact differs')
    selected={j['id'] for j in m['selected_original_jobs']}
    all_ids=[j['id'] for j in fixed['jobs']]
    require(len(set(all_ids))==1004,'Duplicate fixed failure identity')
    require(selected<=set(all_ids), 'Study selected cases outside fixed failure set')
    by_case={case_id:{} for case_id in all_ids}; hashes={}; arms={}; episode_states=[]
    for job in m['jobs']:
        path=output/'episodes'/job['id']/'result.json'
        row=dict(job=job,status='pending',success=None,_valid_completed=False)
        if path.exists():
            try:
                parsed=read(path); hashes[str(path)]=sha(path)
                require(isinstance(parsed,dict),'Episode result is not an object')
                row=parsed
                require(row['job']==job and row['config_sha256']==cs, 'Episode authority differs')
                if row.get('status')=='completed':
                    ep=row['stats']['episodes'][0]
                    require(type(row.get('success')) is bool and ep['success']==row['success']
                            and ep['seed']==job['env_seed'] and row['stats']['horizon']==job['horizon']
                            and 1<=ep['steps']<=job['horizon'] and (row['success'] or ep['steps']==job['horizon']),
                            'Incomplete or altered official outcome')
                    require(row['native_reset_before']==row['native_reset_after']
                            and row['native_reset_before']['native_reset'] is True
                            and row['initial_replay']['exact_match'] is True, 'Missing exact original native reset')
                    expected=m['reference_records'][job['original_id']]['initial_scene']
                    require(row['initial_scene']==expected,'Initial scene differs despite claimed exact match')
                    row['_valid_completed']=True
                else:row['_valid_completed']=False
            except Exception as error:
                row.update(_valid_completed=False,_analysis_error=str(error))
        elif (output/'claims'/(job['id']+'.json')).exists():
            row['status']='missing_result'
        row.setdefault('status','invalid_result')
        row.setdefault('success',None)
        row['_expected_job']=job
        by_case[job['original_id']][job['rescue_arm']]=row
        episode_states.append(row)
    for arm in m['arms']:
        subset=[r for r in episode_states if r['_expected_job']['rescue_arm']==arm]
        completed=[r for r in subset if r.get('_valid_completed')]
        pending=sum(r['status']=='pending' for r in subset)
        successes=sum(r['success'] for r in completed)
        arms[arm]=dict(planned=len(subset),completed=len(completed),pending=pending,
                       unknown=len(subset)-len(completed)-pending,successes=successes,
                       success_rate=successes/len(subset) if len(completed)==len(subset) else None,
                       observed_successes_over_planned=successes/len(subset),
                       assistance_status=dict(Counter(r.get('assistance',{}).get('status','missing') for r in subset)),
                       wall_seconds=sum(r.get('wall_seconds',0) for r in subset))
        arms[arm]['tasks']={task:dict(planned=len(v),completed=sum(r.get('_valid_completed',False) for r in v),
                                     successes=sum(r['success'] for r in v if r.get('_valid_completed')))
                            for task in sorted({r['_expected_job']['task'] for r in subset})
                            if (v:=[r for r in subset if r['_expected_job']['task']==task])}
    cases=[]; prefix_checks=[]; raw_pairs=[]
    for case_id in all_ids:
        pair=by_case[case_id]; prefix=dict(case_id=case_id,exact=None,reason='not_available')
        evidence=dict(valid=False,reason='no_applied_intervention')
        deviations=[]; traces={}
        for arm,row in pair.items():
            replay=row.get('initial_replay')
            expected=m['reference_records'][case_id]['initial_scene']
            if (isinstance(replay,dict) and replay.get('exact_match') is False) or (row.get('initial_scene') and row['initial_scene']!=expected):
                deviations.append(arm)
        if set(pair)=={'control','astra'} and all(r.get('_valid_completed') for r in pair.values()):
            raw_pairs.append(pair)
            try:
                trigger=((pair['astra']['job']['horizon']+31)//32)*16
                for arm,row in pair.items():
                    trace_path=output/'episodes'/row['job']['id']/'query_trace.jsonl'
                    values=[json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines() if line.strip()]
                    expected_steps=list(range(0,row['stats']['episodes'][0]['steps'],16))
                    require([v['step'] for v in values]==expected_steps,'missing_or_duplicate_policy_trace')
                    traces[arm]=values
                a=[r for r in traces['astra'] if r['step']<trigger]
                b=[r for r in traces['control'] if r['step']<trigger]
                prefix.update(exact=a==b and bool(a),reason='matched' if a==b and a else 'prefix_mismatch',
                              queries_astra=len(a),queries_control=len(b),trigger_step=trigger)
                if pair['astra'].get('assistance',{}).get('applied') is True:
                    require(pair['astra']['assistance']['trigger_step']==trigger,'intervention_threshold_changed')
                    evidence=_receipt_evidence(output,pair['astra'],m['reference_records'][case_id],traces['astra'])
            except Exception as error:
                prefix.update(exact=False,reason=str(error))
        if case_id not in selected:
            classification,reason='unknown','not_selected_in_this_run'
        else:classification,reason=_classify_case(pair,bool(deviations),prefix,evidence)
        if case_id in selected:prefix_checks.append(prefix)
        cases.append(dict(case_id=case_id,selected=case_id in selected,classification=classification,reason=reason,
                          control_success=pair.get('control',{}).get('success'),astra_success=pair.get('astra',{}).get('success'),
                          arm_status={arm:dict(status=row.get('status'),valid_completed=row.get('_valid_completed',False),
                                              analysis_error=row.get('_analysis_error')) for arm,row in pair.items()},
                          assistance_status=pair.get('astra',{}).get('assistance',{}).get('status'),
                          initial_deviation_arms=deviations,prefix=prefix,astra_evidence=evidence))
    labels=('rescued','not_rescued','unknown','initial_state_deviation')
    fixed_counts={name:sum(r['classification']==name for r in cases) for name in labels}
    selected_counts={name:sum(r['selected'] and r['classification']==name for r in cases) for name in labels}
    complete=all(r.get('_valid_completed') for r in episode_states)
    strict=selected_counts['rescued']; selected_denominator=len(selected)
    write(output/'analysis/prefix-parity.json',dict(all_exact=bool(prefix_checks) and all(x['exact'] is True for x in prefix_checks),pairs=prefix_checks))
    write(output/'analysis/case-classification.json',dict(schema='astra_strict_case_classification_v1',
          fixed_denominator=1004,selected_denominator=selected_denominator,cases=cases))
    result=dict(schema='astra_native_reset_conditional_rescue_report_v2',complete=complete,
                full_failure_subset=m['full_failure_subset'],selected_original_cases=selected_denominator,
                episodes=sum(r.get('status')!='pending' for r in episode_states),planned_episodes=len(m['jobs']),
                original_episodes=2500,original_successes=1496,original_failures=1004,arms=arms,
                denominators=dict(fixed_original_failure_cases=1004,selected_cases=selected_denominator,
                                  not_selected_cases=1004-selected_denominator,planned_episodes=len(m['jobs'])),
                classification_fixed_1004=fixed_counts,classification_selected=selected_counts,strict_rescued=strict,
                strict_rescued_fraction_of_fixed_1004=strict/1004,
                strict_rescued_fraction_of_selected=strict/selected_denominator,
                strict_rescue_rate_selected=(strict/selected_denominator if selected_counts['unknown']==selected_counts['initial_state_deviation']==0 else None),
                fraction_note='Confirmed rescued counts divided by prespecified denominators; fractions are lower bounds while unknown/deviation cases remain.',
                paired=dict(label='raw completed paired outcomes; astra_only is not the strict rescued count',
                            cases=len(raw_pairs),both_success=sum(p['control']['success'] and p['astra']['success'] for p in raw_pairs),
                            control_only=sum(p['control']['success'] and not p['astra']['success'] for p in raw_pairs),
                            astra_only=sum(not p['control']['success'] and p['astra']['success'] for p in raw_pairs),
                            both_failure=sum(not p['control']['success'] and not p['astra']['success'] for p in raw_pairs)),
                preintervention_prefix_exact=bool(prefix_checks) and all(x['exact'] is True for x in prefix_checks),
                strict_classification_complete=selected_counts['unknown']==selected_counts['initial_state_deviation']==0,
                stop_reasons=stop_reasons or [],supervisor_error=supervisor_error,score_emitted=False,
                protocol=config['protocol'],config_sha256=cs,manifest_sha256=config['manifest_sha256'],result_sha256=hashes,
                limits=['Postselected historical-failure conditional rescue study; not a new 2500-episode benchmark score.',
                        'Original successful cases were not retested, so regression on them is unknown.',
                        'Raw astra_only includes natural retry outcomes and is never labeled rescued.',
                        'Strict rescue requires paired control failure, assisted success, exact initial/prefix identity, and linked actual Astra CLI request/response/event evidence.',
                        'Fixed midpoint intervention, not an outcome-dependent stall detector.',
                        'Unknown, pending, unselected and initial-state-deviation cases remain in the fixed 1004-case listing.'])
    write(output/'analysis/report.json',result)
    print(json.dumps(dict(complete=complete,selected_cases=selected_denominator,strict_rescued=strict,
                         classification_selected=selected_counts,classification_fixed_1004=fixed_counts)))
    return result


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
        analyze(config,stop_reasons=['supervisor_exception'],supervisor_error=error)
        if args.cleanup_servers:
            try: cleanup_servers(config)
            except BaseException: write(output/'cleanup-error.json',dict(error=traceback.format_exc(),unix=time.time()))
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','preflight','episode','run','analyze'])
    p.add_argument('--output',type=Path);p.add_argument('--servers',nargs='+',type=Path)
    p.add_argument('--failure-manifest',type=Path,default=HERE/'failure_manifest.json')
    p.add_argument('--indices',nargs='+',type=int)
    p.add_argument('--arms',nargs='+',choices=['control','astra'],default=['control','astra'])
    p.add_argument('--response-wait-seconds',type=int,default=1200)
    p.add_argument('--config',type=Path);p.add_argument('--config-sha');p.add_argument('--index',type=int)
    p.add_argument('--parallel',type=int,default=72);p.add_argument('--resume',action='store_true')
    p.add_argument('--duration-seconds',type=int,default=13500);p.add_argument('--cleanup-servers',action='store_true')
    a=p.parse_args()
    if a.mode=='prepare':return prepare(a)
    c=read(a.config)
    if a.mode=='preflight':return preflight(a,c)
    if a.mode=='episode':return episode(a,c)
    if a.mode=='run':return run_guarded(a,c)
    if a.mode=='analyze':return analyze(c)

if __name__=='__main__':raise SystemExit(main())
