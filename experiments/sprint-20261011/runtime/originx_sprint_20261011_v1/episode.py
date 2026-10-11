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
        from lora_dev150_final_v1.client import AdapterEvalClient
        from lora_dev150_final_v1.server import check_lora_hello
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
            cls = StageEvalClient if job['policy_id'] == 'B' else AdapterEvalClient if job['policy_id']=='A' else RemoteEvalClient
            p = cls(config['repo'],BASE,model['port'],model['hello'],telemetry_path=out/'rpc.jsonl')
            p.hello = p.wire.hello
            return p
        def validate(hello, _):
            check_hello(hello,model['hello'])
            if job['policy_id'] == 'B': check_stage_hello(hello,model['hello'])
            elif job['policy_id']=='A':
                check_lora_hello(hello,model['hello']);require(not hello.get('stage_serving_identity'),'A-only includes B')
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha',required=True);parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--model-index',type=int,required=True);args=parser.parse_args()
    return episode(args,read(args.config))

if __name__=='__main__':raise SystemExit(main())
