#!/usr/bin/env python3
"""Sixteen prespecified natural-reset randomized development trials, no retries.

Only the new delayed client/observer is introduced. The official evaluator,
processor, observations, horizons and action execution remain unchanged.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import signal
import subprocess
import sys
import tempfile
import time
import traceback

try:
    from . import dev_paired as common
    from . import dev_repeat_audit as audit
except ImportError:
    import dev_paired as common
    import dev_repeat_audit as audit

DESIGN_SHA256 = 'df9cb335933a0f9337df7be85e29014803e25581815605d7bfa4443cf33569f5'
HELPERS = {
    'dev_paired.py': 'de47225b8fe5146d997d2f4936d87d89b7bffcc4316ed29c1839945a19759c29',
    'dev_extend.py': 'd1cbf546043be47e451a487690a650bf3b46d25c090f9b62f220912a83572795',
    'dev_repeat_audit.py': 'f35d79e8a3d193a332c1c33b1a8d56bb94b01c508c1f3fb52952e634faf23588',
}
TASKS = ('ScalePortioning', 'MicrowaveThawingFridge')
ARMS = ('base', 'alpha025')
EVENT_ORDER = ('reset_requested', 'reset_completed', 'initial_fingerprints_recorded',
               'assignment_selected', 'server_connected', 'rng_ack', 'first_infer')


def check_helpers():
    loaded = (common, audit.reference_tools, audit)
    for module in loaded:
        path = Path(module.__file__).resolve()
        if path.parent != Path(__file__).resolve().parent or common.sha(path) != HELPERS[path.name]:
            raise ValueError('Frozen helper path/content differs: ' + str(path))


def load_design(path):
    if common.sha(path) != DESIGN_SHA256:
        raise ValueError('Randomized design is not the frozen approved JSON')
    design = json.loads(Path(path).read_text())
    jobs = design['jobs']
    expected = [(task, seed) for seed in range(800001, 800009) for task in TASKS]
    if ([(j['task'], j['env_seed']) for j in jobs] != expected or
            len({j['trial_id'] for j in jobs}) != 16 or
            any(j['policy_seed'] != j['env_seed'] or j['model'] not in ARMS for j in jobs)):
        raise ValueError('Invalid fixed trial schedule')
    for task in TASKS:
        for arm in ARMS:
            if sum(j['task'] == task and j['model'] == arm for j in jobs) != 4:
                raise ValueError('Randomized task/arm allocation is unbalanced')
    return design


@contextmanager
def preserve_local_rng(evidence):
    """Preserve only local global RNGs, never the remote acknowledged RNG reset."""
    import numpy as np
    import torch
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    def fingerprint():
        return dict(python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                    numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                    torch_cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    evidence['before'] = fingerprint()
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        evidence['after'] = fingerprint()
        if evidence['before'] != evidence['after']:
            raise RuntimeError('Local RNG restoration failed')


def validate_hello(hello, model, run):
    if (hello.get('protocol') != 'robocasa-recovery-rng-v1' or
            hello.get('exclusive_connection') is not True or
            not hello.get('server_instance') or not hello.get('connection_id') or
            Path(hello.get('model_path', '')).resolve() != Path(model['model_path']) or
            hello.get('model_config_sha256') != model['asset_sha256']['config.json'] or
            hello.get('official_server_sha256') != run['official_server_sha256'] or
            hello.get('recovery_server_sha256') != run['recovery_server_sha256']):
        raise ValueError('Actual hello is not the declared dedicated model/server')


class LazyClient:
    """No model choice, processor or connection until the reset observer is ready."""
    def __init__(self, select_model, make_policy, validate, seed, event, instrument=None):
        self.select_model, self.make_policy, self.validate = select_model, make_policy, validate
        self.seed, self.event, self.instrument = seed, event, instrument
        self.policy = self.model = self.forward = self.rng_ack = None
        self.local_rng, self.infer_calls, self.closed = {}, 0, False

    def activate(self, observer):
        if (observer.client is not self or not observer.reset_complete or
                not observer.fingerprints_recorded or self.policy is not None or self.closed):
            raise RuntimeError('Client activation requires one completed, captured natural reset')
        self.model = self.select_model()
        self.event('assignment_selected', model=self.model['name'])
        with preserve_local_rng(self.local_rng):
            self.policy = self.make_policy(self.model)
            self.validate(self.policy.hello, self.model)
            self.event('server_connected', model=self.model['name'], hello=self.policy.hello)
            self.rng_ack = self.policy.reset(seed=self.seed)
            hello, ack = self.policy.hello, self.rng_ack
            if (ack.get('seed') != self.seed or ack.get('requests_since_reset') != 0 or
                    any(ack.get(k) != hello.get(k) for k in ('server_instance', 'connection_id')) or
                    not ack.get('torch_cpu_rng_sha256') or not ack.get('torch_cuda_rng_sha256')):
                self.rng_ack = None
                raise ValueError('RNG acknowledgment does not match the same connection/seed')
            self.event('rng_ack', acknowledgment=ack)
            self.forward = self.instrument(self.policy) if self.instrument else self.policy.eval

    def infer(self, *args, **kwargs):
        if self.closed or self.forward is None or self.rng_ack is None:
            raise RuntimeError('Inference before post-reset activation and RNG acknowledgment')
        if self.infer_calls == 0:
            self.event('first_infer')
        self.infer_calls += 1
        return self.forward.infer(*args, **kwargs)

    def close(self):
        if not self.closed and self.policy is not None:
            self.policy.close()
        self.closed = True


class ResetObserver:
    def __init__(self, env, client, capture, event):
        self.env, self.client, self.capture, self.event = env, client, capture, event
        self.reset_calls, self.reset_complete, self.fingerprints_recorded = 0, False, False

    def reset(self, *args, **kwargs):
        if self.reset_calls:
            raise RuntimeError('A one-episode child may reset only once')
        self.reset_calls += 1
        self.event('reset_requested')
        observation, info = self.env.reset(*args, **kwargs)
        self.reset_complete = True
        self.event('reset_completed')
        self.capture(self.env, observation)
        self.fingerprints_recorded = True
        self.event('initial_fingerprints_recorded')
        self.client.activate(self)
        return observation, info

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()


def randomized_summary(jobs, records):
    planned = {j['trial_id']: j for j in jobs}
    if len(planned) != len(jobs):
        raise ValueError('Duplicate planned trial')
    seen = set()
    for row in records:
        key = row['trial_id']
        if key not in planned or key in seen:
            raise ValueError('Duplicate or unplanned result; replacements are forbidden')
        seen.add(key)
        if row['model'] != planned[key]['model'] or row['status'] not in {'completed', 'error'}:
            raise ValueError('Result assignment/status differs from fixed plan')
        if row['status'] == 'completed' and not isinstance(row.get('success'), bool):
            raise ValueError('Completed outcome must be an official boolean success')
    def aggregate(subset):
        keys = {j['trial_id'] for j in subset}
        rows = [r for r in records if r['trial_id'] in keys]
        completed = [r for r in rows if r['status'] == 'completed']
        successes = sum(r['success'] for r in completed)
        return dict(planned=len(keys), attempted=len(rows), completed=len(completed), successes=successes,
                    failures=len(completed)-successes, infrastructure_unknown=len(rows)-len(completed),
                    not_attempted=len(keys)-len(rows),
                    raw_success_rate=successes/len(keys) if keys and len(completed) == len(keys) else None)
    arms = {}
    for arm in ARMS:
        subset = [j for j in jobs if j['model'] == arm]
        arms[arm] = aggregate(subset)
        arms[arm]['tasks'] = {task: aggregate([j for j in subset if j['task'] == task]) for task in TASKS}
    complete = len(records) == len(jobs) and all(r['status'] == 'completed' for r in records)
    difference = None
    if complete:
        difference = 100 * sum(arms['alpha025']['tasks'][t]['raw_success_rate'] -
                               arms['base']['tasks'][t]['raw_success_rate'] for t in TASKS) / len(TASKS)
    return dict(kind='randomized_task_balanced_exploratory', complete=complete, arms=arms,
                equal_task_weighted_difference_percentage_points=difference,
                p_value=None, significance_claim=False, filtered_episodes=0,
                replacement_episodes=0, historical_reference_reused=False)


def attempt_plan(jobs, invoke, save):
    """One attempt per literal job; failures are retained, never resampled."""
    records = []
    for job in jobs:
        try:
            row = invoke(job)
        except Exception as error:
            row = dict(trial_id=job['trial_id'], model=job['model'], status='error',
                       error=f'{type(error).__name__}: {error}', safe_to_continue=False)
        records.append(row)
        save(records, randomized_summary(jobs, records))
        if row['status'] == 'error' and not row.get('safe_to_continue', False):
            break
    return records


def check_sources(run, output):
    check_helpers()
    if common.sha(output / 'design.json') != DESIGN_SHA256:
        raise ValueError('Copied frozen design changed')
    for path, digest in run['source_sha256'].items():
        if common.sha(path) != digest:
            raise ValueError('Pinned source changed: ' + path)
    for name, digest in run['copied_code_sha256'].items():
        if common.sha(output / 'code' / name) != digest:
            raise ValueError('Pinned copied runner/helper changed')
    for model in run['models']:
        common.assert_assets(model)
        audit.assert_server_process(run['servers'][model['name']], model, run['repo'])


def import_entry(repo):
    path = Path(repo) / 'eval_robocasa365/entry.py'
    spec = importlib.util.spec_from_file_location('randomized_official_entry', path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    original_client = entry.Client
    if Path(inspect.getfile(original_client)).resolve() != Path(repo) / 'deploy/client.py':
        raise ValueError('Official entry imported a different client source')
    class BoundedClient(original_client):
        def _connect_with_retry(self, max_retries=None, retry_interval=1):
            return super()._connect_with_retry(max_retries=3, retry_interval=1)
    entry.Client = BoundedClient
    return entry


def episode(args):
    output = args.output.resolve(strict=True)
    if common.sha(output / 'run_manifest.json') != args.manifest_sha256:
        raise ValueError('Run manifest changed before child launch')
    run = json.loads((output / 'run_manifest.json').read_text())
    design = load_design(output / 'design.json')
    engineering = args.engineering
    if engineering:
        if (args.task, args.seed) != (design['engineering_check']['task'], design['engineering_check']['env_seed']):
            raise ValueError('Invalid engineering trial')
    elif (args.task, args.seed) not in {(j['task'], j['env_seed']) for j in design['jobs']}:
        raise ValueError('Unplanned trial')
    expected = renderer_environment(run['renderer_gpu'])
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise ValueError('Child renderer environment differs')
    check_sources(run, output)
    if {n: importlib.metadata.version(n) for n in audit.reference_tools.RUNTIME_PACKAGES} != run['simulator_packages']:
        raise ValueError('Child simulator packages changed')
    destination = output / ('engineering' if engineering else 'episodes') / f'{args.task}-{args.seed}'
    events, initial, instrumentation = [], {}, {}
    def event(name, **fields):
        row = dict(event=name, sequence=len(events), monotonic_seconds=time.monotonic(), **fields)
        events.append(row)
        with (destination / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
    # This callback is the only lookup of this child's model assignment.
    def select_model():
        arm = design['engineering_check']['fixed_arm'] if engineering else next(
            j['model'] for j in design['jobs'] if (j['task'], j['env_seed']) == (args.task, args.seed))
        return next(m for m in run['models'] if m['name'] == arm)
    # Model-independent imports/entry setup precede the common environment seeding.
    entry = import_entry(run['repo'])
    sys.path.insert(0, run['recovery_parent'])
    from recovery.policy import OfficialEvalPolicy
    if Path(inspect.getfile(OfficialEvalPolicy)).resolve() != Path(run['recovery_parent']) / 'recovery/policy.py':
        raise ValueError('Imported a different recovery policy adapter')
    import numpy as np
    import torch
    import gymnasium as gym
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    if (Path(robocasa.__file__).resolve().parent != Path(run['package_root']) or
            int(get_task_horizon(args.task)) != design['protocol']['horizons'][args.task]):
        raise ValueError('RoboCasa import path or official horizon differs')
    def make_policy(model):
        return OfficialEvalPolicy(run['repo'], model['model_path'], host='127.0.0.1',
                                  port=model['server_port'], timeout=180, _entry=entry)
    def instrument(policy):
        request = audit.FirstRequestCapture(destination, policy.eval.client._send_with_length_prefix)
        policy.eval.client._send_with_length_prefix = request
        action = audit.FirstActionCapture(policy.eval, destination)
        instrumentation.update(request=request, action=action)
        return action
    lazy = LazyClient(select_model, make_policy, lambda h, m: validate_hello(h, m, run),
                      args.seed, event, instrument)
    def capture(env, observation):
        underlying = env.unwrapped.env
        initial.update(audit.capture_initial(destination, observation, entry.observation_to_state(observation), underlying.sim))
        initial.update(layout_id=int(underlying.layout_id), style_id=int(underlying.style_id),
                       xml_sha256=common.sha(destination / 'initial.xml'))
        common.atomic_json(destination / 'initial_scene.json', initial)
    observers = []
    class ObservedGym:
        @staticmethod
        def make(*a, **kw):
            if observers:
                raise RuntimeError('One child must create exactly one environment')
            observer = ResetObserver(gym.make(*a, **kw), lazy, capture, event)
            observers.append(observer)
            return observer
    # Common parser metadata only; evaluate_task uses the passed LazyClient.
    argv = ['--model-path', run['models'][0]['model_path'], '--split', 'pretrain', '--task-set', 'pretrain300',
            '--task-name', args.task, '--num-trials', '1', '--seed', str(args.seed), '--replan-steps', '16',
            '--obs-history', '4', '--obs-interval', '2', '--crop-ratio', '0.95']
    if engineering:
        argv += ['--horizon', '1']
    official = entry.parse_args(argv)
    entry.validate_args(official)
    random.seed(args.seed)
    np.random.seed(args.seed)
    started, record = time.monotonic(), None
    try:
        stats = entry.evaluate_task(args.task, 0, official, lazy, ObservedGym, get_task_horizon,
                                    convert_action, destination / 'official', episode_indices=[0],
                                    show_progress=False, write_task_stats=False)
        if (tuple(e['event'] for e in events) != EVENT_ORDER or len(observers) != 1 or
                observers[0].reset_calls != 1 or len(stats['episodes']) != 1 or
                not all(instrumentation[k].captured for k in ('request', 'action'))):
            raise ValueError('Missing activation order or official first-input/action evidence')
        result = stats['episodes'][0]
        horizon = 1 if engineering else design['protocol']['horizons'][args.task]
        if (result['seed'] != args.seed or result['episode'] != 0 or result['global_episode_index'] != 0 or
                not isinstance(result['success'], bool) or not 1 <= result['steps'] <= horizon or
                stats['horizon'] != horizon or (engineering and (result['steps'] != 1 or lazy.infer_calls != 1))):
            raise ValueError('Official result differs from fixed single-episode protocol')
        check_sources(run, output)
        record = dict(status='completed', success=result['success'], steps=result['steps'], horizon=horizon,
                      official_episode=result, artifact_sha256={n: common.sha(destination / n) for n in audit.ARTIFACTS})
    except Exception as error:
        record = dict(status='error', error=f'{type(error).__name__}: {error}',
                      safe_to_continue=isinstance(error, (ConnectionError, TimeoutError)))
        (destination / 'exception.txt').write_text(traceback.format_exc())
    finally:
        try:
            lazy.close()
        except Exception as error:
            record = dict(status='error', error=f'Cleanup {type(error).__name__}: {error}', safe_to_continue=False)
        if record is not None:
            record.update(trial_id=f'{args.task}-{args.seed}', task=args.task, seed=args.seed, policy_seed=args.seed,
                          model=lazy.model['name'] if lazy.model else None,
                          model_hash=lazy.model['model_hash'] if lazy.model else None, engineering=engineering,
                          initial_observation=initial, rng_ack=lazy.rng_ack, local_rng=lazy.local_rng,
                          server_identity=lazy.policy.hello if lazy.policy else None, infer_calls=lazy.infer_calls,
                          events=events, duration_seconds=time.monotonic()-started)
            common.atomic_json(destination / 'result.json', record)
    return 0 if record and record['status'] == 'completed' else 1


def renderer_environment(gpu):
    return dict(PYTHONHASHSEED='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
                MUJOCO_EGL_DEVICE_ID=str(gpu), MUJOCO_GL='egl', TOKENIZERS_PARALLELISM='false')


def guarded_spawn(command, register, **kwargs):
    """Register the owned PID before a pending parent termination can arrive."""
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        # Popen inherits signal masks; exec preserves the PID while unblocking
        # the child before entering the normal evaluator process.
        trampoline = ('import os,signal,sys; '
                      'signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM,signal.SIGINT}); '
                      'os.execv(sys.argv[1],sys.argv[1:])')
        process = subprocess.Popen([sys.executable, '-c', trampoline, *command], **kwargs)
        register(process)
        return process
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def validate_child_record(row, job, model, engineering):
    """Check raw identity before the parent adds planned assignment metadata."""
    if row.get('status') == 'completed':
        expected = dict(trial_id=job['trial_id'], task=job['task'], seed=job['env_seed'],
                        policy_seed=job['env_seed'], model=job['model'], model_hash=model['model_hash'],
                        engineering=engineering)
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError('Completed child identity differs from the fixed trial/model')
        if set(row.get('artifact_sha256', {})) != set(audit.ARTIFACTS):
            raise ValueError('Completed child lacks required initial/input/action artifacts')
        if tuple(event['event'] for event in row.get('events', [])) != EVENT_ORDER:
            raise ValueError('Completed child activation order is invalid')
    elif row.get('status') != 'error':
        raise ValueError('Unknown child result status')
    if row.get('model') not in (None, job['model']):
        raise ValueError('Actual post-reset model differs from precommitted assignment')


def task_protocol(package):
    """Validate only this fixed broader pair against the untouched official registry."""
    registry = Path(package) / 'utils/dataset_registry.py'
    wanted = {'TARGET_TASKS', 'PRETRAINING_TASKS', 'COMPOSITE_TASK_DATASETS'}
    values = {}
    for node in ast.parse(registry.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = common.literal(node.value)
    if set(values) != wanted:
        raise ValueError('Cannot read the official task allowlists')
    target50 = set(sum(values['TARGET_TASKS'].values(), []))
    for task in TASKS:
        if task in target50 or task not in values['PRETRAINING_TASKS']['pretrain300']:
            raise ValueError(f'Development task is not strictly outside all target50: {task}')
    horizons = {task: values['COMPOSITE_TASK_DATASETS'][task]['horizon'] for task in TASKS}
    if horizons != {'ScalePortioning': 2100, 'MicrowaveThawingFridge': 2700}:
        raise ValueError('Official broad-task horizons changed')
    return dict(tasks=list(TASKS), seeds=list(range(800001, 800009)), registry=str(registry),
                registry_sha256=common.sha(registry), horizons=horizons, split='pretrain',
                replan_steps=16, obs_history=4, obs_interval=2, crop_ratio=0.95,
                python_hash_seed=0, environment_global_and_policy_seed='same fixed episode seed',
                uses_stage_oracle=False, changes_observations=False)


def run(args):
    check_helpers()
    design = load_design(args.design)
    if args.output.exists():
        raise FileExistsError('Output must be new; no overwrite, resume or retry')
    repo, package, recovery = (p.resolve(strict=True) for p in (args.xr1_repo, args.robocasa_package, args.recovery_parent))
    protocol = task_protocol(package)
    if (protocol['registry_sha256'] != design['registry_sha256_at_design'] or
            protocol['horizons'] != design['protocol']['horizons']):
        raise ValueError('Task allowlist or horizons changed since design')
    sources = {repo / 'eval_robocasa365/entry.py': design['source_hashes_at_design']['official_entry'],
               repo / 'deploy/client.py': design['source_hashes_at_design']['official_client'],
               recovery / 'recovery/policy.py': design['source_hashes_at_design']['policy_adapter'],
               recovery / 'recovery/seeded_server.py': design['source_hashes_at_design']['seeded_server'],
               Path(protocol['registry']): protocol['registry_sha256']}
    for path, digest in sources.items():
        if common.sha(path) != digest:
            raise ValueError('Source differs from frozen design: ' + str(path))
    source_paths = [Path(__file__).resolve(), repo / 'deploy/server.py',
                    package / 'utils/env_utils.py', package / 'utils/dataset_registry_utils.py']
    source_paths += list((recovery / 'recovery').glob('*.py'))
    source_paths += [Path(__file__).resolve().parent / n for n in HELPERS]
    sources.update({p: common.sha(p) for p in source_paths})
    renderer = design['resources']['renderer_gpu']
    models = [common.model_record(dict(name=arm, model_path=design['models'][arm]['model_path'],
                                      server_port=design['models'][arm]['preferred_existing_seeded_port'],
                                      inference_gpu=design['resources']['inference_gpu']), renderer) for arm in ARMS]
    servers = {m['name']: audit.server_process(getattr(args, m['name'] + '_server_pid'), m, repo) for m in models}
    output = args.output.resolve()
    if any(output.is_relative_to(p) for p in (repo, package, recovery / 'recovery', *(Path(m['model_path']) for m in models))):
        raise ValueError('Output cannot overwrite a source repository or model')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'code').mkdir()
    copied = {}
    for source in [Path(__file__).resolve(), *(Path(__file__).resolve().parent / n for n in HELPERS)]:
        target = output / 'code' / source.name
        target.write_bytes(source.read_bytes())
        copied[source.name] = common.sha(target)
        if copied[source.name] != sources[source]:
            raise ValueError('Source changed during snapshot')
        target.chmod(0o444)
    (output / 'design.json').write_bytes(args.design.read_bytes())
    run_record = dict(format_version=1, design_sha256=DESIGN_SHA256, repo=str(repo), package_root=str(package),
                      recovery_parent=str(recovery), renderer_gpu=renderer, models=models, servers=servers,
                      source_sha256={str(p): h for p, h in sources.items()}, copied_code_sha256=copied,
                      official_server_sha256=sources[repo / 'deploy/server.py'],
                      recovery_server_sha256=design['source_hashes_at_design']['seeded_server'],
                      simulator_packages={n: importlib.metadata.version(n) for n in audit.reference_tools.RUNTIME_PACKAGES},
                      created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      owner_pid=os.getpid(), owner_start_ticks=audit.proc_start(os.getpid()))
    common.atomic_json(output / 'run_manifest.json', run_record)
    manifest_sha = common.sha(output / 'run_manifest.json')
    child, child_start = None, None
    def stop_owned():
        if child is not None and child.poll() is None:
            if child_start is not None and audit.proc_start(child.pid) != child_start:
                raise RuntimeError('Owned child process identity changed')
            audit.reference_tools._terminate_owned(child)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    evidence_hashes = {}
    def invoke(job, engineering=False):
        nonlocal child, child_start
        check_sources(run_record, output)
        if common.sha(output / 'run_manifest.json') != manifest_sha:
            raise ValueError('Manifest changed during evaluation')
        if common.gpu_pids(renderer):
            raise RuntimeError('Renderer GPU occupied; no concurrent rendering')
        directory = output / ('engineering' if engineering else 'episodes') / job['trial_id']
        directory.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-u', str(output / 'code/dev_randomized_broad_alpha.py'), 'episode', '--output', str(output),
                   '--manifest-sha256', manifest_sha, '--task', job['task'], '--seed', str(job['env_seed'])]
        if engineering:
            command.append('--engineering')
        with (directory / 'process.log').open('xb') as log:
            identity = {}
            def register(process):
                nonlocal child, child_start
                child, child_start = process, None
                child_start = audit.proc_start(child.pid)
                identity.update(pid=child.pid, process_start_ticks=child_start, parent_pid=os.getpid(), command=command,
                                run_manifest_sha256=manifest_sha, renderer_gpu=renderer)
                common.atomic_json(directory / 'process_identity.json', identity)
            guarded_spawn(command, register, env=dict(os.environ, **renderer_environment(renderer)),
                          stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            returncode = child.wait()
        common.atomic_json(directory / 'process_exit.json', dict(returncode=returncode, **identity))
        path = directory / 'result.json'
        row = json.loads(path.read_text()) if path.exists() else dict(status='error', error=f'Child exited {returncode} without result', safe_to_continue=False)
        model = next(m for m in models if m['name'] == job['model'])
        validate_child_record(row, job, model, engineering)
        row.update(trial_id=job['trial_id'], model=job['model'], task=job['task'], seed=job['env_seed'])
        if (returncode == 0) != (row['status'] == 'completed') or returncode < 0:
            row.update(status='error', error=f'Child process failure: exit {returncode}', safe_to_continue=False)
        for name, digest in row.get('artifact_sha256', {}).items():
            if name not in audit.ARTIFACTS or common.sha(directory / name) != digest:
                raise ValueError('Child artifact identity changed')
        for p in directory.iterdir():
            if p.is_file():
                evidence_hashes[str(p)] = common.sha(p)
        print(json.dumps({k: row.get(k) for k in ('trial_id', 'model', 'status', 'success', 'steps')}), flush=True)
        return row
    def save(records, summary):
        common.atomic_json(output / 'results.json', records)
        common.atomic_json(output / 'summary.json', summary)
    lock_path = Path(tempfile.gettempdir()) / f'robocasa-dev-render-{os.getuid()}-gpu-{renderer}.lock'
    try:
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check = design['engineering_check']
            engineering = dict(task=check['task'], env_seed=check['env_seed'], model=check['fixed_arm'],
                               trial_id=f"{check['task']}-{check['env_seed']}")
            gate = invoke(engineering, engineering=True)
            common.atomic_json(output / 'engineering_gate.json', gate)
            if gate['status'] != 'completed':
                save([], randomized_summary(design['jobs'], []))
                raise RuntimeError('The single engineering attempt failed; no formal trials or retries')
            records = attempt_plan(design['jobs'], invoke, save)
            check_sources(run_record, output)
            if common.sha(output / 'run_manifest.json') != manifest_sha or any(common.sha(p) != h for p, h in evidence_hashes.items()):
                raise ValueError('Run or episode evidence changed before completion')
            common.atomic_json(output / 'evidence_sha256.json', evidence_hashes)
            summary = randomized_summary(design['jobs'], records)
            common.atomic_json(output / 'run_status.json', dict(status='complete' if summary['complete'] else 'incomplete', attempted=len(records)))
    except BaseException as error:
        stop_owned()
        common.atomic_json(output / 'run_status.json', dict(status='stopped', error=f'{type(error).__name__}: {error}'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    launch = sub.add_parser('run')
    launch.add_argument('--design', type=Path, default=Path(__file__).with_name('randomized_broad_alpha_design_v2.json'))
    launch.add_argument('--xr1-repo', type=Path, required=True)
    launch.add_argument('--robocasa-package', type=Path, required=True)
    launch.add_argument('--recovery-parent', type=Path, required=True)
    launch.add_argument('--base-server-pid', type=int, required=True)
    launch.add_argument('--alpha025-server-pid', type=int, required=True)
    launch.add_argument('--output', type=Path, required=True)
    child = sub.add_parser('episode')
    child.add_argument('--output', type=Path, required=True)
    child.add_argument('--manifest-sha256', required=True)
    child.add_argument('--task', choices=TASKS, required=True)
    child.add_argument('--seed', type=int, required=True)
    child.add_argument('--engineering', action='store_true')
    args = parser.parse_args()
    if args.command == 'episode':
        raise SystemExit(episode(args))
    run(args)


if __name__ == '__main__':
    main()
