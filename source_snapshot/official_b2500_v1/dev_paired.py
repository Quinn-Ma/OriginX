#!/usr/bin/env python3
"""Fixed outside-target50 development comparison using the unmodified official rollout."""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

TASKS = ('CupcakeCleanup', 'DumpLeftovers')
SEEDS = tuple(range(500001, 500011))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stamp(path):
    s = Path(path).stat()
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino]


def atomic_json(path, value):
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def literal(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'dict', 'OrderedDict'} and not node.args:
        return {key.arg: literal(key.value) for key in node.keywords}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def task_protocol(package):
    registry = package/'utils/dataset_registry.py'
    wanted = {'TARGET_TASKS', 'PRETRAINING_TASKS', 'COMPOSITE_TASK_DATASETS'}
    values = {}
    for node in ast.parse(registry.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = literal(node.value)
    if set(values) != wanted:
        raise ValueError('Cannot read the official task allowlists')
    target50 = set(sum(values['TARGET_TASKS'].values(), []))
    for task in TASKS:
        if task in target50 or task not in values['PRETRAINING_TASKS']['pretrain300']:
            raise ValueError(f'Development task is not strictly outside all target50: {task}')
    return dict(tasks=list(TASKS), seeds=list(SEEDS), registry=str(registry), registry_sha256=sha(registry),
                horizons={task: values['COMPOSITE_TASK_DATASETS'][task]['horizon'] for task in TASKS},
                split='pretrain', replan_steps=16, obs_history=4, obs_interval=2, crop_ratio=0.95,
                python_hash_seed=0, environment_global_and_policy_seed='same fixed episode seed',
                uses_stage_oracle=False, changes_observations=False)


def model_record(item, renderer_gpu):
    name = item['name']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Model names must be simple identifiers')
    path = Path(item['model_path']).resolve(strict=True)
    if str(item['inference_gpu']) == str(renderer_gpu):
        raise ValueError('Inference and rendering must use different physical GPUs')
    if not 1024 <= item['server_port'] <= 65535:
        raise ValueError('Invalid dedicated seeded-server port')
    index = path/'model.safetensors.index.json'
    weights = set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
    assets = weights | {p.name for p in path.iterdir() if p.is_file()
                        and p.suffix in {'.json', '.py', '.jinja', '.txt', '.model'}}
    hashes, stamps = {}, {}
    for filename in sorted(assets):
        if Path(filename).name != filename:
            raise ValueError('Checkpoint assets must be local flat files')
        stamps[filename] = stamp(path/filename)
        hashes[filename] = sha(path/filename)
        if stamp(path/filename) != stamps[filename]:
            raise ValueError('Model changed during hashing')
    return dict(item, model_path=str(path), asset_sha256=hashes, asset_stamps=stamps,
                model_hash=jhash(hashes))


def assert_assets(record):
    path = Path(record['model_path'])
    if any(stamp(path/name) != value for name, value in record['asset_stamps'].items()):
        raise ValueError(f'Model changed during the development run: {record["name"]}')


def gpu_pids(index):
    xml = subprocess.check_output(['nvidia-smi', '-i', str(index), '-q', '-x'], text=True)
    return sorted({int(node.text) for node in ET.fromstring(xml).findall('.//process_info/pid')})


def paired_summary(records, names, reference):
    keyed = {(row['model'], row['task'], row['seed']): row for row in records}
    expected = {(name, task, seed) for name in names for task in TASKS for seed in SEEDS}
    if len(keyed) != len(records) or set(keyed) != expected:
        raise ValueError('Refusing summary with missing, duplicate or unexpected episodes')
    per_model = {}
    for name in names:
        rows = [row for row in records if row['model'] == name]
        per_model[name] = dict(episodes=len(rows), successes=sum(row['success'] for row in rows),
                              success_rate=sum(row['success'] for row in rows)/len(rows),
                              mean_steps=sum(row['steps'] for row in rows)/len(rows),
                              tasks={task: dict(episodes=len(SEEDS), successes=sum(row['success'] for row in rows if row['task'] == task)) for task in TASKS})
    comparisons = {}
    for name in names:
        if name == reference:
            continue
        counts = dict(both_success=0, reference_only=0, candidate_only=0, both_failure=0)
        mismatches = []
        for task in TASKS:
            for seed in SEEDS:
                base, candidate = keyed[reference, task, seed], keyed[name, task, seed]
                if (base['initial_observation'] != candidate['initial_observation'] or
                        any(base['rng_ack'][key] != candidate['rng_ack'][key]
                            for key in ('torch_cpu_rng_sha256', 'torch_cuda_rng_sha256'))):
                    mismatches.append(dict(task=task, seed=seed,
                                           initial_observation_equal=base['initial_observation'] == candidate['initial_observation'],
                                           policy_rng_equal=all(base['rng_ack'][key] == candidate['rng_ack'][key] for key in ('torch_cpu_rng_sha256', 'torch_cuda_rng_sha256'))))
                label = ('both_success' if candidate['success'] else 'reference_only') if base['success'] else ('candidate_only' if candidate['success'] else 'both_failure')
                counts[label] += 1
        discordant = counts['reference_only']+counts['candidate_only']
        p_value = min(1.0, 2*sum(math.comb(discordant, k) for k in range(min(counts['reference_only'], counts['candidate_only'])+1))/2**discordant) if discordant else 1.0
        comparisons[name] = dict(reference=reference, pairs=len(TASKS)*len(SEEDS), **counts,
                                 success_difference_percentage_points=100*(counts['candidate_only']-counts['reference_only'])/(len(TASKS)*len(SEEDS)),
                                 exact_paired_test_p_value=p_value if not mismatches else None,
                                 initial_pairing_valid=not mismatches, pairing_mismatches=mismatches)
    return dict(models=per_model, paired=comparisons,
                pairing_valid=all(item['initial_pairing_valid'] for item in comparisons.values()))


def episode(args):
    run = json.loads((args.output/'run_manifest.json').read_text())
    model = next(item for item in run['models'] if item['name'] == args.model)
    if args.task not in TASKS or args.seed not in SEEDS:
        raise ValueError('Episode not in the fixed development plan')
    expected_env = dict(PYTHONHASHSEED='0', MUJOCO_GL='egl', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                        CUDA_VISIBLE_DEVICES=str(run['renderer_gpu']), MUJOCO_EGL_DEVICE_ID=str(run['renderer_gpu']))
    if any(os.environ.get(key) != value for key, value in expected_env.items()):
        raise ValueError('Episode resource/hash environment differs from the fixed plan')
    assert_assets(model)
    sources = {
        Path(run['repo'])/'eval_robocasa365/entry.py': run['official_entry_sha256'],
        Path(run['repo'])/'deploy/client.py': run['official_client_sha256'],
        Path(run['repo'])/'deploy/server.py': run['official_server_sha256'],
        Path(run['recovery_parent'])/'recovery/policy.py': run['recovery_policy_sha256'],
        Path(run['recovery_parent'])/'recovery/seeded_server.py': run['recovery_server_sha256'],
        Path(__file__): run['runner_sha256'],
    }
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Evaluator or seeded-server source changed during this run')
    if sha(Path(run['protocol']['registry'])) != run['protocol']['registry_sha256']:
        raise ValueError('Task registry changed')
    sys.path.insert(0, run['recovery_parent'])
    from recovery.policy import OfficialEvalPolicy
    import numpy as np
    policy = OfficialEvalPolicy(run['repo'], model['model_path'], host='127.0.0.1',
                                port=model['server_port'], timeout=180)
    try:
        hello = policy.hello
        if (Path(hello['model_path']).resolve() != Path(model['model_path']) or
                hello['model_config_sha256'] != model['asset_sha256']['config.json'] or
                hello['official_server_sha256'] != run['official_server_sha256'] or
                hello['recovery_server_sha256'] != run['recovery_server_sha256']):
            raise ValueError('Connected seeded server is not the declared model/code')
        rng_ack = policy.reset(seed=args.seed)
        entry = policy.entry
        import gymnasium as gym
        import robocasa
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        from robocasa.utils.env_utils import convert_action
        if Path(robocasa.__file__).resolve().parent != Path(run['package_root']):
            raise ValueError('Simulator imported a different RoboCasa checkout')
        if int(get_task_horizon(args.task)) != run['protocol']['horizons'][args.task]:
            raise ValueError('Official task horizon changed')
        random.seed(args.seed)
        np.random.seed(args.seed)
        initial = {}
        def array_hash(value):
            value = np.asarray(value)
            digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
            return digest.hexdigest()
        class ResetObserver:
            def __init__(self, env):
                self.env = env
            def reset(self, *a, **kw):
                observation, info = self.env.reset(*a, **kw)
                initial.update(rgb={key: array_hash(observation[key]) for key in entry.CAMERA_KEYS},
                               proprioception=array_hash(entry.observation_to_state(observation)),
                               physics_state=array_hash(self.env.unwrapped.env.sim.get_state().flatten()),
                               instruction=observation['annotation.human.task_description'])
                return observation, info
            def step(self, action):
                return self.env.step(action)
            def close(self):
                return self.env.close()
        class ObservedGym:
            @staticmethod
            def make(*a, **kw):
                return ResetObserver(gym.make(*a, **kw))
        destination = args.output/'episodes'/args.task/str(args.seed)/args.model
        official = entry.parse_args(['--model-path', model['model_path'], '--split', 'pretrain',
                                     '--task-set', 'pretrain300', '--task-name', args.task,
                                     '--num-trials', '1', '--seed', str(args.seed),
                                     '--replan-steps', '16', '--obs-history', '4', '--obs-interval', '2',
                                     '--crop-ratio', '0.95'])
        entry.validate_args(official)
        stats = entry.evaluate_task(args.task, 0, official, policy.eval, ObservedGym,
                                    get_task_horizon, convert_action, destination/'official',
                                    episode_indices=[0], show_progress=False, write_task_stats=False)
        result = stats['episodes'][0]
        if result['seed'] != args.seed or not initial:
            raise ValueError('Official episode did not use the requested seed')
        record = dict(model=args.model, model_hash=model['model_hash'], task=args.task,
                      seed=args.seed, global_seed=args.seed, policy_seed=args.seed,
                      success=bool(result['success']), steps=int(result['steps']), horizon=stats['horizon'],
                      rng_ack=rng_ack, initial_observation=initial, official_episode=result,
                      server_identity=hello)
        atomic_json(destination/'result.json', record)
    finally:
        policy.close()


def run(args):
    if args.renderer_gpu < 0:
        raise ValueError('Renderer GPU must be a nonnegative physical index')
    if args.output.exists():
        raise FileExistsError('Development output must be a new directory')
    repo, package, recovery = args.repo.resolve(strict=True), args.package_root.resolve(strict=True), args.recovery_parent.resolve(strict=True)
    plan = json.loads(args.plan.read_text())
    if plan.get('format_version') != 1 or len(plan['models']) < 2:
        raise ValueError('A model plan must name at least the reference and one candidate')
    models = [model_record(item, args.renderer_gpu) for item in plan['models']]
    names = [item['name'] for item in models]
    if len(set(names)) != len(names) or len({item['server_port'] for item in models}) != len(models) or args.reference not in names:
        raise ValueError('Require distinct model names/ports and a named reference')
    protocol = task_protocol(package)
    manifest = dict(format_version=1, repo=str(repo), package_root=str(package), recovery_parent=str(recovery),
                    models=models, reference=args.reference, protocol=protocol, renderer_gpu=args.renderer_gpu,
                    official_entry_sha256=sha(repo/'eval_robocasa365/entry.py'),
                    official_client_sha256=sha(repo/'deploy/client.py'),
                    official_server_sha256=sha(repo/'deploy/server.py'),
                    recovery_server_sha256=sha(recovery/'recovery/seeded_server.py'),
                    recovery_policy_sha256=sha(recovery/'recovery/policy.py'), runner_sha256=sha(__file__),
                    simulator_packages={name: importlib.metadata.version(name) for name in
                                        ('torch', 'numpy', 'mujoco', 'robocasa', 'robosuite', 'transformers')},
                    created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/'run_manifest.json', manifest)
    # Independent physical GPUs may render concurrently; each dev GPU is exclusive.
    lock_path = Path(tempfile.gettempdir())/f'robocasa-dev-render-{os.getuid()}-gpu-{args.renderer_gpu}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        env = dict(os.environ, PYTHONHASHSEED='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUDA_VISIBLE_DEVICES=str(args.renderer_gpu), MUJOCO_EGL_DEVICE_ID=str(args.renderer_gpu),
                   MUJOCO_GL='egl', TOKENIZERS_PARALLELISM='false')
        records = []
        for task in TASKS:
            for seed in SEEDS:
                for model in models:
                    if gpu_pids(args.renderer_gpu):
                        raise RuntimeError('Renderer GPU is occupied; refusing concurrent evaluation')
                    destination = args.output/'episodes'/task/str(seed)/model['name']
                    destination.mkdir(parents=True, exist_ok=False)
                    command = [sys.executable, '-u', str(Path(__file__).resolve()), 'episode',
                               '--output', str(args.output.resolve()), '--model', model['name'],
                               '--task', task, '--seed', str(seed)]
                    with (destination/'process.log').open('xb') as log:
                        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                        try:
                            code = child.wait()
                        except BaseException:
                            if child.poll() is None:
                                child.terminate()
                                try:
                                    child.wait(timeout=20)
                                except subprocess.TimeoutExpired:
                                    child.kill()
                                    child.wait()
                            raise
                    atomic_json(destination/'process_exit.json', dict(returncode=code))
                    if code:
                        raise RuntimeError(f'Dev episode failed ({code}); preserved {destination}; no automatic retry')
                    record = json.loads((destination/'result.json').read_text())
                    if (record['model'], record['task'], record['seed'], record['model_hash']) != (model['name'], task, seed, model['model_hash']):
                        raise ValueError('Episode result does not match its scheduled model/task/seed')
                    records.append(record)
                    print(json.dumps({key: records[-1][key] for key in ('model', 'task', 'seed', 'success', 'steps')}), flush=True)
        summary = paired_summary(records, names, args.reference)
        atomic_json(args.output/'summary.json', summary)
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    launch = sub.add_parser('run', help='Run the fixed development plan using already-running dedicated servers')
    launch.add_argument('--repo', type=Path, required=True)
    launch.add_argument('--package-root', type=Path, required=True)
    launch.add_argument('--recovery-parent', type=Path, required=True)
    launch.add_argument('--plan', type=Path, required=True)
    launch.add_argument('--output', type=Path, required=True)
    launch.add_argument('--reference', default='base')
    launch.add_argument('--renderer-gpu', type=int, default=5)
    child = sub.add_parser('episode', help=argparse.SUPPRESS)
    child.add_argument('--output', type=Path, required=True)
    child.add_argument('--model', required=True)
    child.add_argument('--task', choices=TASKS, required=True)
    child.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run(args)
    else:
        episode(args)


if __name__ == '__main__':
    main()
