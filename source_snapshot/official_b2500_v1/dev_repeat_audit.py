#!/usr/bin/env python3
"""Four fresh official base rollouts to audit reset/render/inference repeatability.

Two outside-target50 tasks, seed 500001, two repeats each. No state restoration,
retries, filtering, resume, server launch, or significance claim. The official
rollout, processor and server are unmodified; observers only copy their outputs.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time

try:
    from . import dev_paired as original
    from . import dev_extend as reference_tools
except ImportError:
    import dev_paired as original
    import dev_extend as reference_tools

FROZEN_EXTEND_SHA256 = 'd1cbf546043be47e451a487690a650bf3b46d25c090f9b62f220912a83572795'
SEED = 500001
REPEATS = (0, 1)
ARTIFACTS = ('initial_observation.npz', 'initial.xml', 'first_request.npz',
             'first_request.json', 'first_action_chunk.npy')


def plan():
    return [(task, SEED, repeat) for task in original.TASKS for repeat in REPEATS]


def array_hash(value):
    import numpy as np
    value = np.asarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def write_npz(path, arrays):
    import numpy as np
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def proc_start(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def server_process(pid, model, repo):
    """Read only the declared server process and the relevant environment keys."""
    start = proc_start(pid)
    command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    values = {}
    for raw in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0'):
        key, _, value = raw.partition(b'=')
        if key in {b'CUDA_VISIBLE_DEVICES', b'CUDA_DEVICE_ORDER'}:
            values[key.decode()] = value.decode()
    def argument(flag):
        if command.count(flag) != 1:
            raise ValueError('Dedicated server command must explicitly declare ' + flag)
        return command[command.index(flag) + 1]
    if ('recovery.seeded_server' not in command or
            Path(argument('--model')).resolve() != Path(model['model_path']) or
            Path(argument('--xr1-repo')).resolve() != Path(repo) or
            int(argument('--port')) != model['server_port'] or
            values.get('CUDA_VISIBLE_DEVICES') != str(model['inference_gpu']) or
            values.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or proc_start(pid) != start):
        raise ValueError('Declared dedicated server process/model/physical GPU differs')
    return dict(pid=pid, process_start_ticks=start, command=command, environment=values)


def assert_server_process(identity, model, repo):
    if server_process(identity['pid'], model, repo) != identity:
        raise ValueError('Dedicated inference server process changed during the audit')


def capture_initial(directory, observation, state, sim):
    """Copy reset data without another render, simulation step or RNG draw."""
    import numpy as np
    arrays = {key: np.array(observation[key], copy=True) for key in reference_tools.CAMERAS}
    for key, value in arrays.items():
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f'Unexpected official RGB array: {key}')
    arrays.update(proprioception=np.array(state, copy=True),
                  physics_state=np.array(sim.get_state().flatten(), copy=True),
                  qpos=np.array(sim.data.qpos, copy=True), qvel=np.array(sim.data.qvel, copy=True),
                  ctrl=np.array(sim.data.ctrl, copy=True))
    xml = sim.model.get_xml()
    with (directory / 'initial.xml').open('x') as stream:
        stream.write(xml)
    write_npz(directory / 'initial_observation.npz', arrays)
    return dict(rgb={key: array_hash(arrays[key]) for key in reference_tools.CAMERAS},
                proprioception=array_hash(arrays['proprioception']),
                physics_state=array_hash(arrays['physics_state']),
                instruction=observation['annotation.human.task_description'])


class FirstRequestCapture:
    """Observe the actual processor request once, immediately before serialization.

    Raw tensor bytes preserve BF16/integer precision; float32 copies support
    numerical comparisons. These are client tensors, before the unchanged
    server's model-dtype/device conversion. The same request object is forwarded.
    """
    def __init__(self, directory, send):
        self.directory, self.send, self.captured = directory, send, False

    def __call__(self, request):
        if not self.captured and '_recovery_control' not in request:
            import numpy as np
            import torch
            if request.get('task_id') != 'robocasa365':
                raise ValueError('Unexpected first inference task')
            arrays, fields = {}, {}
            for index, (key, value) in enumerate(sorted(request.items())):
                prefix = f'field_{index}'
                if isinstance(value, torch.Tensor):
                    tensor = value.detach().cpu().contiguous()
                    raw = tensor.reshape(-1).view(torch.uint8).numpy().copy()
                    numeric = tensor.float().numpy().copy() if tensor.is_floating_point() else tensor.numpy().copy()
                    dtype, shape = str(tensor.dtype), list(tensor.shape)
                    kind = 'torch.Tensor'
                elif isinstance(value, np.ndarray):
                    if value.dtype.hasobject:
                        raise ValueError('Object arrays are not valid processor inputs')
                    numeric = np.array(value, copy=True)
                    raw = np.ascontiguousarray(value).view(np.uint8).reshape(-1).copy()
                    dtype, shape, kind = value.dtype.str, list(value.shape), 'numpy.ndarray'
                else:
                    # Fail if a future processor returns non-JSON objects.
                    json.dumps(value, allow_nan=False)
                    fields[key] = dict(kind='json', value=value)
                    continue
                arrays[prefix + '_raw'], arrays[prefix + '_values'] = raw, numeric
                fields[key] = dict(kind=kind, dtype=dtype, shape=shape,
                                   raw_key=prefix + '_raw', values_key=prefix + '_values',
                                   raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
            if not any(key.startswith('pixel_values') and value['kind'] != 'json'
                       for key, value in fields.items()):
                raise ValueError('First actual processor request contains no pixel_values tensor')
            write_npz(self.directory / 'first_request.npz', arrays)
            original.atomic_json(self.directory / 'first_request.json',
                                 dict(format_version=1, capture_point='client_before_serialization_and_server_dtype_conversion',
                                      fields=fields, request_sha256=original.jhash(fields)))
            self.captured = True
        return self.send(request)


class FirstActionCapture:
    def __init__(self, client, directory):
        self.client, self.directory, self.captured = client, directory, False

    def infer(self, *args, **kwargs):
        result = self.client.infer(*args, **kwargs)
        if not self.captured:
            import numpy as np
            if result.shape != (16, 12) or result.dtype != np.float32 or not np.isfinite(result).all():
                raise ValueError('Unexpected first decoded official action chunk')
            with (self.directory / 'first_action_chunk.npy').open('xb') as stream:
                np.save(stream, result, allow_pickle=False)
            self.captured = True
        return result


def check_sources(run, output):
    for path, digest in run['audit']['source_sha256'].items():
        if original.sha(path) != digest:
            raise ValueError('Pinned audit/evaluator dependency changed: ' + path)
    for name, digest in run['audit']['copied_code_sha256'].items():
        if original.sha(output / 'code' / name) != digest:
            raise ValueError('Copied audit code changed')
    if original.sha(Path(run['audit']['reference_run']) / 'run_manifest.json') != run['audit']['reference_manifest_sha256']:
        raise ValueError('Reference manifest changed')
    original.assert_assets(run['models'][0])
    assert_server_process(run['audit']['server_process'], run['models'][0], run['repo'])


def episode(args):
    output = args.output.resolve(strict=True)
    if original.sha(output / 'run_manifest.json') != args.manifest_sha256:
        raise ValueError('Pinned audit manifest changed before episode')
    run = json.loads((output / 'run_manifest.json').read_text())
    if (args.task, SEED, args.repeat) not in plan():
        raise ValueError('Episode is outside the four fixed audit rollouts')
    expected = dict(PYTHONHASHSEED='0', MUJOCO_GL='egl', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES=str(run['renderer_gpu']), MUJOCO_EGL_DEVICE_ID=str(run['renderer_gpu']))
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise ValueError('Audit renderer environment differs')
    check_sources(run, output)
    if {name: importlib.metadata.version(name) for name in reference_tools.RUNTIME_PACKAGES} != run['simulator_packages']:
        raise ValueError('Audit episode runtime differs from the reference')
    model = run['models'][0]
    destination = output / 'episodes' / args.task / str(SEED) / f'repeat_{args.repeat}'
    sys.path.insert(0, run['recovery_parent'])
    from recovery.policy import OfficialEvalPolicy
    import numpy as np
    policy = OfficialEvalPolicy(run['repo'], model['model_path'], host='127.0.0.1', port=model['server_port'], timeout=180)
    try:
        hello = policy.hello
        if (Path(hello['model_path']).resolve() != Path(model['model_path']) or
                hello['model_config_sha256'] != model['asset_sha256']['config.json'] or
                hello['official_server_sha256'] != run['official_server_sha256'] or
                hello['recovery_server_sha256'] != run['recovery_server_sha256']):
            raise ValueError('Audit server is not the declared base checkpoint/code')
        rng_ack = policy.reset(seed=SEED)
        request_capture = FirstRequestCapture(destination, policy.eval.client._send_with_length_prefix)
        policy.eval.client._send_with_length_prefix = request_capture
        action_capture = FirstActionCapture(policy.eval, destination)
        entry = policy.entry
        import gymnasium as gym
        import robocasa
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        from robocasa.utils.env_utils import convert_action
        if (Path(robocasa.__file__).resolve().parent != Path(run['package_root']) or
                int(get_task_horizon(args.task)) != run['protocol']['horizons'][args.task]):
            raise ValueError('Audit simulator checkout/horizon differs from the reference')
        random.seed(SEED)
        np.random.seed(SEED)
        initial = {}
        class ResetObserver:
            def __init__(self, env):
                self.env = env
            def reset(self, *a, **kw):
                observation, info = self.env.reset(*a, **kw)
                if initial:
                    raise ValueError('Official one-episode rollout reset more than once')
                initial.update(capture_initial(destination, observation, entry.observation_to_state(observation),
                                               self.env.unwrapped.env.sim))
                return observation, info
            def step(self, action):
                return self.env.step(action)
            def close(self):
                return self.env.close()
        class ObservedGym:
            @staticmethod
            def make(*a, **kw):
                return ResetObserver(gym.make(*a, **kw))
        official = entry.parse_args(['--model-path', model['model_path'], '--split', 'pretrain',
                                     '--task-set', 'pretrain300', '--task-name', args.task,
                                     '--num-trials', '1', '--seed', str(SEED), '--replan-steps', '16',
                                     '--obs-history', '4', '--obs-interval', '2', '--crop-ratio', '0.95'])
        entry.validate_args(official)
        stats = entry.evaluate_task(args.task, 0, official, action_capture, ObservedGym,
                                    get_task_horizon, convert_action, destination / 'official',
                                    episode_indices=[0], show_progress=False, write_task_stats=False)
        if len(stats['episodes']) != 1 or not initial or not request_capture.captured or not action_capture.captured:
            raise ValueError('Missing official episode or first-observation/request/action evidence')
        result = stats['episodes'][0]
        record = dict(model=model['name'], model_hash=model['model_hash'], task=args.task, seed=SEED,
                      global_seed=SEED, policy_seed=SEED, repeat=args.repeat, success=bool(result['success']),
                      steps=int(result['steps']), horizon=stats['horizon'], rng_ack=rng_ack,
                      initial_observation=initial, official_episode=result, server_identity=hello,
                      artifact_sha256={name: original.sha(destination / name) for name in ARTIFACTS},
                      initial_xml_sha256=original.sha(destination / 'initial.xml'))
        reference_tools.validate_episode_record(record, model, args.task, SEED, run)
        check_sources(run, output)
        original.atomic_json(destination / 'result.json', record)
    finally:
        policy.close()


def numeric_difference(left, right):
    import numpy as np
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(shape_equal=left.shape == right.shape, dtype_equal=left.dtype == right.dtype, exact=False)
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite audit data')
    return dict(shape_equal=True, dtype_equal=True, exact=bool(np.array_equal(left, right)),
                max_abs=float(delta.max(initial=0)), mean_abs=float(delta.mean()) if delta.size else 0,
                nonzero_values=int(np.count_nonzero(delta)), values=int(delta.size))


def rgb_difference(left, right):
    import numpy as np
    result = numeric_difference(left, right)
    if not result['shape_equal'] or not result['dtype_equal']:
        return result
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    pixels = np.any(delta != 0, axis=-1)
    indices = np.argwhere(pixels)
    result.update(nonzero_pixels=int(pixels.sum()), pixels=int(pixels.size),
                  fraction_values_gt5=float((delta > 5).mean()),
                  p95_abs=float(np.percentile(delta, 95)), p99_abs=float(np.percentile(delta, 99)),
                  changed_bbox_yx=None if not indices.size else [indices.min(0).tolist(), indices.max(0).tolist()],
                  left_black_fraction=float(np.all(left == 0, axis=-1).mean()),
                  right_black_fraction=float(np.all(right == 0, axis=-1).mean()))
    return result


def read_result(output, run, task, repeat):
    directory = output / 'episodes' / task / str(SEED) / f'repeat_{repeat}'
    if json.loads((directory / 'process_exit.json').read_text()).get('returncode') != 0:
        raise ValueError('Audit episode failed; no exclusion or replacement')
    record = json.loads((directory / 'result.json').read_text())
    reference_tools.validate_episode_record(record, run['models'][0], task, SEED, run)
    if record.get('repeat') != repeat or set(record.get('artifact_sha256', {})) != set(ARTIFACTS):
        raise ValueError('Missing repeat identity or audit artifacts')
    if any(original.sha(directory / name) != digest for name, digest in record['artifact_sha256'].items()):
        raise ValueError('Audit artifact changed')
    if record['initial_xml_sha256'] != record['artifact_sha256']['initial.xml']:
        raise ValueError('Initial XML identity disagrees')
    return directory, record


def summarize(output, run):
    import numpy as np
    rows, comparisons = [], {}
    for task in original.TASKS:
        pairs = [read_result(output, run, task, repeat) for repeat in REPEATS]
        (left_dir, left), (right_dir, right) = pairs
        rows.extend({key: record[key] for key in ('task', 'seed', 'repeat', 'success', 'steps', 'horizon')}
                    for _, record in pairs)
        with np.load(left_dir / 'initial_observation.npz', allow_pickle=False) as a, np.load(right_dir / 'initial_observation.npz', allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError('Different initial audit array fields')
            rgb = {key: rgb_difference(a[key], b[key]) for key in reference_tools.CAMERAS}
            state = {key: numeric_difference(a[key], b[key]) for key in a.files if key not in reference_tools.CAMERAS}
            for data, record in ((a, left), (b, right)):
                if (any(array_hash(data[key]) != record['initial_observation']['rgb'][key] for key in reference_tools.CAMERAS)
                        or any(array_hash(data[key]) != record['initial_observation'][key] for key in ('proprioception', 'physics_state'))):
                    raise ValueError('Initial observation hashes disagree with saved arrays')
        a_meta, b_meta = [json.loads((directory / 'first_request.json').read_text()) for directory in (left_dir, right_dir)]
        request = {}
        with np.load(left_dir / 'first_request.npz', allow_pickle=False) as a, np.load(right_dir / 'first_request.npz', allow_pickle=False) as b:
            for data, metadata in ((a, a_meta), (b, b_meta)):
                if metadata['request_sha256'] != original.jhash(metadata['fields']):
                    raise ValueError('Processor request descriptor changed')
                for descriptor in metadata['fields'].values():
                    if descriptor['kind'] != 'json' and hashlib.sha256(data[descriptor['raw_key']].tobytes()).hexdigest() != descriptor['raw_sha256']:
                        raise ValueError('Processor raw tensor hash changed')
            for key in sorted(set(a_meta['fields']) | set(b_meta['fields'])):
                x, y = a_meta['fields'].get(key), b_meta['fields'].get(key)
                if not x or not y or x['kind'] == 'json' or y['kind'] == 'json':
                    request[key] = dict(exact=x == y)
                else:
                    request[key] = dict(raw_exact=(x['dtype'], x['shape'], x['raw_sha256']) == (y['dtype'], y['shape'], y['raw_sha256']),
                                        numeric=numeric_difference(a[x['values_key']], b[y['values_key']]))
        actions = numeric_difference(np.load(left_dir / 'first_action_chunk.npy', allow_pickle=False),
                                     np.load(right_dir / 'first_action_chunk.npy', allow_pickle=False))
        comparisons[task] = dict(initial_xml_exact=left['initial_xml_sha256'] == right['initial_xml_sha256'],
                                 initial_instruction_exact=left['initial_observation']['instruction'] == right['initial_observation']['instruction'],
                                 policy_rng_exact=all(left['rng_ack'][key] == right['rng_ack'][key] for key in ('torch_cpu_rng_sha256', 'torch_cuda_rng_sha256')),
                                 same_server_instance=left['server_identity']['server_instance'] == right['server_identity']['server_instance'],
                                 initial_rgb=rgb, initial_state=state, first_processor_request=request,
                                 first_processor_request_exact=a_meta['request_sha256'] == b_meta['request_sha256'],
                                 first_action_chunk=actions, final_success_equal=left['success'] == right['success'],
                                 steps_difference=right['steps'] - left['steps'])
    return dict(kind='four_fresh_base_repeats_v1', episodes=rows, comparisons=comparisons,
                scope='Two repeats per task screen gross variability; they do not estimate a reliable variance or causal source.',
                raw_rgb_identity_is_not_a_statistical_validity_requirement=True, filtered_episodes=0,
                restored_states=False, p_value=None)


def run(args):
    if args.output.exists():
        raise FileExistsError('Audit output must be new; no overwrite or resume')
    if args.renderer_gpu < 0 or args.inference_gpu < 0 or args.renderer_gpu == args.inference_gpu:
        raise ValueError('Rendering and inference need distinct physical GPUs')
    if original.sha(reference_tools.__file__) != FROZEN_EXTEND_SHA256:
        raise ValueError('Reference validation helper changed')
    reference = reference_tools.validate_reference_run(args.reference_run, args.reference_manifest_sha256)
    runtime = {name: importlib.metadata.version(name) for name in reference_tools.RUNTIME_PACKAGES}
    if runtime != reference['manifest']['simulator_packages']:
        raise ValueError('Audit runtime differs from the original development run')
    if not 1024 <= args.server_port <= 65535 or args.server_port in {m['server_port'] for m in reference['manifest']['models']}:
        raise ValueError('A separate dedicated seeded-server port is required')
    model = dict(reference['model'], server_port=args.server_port, inference_gpu=args.inference_gpu)
    server = server_process(args.server_pid, model, reference['manifest']['repo'])
    if any(args.output.resolve().is_relative_to(path) for path in (reference['root'], Path(model['model_path']))):
        raise ValueError('Audit output must be outside reference run and checkpoint')
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    (output / 'code').mkdir()
    sources = dict(reference['sources'])
    sources.update({Path(reference_tools.__file__).resolve(): FROZEN_EXTEND_SHA256, Path(__file__).resolve(): original.sha(__file__)})
    copied = {}
    for source in (Path(original.__file__), Path(reference_tools.__file__), Path(__file__)):
        target = output / 'code' / source.name
        target.write_bytes(source.read_bytes())
        copied[source.name] = original.sha(source)
        if original.sha(target) != copied[source.name]:
            raise ValueError('Code copy changed while pinning')
        target.chmod(0o444)
    run = copy.deepcopy(reference['manifest'])
    run.update(models=[model], renderer_gpu=args.renderer_gpu, simulator_packages=runtime,
               audit=dict(kind='four_fresh_base_repeats_v1', plan=[dict(task=t, seed=s, repeat=r) for t, s, r in plan()],
                          reference_run=str(reference['root']), reference_manifest_sha256=reference['manifest_sha256'],
                          source_sha256={str(p): d for p, d in sources.items()}, copied_code_sha256=copied,
                          server_process=server, owner_pid=os.getpid(), owner_start_ticks=proc_start(os.getpid()),
                          created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          saves_initial_pixels=True, captures_actual_first_processor_request=True,
                          state_restoration=False, retry=False, resume=False, filtering=False))
    original.atomic_json(output / 'run_manifest.json', run)
    manifest_sha = original.sha(output / 'run_manifest.json')
    child, child_identity = None, None
    def stop_owned():
        if child is not None and child.poll() is None:
            if child_identity is not None and proc_start(child.pid) != child_identity['process_start_ticks']:
                raise RuntimeError('Owned child PID identity changed; refusing termination')
            reference_tools._terminate_owned(child)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    env = dict(os.environ, PYTHONHASHSEED='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(args.renderer_gpu),
               MUJOCO_EGL_DEVICE_ID=str(args.renderer_gpu), MUJOCO_GL='egl', TOKENIZERS_PARALLELISM='false')
    lock_path = Path(tempfile.gettempdir()) / f'robocasa-dev-render-{os.getuid()}-gpu-{args.renderer_gpu}.lock'
    pinned_results = {}
    try:
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for task, seed, repeat in plan():
                check_sources(run, output)
                if original.sha(output / 'run_manifest.json') != manifest_sha:
                    raise ValueError('Audit manifest changed')
                if original.gpu_pids(args.renderer_gpu):
                    raise RuntimeError('Renderer GPU occupied; refusing concurrent rendering')
                directory = output / 'episodes' / task / str(seed) / f'repeat_{repeat}'
                directory.mkdir(parents=True, exist_ok=False)
                command = [sys.executable, '-u', str(output / 'code' / Path(__file__).name), 'episode',
                           '--output', str(output), '--manifest-sha256', manifest_sha, '--task', task, '--repeat', str(repeat)]
                with (directory / 'process.log').open('xb') as log:
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    child_identity = dict(pid=child.pid, process_start_ticks=proc_start(child.pid), command=command,
                                          cwd=str(Path.cwd()), parent_pid=os.getpid(), renderer_gpu=args.renderer_gpu,
                                          run_manifest_sha256=manifest_sha)
                    original.atomic_json(directory / 'process_identity.json', child_identity)
                    returncode = child.wait()
                original.atomic_json(directory / 'process_exit.json', dict(returncode=returncode, **child_identity))
                if returncode:
                    raise RuntimeError(f'Audit episode exited {returncode}; preserve artifacts without retry')
                _, record = read_result(output, run, task, repeat)
                for name in ('result.json', 'process_exit.json', 'process_identity.json', 'process.log'):
                    pinned_results[str(directory / name)] = original.sha(directory / name)
                print(json.dumps({k: record[k] for k in ('task', 'seed', 'repeat', 'success', 'steps')}), flush=True)
            check_sources(run, output)
            if (original.sha(output / 'run_manifest.json') != manifest_sha or
                    any(original.sha(p) != d for p, d in pinned_results.items())):
                raise ValueError('Completed audit evidence changed before summary')
            summary = summarize(output, run)
            original.atomic_json(output / 'summary.json', summary)
            original.atomic_json(output / 'audit_status.json', dict(status='complete', episodes=4))
            print(json.dumps(summary), flush=True)
    except BaseException as error:
        stop_owned()
        original.atomic_json(output / 'audit_status.json', dict(status='failed', error=f'{type(error).__name__}: {error}'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    parent = sub.add_parser('run')
    parent.add_argument('--reference-run', type=Path, required=True)
    parent.add_argument('--reference-manifest-sha256', required=True)
    parent.add_argument('--server-port', type=int, required=True)
    parent.add_argument('--server-pid', type=int, required=True)
    parent.add_argument('--inference-gpu', type=int, default=3)
    parent.add_argument('--renderer-gpu', type=int, default=5)
    parent.add_argument('--output', type=Path, required=True)
    child = sub.add_parser('episode')
    child.add_argument('--output', type=Path, required=True)
    child.add_argument('--manifest-sha256', required=True)
    child.add_argument('--task', choices=original.TASKS, required=True)
    child.add_argument('--repeat', type=int, choices=REPEATS, required=True)
    args = parser.parse_args()
    (run if args.command == 'run' else episode)(args)


if __name__ == '__main__':
    main()
