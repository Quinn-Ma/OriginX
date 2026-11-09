"""Closed-loop RoboCasa365 evaluation with ORDINARY inputs (original task instruction, clean RGB, state).

One process = one policy (official checkpoint, or a trained BranchBox LoRA adapter) on one physical GPU,
iterating over tasks and fixed seeds. Results are appended per episode so the run can be resumed.
"""
from pathlib import Path
import argparse, json, os, sys, time, hashlib, traceback
import numpy as np

ROOT = Path('/dev/shm/qinzhen/sbl')
NATIVE = Path('/dev/shm/qinzhen/lerobot-three-model-adapters-20261106')
TRAIN = Path('/dev/shm/qinzhen/lerobot-training-ready-20261106')
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
SEEN = ['DeliverStraw', 'GetToastedBread', 'PackIdenticalLunches', 'SearingMeat', 'StirVegetables']
UNSEEN = ['ArrangeTea', 'CuttingToolSelection', 'RecycleBottlesByType', 'WaffleReheat', 'WeighIngredients']


def parse_seeds(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def write_video(path, frames, fps=5):
    import av
    container = av.open(str(path), mode='w'); stream = container.add_stream('libx264', rate=fps)
    stream.width, stream.height, stream.pix_fmt = 256, 256, 'yuv420p'; stream.options = {'crf': '23'}
    for fr in frames:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def first_attr(env, name):
    for getter in (lambda: env.call(name), lambda: env.get_attr(name)):
        try:
            v = getter()
            if isinstance(v, (tuple, list)):
                v = v[0]
            if v is not None:
                return v
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['xr1', 'awomo_i', 'awomo_v'], required=True)
    ap.add_argument('--condition', choices=['original', 'S', 'BL', 'F', 'FB'], required=True)
    ap.add_argument('--adapter', default=None); ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--tasks', default=','.join(SEEN + UNSEEN)); ap.add_argument('--seeds', default='0-19')
    ap.add_argument('--out', default=None); ap.add_argument('--video', action='store_true'); ap.add_argument('--max-episodes', type=int, default=0)
    ap.add_argument('--policy-seed', type=int, default=4242); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    if a.gpu in (2, 3):
        raise SystemExit('GPUs 2/3 are faulted (Recovery Action: Reset)')
    if a.condition != 'original' and not a.adapter:
        raise SystemExit('--adapter required for trained conditions')
    out = Path(a.out) if a.out else ROOT / 'eval' / ('%s-%s%s' % (a.model, a.condition, a.tag))
    (out / 'videos').mkdir(parents=True, exist_ok=True); (out / 'frames').mkdir(exist_ok=True)
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(a.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      PYNPUT_BACKEND='dummy', NUMBA_CACHE_DIR=str(ROOT / 'cache/numba'), XDG_CACHE_HOME=str(ROOT / 'cache'),
                      TMPDIR=str(ROOT / 'tmp'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # the simulator process only needs EGL; the policy worker pins its own GPU
    for d in ['cache/numba', 'cache', 'tmp', 'logs']:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(NATIVE)); sys.path.insert(0, str(TRAIN / 'scripts'))
    sys.path.insert(0, '/dev/shm/qinzhen/lerobot-robocasa-20261105/src/lerobot/src')
    from lerobot.envs.configs import RoboCasaEnvConfig
    from lerobot.envs.factory import make_env
    results_path = out / 'results.jsonl'; done = set()
    if results_path.exists():
        for line in open(results_path):
            r = json.loads(line); done.add((r['task'], r['seed']))
    seeds = parse_seeds(a.seeds); tasks = a.tasks.split(',')
    todo = [(t, s) for t in tasks for s in seeds if (t, s) not in done]
    if a.max_episodes:
        todo = todo[:a.max_episodes]
    print('EVAL', a.model, a.condition, 'gpu', a.gpu, 'todo', len(todo), 'done', len(done), flush=True)
    if not todo:
        print('NOTHING_TO_DO'); return 0
    t_load = time.time()
    if a.condition == 'original':
        from adapter import NativeRoboCasaAdapter
        policy = NativeRoboCasaAdapter(a.model, physical_gpu=a.gpu, seed=a.policy_seed)
    elif a.condition == 'FB':
        sys.path.insert(0, str(ROOT / 'eval'))
        from feedback_adapter import FeedbackAdapter
        policy = FeedbackAdapter(a.model, a.adapter, physical_gpu=a.gpu, seed=a.policy_seed)
    else:
        from adapter_inference import BranchBoxInferenceAdapter
        policy = BranchBoxInferenceAdapter(a.model, a.adapter, physical_gpu=a.gpu, seed=a.policy_seed)
    print('POLICY_LOADED', round(time.time() - t_load, 1), 's', json.dumps({k: str(v)[:200] for k, v in policy.load_receipt.items()}), flush=True)
    meta = {'model': a.model, 'condition': a.condition, 'adapter': a.adapter, 'gpu': a.gpu, 'policy_seed': a.policy_seed,
            'adapter_sha256': hashlib.sha256(Path(a.adapter).read_bytes()).hexdigest() if a.adapter else None,
            'inputs': 'ordinary: original task instruction, clean RGB, raw state16; no box/stage prompts; no feedback'}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    try:
        for task in tasks:
            task_todo = [s for (t, s) in todo if t == task]
            if not task_todo:
                continue
            cfg = RoboCasaEnvConfig(task=task, split='pretrain', obj_registries=['objaverse', 'lightwheel'])
            env = make_env(cfg, n_envs=1, use_async_envs=False)[task][0]
            horizon = first_attr(env, '_max_episode_steps')
            try:
                horizon = int(horizon)
            except Exception:
                from lerobot.sim.integrations.robocasa import _get_task_horizon
                horizon = int(_get_task_horizon(task))
            for seed in task_todo:
                t0 = time.time(); rec = {'model': a.model, 'condition': a.condition, 'task': task, 'seed': seed, 'horizon': horizon,
                                         'split': 'seen' if task in SEEN else 'unseen'}
                try:
                    obs, info = env.reset(seed=seed); policy.reset()
                    instruction = first_attr(env, 'task_description') or task
                    rec['instruction'] = str(instruction)
                    first = obs['observation.images.robot0_agentview_left'][0]
                    rec['first_frame_sha256'] = hashlib.sha256(np.ascontiguousarray(first).tobytes()).hexdigest()
                    rec['first_state'] = [round(float(x), 5) for x in np.asarray(obs['observation.state']).reshape(-1)]
                    frames = [first.copy()]; success = False; steps = 0; terminated = truncated = False; latencies = []
                    while True:
                        batch = {k: v for k, v in obs.items()}; batch['task'] = [str(instruction)]
                        tq = time.time(); action = policy.select_action(batch); latencies.append(time.time() - tq)
                        obs, reward, term, trunc, info = env.step(action.numpy().astype(np.float32))
                        steps += 1
                        success = bool(np.asarray(info['is_success']).reshape(-1)[0])
                        terminated = bool(np.asarray(term).reshape(-1)[0]); truncated = bool(np.asarray(trunc).reshape(-1)[0])
                        if steps % 4 == 0:
                            frames.append(obs['observation.images.robot0_agentview_left'][0].copy())
                        if success or terminated or truncated or steps >= horizon:
                            break
                    rec.update(location_trace=getattr(policy, 'location_trace', None), success=success, steps=steps, terminated=terminated, truncated=truncated,
                               policy_queries=len(policy.query_receipts), elapsed_s=round(time.time() - t0, 1),
                               query_latency_median_s=round(float(np.median(latencies)), 4) if latencies else None,
                               status='ok')
                    last = obs['observation.images.robot0_agentview_left'][0]
                    from PIL import Image
                    Image.fromarray(first).save(out / 'frames' / ('%s-s%02d-first.png' % (task, seed)))
                    Image.fromarray(last).save(out / 'frames' / ('%s-s%02d-last.png' % (task, seed)))
                    if a.video:
                        frames.append(last.copy()); write_video(out / 'videos' / ('%s-s%02d.mp4' % (task, seed)), frames)
                except Exception:
                    rec.update(status='error', error=traceback.format_exc()[-4000:], elapsed_s=round(time.time() - t0, 1), success=None)
                with results_path.open('a') as f:
                    f.write(json.dumps(rec) + '\n')
                print('EPISODE', task, seed, rec.get('status'), 'success=%s' % rec.get('success'), 'steps=%s' % rec.get('steps'),
                      'elapsed=%ss' % rec.get('elapsed_s'), flush=True)
            env.close()
    finally:
        policy.close()
    print('EVAL_DONE', a.model, a.condition, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
