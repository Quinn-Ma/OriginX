"""train_prepared.py with (1) the decoded-frame window reader and (2) a background prefetch thread.

Everything else (preflight gates, losses, optimizer, logging, checkpoints, arm matching) is train_prepared.execute.
"""
from pathlib import Path
import argparse, json, os, queue, random, sys, threading, time
ROOT = Path('/dev/shm/qinzhen/lerobot-training-ready-20261106')
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import training_data
from fast_windows import FastWindows
training_data.DemonstrationWindows = FastWindows  # execute() imports the name from this module at call time
import train_prepared


def execute_prefetch(config, panel, gpu, output):
    """Same loop as train_prepared.execute, but windows are prepared ahead in a thread so the GPU never waits on I/O."""
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1',
                      TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'tmp/cache'), HF_HOME=str(ROOT / 'tmp/hf'), PYNPUT_BACKEND='dummy')
    import torch
    from lerobot_policy_branchbox_train import BranchBoxTrainConfig
    from lerobot.policies.factory import make_policy
    from lerobot.envs.configs import RoboCasaEnvConfig
    from lerobot.optim.optimizers import AdamWConfig
    from training_data import paint_frames
    if gpu == 3:
        raise ValueError('GPU3 is not cleared for use')
    if output.exists():
        raise FileExistsError('Refuse to overwrite an existing run')
    output.mkdir(parents=True); (output / 'frozen-config.json').write_text(json.dumps(config, indent=2))
    seed = config['seed']; rng = random.Random(seed); torch.manual_seed(seed)
    rows = panel['windows']; datasets = {t: FastWindows(t, 'train') for t in sorted({r['task'] for r in rows})}
    by_task = {t: [r for r in rows if r['task'] == t] for t in datasets}; tasks = list(by_task)
    policy = None; updates = 0; micro = 0; started = time.monotonic(); status = 'incomplete'
    total_micro = config['optimizer_updates'] * config['gradient_accumulation']
    # Pre-draw the sampling sequence with the SAME rng call pattern as train_prepared (rng.choice per microbatch).
    schedule = [rng.choice(by_task[tasks[m % len(tasks)]]) for m in range(total_micro)]
    try:
        cfg = BranchBoxTrainConfig(**{k: v for k, v in config['policy'].items() if k != 'type'}, physical_gpu=gpu)
        policy = make_policy(cfg, env_cfg=RoboCasaEnvConfig(task=tasks[0]))
        horizon = policy.bridge.initial['raw_action_horizon']
        q = queue.Queue(maxsize=16)

        def producer():
            for m, row in enumerate(schedule):
                try:
                    w = datasets[row['task']].window(cfg.variant, row['episode'], row['anchor'], horizon)
                    w['annotation'] = row['annotation']; paint_frames(w, 'F')  # validate annotation for every arm
                    q.put((m, w))
                except Exception as exc:  # surface loader errors in the consumer
                    q.put((m, exc)); return
            q.put(None)
        th = threading.Thread(target=producer, daemon=True); th.start()
        optim_cfg = AdamWConfig(**{k: v for k, v in config['optimizer'].items() if k != 'type'})
        optimizer = optim_cfg.build(policy.get_optim_params()); accum = config['gradient_accumulation']
        with (output / 'training.jsonl').open('a', buffering=1) as log:
            for step in range(config['optimizer_updates']):
                optimizer.zero_grad(set_to_none=True); t_step = time.monotonic()
                for j in range(accum):
                    if time.monotonic() - started >= config['max_wall_seconds']:
                        raise TimeoutError('Approved wall-time budget reached')
                    item = q.get()
                    if item is None:
                        raise RuntimeError('Prefetch ended early')
                    m, w = item
                    if isinstance(w, Exception):
                        raise w
                    assert m == micro
                    loss, terms = policy({'window': w, 'seed': seed + micro}); (loss / accum).backward()
                    log.write(json.dumps({'update': step, 'microbatch': micro, 'task': w['task'], 'episode': w['episode'],
                                          'anchor': w['anchor'], 'stage': w['stage'], 'loss': float(loss.detach()), 'terms': terms}) + '\n'); micro += 1
                torch.nn.utils.clip_grad_norm_(policy.get_optim_params(), optim_cfg.grad_clip_norm, error_if_nonfinite=True)
                optimizer.step(); updates += 1
                if updates % 10 == 0:
                    print('update', updates, '/', config['optimizer_updates'], 'loss', round(float(loss.detach()), 4),
                          'step_s', round(time.monotonic() - t_step, 2), 'elapsed_min', round((time.monotonic() - started) / 60, 1), flush=True)
                if updates % 50 == 0:
                    policy.save_pretrained(output / 'latest')
                    torch.save({'optimizer': optimizer.state_dict(), 'updates': updates, 'microbatches': micro,
                                'python_rng': rng.getstate(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, output / 'optimizer.pt')
            status = 'completed'
    finally:
        if policy is not None:
            policy.save_pretrained(output / 'final'); policy.close()
        (output / 'status.json').write_text(json.dumps({'status': status, 'optimizer_updates': updates, 'microbatches': micro,
                                                         'elapsed_seconds': time.monotonic() - started, 'scored_rollouts': 0}, indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--execute', action='store_true')
    p.add_argument('--gpu', type=int, default=0); p.add_argument('--output', type=Path)
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); blockers, panel = train_prepared.preflight(cfg)
    print(json.dumps({'preflight_only': not args.execute, 'ready': not blockers, 'blockers': blockers}, indent=2), flush=True)
    if not args.execute:
        return 0
    if blockers:
        raise SystemExit('Launch refused: resolve the reported blockers first')
    if args.output is None:
        raise SystemExit('--output is required for execution')
    execute_prefetch(cfg, panel, args.gpu, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
