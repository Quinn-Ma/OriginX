"""Daemon: when a training run completes, launch its ordinary-input evaluation in 4 seed shards, memory-aware.

Runs on the cluster. Idempotent: a shard is launched once (marker files), and eval_runner itself resumes.
"""
from pathlib import Path
import json, os, subprocess, time
ROOT = Path('/dev/shm/qinzhen/sbl'); PY = '/dev/shm/qinzhen/lerobot-robocasa-20261105/venv/bin/python'
GPUS = [0, 1, 4, 5, 6, 7]
NEED_GB = {'xr1': 13, 'awomo_i': 21, 'awomo_v': 27}
SHARDS = [('q0', '0-4'), ('q1', '5-9'), ('q2', '10-14'), ('q3', '15-19')]
MARK = ROOT / 'eval/chain-markers'; MARK.mkdir(parents=True, exist_ok=True)
LOG = (ROOT / 'logs/chain.log').open('a', buffering=1)
pending = []  # (gpu, need_gb, launch_time) reservations while a worker loads


def log(*a):
    LOG.write(time.strftime('%Y-%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a) + '\n')


def free_gb():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
    free = {}
    for line in out.strip().splitlines():
        i, used, total = [int(x) for x in line.split(',')]
        free[i] = (total - used) / 1024
    now = time.time()
    for g, need, t in pending:
        if now - t < 240:
            free[g] = free.get(g, 0) - need
    return free


def launch(model, cond, adapter, tag, seeds):
    free = free_gb(); need = NEED_GB[model]
    cands = [(free[g], g) for g in GPUS if free[g] >= need + 4]
    if not cands:
        return False
    gpu = max(cands)[1]
    out = ROOT / 'eval' / ('%s-%s-%s' % (model, cond, tag))
    cmd = [PY, str(ROOT / 'eval/eval_runner.py'), '--model', model, '--condition', cond, '--adapter', str(adapter), '--gpu', str(gpu),
           '--seeds', seeds, '--video', '--out', str(out)]
    logf = (ROOT / 'logs' / ('eval-%s-%s-%s.log' % (model, cond, tag))).open('ab')
    subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd=ROOT)
    pending.append((gpu, need, time.time())); (MARK / ('%s-%s-%s' % (model, cond, tag))).write_text(str(gpu))
    log('LAUNCH', model, cond, tag, 'gpu', gpu, 'free', round(free[gpu], 1))
    return True


def main():
    log('chain start')
    while True:
        try:
            for run in sorted((ROOT / 'train/runs').glob('*-*')):
                st = run / 'status.json'
                if not st.exists():
                    continue
                if json.loads(st.read_text()).get('status') != 'completed':
                    continue
                model, cond = run.name.split('-'); adapter = run / 'final/adapter.pt'
                if not adapter.exists():
                    continue
                for tag, seeds in SHARDS:
                    if (MARK / ('%s-%s-%s' % (model, cond, tag))).exists():
                        continue
                    if launch(model, cond, adapter, tag, seeds):
                        time.sleep(20)
            if (ROOT / 'eval/chain.STOP').exists():
                log('STOP file found'); return
        except Exception as exc:
            log('ERROR', repr(exc))
        time.sleep(60)


if __name__ == '__main__':
    main()
