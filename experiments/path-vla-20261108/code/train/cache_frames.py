"""Decode every selected demonstration video once into uint8 .npy arrays on the RAM disk (CPU only).

Output: /dev/shm/qinzhen/sbl/framecache/<task>/e<ep>/<camera>.npy with shape (N,256,256,3).
Pixels are decoded with PyAV rgb24, the same decoder LeRobot's pyav backend uses.
"""
from pathlib import Path
import argparse, json, hashlib
from concurrent.futures import ProcessPoolExecutor
import av, numpy as np
DATA = Path('/dev/shm/qinzhen/lerobot-training-ready-20261106/data')
OUT = Path('/dev/shm/qinzhen/sbl/framecache')
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')


def work(job):
    task, ep = job; d = OUT / task / ('e%03d' % ep); d.mkdir(parents=True, exist_ok=True); info = {}
    for cam in CAMERAS:
        f = d / (cam + '.npy')
        if f.exists():
            info[cam] = int(np.load(f, mmap_mode='r').shape[0]); continue
        path = DATA / task / 'lerobot_old/videos/chunk-000' / ('observation.images.' + cam) / ('episode_%06d.mp4' % ep)
        frames = []; c = av.open(str(path))
        for fr in c.decode(video=0):
            frames.append(fr.to_ndarray(format='rgb24'))
        c.close(); arr = np.stack(frames).astype(np.uint8)
        tmp = d / (cam + '.tmp.npy'); np.save(tmp, arr); tmp.rename(f); info[cam] = int(arr.shape[0])
    return {'task': task, 'episode': ep, 'frames': info}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=12); a = ap.parse_args()
    manifest = json.loads((DATA.parent / 'receipts/training-data-manifest.json').read_text())
    jobs = [(r['task'], ep) for r in manifest['tasks'] for ep in r['train_episodes'] + r['dev_episodes']]
    OUT.mkdir(parents=True, exist_ok=True); rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(work, jobs, chunksize=1)):
            rows.append(r)
            if i % 10 == 0:
                print('cached', i + 1, '/', len(jobs), flush=True)
    (OUT / 'index.json').write_text(json.dumps(rows, indent=1)); print('CACHE_DONE', len(rows), flush=True)


if __name__ == '__main__':
    main()
