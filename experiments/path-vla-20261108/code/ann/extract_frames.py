"""Decode the current frame of every queued window (3 cameras) to PNG. CPU only."""
from pathlib import Path
import argparse, collections, hashlib, json
from concurrent.futures import ProcessPoolExecutor
import av
from PIL import Image

DATA = Path('/dev/shm/qinzhen/lerobot-training-ready-20261106/data')
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')


def work(args):
    task, episode, anchors, out = args
    anchors = sorted(set(anchors)); rows = []
    for cam in CAMERAS:
        path = DATA / task / 'lerobot_old/videos/chunk-000' / ('observation.images.' + cam) / ('episode_%06d.mp4' % episode)
        want = set(anchors); container = av.open(str(path)); n = 0; got = {}
        for frame in container.decode(video=0):
            if n in want:
                got[n] = frame.to_ndarray(format='rgb24')
            n += 1
        container.close()
        missing = want - set(got)
        if missing:
            raise RuntimeError('%s e%d %s: frames %s missing out of %d' % (task, episode, cam, sorted(missing), n))
        d = out / task / ('e%03d' % episode); d.mkdir(parents=True, exist_ok=True)
        for a, arr in got.items():
            f = d / ('f%05d-%s.png' % (a, cam))
            Image.fromarray(arr).save(f)
            rows.append({'task': task, 'episode': episode, 'anchor': a, 'camera': cam, 'path': str(f),
                         'rgb_sha256': hashlib.sha256(arr.tobytes()).hexdigest(), 'video_frames': n})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--queue', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--splits', nargs='+', default=['train', 'dev']); ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    groups = collections.defaultdict(list)
    for split in a.splits:
        for line in open(Path(a.queue) / (split + '.jsonl')):
            r = json.loads(line)
            if r['kind'] == 'done':
                continue
            groups[(r['task'], r['episode'])].append(r['anchor'])
    jobs = [(t, e, anc, out) for (t, e), anc in sorted(groups.items())]
    print('episodes', len(jobs), 'windows', sum(len(set(j[2])) for j in jobs), flush=True)
    with ProcessPoolExecutor(a.workers) as ex, (out / 'frames-index.jsonl').open('w') as f:
        for i, rows in enumerate(ex.map(work, jobs, chunksize=1)):
            for r in rows:
                f.write(json.dumps(r) + '\n')
            if i % 10 == 0:
                print('done', i + 1, '/', len(jobs), flush=True)
    print('EXTRACT_DONE', flush=True)


if __name__ == '__main__':
    main()
