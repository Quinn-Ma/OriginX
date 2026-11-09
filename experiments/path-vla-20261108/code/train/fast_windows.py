"""Drop-in replacement for training_data.DemonstrationWindows backed by the decoded frame cache.

Produces exactly the same window dict (frames / states / actions / instruction / stage) as the LeRobot-backed reader,
but reads frames from memory-mapped uint8 arrays and the per-episode v2.1 parquet instead of decoding video per index.
"""
from pathlib import Path
import json
import numpy as np
ROOT = Path('/dev/shm/qinzhen/lerobot-training-ready-20261106')
CACHE = Path('/dev/shm/qinzhen/sbl/framecache')
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
OFFSETS = {'xr1': [-6, -4, -2, 0], 'awomo_i': list(range(-360, 1, 40)) + [16], 'awomo_v': list(range(-96, 33, 4))}


class FastWindows:
    def __init__(self, task, split='train'):
        import pyarrow.parquet as pq
        if split not in ['train', 'dev']:
            raise ValueError(split)
        manifest = json.loads((ROOT / 'receipts/training-data-manifest.json').read_text())
        row = next((r for r in manifest['tasks'] if r['task'] == task), None)
        if row is None:
            raise ValueError('Task outside fixed seen training panel')
        self.task = task; self.split = split; self.episodes = row[split + '_episodes']
        old = Path(row['root']).with_name('lerobot_old')
        names = {r['task_index']: r['task'] for r in map(json.loads, (old / 'meta/tasks.jsonl').read_text().splitlines())}
        self.rows = {}; self.tables = {}; self.frames = {}
        for ep in self.episodes:
            t = pq.read_table(old / 'data' / ('chunk-%03d' % (ep // 1000)) / ('episode_%06d.parquet' % ep),
                              columns=['frame_index', 'observation.state', 'action', 'task_index', 'subtask_idx']).to_pydict()
            n = len(t['frame_index']); assert t['frame_index'] == list(range(n))
            self.tables[ep] = {'state': np.asarray(t['observation.state'], dtype=np.float32), 'action': np.asarray(t['action'], dtype=np.float32),
                               'instruction': [names[i] for i in t['task_index']], 'stage': np.asarray(t['subtask_idx'], dtype=np.int64)}
            self.rows[ep] = np.arange(n)
            self.frames[ep] = {c: np.load(CACHE / task / ('e%03d' % ep) / (c + '.npy'), mmap_mode='r') for c in CAMERAS}
            for c in CAMERAS:
                assert self.frames[ep][c].shape[0] == n, (task, ep, c, self.frames[ep][c].shape, n)

    def window(self, variant, episode, anchor, horizon=32):
        if episode not in self.rows:
            raise ValueError('Episode outside frozen split')
        n = len(self.rows[episode]); tb = self.tables[episode]
        if anchor % 16 or anchor < 0 or anchor + max(horizon, 32) >= n:
            raise ValueError('Full future horizon required; anchor must be a multiple of 16')
        offsets = OFFSETS[variant]; fr = self.frames[episode]
        frames = {o: np.stack([np.ascontiguousarray(fr[c][max(0, anchor + o)]) for c in CAMERAS]) for o in offsets}
        return {'task': self.task, 'episode': episode, 'anchor': anchor, 'instruction': tb['instruction'][anchor],
                'state': tb['state'][anchor].copy(), 'states': {o: tb['state'][max(0, anchor + o)].copy() for o in offsets if o <= 0},
                'frames': frames, 'actions': tb['action'][anchor:anchor + horizon].copy(), 'stage': int(tb['stage'][anchor]),
                'split': self.split, 'annotation': None, 'provenance': 'official_box_human_pretrain'}


def check_against_lerobot(n_windows=4):
    """Exactness check versus the LeRobot-backed reader on random windows."""
    import random, sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    from training_data import DemonstrationWindows
    rng = random.Random(3); report = []
    for task in ['DeliverStraw', 'SearingMeat']:
        slow = DemonstrationWindows(task, 'train'); fast = FastWindows(task, 'train')
        for _ in range(n_windows):
            ep = rng.choice(slow.episodes); n = len(slow.rows[ep]); anchor = rng.randrange(1, (n - 40) // 16) * 16
            for variant in ['xr1', 'awomo_v']:
                a = slow.window(variant, ep, anchor); b = fast.window(variant, ep, anchor)
                same_frames = all(np.array_equal(a['frames'][o], b['frames'][o]) for o in a['frames'])
                maxdiff = max(int(np.abs(a['frames'][o].astype(int) - b['frames'][o].astype(int)).max()) for o in a['frames'])
                same = (same_frames and np.allclose(a['state'], b['state']) and np.allclose(a['actions'], b['actions'])
                        and a['instruction'] == b['instruction'] and a['stage'] == b['stage']
                        and all(np.allclose(a['states'][o], b['states'][o]) for o in a['states']))
                report.append({'task': task, 'episode': ep, 'anchor': anchor, 'variant': variant, 'identical': bool(same), 'max_pixel_diff': maxdiff})
                print(report[-1], flush=True)
    return report


if __name__ == '__main__':
    import json as _j
    r = check_against_lerobot()
    print('ALL_IDENTICAL', all(x['identical'] for x in r))
