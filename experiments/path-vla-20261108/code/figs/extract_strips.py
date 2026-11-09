"""Frame library for figure design: per-stage frames of a few training demonstrations, clean and with the
pseudo-label target box, plus metadata (stage description, target, box). Runs on the cluster; CPU only."""
from pathlib import Path
import json, collections
from PIL import Image, ImageDraw
ROOT = Path('/dev/shm/qinzhen/sbl'); OUT = ROOT / 'figs/library'; OUT.mkdir(parents=True, exist_ok=True)
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
anns = json.loads((ROOT / 'ann/panel/annotations-train.json').read_text())
by_ep = collections.defaultdict(list)
for a in anns:
    by_ep[(a['task'], a['episode'])].append(a)
meta = []
for task in ['DeliverStraw', 'SearingMeat', 'GetToastedBread', 'StirVegetables', 'PackIdenticalLunches']:
    eps = sorted([k for k in by_ep if k[0] == task], key=lambda k: -len({a['stage'] for a in by_ep[k] if a['frames']['0'][0]['visible']}))
    for (t, ep) in eps[:2]:
        rows = by_ep[(t, ep)]; stages = sorted({a['stage'] for a in rows})
        for st in stages:
            cand = [a for a in rows if a['stage'] == st]
            vis0 = [a for a in cand if a['frames']['0'][0]['visible']] or [a for a in cand if a['visible_cameras'] > 0]
            if not vis0:
                continue
            a = vis0[len(vis0) // 2]
            for ci, cam in enumerate(CAMERAS):
                src = ROOT / 'frames' / t / ('e%03d' % ep) / ('f%05d-%s.png' % (a['anchor'], cam))
                im = Image.open(src).convert('RGB'); base = 'strip-%s-e%03d-s%02d-%s' % (t, ep, st, ['left', 'right', 'wrist'][ci])
                im.save(OUT / (base + '-clean.png'))
                e = a['frames']['0'][ci]; boxed = im.copy()
                if e['visible']:
                    d = ImageDraw.Draw(boxed); b = [v * 255 for v in e['box_xyxy']]; d.rectangle(b, outline=(255, 0, 0), width=3)
                boxed.save(OUT / (base + '-box.png'))
            meta.append({'task': t, 'episode': ep, 'stage': st, 'anchor': a['anchor'], 'kind': a['kind'], 'description': a['description'],
                         'target_prompt': a['target_prompt'], 'visible': [e['visible'] for e in a['frames']['0']],
                         'box_xyxy': [e['box_xyxy'] for e in a['frames']['0']], 'files_prefix': 'strip-%s-e%03d-s%02d' % (t, ep, st)})
(OUT / 'library.json').write_text(json.dumps(meta, indent=1))
print('stages', len(meta), 'files', len(list(OUT.glob('*.png'))))
