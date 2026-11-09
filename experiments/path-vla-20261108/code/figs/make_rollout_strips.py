"""Qualitative strips: 6 evenly spaced frames from the saved 5-fps agent-view videos of selected episodes (run on cluster)."""
from pathlib import Path
import json, sys
import av, numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path('/dev/shm/qinzhen/sbl'); OUT = ROOT / 'figs'; OUT.mkdir(exist_ok=True)
SEL = json.loads(sys.argv[1])  # list of [model, cond, task, seed, label]
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)


def frames_of(model, cond, task, seed, n=6):
    for d in ROOT.glob('eval/%s-%s-q*' % (model, cond)):
        p = d / 'videos' / ('%s-s%02d.mp4' % (task, seed))
        if p.exists():
            c = av.open(str(p)); fr = [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]; c.close()
            idx = np.linspace(0, len(fr) - 1, n).round().astype(int)
            res = [json.loads(l) for l in open(d / 'results.jsonl') if json.loads(l)['task'] == task and json.loads(l)['seed'] == seed][0]
            return [fr[i] for i in idx], [int(i * 4) for i in idx], res
    raise FileNotFoundError((model, cond, task, seed))


rows = []
for model, cond, task, seed, label in SEL:
    fr, steps, res = frames_of(model, cond, task, seed)
    rows.append((fr, steps, res, label))
W = 6 * 256 + 5 * 4; H = 256 + 22
sheet = Image.new('RGB', (W, len(rows) * (H + 6)), 'white'); d = ImageDraw.Draw(sheet)
for r, (fr, steps, res, label) in enumerate(rows):
    y0 = r * (H + 6)
    d.text((2, y0 + 2), '%s  |  %s  |  %s' % (label, res['instruction'][:95], 'SUCCESS at step %d' % res['steps'] if res['success'] else 'FAILURE (timeout %d)' % res['steps']), fill='black', font=font)
    for i, (im, st) in enumerate(zip(fr, steps)):
        img = Image.fromarray(im); dd = ImageDraw.Draw(img); dd.text((4, 240), 't=%d' % st, fill='white', font=font)
        sheet.paste(img, (i * 260, y0 + 22))
sheet.save(OUT / 'rollouts.png'); print('saved', len(rows))
