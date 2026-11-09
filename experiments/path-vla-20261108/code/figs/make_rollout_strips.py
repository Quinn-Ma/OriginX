"""Qualitative strips: 6 evenly spaced frames from the saved 5-fps agent-view videos of selected episodes (run on cluster).

Writes figs/rollouts.png (4 rows x 6 frames of 256 px at x = 260 i, y = 284 r + 22) and figs/rollouts.json (the step of
every frame). The paper figure is re-composed from this sheet by figs/recompose_rollouts.py, which crops the frames,
stamps the step index at print size and leaves the row titles to LaTeX; the sheet itself carries a short row label and
small stamps for inspection only.

Step of a frame (eval/eval_runner.py): the first observation (step 0), one frame after every 4th step, then the final
observation appended, so a video of an S-step episode has S // 4 + 2 frames; frame i is step 4 i except the last, which
is step S. (An earlier version labelled the last frame 4 i, i.e. up to 4 steps past the end of the episode.)
"""
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
            assert len(fr) == res['steps'] // 4 + 2, (len(fr), res['steps'])
            return [fr[i] for i in idx], [res['steps'] if i == len(fr) - 1 else int(i * 4) for i in idx], res
    raise FileNotFoundError((model, cond, task, seed))


rows = []
for model, cond, task, seed, label in SEL:
    fr, steps, res = frames_of(model, cond, task, seed)
    rows.append((fr, steps, res, label))
W = 6 * 256 + 5 * 4; H = 256 + 22
sheet = Image.new('RGB', (W, len(rows) * (H + 6)), 'white'); d = ImageDraw.Draw(sheet)
for r, (fr, steps, res, label) in enumerate(rows):
    y0 = r * (H + 6)
    d.text((2, y0 + 2), '%s  |  %s' % (label, 'success at step %d' % res['steps'] if res['success'] else 'timeout at %d' % res['steps']), fill='black', font=font)
    for i, (im, st) in enumerate(zip(fr, steps)):
        img = Image.fromarray(im); dd = ImageDraw.Draw(img); dd.text((4, 240), 't=%d' % st, fill='white', font=font)
        sheet.paste(img, (i * 260, y0 + 22))
sheet.save(OUT / 'rollouts.png')
(OUT / 'rollouts.json').write_text(json.dumps([{'label': lab, 'task': res['task'], 'seed': res['seed'], 'steps': res['steps'],
                                                'success': res['success'], 'frame_steps': st} for fr, st, res, lab in rows], indent=1))
print('saved', len(rows))
