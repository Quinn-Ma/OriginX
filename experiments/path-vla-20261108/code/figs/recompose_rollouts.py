"""Figure fig:rollouts, re-composed locally from the frames that make_rollout_strips.py extracted on the cluster.

The 5-fps agent-view videos live on the cluster only; their six selected frames per episode are kept in
paper/figures/fig_rollouts_frames.png (the sheet written by make_rollout_strips.py, renamed: 4 rows x 6 frames of
256 px at x = 260 i, y = 284 r + 22, with old step stamps in the bottom 24 px and a title band above each row).
This script crops every frame to rows 40..200 (dropping the cabinet band at the top, and the robot base and the old
stamps at the bottom; every object the caption mentions stays in view),
stamps the true step index, and writes one strip per row (paper/figures/rollouts/row<r>.png). Row titles are set in
LaTeX next to each strip, so all text prints at >= 7 pt.

Step index of a frame (eval/eval_runner.py): the runner stores the first observation (step 0) and one frame after every
4th step, then appends the final observation, so a video of an S-step episode has n = S // 4 + 2 frames; frame i < n-1
is step 4 i and the last frame is step S. make_rollout_strips.py took i = round(linspace(0, n-1, 6)).
Usage: python -X utf8 figs/recompose_rollouts.py
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / 'paper/figures/fig_rollouts_frames.png'
OUT = ROOT / 'paper/figures/rollouts'
# rows of the sheet, in order: (model, condition, task, seed); steps come from the episode logs
ROWS = [('xr1', 'original', 'WeighIngredients', 12), ('xr1', 'F', 'WeighIngredients', 12),
        ('awomo_v', 'original', 'PackIdenticalLunches', 17), ('awomo_v', 'F', 'PackIdenticalLunches', 17)]
Y0, Y1 = 40, 200            # kept rows of each 256-px frame
GAP = 4                     # px between frames in a strip
FONT = ImageFont.truetype(str(Path(__import__("matplotlib").get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 24)  # 24 px = 6.8 pt in print


def episode(model, cond, task, seed):
    hits = []
    for f in sorted((ROOT / 'results/eval').glob('%s-%s-q*/results.jsonl' % (model, cond))):
        for line in open(f):
            r = json.loads(line)
            if r['task'] == task and r['seed'] == seed:
                hits.append(r)
    assert len(hits) == 1, (model, cond, task, seed, len(hits))
    return hits[0]


def frame_steps(S, k=6):
    n = S // 4 + 2
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return [S if i == n - 1 else 4 * int(i) for i in idx], [4 * int(i) for i in idx]


sheet = Image.open(SHEET).convert('RGB')
assert sheet.size == (6 * 256 + 5 * 4, 4 * 284), sheet.size
OUT.mkdir(parents=True, exist_ok=True)
meta = []
for r, (model, cond, task, seed) in enumerate(ROWS):
    rec = episode(model, cond, task, seed)
    steps, old_stamps = frame_steps(rec['steps'])
    strip = Image.new('RGB', (6 * 256 + 5 * GAP, Y1 - Y0), 'white')
    for i, st in enumerate(steps):
        fr = sheet.crop((i * 260, r * 284 + 22 + Y0, i * 260 + 256, r * 284 + 22 + Y1))
        d = ImageDraw.Draw(fr, 'RGBA')
        txt = 't = {:,}'.format(st)
        x0, y0, x1, y1 = d.textbbox((0, 0), txt, font=FONT)
        w, h = x1 - x0, y1 - y0
        d.rectangle((0, fr.height - h - 12, w + 12, fr.height), fill=(0, 0, 0, 165))
        d.text((6 - x0, fr.height - h - 6 - y0), txt, fill='white', font=FONT)
        strip.paste(fr, (i * (256 + GAP), 0))
    strip.save(OUT / ('row%d.png' % r))
    meta.append({'row': r, 'model': model, 'condition': cond, 'task': task, 'seed': seed, 'success': rec['success'],
                 'steps': rec['steps'], 'horizon': rec['horizon'], 'first_frame_sha256': rec['first_frame_sha256'],
                 'frame_steps': steps, 'old_stamps_4i': old_stamps})
(OUT / 'rows.json').write_text(json.dumps(meta, indent=1))
for m in meta:
    print(m['row'], m['model'], m['condition'], m['task'], m['seed'], 'success' if m['success'] else 'timeout', m['steps'], m['frame_steps'])
print('same first frame within each pair:', meta[0]['first_frame_sha256'] == meta[1]['first_frame_sha256'],
      meta[2]['first_frame_sha256'] == meta[3]['first_frame_sha256'])
