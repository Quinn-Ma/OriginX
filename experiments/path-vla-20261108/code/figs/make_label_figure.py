"""Compose the label-example figure: 4 windows x 3 cameras with the selected target box, from the frozen panel."""
from pathlib import Path
import json, random
from PIL import Image, ImageDraw, ImageFont
ROOT = Path('/dev/shm/qinzhen/sbl'); CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
anns = json.loads((ROOT / 'ann/panel/annotations-train.json').read_text())
WANT = [('PackIdenticalLunches', 47, 496), ('SearingMeat', 0, 2288), ('DeliverStraw', 63, 64), ('GetToastedBread', 22, 1312), ('SearingMeat', 23, 976), ('StirVegetables', 67, 352)]
idx = {(a['task'], a['episode'], a['anchor']): a for a in anns}
picks = [idx[k] for k in WANT]
out = ROOT / 'figs'; out.mkdir(exist_ok=True)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
W = 3 * 256 + 2 * 6; H = 256 + 18
sheet = Image.new('RGB', (W * 2 + 20, H * 3 + 10), 'white')
for i, a in enumerate(picks):
    col, row = i % 2, i // 2; x0 = col * (W + 20); y0 = row * (H + 5)
    d = ImageDraw.Draw(sheet); d.text((x0 + 2, y0), '%s: "%s"' % (a['task'], a['description']), fill='black', font=font)
    for ci, cam in enumerate(CAMERAS):
        im = Image.open(ROOT / 'frames' / a['task'] / ('e%03d' % a['episode']) / ('f%05d-%s.png' % (a['anchor'], cam))).convert('RGB')
        e = a['frames']['0'][ci]; dd = ImageDraw.Draw(im)
        if e['visible']:
            b = [v * 255 for v in e['box_xyxy']]; dd.rectangle(b, outline=(255, 0, 0), width=2)
        sheet.paste(im, (x0 + ci * 262, y0 + 18))
        Image.fromarray(__import__('numpy').array(im)).save(out / ('label-%d-%d.png' % (i, ci)))
sheet.save(out / 'label_examples.png'); json.dump([{k: a[k] for k in ['task', 'episode', 'anchor', 'kind', 'description']} for a in picks], open(out / 'label_examples.json', 'w'), indent=1)
print('ok', [a['description'] for a in picks])
