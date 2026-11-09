"""Render a random audit sample of final annotations: raw frame + selected box per camera. For visual inspection only."""
from pathlib import Path
import argparse, json, random
from PIL import Image, ImageDraw, ImageFont
ROOT = Path('/dev/shm/qinzhen/sbl')
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--annotations', default=str(ROOT / 'ann/panel/annotations-train.json'))
    ap.add_argument('--n', type=int, default=48); ap.add_argument('--seed', type=int, default=7); ap.add_argument('--out', default=str(ROOT / 'ann/review'))
    ap.add_argument('--per-page', type=int, default=8); ap.add_argument('--kind', default=None)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    anns = json.loads(Path(a.annotations).read_text())
    if a.kind:
        anns = [x for x in anns if x['kind'] == a.kind]
    rng = random.Random(a.seed); sample = rng.sample(anns, min(a.n, len(anns)))
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except Exception:
        font = ImageFont.load_default()
    manifest = []
    for p in range(0, len(sample), a.per_page):
        rows = sample[p:p + a.per_page]; H = 256 + 36
        page = Image.new('RGB', (3 * 256 + 20, len(rows) * H), 'white'); d = ImageDraw.Draw(page)
        for i, x in enumerate(rows):
            y = i * H
            d.text((4, y + 2), '%d) %s e%d f%d [%s] %s -> %s' % (p + i + 1, x['task'], x['episode'], x['anchor'], x['kind'], x['description'][:60], x['target_prompt']), fill='black', font=font)
            for ci, cam in enumerate(CAMERAS):
                im = Image.open(ROOT / 'frames' / x['task'] / ('e%03d' % x['episode']) / ('f%05d-%s.png' % (x['anchor'], cam))).convert('RGB')
                e = x['frames']['0'][ci]; dd = ImageDraw.Draw(im)
                if e['visible']:
                    b = [v * 255 for v in e['box_xyxy']]; dd.rectangle(b, outline=(255, 0, 0), width=2)
                    cx, cy = [v * 255 for v in e['location_xy']]; dd.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 255, 0))
                else:
                    dd.text((4, 236), 'not visible (%d cand.)' % e.get('n_candidates', 0), fill=(255, 255, 0), font=font)
                page.paste(im, (ci * 256 + ci * 10, y + 20))
            manifest.append({'index': p + i + 1, 'task': x['task'], 'episode': x['episode'], 'anchor': x['anchor'], 'kind': x['kind'], 'description': x['description']})
        page.save(out / ('page%02d.png' % (p // a.per_page)))
    (out / 'sample.json').write_text(json.dumps(manifest, indent=1))
    print('REVIEW_PAGES', (len(sample) + a.per_page - 1) // a.per_page, 'windows', len(sample))


if __name__ == '__main__':
    main()
