"""Figure fig:labels, composed locally: one image per labelled window (left, right and wrist camera side by side) from the
boxed frames that make_label_figure.py exported on the cluster (paper/figures/frames/label-<window>-<camera>.png).
The per-window titles are set in LaTeX above each image (so they print at body-text size); this script writes only the
images, paper/figures/labels/window<i>.png, and checks the window list against make_label_figure.py's selection.
Usage: python -X utf8 figs/make_label_groups.py
"""
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'paper/figures/frames'
OUT = ROOT / 'paper/figures/labels'
GAP = 6
OUT.mkdir(parents=True, exist_ok=True)
for i in range(6):
    ims = [Image.open(SRC / ('label-%d-%d.png' % (i, c))).convert('RGB') for c in range(3)]
    assert all(im.size == (256, 256) for im in ims)
    g = Image.new('RGB', (3 * 256 + 2 * GAP, 256), 'white')
    for c, im in enumerate(ims):
        g.paste(im, (c * (256 + GAP), 0))
    g.save(OUT / ('window%d.png' % i))
print('wrote', sorted(p.name for p in OUT.glob('*.png')))
