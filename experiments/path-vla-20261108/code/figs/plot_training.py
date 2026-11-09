"""Training-loss curves for the nine runs (action / stage / location terms), smoothed per 20 updates.
Drawn at its final printed size (6.19 in wide; include it unscaled in the supplement), all text >= 7 pt.
Usage: python -X utf8 figs/plot_training.py train/runs paper/figures"""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '/dev/shm/qinzhen/sbl/train/runs')
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else '/dev/shm/qinzhen/sbl/figs')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman'], 'mathtext.fontset': 'stix',
                     'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
MODELS = [('xr1', 'XR-1'), ('awomo_i', 'Awomo-0.5I'), ('awomo_v', 'Awomo-0.5V')]
ARMS = [('S', '#1f77b4'), ('BL', '#ff7f0e'), ('F', '#2ca02c')]
TERMS = [('action', 'action loss'), ('stage', 'stage CE'), ('location', 'location smooth-L1')]
fig, axes = plt.subplots(3, 3, figsize=(6.19, 4.25), sharex=True)
for r, (mk, mn) in enumerate(MODELS):
    for arm, col in ARMS:
        f = ROOT / ('%s-%s' % (mk, arm)) / 'training.jsonl'
        if not f.exists():
            continue
        rows = [json.loads(l) for l in open(f)]
        upd = np.array([x['update'] for x in rows])
        for c, (term, title) in enumerate(TERMS):
            vals = np.array([x['terms'][term] for x in rows])
            if term == 'stage' and arm == 'BL' or term == 'location' and arm == 'S':
                continue  # loss switched off in this arm
            n = upd.max() + 1; per = np.array([vals[upd == u].mean() for u in range(n)])
            k = 20; sm = np.convolve(per, np.ones(k) / k, mode='valid')
            axes[r, c].plot(np.arange(len(sm)) + k, sm, color=col, label=arm, lw=0.9)
            if r == 0:
                axes[r, c].set_title(title)
            axes[r, c].grid(alpha=.3, lw=0.5)
    axes[r, 0].set_ylabel(mn, fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel('optimizer update')
h, l = axes[0, 0].get_legend_handles_labels()
axes[0, 1].legend(h, l, loc='upper right', ncol=3, frameon=False, handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
plt.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.6); OUT.mkdir(exist_ok=True)
assert abs(fig.get_size_inches()[0] - 6.19) < 1e-9
plt.savefig(OUT / 'training_curves.pdf'); plt.savefig(OUT / 'training_curves.png', dpi=200)   # page = figsize exactly
print('saved', OUT / 'training_curves.pdf')
