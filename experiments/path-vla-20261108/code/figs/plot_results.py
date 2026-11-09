"""Main results figure: (a) success rate per model and arm with Wilson 95% CIs; (b) paired contrasts with bootstrap CIs."""
from pathlib import Path
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
summary = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2]); out.mkdir(exist_ok=True)
MODELS = [('xr1', 'XR-1'), ('awomo_i', 'Awomo-0.5I'), ('awomo_v', 'Awomo-0.5V')]
CONDS = [('original', 'official', '#7f7f7f'), ('S', 'S: stage', '#1f77b4'), ('BL', 'BL: box+loc', '#ff7f0e'), ('F', 'F: full', '#2ca02c')]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={'width_ratios': [1.25, 1]})
w = 0.19
for mi, (mk, mn) in enumerate(MODELS):
    for ci, (ck, cn, col) in enumerate(CONDS):
        v = summary['per_policy']['%s-%s' % (mk, ck)]; k, n = v['all']; p = 100 * k / n; lo, hi = wilson(k, n)
        x = mi + (ci - 1.5) * w
        ax1.bar(x, p, w * 0.92, color=col, label=cn if mi == 0 else None, edgecolor='white', linewidth=.5)
        ax1.errorbar(x, p, yerr=[[p - 100 * lo], [100 * hi - p]], color='black', lw=1, capsize=2)
        ax1.text(x, 100 * hi + 1.2, '%d' % k, ha='center', va='bottom', fontsize=7)
ax1.set_xticks(range(3)); ax1.set_xticklabels([mn for _, mn in MODELS]); ax1.set_ylabel('success rate (%)  [n = 200 episodes]')
ax1.set_ylim(0, 62); ax1.legend(fontsize=8, ncol=2, loc='upper left', frameon=False); ax1.grid(axis='y', alpha=.3)
ax1.set_title('(a) Ordinary-input success on 10 tasks x 20 seeds', fontsize=10)
# contrasts
order = [('F', 'BL', True), ('F', 'S', True), ('F', 'original', False), ('S', 'original', False), ('BL', 'original', False)]
ys = []; labels = []; y = 0
for mi, (mk, mn) in enumerate(MODELS):
    for x_, y_, primary in order:
        c = summary['contrasts']['%s:%s-%s' % (mk, x_, y_)]
        lo, hi = c['ci95']; d = c['diff_pp']
        col = '#d62728' if primary else '#555555'
        ax2.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt='o', color=col, ms=4, lw=1.3, capsize=2)
        labels.append('%s  %s - %s' % (mn if x_ == 'F' and y_ == 'BL' else '', x_, 'official' if y_ == 'original' else y_))
        ys.append(y); y -= 1
    y -= 0.8
ax2.axvline(0, color='black', lw=.8); ax2.axvspan(-5, 5, color='gray', alpha=.08)
ax2.set_yticks(ys); ax2.set_yticklabels(labels, fontsize=7.5); ax2.set_xlabel('paired success difference (pp), 95% task-stratified bootstrap CI')
ax2.set_title('(b) Paired contrasts (red = pre-registered primary)', fontsize=10); ax2.grid(axis='x', alpha=.3)
plt.tight_layout(); plt.savefig(out / 'results_main.pdf'); plt.savefig(out / 'results_main.png', dpi=140); print('saved')
