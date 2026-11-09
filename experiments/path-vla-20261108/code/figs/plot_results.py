"""Main results figure (fig:main), drawn at its final printed size (6.875 in = \\textwidth; include it unscaled):
(a) successes out of 200 per model and arm with Wilson 95% CIs; the axis is in successes, so the count printed above
    each bar is the bar's value;
(b) paired contrasts with task-stratified bootstrap 95% CIs, one small panel per model.
Times New Roman, like Figures 1, 2 and 4 (analysis/task_effects.py).
Usage: python -X utf8 figs/plot_results.py results/analysis/summary.json paper/figures"""
from pathlib import Path
import json, math, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
summary = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2]); out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman'], 'mathtext.fontset': 'stix',
                     'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
MINUS = '−'
MODELS = [('xr1', 'XR-1'), ('awomo_i', 'Awomo-0.5I'), ('awomo_v', 'Awomo-0.5V')]
CONDS = [('original', 'official', '#7f7f7f'), ('S', 'S: progress', '#1f77b4'), ('BL', 'BL: target', '#ff7f0e'),
         ('F', 'F: PaTH-VLA (both)', '#2ca02c')]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


W, H = 6.875, 2.2
fig = plt.figure(figsize=(W, H))
# (a) on the left; (b) three small forest plots sharing the contrast labels
L, B, T = 0.055, 0.165, 0.878                     # margins as figure fractions
ax1 = fig.add_axes([L, B, 0.385, T - B])
xb0, wb, gap = 0.565, 0.135, 0.013
axb = [fig.add_axes([xb0 + i * (wb + gap), B, wb, T - B]) for i in range(3)]
w = 0.2
for mi, (mk, mn) in enumerate(MODELS):
    for ci, (ck, cn, col) in enumerate(CONDS):
        v = summary['per_policy']['%s-%s' % (mk, ck)]; k, n = v['all']; lo, hi = wilson(k, n)
        assert n == 200, (mk, ck, n)
        x = mi + (ci - 1.5) * w
        ax1.bar(x, k, w * 0.9, color=col, label=cn if mi == 0 else None, edgecolor='white', linewidth=.4)
        ax1.errorbar(x, k, yerr=[[k - n * lo], [n * hi - k]], color='black', lw=0.7, capsize=1.5, capthick=0.7)
        ax1.text(x, n * hi + 2.0, '%d' % k, ha='center', va='bottom', fontsize=7)
ax1.set_xticks(range(3)); ax1.set_xticklabels([mn for _, mn in MODELS]); ax1.set_ylabel('successes (of 200 episodes)')
ax1.set_xlim(-0.5, 2.5); ax1.set_ylim(0, 150); ax1.set_yticks(range(0, 121, 20))
ax1.legend(ncol=2, loc='upper center', frameon=False, handlelength=1.0, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1)
ax1.grid(axis='y', alpha=.3, lw=0.5); ax1.set_axisbelow(True)
ax1.set_title('(a) Ordinary-input success, 10 tasks × 20 seeds', loc='left')
for s in ('top', 'right'):
    ax1.spines[s].set_visible(False)
# (b) contrasts
order = [('F', 'BL', True), ('F', 'S', True), ('F', 'original', False), ('S', 'original', False), ('BL', 'original', False)]
labels = ['%s%s%s' % (x_, MINUS, 'official' if y_ == 'original' else y_) for x_, y_, _ in order]   # "F−BL", as in the text
for mi, (mk, mn) in enumerate(MODELS):
    ax = axb[mi]
    ax.axvspan(-5, 5, color='gray', alpha=.12, lw=0); ax.axvline(0, color='black', lw=.6)
    for r, (x_, y_, primary) in enumerate(order):
        c = summary['contrasts']['%s:%s-%s' % (mk, x_, y_)]
        lo, hi = c['ci95']; d = c['diff_pp']
        col = '#d62728' if primary else '#4d4d4d'
        ax.errorbar(d, -r, xerr=[[d - lo], [hi - d]], fmt='o', color=col, ms=3, lw=1.0, capsize=1.5, capthick=0.8)
    ax.set_ylim(-len(order) + 0.4, 0.6); ax.set_xlim(-22, 17); ax.set_xticks([-20, -10, 0, 10])
    ax.set_xticklabels([('%d' % t).replace('-', MINUS) for t in [-20, -10, 0, 10]])
    ax.set_yticks([-r for r in range(len(order))])
    ax.set_yticklabels(labels if mi == 0 else [])
    if mi:
        ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=.3, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(('(b) ' if mi == 0 else '') + mn, loc='left' if mi == 0 else 'center')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axb[1].set_xlabel('paired success difference (pp), 95% bootstrap CI')
assert abs(fig.get_size_inches()[0] - W) < 1e-9
for ext in ('pdf', 'png'):
    fig.savefig(out / ('results_main.' + ext), dpi=200)     # no bbox_inches: the page is exactly W x H inches
print('saved', out / 'results_main.pdf')
