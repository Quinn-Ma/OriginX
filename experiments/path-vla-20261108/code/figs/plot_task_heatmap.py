"""Per-task paired success differences (adapted arm minus official checkpoint), as a heatmap per model."""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
summary = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2])
MODELS = [('xr1', 'XR-1'), ('awomo_i', 'Awomo-0.5I'), ('awomo_v', 'Awomo-0.5V')]
ARMS = [('S', 'S'), ('BL', 'BL'), ('F', 'F')]
TASKS = ['DeliverStraw', 'GetToastedBread', 'PackIdenticalLunches', 'SearingMeat', 'StirVegetables', 'ArrangeTea', 'CuttingToolSelection', 'RecycleBottlesByType', 'WaffleReheat', 'WeighIngredients']
SHORT = ['Straw', 'Toast', 'Lunch', 'Sear', 'Stir', 'Tea*', 'Cut*', 'Recycle*', 'Waffle*', 'Weigh*']
fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
for ax, (mk, mn) in zip(axes, MODELS):
    off = summary['per_policy']['%s-original' % mk]['tasks']
    M = np.zeros((3, 10))
    for i, (ak, an) in enumerate(ARMS):
        arm = summary['per_policy']['%s-%s' % (mk, ak)]['tasks']
        for j, t in enumerate(TASKS):
            M[i, j] = arm[t]['success'] - off[t]['success']
    im = ax.imshow(M, cmap='RdBu', vmin=-10, vmax=10, aspect='auto')
    for i in range(3):
        for j in range(10):
            v = int(M[i, j]); ax.text(j, i, '%+d' % v if v else '0', ha='center', va='center', fontsize=7.5, color='black' if abs(v) < 6 else 'white')
    ax.set_xticks(range(10)); ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=7.5); ax.set_yticks(range(3)); ax.set_yticklabels([a for _, a in ARMS], fontsize=8)
    ax.set_title('%s: official %d/200' % (mn, summary['per_policy']['%s-original' % mk]['all'][0]), fontsize=9)
    ax.axvline(4.5, color='black', lw=.8)
cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01); cb.set_label('successes vs. official (of 20)', fontsize=8)
plt.savefig(out / 'task_heatmap.pdf', bbox_inches='tight'); plt.savefig(out / 'task_heatmap.png', dpi=140, bbox_inches='tight'); print('saved')
