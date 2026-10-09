"""Retrospective, count-verified plots; no experiments, model calls, or source edits."""
from pathlib import Path
import collections
import csv
import hashlib
import json

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
AUDIT_PATH = ROOT / 'work/paper_revision_20261009/evidence_audit.json'
JOINT_PATH = ROOT / 'outputs/OriginX_Astra救援复测结果.json'
BASE_PATH = ROOT / 'outputs/RoboCasa_原生评测结果.json'
FAIL_PATH = ROOT / 'work/astra_rescue_20261009/failure_manifest.json'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

audit, joint, base = load(AUDIT_PATH), load(JOINT_PATH), load(BASE_PATH)
assert base['complete'] and base['episodes'] == 2500 and base['successes'] == 1496
assert len(joint['cases']) == len({c['case_id'] for c in joint['cases']}) == 1004
for rel, digest in audit['sources'].items():
    assert sha(ROOT / rel) == digest, rel

def summarize(cases):
    N = len(cases)
    E = sum(c['verified_astra_matched_pair'] for c in cases)
    R = sum(c['classification'] == 'rescued' for c in cases)
    assert R <= E <= N
    return {'N': N, 'E': E, 'R': R, 'coverage_E_over_N': E/N if N else None,
            'conditional_R_over_E': R/E if E else None,
            'confirmed_R_over_N': R/N if N else None}

LABELS = {'atomic_seen': 'Atomic-Seen', 'composite_seen': 'Composite-Seen',
          'composite_unseen': 'Composite-Unseen', 'all': 'All failures'}
COLORS = {'atomic_seen': '#26749a', 'composite_seen': '#b47818',
          'composite_unseen': '#944a80', 'all': '#39434d'}
MARKERS = {'atomic_seen': 'o', 'composite_seen': 's', 'composite_unseen': '^'}
strata = []
for name in [*list(LABELS)[:3], 'all']:
    s = summarize([c for c in joint['cases'] if name == 'all' or c['stratum'] == name])
    s['stratum'] = name
    reference = audit['summary'] if name == 'all' else audit['by_stratum'][name]
    assert (s['N'], s['E'], s['R']) == (reference['n'], reference['E'], reference['rescued'])
    assert np.isclose(s['coverage_E_over_N'] * s['conditional_R_over_E'], s['confirmed_R_over_N'])
    strata.append(s)

tasks = []
for name, original in sorted(base['tasks'].items()):
    cases = [c for c in joint['cases'] if c['task'] == name]
    s = summarize(cases)
    assert s['N'] == original['episodes'] - original['successes']
    if s['N']:
        ref = next(t for t in audit['by_task'] if t['task'] == name)
        assert (s['N'], s['E'], s['R']) == (ref['n'], ref['E'], ref['rescued'])
    s.update(task=name, stratum=original['stratum'],
             historical_successes=original['successes'], historical_episodes=original['episodes'],
             historical_success_rate=original['success_rate'])
    if s['E']:
        assert np.isclose(s['coverage_E_over_N'] * s['conditional_R_over_E'], s['confirmed_R_over_N'])
    tasks.append(s)

correlations = {}
for metric in ['coverage_E_over_N', 'conditional_R_over_E', 'confirmed_R_over_N']:
    eligible = [t for t in tasks if t[metric] is not None]
    rho = stats.spearmanr([t['historical_success_rate'] for t in eligible], [t[metric] for t in eligible]).statistic
    correlations[metric] = {'tasks': len(eligible), 'spearman_rho': float(rho),
                            'weighting': 'one observation per task; average ranks for ties',
                            'inference': 'descriptive only; no p-value or causal claim'}

hardest = sorted(tasks, key=lambda t: (t['historical_success_rate'], t['task']))[:5]
hardest_names = [t['task'] for t in hardest]
assert hardest_names == audit['task_concentration']['original_bottom_five_tasks']
hardest_summary = summarize([c for c in joint['cases'] if c['task'] in hardest_names])
rest_summary = summarize([c for c in joint['cases'] if c['task'] not in hardest_names])

with (OUT / 'coverage_by_task.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(tasks[0]))
    writer.writeheader(); writer.writerows(tasks)
with (OUT / 'coverage_by_split.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(strata[0]))
    writer.writeheader(); writer.writerows(strata)

receipt = {
    'schema': 'originx_coverage_decomposition_figures_v1',
    'source_sha256': {str(p): sha(p) for p in [AUDIT_PATH, JOINT_PATH, BASE_PATH, FAIL_PATH]},
    'no_rollouts': True, 'no_model_calls': True, 'no_training': True, 'no_source_modifications': True,
    'cohort': 'All 1004 unique original B-model failures, each counted once',
    'coverage_definition': 'Verified applied-assistance matched pairs E divided by all original failures N',
    'conditional_definition': 'Strict rescues R divided by verified pairs E; undefined at E=0',
    'confirmed_definition': 'Strict rescues R divided by all original failures N; unknowns/deviations retained',
    'identity': '(E/N)*(R/E)=R/N for E>0; E=0 has undefined conditional rate, not zero',
    'by_split': strata,
    'descriptive_task_correlations': correlations,
    'original_bottom_five_tasks': hardest_names,
    'original_bottom_five': hardest_summary,
    'other_tasks': rest_summary,
    'excluded_from_all_task_rescue_plots': [t['task'] for t in tasks if t['N'] == 0],
    'additional_task_excluded_from_conditional_plot': [t['task'] for t in tasks if t['N'] > 0 and t['E'] == 0],
    'limitations': [
        'The multiplicative factorization is an accounting identity, not a causal decomposition.',
        'Historical success is a noisy policy-specific difficulty proxy from 50 trials per task.',
        'All tasks and splits are fixed observed groups; group comparisons are retrospective.',
        'Spearman associations across tasks are weak and should not be described as a monotonic hardness law.',
        'The outcome cohort is selected on historical failure; no full-benchmark score is estimated.',
        'Tasks with identical coordinates overplot; marker sizes are fixed, not false displacements.',
    ],
}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'xtick.labelsize': 7,
    'ytick.labelsize': 8, 'legend.fontsize': 7.2,
    'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .5,
    'xtick.major.width': .5, 'ytick.major.width': .5, 'savefig.facecolor': 'white',
})

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15), sharey=True,
                          gridspec_kw={'width_ratios': [1.06, 1, 1]})
plt.subplots_adjust(left=.175, right=.986, bottom=.19, top=.83, wspace=.14)
metrics = ['coverage_E_over_N', 'conditional_R_over_E', 'confirmed_R_over_N']
titles = ['(a) Evidence coverage $E/N$', '(b) Conditional rescue $R/E$', '(c) Confirmed rescue $R/N$']
maxes = [1, .125, .10]
for aidx, (ax, metric, title, xmax) in enumerate(zip(axes, metrics, titles, maxes)):
    for i, s in enumerate(strata):
        ax.barh(i, s[metric], height=.61, color=COLORS[s['stratum']], alpha=.23, linewidth=0, zorder=2)
        label = f"{100*s[metric]:.2f}%"
        den = s['N'] if metric != 'conditional_R_over_E' else s['E']
        num = s['E'] if metric == 'coverage_E_over_N' else s['R']
        ax.text(xmax*.025, i, f'{label} ({num}/{den})', va='center', ha='left', fontsize=7.7,
                color='#20252a', zorder=3)
    ax.set_xlim(0, xmax)
    ax.set_ylim(3.55, -.55)
    ax.set_title(title, pad=7)
    ax.set_yticks(range(4), [LABELS[s['stratum']] for s in strata])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xticks([0, .5, 1] if aidx == 0 else ([0,.05,.1] if aidx == 1 else [0,.05,.10]))
    ax.grid(axis='x', color='#e2e5e7', linewidth=.45, zorder=0)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.spines['left'].set_visible(False)
    ax.axhline(2.5, color='#929ba3', linewidth=.45, linestyle=(0,(2,2)))
fig.text(.56, .05, r'All failures:  $(869/1004)\ \times\ (56/869)\ =\ 56/1004\ =\ 5.58\%$', ha='center', fontsize=8.5)
fig.savefig(OUT / 'coverage_decomposition.pdf')
fig.savefig(OUT / 'coverage_decomposition.png', dpi=350)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), sharex=True)
plt.subplots_adjust(left=.083, right=.99, bottom=.26, top=.79, wspace=.37)
for ax, metric, title in zip(axes, metrics, ['(a) Evidence coverage', '(b) Conditional rescue', '(c) Confirmed rescue']):
    for name in list(LABELS)[:3]:
        group = [t for t in tasks if t['stratum'] == name and t[metric] is not None]
        ax.scatter([t['historical_success_rate'] for t in group], [t[metric] for t in group],
                   s=23, marker=MARKERS[name], color=COLORS[name], alpha=.78,
                   edgecolors='white', linewidths=.45, label=LABELS[name], zorder=3)
    ax.set_xlim(-.035,1.035)
    ax.set_ylim((-.055,1.055) if metric == 'coverage_E_over_N' else (-.027,.54))
    ax.set_xticks([0,.5,1]); ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_yticks([0,.5,1] if metric == 'coverage_E_over_N' else [0,.25,.5])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel('Historical task success rate', labelpad=3)
    ax.set_title(title, pad=15)
    ax.grid(color='#e6e8ea', linewidth=.45, zorder=0)
    c = correlations[metric]
    ax.text(.5,1.02,fr"$n={c['tasks']}$ tasks; Spearman $\rho={c['spearman_rho']:.2f}$",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=7.1)
    ax.set_ylabel({'coverage_E_over_N':r'$E/N$', 'conditional_R_over_E':r'$R/E$', 'confirmed_R_over_N':r'$R/N$'}[metric], labelpad=2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53,.995), ncol=3, frameon=False,
           columnspacing=1.5, handletextpad=.35)
fig.text(.52,.045, 'One point per task; coincident points overlap.',
         ha='center', fontsize=7.2)
fig.savefig(OUT / 'task_difficulty_coverage.pdf')
fig.savefig(OUT / 'task_difficulty_coverage.png', dpi=350)
plt.close(fig)

receipt['figure_sha256'] = {name:sha(OUT/name) for name in [
    'coverage_decomposition.pdf', 'coverage_decomposition.png',
    'task_difficulty_coverage.pdf', 'task_difficulty_coverage.png']}
(OUT / 'coverage_figure_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(OUT / 'figure_notes.md').write_text('''# Coverage factorization and difficulty: verified retrospective figures

## Recommended main figure

`coverage_decomposition.pdf` is sized 7.1 x 2.15 inches for a CVPR double-column figure.

Suggested caption: **Evidence coverage and rescue efficacy describe different limitations.** Each row factorizes the confirmed rescue fraction of the fixed historical failure cohort as $(E/N)(R/E)=R/N$. Counts are shown beside the observed percentages; these are exact cohort summaries, not fitted estimates. Composite-Unseen has the highest evidence coverage (489/530, 92.26%) yet the lowest rescue fraction among verified pairs (19/489, 3.89%). Its lower confirmed fraction therefore persists after restricting to verified pairs. Composite-Seen has lower coverage (243/321, 75.70%) and higher conditional rescue (25/243, 10.29%). This is an accounting identity and retrospective comparison, not a causal attribution to task novelty.

## Recommended supplementary figure

`task_difficulty_coverage.pdf` is sized 7.1 x 2.6 inches and includes all tasks for which each metric is defined.

Suggested caption: **Task-level heterogeneity does not imply a monotonic difficulty relationship.** Historical success across 50 original episodes per task serves only as a policy-specific difficulty proxy. Each point is one task, colored by split, with unweighted rank associations shown descriptively. CloseFridge contributes no original failures and is excluded from all rescue plots; OpenStandMixerHead has no verified pair and is additionally excluded from $R/E$. Identical coordinates overlap without jitter. Despite low recovery among the five hardest original tasks, the overall task-level association between historical success and conditional rescue is weak ($\\rho=0.12$, 48 tasks). No regression, p-value, causal trend, or missing-at-random assumption is asserted.

## Validity details

- Source hashes are verified against the previous detailed audit; case counts and all split/task numerators and denominators are recomputed from the final per-case joint ledger.
- Task failure counts are checked against the independent historical 2,500-episode report.
- The full-cohort identity is 869/1004 x 56/869 = 56/1004. Displayed rounded percentages are illustrative; exact counts define the identity.
- For E=0, R/E is undefined and is never silently encoded as zero.
- Case evidence verification comes from the existing detailed audit; this script checks its source hashes and independently recomputes every plotted aggregate.
- Baseline task difficulty and rescue rates have shared finite-sample denominators; these exploratory task correlations do not identify a mechanism.
- No experiments, policy updates, model calls, or source evidence edits were performed.
''', encoding='utf-8')
print(json.dumps({'split_counts':strata,'correlations':correlations,'output':str(OUT)},indent=2))
