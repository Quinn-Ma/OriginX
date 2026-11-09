"""Sensitivity analysis (post hoc): repeat every paired contrast of the main analysis (the two primary contrasts with
their Holm step, and the secondary contrasts against the official checkpoint and S-BL) on the seeds whose first
observation (SHA-256 of the left agent-view frame) and initial state are identical across all 12 primary policies.

Writes analysis/out/sensitivity.json, paper/tables/macros_sensitivity.tex and paper/tables/sensitivity_table.tex.
Macro values are math-mode strings (use them inside $...$).
"""
from pathlib import Path
import collections, json, sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from analyze import load, paired_contrast, exact_mcnemar, MODELS

data = load(ROOT / 'results/eval')
hashes = collections.defaultdict(set); states = collections.defaultdict(set)
for (m, c), recs in data.items():
    for k, r in recs.items():
        hashes[k].add(r['first_frame_sha256'])
        states[k].add(tuple(round(x, 4) for x in r['first_state']))
frame_var = sorted(k for k in hashes if len(hashes[k]) > 1)
state_var = sorted(k for k in states if len(states[k]) > 1)
keep = set(hashes) - set(frame_var) - set(state_var)
NAMES = {'xr1': 'XR', 'awomo_i': 'AI', 'awomo_v': 'AV'}
LABEL = {'xr1': r'\xr', 'awomo_i': r'\awi', 'awomo_v': r'\awv'}
PRIMARY = [('F', 'BL'), ('F', 'S')]
CONTRASTS = PRIMARY + [('F', 'original'), ('S', 'original'), ('BL', 'original'), ('S', 'BL')]
EOL = ' ' + '\\' * 2   # LaTeX row end


def exact(v, n):
    """paired_contrast() rounds to two decimals, but every difference and bootstrap quantile is 100*k/n for an
    integer k; recover that exact value before rounding to one decimal (avoids double rounding: 15/191 = 7.853 pp
    would otherwise print as 7.85 -> +7.8 instead of +7.9)."""
    k = round(v * n / 100)
    assert abs(100 * k / n - v) < 0.006, (v, n)
    return 100 * k / n


def sg(x):
    """one-decimal signed number for math mode; zero without a sign"""
    s = '%.1f' % abs(x)
    return s if float(s) == 0 else ('+' if x > 0 else '-') + s


out = {'n_seeds_total': len(hashes), 'n_identical': len(keep), 'post_hoc': True,
       'frame_variant_seeds': [list(k) for k in frame_var], 'state_variant_seeds': [list(k) for k in state_var], 'contrasts': {}}
macros = [r'\newcommand{\sensN}{%d}' % len(keep), r'\newcommand{\sensNvar}{%d}' % len(frame_var),
          r'\newcommand{\sensNstate}{%d}' % len(state_var)]
by_task = collections.defaultdict(list)
for t, s in frame_var:
    by_task[t].append(str(s))
macros.append(r'\newcommand{\sensSeeds}{%s}' % '; '.join('%s %s' % (t, ', '.join(v)) for t, v in sorted(by_task.items())))
by_task_state = collections.defaultdict(list)
for t, s in state_var:
    by_task_state[t].append(str(s))
macros.append(r'\newcommand{\sensStateSeeds}{%s}' % '; '.join('%s %s' % (t, ' and '.join(v) if len(v) == 2 else ', '.join(v)) for t, v in sorted(by_task_state.items())))
rows = []
for m in MODELS:
    res = {}
    for x, y in CONTRASTS:
        a = {k: v for k, v in data[(m, x)].items() if k in keep}
        b = {k: v for k, v in data[(m, y)].items() if k in keep}
        c = paired_contrast(a, b, n_boot=10000)
        n = c['n_pairs']
        c['mcnemar_p'] = exact_mcnemar(c['wins'], c['losses'])  # unrounded (paired_contrast rounds to 4 decimals)
        c['fmt'] = (exact(c['diff_pp'], n), exact(c['ci95'][0], n), exact(c['ci95'][1], n))
        c['fmt975'] = (exact(c['ci975'][0], n), exact(c['ci975'][1], n))
        res[(x, y)] = c
    # Holm over the two primary contrasts of the model, exactly as in analyze.py: the contrast with the smaller p is
    # judged on its 97.5% interval and doubled p, the other only if the first is rejected, on its 95% interval; the
    # adjusted p-values are monotone, p_adj(2) = max(p_adj(1), p(2))
    ps = sorted(PRIMARY, key=lambda xy: res[xy]['mcnemar_p'])
    prev_holm, prev_rejected = 0.0, True
    for rank, xy in enumerate(ps):
        c = res[xy]
        c['holm_p'] = max(prev_holm, min(1.0, c['mcnemar_p'] * (2 - rank)))
        c['holm_ci'] = c['fmt975'] if rank == 0 else c['fmt'][1:]
        rejected = prev_rejected and (c['holm_ci'][0] > 0 or c['holm_ci'][1] < 0)
        c['supported'] = bool(rejected and c['holm_ci'][0] > 0 and c['fmt'][0] >= 5)
        prev_holm, prev_rejected = c['holm_p'], rejected
    for x, y in CONTRASTS:
        c = res[(x, y)]
        out['contrasts']['%s:%s-%s' % (m, x, y)] = dict({k: c[k] for k in ['n_pairs', 'a_success', 'b_success', 'diff_pp', 'ci95', 'ci975', 'wins', 'losses', 'mcnemar_p']},
                                                     **({k: c[k] for k in ['holm_p', 'holm_ci', 'supported']} if (x, y) in PRIMARY else {}))
        tag = 'sens%s%s%s' % (NAMES[m], x, 'off' if y == 'original' else y)
        macros += [r'\newcommand{\%sdiff}{%s}' % (tag, sg(c['fmt'][0])), r'\newcommand{\%sci}{[%s, %s]}' % (tag, sg(c['fmt'][1]), sg(c['fmt'][2])),
                   r'\newcommand{\%sp}{%.3f}' % (tag, c['mcnemar_p'])]
        if (x, y) in PRIMARY:
            macros += [r'\newcommand{\%sholm}{%.3f}' % (tag, c['holm_p']),
                       r'\newcommand{\%sholmci}{[%s, %s]}' % (tag, sg(c['holm_ci'][0]), sg(c['holm_ci'][1]))]
        rows.append((m, x, y, c))
n_supported = sum(1 for v in out['contrasts'].values() if v.get('supported'))
macros.append(r'\newcommand{\sensNsupported}{%d}' % n_supported)
(HERE / 'out').mkdir(exist_ok=True)
(HERE / 'out/sensitivity.json').write_text(json.dumps(out, indent=1))
(ROOT / 'paper/tables/macros_sensitivity.tex').write_text('\n'.join(macros) + '\n')
# the interval is set in two right-aligned columns ("[lo," and "hi]") so that brackets and commas line up
lines = [r'\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{@{}llrr@{\,}rrr@{}}', r'\toprule',
         r'Model & Contrast & $\Delta$ (pp) & \multicolumn{2}{c}{95\% CI} & $p$ & Holm $p$' + EOL, r'\midrule']
per_model = len(CONTRASTS)
for i, (m, x, y, c) in enumerate(rows):
    first = i % per_model == 0
    lab = '%s$-$%s' % (x, 'official' if y == 'original' else y)
    if (x, y) in PRIMARY:
        lab = r'\textbf{' + lab + '}'
    holm = '%.3f' % c['holm_p'] if (x, y) in PRIMARY else ''
    lines.append((r'\multirow{%d}{*}{%s}' % (per_model, LABEL[m]) if first else '') + r' & %s & $%s$ & $[%s,$ & $%s]$ & %.3f & %s' % (
        lab, sg(c['fmt'][0]), sg(c['fmt'][1]), sg(c['fmt'][2]), c['mcnemar_p'], holm) + EOL)
    if i % per_model == per_model - 1 and i != len(rows) - 1:
        lines.append(r'\midrule')
lines += [r'\bottomrule', r'\end{tabular}',
          r'\caption{\textbf{Sensitivity to scene identity (post hoc).} Every contrast recomputed on the %d of %d seeds whose first '
          r'observation and initial state are identical across all 12 policies (task-stratified paired bootstrap, 10{,}000 draws; '
          r'exact McNemar $p$, uncorrected). Bold: pre-registered primary contrasts, with the Holm-corrected $p$; %s of them '
          r'is supported under the pre-registered rule.}' % (len(keep), len(hashes), 'none' if n_supported == 0 else str(n_supported)),
          r'\label{tab:sensitivity}', r'\end{table}']
(ROOT / 'paper/tables/sensitivity_table.tex').write_text('\n'.join(lines) + '\n')
print(json.dumps({k: out[k] for k in ['n_seeds_total', 'n_identical', 'frame_variant_seeds', 'state_variant_seeds']}))
for m, x, y, c in rows:
    print(m, x, '-', y, c['n_pairs'], c['fmt'], c['mcnemar_p'], 'holm', c.get('holm_p'), c.get('holm_ci'), c.get('supported'))
