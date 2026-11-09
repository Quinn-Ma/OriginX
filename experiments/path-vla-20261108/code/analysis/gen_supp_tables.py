"""Supplement tables: seen/unseen split contrasts, episode statistics and rescues/regressions from secondary.json.
Usage: python -X utf8 analysis/gen_supp_tables.py analysis/out/secondary.json paper/tables
Numbers are typeset in math mode (true minus sign, unsigned zero), step counts with thousands separators."""
from pathlib import Path
import json, sys
sec = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2])
MODELS = [('xr1', r'\xr{}'), ('awomo_i', r'\awi{}'), ('awomo_v', r'\awv{}')]
order = [('F', 'BL'), ('F', 'S'), ('F', 'original'), ('S', 'original'), ('BL', 'original')]
EOL = ' ' + '\\' * 2   # LaTeX row end


def sg(x, nd=0):
    """signed number for math mode; zero is printed without a sign"""
    s = '%.*f' % (nd, abs(x))
    return s if float(s) == 0 else ('+' if x > 0 else '-') + s


def steps(x):
    if x is None:
        return '--'
    s = '{:,.1f}'.format(x) if x != int(x) else '{:,}'.format(int(x))
    return s.replace(',', '{,}')


# each interval is set in two right-aligned columns ("[lo," and "hi]") so that brackets and commas line up
lines = [r'\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}llrr@{\,}rrr@{\,}r@{}}', r'\toprule',
         r'& & \multicolumn{3}{c}{seen tasks} & \multicolumn{3}{c}{unseen tasks}' + EOL, r'\cmidrule(lr){3-5}\cmidrule(l){6-8}',
         r'Model & Contrast & $\Delta$ (pp) & \multicolumn{2}{c}{95\% CI} & $\Delta$ (pp) & \multicolumn{2}{c}{95\% CI}' + EOL, r'\midrule']
for mk, mn in MODELS:
    first = True
    for x, y in order:
        a = sec['split_contrasts']['%s:%s-%s:seen' % (mk, x, y)]; b = sec['split_contrasts']['%s:%s-%s:unseen' % (mk, x, y)]
        lab = '%s$-$%s' % (x, 'official' if y == 'original' else y)
        lines.append((r'\multirow{5}{*}{%s}' % mn if first else '') + ' & %s & $%s$ & $[%s,$ & $%s]$ & $%s$ & $[%s,$ & $%s]$' % (
            lab, sg(a['diff_pp']), sg(a['ci95'][0]), sg(a['ci95'][1]), sg(b['diff_pp']), sg(b['ci95'][0]), sg(b['ci95'][1])) + EOL)
        first = False
    lines.append(r'\midrule')
lines[-1] = r'\bottomrule'
lines += [r'\end{tabular}',
          r'\caption{\textbf{Contrasts split by task group.} Paired success differences on the five composite-seen tasks, whose '
          r'demonstrations were used for adaptation, and on the five composite-unseen tasks (100 pairs each), with task-stratified '
          r'paired-bootstrap 95\% intervals (10{,}000 draws).}', r'\label{tab:split}', r'\end{table}']
(out / 'split_table.tex').write_text('\n'.join(lines) + '\n')
lines = [r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}llr@{}lrr@{}}', r'\toprule',
         r'Model & Policy & \multicolumn{2}{c}{steps$^{a}$} & timeouts & time (s)$^{b}$' + EOL, r'\midrule']
for mk, mn in MODELS:
    first = True
    for c, cn in [('original', 'official'), ('S', 'S'), ('BL', 'BL'), ('F', 'F')]:
        v = sec['steps']['%s-%s' % (mk, c)]
        whole, _, frac = steps(v['median_steps_success']).partition('.')
        lines.append((r'\multirow{4}{*}{%s}' % mn if first else '') + ' & %s & %s & %s & %d & %.0f' % (
            cn, whole, '.' + frac if frac else '', v['timeouts'], v['mean_elapsed_s']) + EOL)
        first = False
    lines.append(r'\midrule')
lines[-1] = r'\bottomrule'
lines += [r'\end{tabular}',
          r'\caption{\textbf{Episode statistics.} $^{a}$Median length (steps) of the successful episodes. $^{b}$Mean wall-clock per '
          r'episode, including simulation and policy inference on shared GPUs. Every failure is a horizon timeout (no early '
          r'terminations occurred).}', r'\label{tab:steps}', r'\end{table}']
(out / 'steps_table.tex').write_text('\n'.join(lines) + '\n')
# rescues / regressions
lines = [r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
         r'Model & F rescues & F regressions & net' + EOL, r'\midrule']
for mk, mn in MODELS:
    r = sum(sec['rescues'][mk]['rescued'].values()); g = sum(sec['rescues'][mk]['regressed'].values())
    lines.append(r'%s & %d & %d & $%s$' % (mn, r, g, sg(r - g)) + EOL)
lines += [r'\bottomrule', r'\end{tabular}',
          r'\caption{\textbf{Rescues and regressions.} Episodes the F arm solves that the official checkpoint fails (rescues) and '
          r'vice versa (regressions), out of 200 paired episodes.}', r'\label{tab:rescue}', r'\end{table}']
(out / 'rescue_table.tex').write_text('\n'.join(lines) + '\n'); print('ok')
