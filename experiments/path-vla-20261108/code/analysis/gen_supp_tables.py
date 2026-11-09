"""Supplement tables: seen/unseen split contrasts and episode-length statistics from secondary.json."""
from pathlib import Path
import json, sys
sec = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2])
MODELS = [('xr1', 'XR-1'), ('awomo_i', 'Awomo-0.5I'), ('awomo_v', 'Awomo-0.5V')]
order = [('F', 'BL'), ('F', 'S'), ('F', 'original'), ('S', 'original'), ('BL', 'original')]
lines = ['\\begin{table}[h]\\centering\\small\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{llrrrr}', '\\toprule',
         'Model & Contrast & \\multicolumn{2}{c}{seen (100 pairs)} & \\multicolumn{2}{c}{unseen (100 pairs)} \\\\',
         ' & & $\\Delta$ (pp) & 95\\% CI & $\\Delta$ (pp) & 95\\% CI \\\\', '\\midrule']
for mk, mn in MODELS:
    first = True
    for x, y in order:
        a = sec['split_contrasts']['%s:%s-%s:seen' % (mk, x, y)]; b = sec['split_contrasts']['%s:%s-%s:unseen' % (mk, x, y)]
        lab = '%s$-$%s' % (x, 'official' if y == 'original' else y)
        lines.append(('\\multirow{5}{*}{%s}' % mn if first else '') + ' & %s & %+.0f & [%+.0f, %+.0f] & %+.0f & [%+.0f, %+.0f] \\\\' % (
            lab, a['diff_pp'], a['ci95'][0], a['ci95'][1], b['diff_pp'], b['ci95'][0], b['ci95'][1]))
        first = False
    lines.append('\\midrule')
lines[-1] = '\\bottomrule'
lines += ['\\end{tabular}', '\\caption{Paired contrasts split by task group (composite-seen tasks whose demonstrations were used for adaptation vs.\\ composite-unseen tasks). Task-stratified paired bootstrap, 4{,}000 draws.}', '\\label{tab:split}', '\\end{table}']
(out / 'split_table.tex').write_text('\n'.join(lines) + '\n')
lines = ['\\begin{table}[h]\\centering\\small\\setlength{\\tabcolsep}{4pt}', '\\begin{tabular}{llrrr}', '\\toprule',
         'Model & Policy & median steps (success) & timeouts & mean wall-clock (s) \\\\', '\\midrule']
for mk, mn in MODELS:
    first = True
    for c, cn in [('original', 'official'), ('S', 'S'), ('BL', 'BL'), ('F', 'F')]:
        v = sec['steps']['%s-%s' % (mk, c)]
        lines.append(('\\multirow{4}{*}{%s}' % mn if first else '') + ' & %s & %s & %d & %.0f \\\\' % (cn, v['median_steps_success'], v['timeouts'], v['mean_elapsed_s']))
        first = False
    lines.append('\\midrule')
lines[-1] = '\\bottomrule'
lines += ['\\end{tabular}', '\\caption{Episode statistics. Every failure is a horizon timeout (no early terminations occurred). Wall-clock includes simulation and policy inference on shared GPUs.}', '\\label{tab:steps}', '\\end{table}']
(out / 'steps_table.tex').write_text('\n'.join(lines) + '\n')
# rescues / regressions
lines = ['\\begin{table}[h]\\centering\\small\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{lrrr}', '\\toprule', 'Model & F rescues & F regressions & net \\\\', '\\midrule']
for mk, mn in MODELS:
    r = sum(sec['rescues'][mk]['rescued'].values()); g = sum(sec['rescues'][mk]['regressed'].values())
    lines.append('%s & %d & %d & %+d \\\\' % (mn, r, g, r - g))
lines += ['\\bottomrule', '\\end{tabular}', '\\caption{Episodes the F arm solves that the official checkpoint fails (rescues) and vice versa (regressions), out of 200 paired episodes.}', '\\label{tab:rescue}', '\\end{table}']
(out / 'rescue_table.tex').write_text('\n'.join(lines) + '\n'); print('ok')
