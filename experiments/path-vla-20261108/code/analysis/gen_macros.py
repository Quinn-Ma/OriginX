"""Generate paper/tables/macros.tex and results tables from analysis/summary.json (+ panel summary, training status)."""
from pathlib import Path
import argparse, json
MODELS = [('xr1', 'XR-1'), ('awomo_i', 'Awomo-0.5I'), ('awomo_v', 'Awomo-0.5V')]
MK = {'xr1': 'XR', 'awomo_i': 'AI', 'awomo_v': 'AV'}
CONDS = [('original', 'official'), ('S', 'S'), ('BL', 'BL'), ('F', 'F')]
SEEN = ['DeliverStraw', 'GetToastedBread', 'PackIdenticalLunches', 'SearingMeat', 'StirVegetables']
UNSEEN = ['ArrangeTea', 'CuttingToolSelection', 'RecycleBottlesByType', 'WaffleReheat', 'WeighIngredients']
SHORT = {'DeliverStraw': 'Straw', 'GetToastedBread': 'Toast', 'PackIdenticalLunches': 'Lunch', 'SearingMeat': 'Sear', 'StirVegetables': 'Stir',
         'ArrangeTea': 'Tea', 'CuttingToolSelection': 'Cut', 'RecycleBottlesByType': 'Recycle', 'WaffleReheat': 'Waffle', 'WeighIngredients': 'Weigh'}


def sg(x):
    """one-decimal signed number for math mode; zero is printed without a sign"""
    s = '%.1f' % abs(x)
    return s if float(s) == 0 else ('+' if x > 0 else '-') + s


def fmt_ci(c):
    """interval for math mode (true minus sign, unsigned zero)"""
    return '[%s, %s]' % (sg(c[0]), sg(c[1]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--summary', required=True); ap.add_argument('--panel', default=None)
    ap.add_argument('--out', required=True); ap.add_argument('--extra', default=None)
    a = ap.parse_args(); s = json.loads(Path(a.summary).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    extra = json.loads(Path(a.extra).read_text()) if a.extra else {}
    macros = []
    def m(name, value):
        macros.append('\\newcommand{\\%s}{%s}' % (name, value))
    total_eps = sum(v['all'][1] for v in s['per_policy'].values())
    m('nEvalEpisodes', '{:,}'.format(total_eps))
    if a.panel:
        p = json.loads(Path(a.panel).read_text()); m('nPanelWindows', '{:,}'.format(p['panel_windows']))
    for k, v in extra.items():
        m(k, v)
    # per-policy rates
    for mk, mn in MODELS:
        for ck, cn in CONDS:
            v = s['per_policy'].get('%s-%s' % (mk, ck))
            if v:
                m('rate%s%s' % (MK[mk], ck), '%.1f' % (v['rate'] or 0))
                m('succ%s%s' % (MK[mk], ck), '%d/%d' % tuple(v['all']))
    for key, c in s['contrasts'].items():
        mk, pair = key.split(':'); x, y = pair.split('-'); nm = 'c%s%s%s' % (MK[mk], x, y)
        m(nm + 'diff', '\\ensuremath{%s}' % sg(c['diff_pp'])); m(nm + 'ci', '\\ensuremath{%s}' % fmt_ci(c['ci95'])); m(nm + 'p', '%.3f' % c['mcnemar_p'])
        m(nm + 'wl', '%d/%d' % (c['wins'], c['losses']))
        if 'holm_p' in c:
            m(nm + 'holm', '%.3f' % c['holm_p'])
    (out / 'macros.tex').write_text('\n'.join(macros) + '\n')
    # main results table: success counts per task per policy
    lines = ['\\begin{table*}[t]\\centering\\small', '\\setlength{\\tabcolsep}{4pt}',
             '\\begin{tabular}{ll' + 'c' * 10 + 'ccc}', '\\toprule',
             ' & & \\multicolumn{5}{c}{composite-seen} & \\multicolumn{5}{c}{composite-unseen} & & & \\\\',
             '\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}',
             'Model & Policy & ' + ' & '.join(SHORT[t] for t in SEEN) + ' & ' + ' & '.join(SHORT[t] for t in UNSEEN) + ' & Seen & Unseen & All \\\\', '\\midrule']
    for mk, mn in MODELS:
        first = True
        for ck, cn in CONDS:
            v = s['per_policy'].get('%s-%s' % (mk, ck))
            if not v:
                continue
            cells = ['%d' % v['tasks'][t]['success'] for t in SEEN + UNSEEN]
            n_t = v['tasks'][SEEN[0]]['n']
            row = ('\\multirow{4}{*}{%s}' % mn if first else '') + ' & ' + cn + ' & ' + ' & '.join(cells)
            pct = lambda x: (100 * x[0] / x[1]) if x[1] else 0.0
            row += ' & %.0f & %.0f & \\textbf{%.1f} \\\\' % (pct(v['seen']), pct(v['unseen']), v['rate'] or 0.0)
            lines.append(row); first = False
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines += ['\\end{tabular}', '\\caption{\\textbf{Per-task success counts.} Successes out of 20 seeded episodes per task, with seen, unseen and overall success rates (\\%) for the official checkpoint and the three adapted arms of each model. All arms are evaluated with ordinary inputs on the same 200 seeded episodes.}',
              '\\label{tab:main}', '\\end{table*}']
    (out / 'main_table.tex').write_text('\n'.join(lines) + '\n')
    # contrasts table; the interval is set in two right-aligned columns ("[lo," and "hi]") so that brackets and commas line up
    lines = ['\\begin{table}[t]\\centering\\footnotesize', '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{@{}llrr@{\\,}rrr@{}}', '\\toprule',
             'Model & Contrast & $\\Delta$ (pp) & \\multicolumn{2}{c}{95\\% CI} & W/L & $p$ \\\\', '\\midrule']
    order = [('F', 'BL'), ('F', 'S'), ('F', 'original'), ('S', 'original'), ('BL', 'original')]
    for mk, mn in MODELS:
        first = True
        for x, y in order:
            c = s['contrasts'].get('%s:%s-%s' % (mk, x, y))
            if not c:
                continue
            lab = '%s$-$%s' % (x, 'official' if y == 'original' else y)
            if c.get('primary'):
                lab = '\\textbf{' + lab + '}'
            lines.append(('\\multirow{5}{*}{%s}' % mn if first else '') + ' & %s & $%s$ & $[%s,$ & $%s]$ & %d/%d & %.3f \\\\' % (
                lab, sg(c['diff_pp']), sg(c['ci95'][0]), sg(c['ci95'][1]), c['wins'], c['losses'], c['mcnemar_p']))
            first = False
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines += ['\\end{tabular}', '\\caption{\\textbf{Paired contrasts.} Pooled over 200 (task, seed) pairs per model: success-rate difference, task-stratified paired-bootstrap 95\\% interval, wins/losses among discordant pairs, exact McNemar $p$ (uncorrected). Bold rows are the two pre-registered primary contrasts; their Holm-corrected verdicts are in the text.}',
              '\\label{tab:contrasts}', '\\end{table}']
    (out / 'contrast_table.tex').write_text('\n'.join(lines) + '\n')
    print('wrote', out / 'macros.tex', len(macros), 'macros')


if __name__ == '__main__':
    main()
