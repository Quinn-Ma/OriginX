"""Per-task effects of adaptation: Figure fig:tasks and paper/sections/results_tasks.tex.

Every number quoted in results_tasks.tex (and its caption) is printed by this script as a numbered fact
[T01], [T02], ...; nothing in that text is typed by hand.

Sources
  counts       results/analysis/summary.json      per_policy['<model>-<cond>']['tasks'][task] = {n, success}
  cross-checks results/analysis/summary.json      seen/unseen/all totals
               analysis/out/secondary.json        seen/unseen split contrasts; F-vs-official rescues/regressions per task
  pairing      results/eval/<model>-<cond>-q*/results.jsonl   episode logs, used only for the per-cell exact McNemar
                                                  tests, after checking that they reproduce every count in summary.json
Figure
  paper/figures/task_heatmap.pdf (+ .png preview): one column (3.25 in) of a two-column paper wide, drawn at final size
  (include without scaling), Times New Roman, smallest text 7 pt.

usage: python -X utf8 analysis/task_effects.py [--root D:/codex/sbl-study-20261108] [--no-fig]
"""
from pathlib import Path
import argparse, collections, json, math

MODELS = [('xr1', 'XR-1'), ('awomo_i', 'Awomo-0.5I'), ('awomo_v', 'Awomo-0.5V')]
ARMS = ['S', 'BL', 'F']
SEEN = ['DeliverStraw', 'GetToastedBread', 'PackIdenticalLunches', 'SearingMeat', 'StirVegetables']
UNSEEN = ['ArrangeTea', 'CuttingToolSelection', 'RecycleBottlesByType', 'WaffleReheat', 'WeighIngredients']
TASKS = SEEN + UNSEEN
SHORT = dict(zip(TASKS, ['Straw', 'Toast', 'Lunch', 'Sear', 'Stir', 'Tea', 'Cut', 'Recycle', 'Waffle', 'Weigh']))  # = Table 1 headers
MINUS = '\u2212'
N_SEEDS = 20  # seeds per task fixed by the protocol; checked against every count and every log below


def exact_mcnemar(b, c):
    """Two-sided exact binomial test on the discordant pairs (same definition as analysis/analyze.py)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def sgn(v):
    return ('+%d' % v) if v > 0 else ('%s%d' % (MINUS, -v) if v < 0 else '0')


def load_logs(root):
    """(model, cond) -> {(task, seed): success}; refuses duplicate episodes."""
    data = collections.defaultdict(dict)
    for d in sorted(Path(root).glob('*-*')):
        parts = d.name.split('-')
        if len(parts) < 2 or parts[0] not in dict(MODELS) or parts[1] not in ['original'] + ARMS:
            continue
        f = d / 'results.jsonl'
        if not f.exists():
            continue
        for line in f.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r['task'], int(r['seed']))
            if key in data[(parts[0], parts[1])]:
                raise ValueError('duplicate episode %s %s' % (d.name, key))
            data[(parts[0], parts[1])][key] = bool(r['success'])
    return data


def compute(root):
    summary = json.loads((root / 'results/analysis/summary.json').read_text(encoding='utf-8'))
    secondary = json.loads((root / 'analysis/out/secondary.json').read_text(encoding='utf-8'))
    pp = summary['per_policy']
    off = {m: {t: pp['%s-original' % m]['tasks'][t]['success'] for t in TASKS} for m, _ in MODELS}
    n_ep = {pp['%s-%s' % (m, c)]['tasks'][t]['n'] for m, _ in MODELS for c in ['original'] + ARMS for t in TASKS}
    assert n_ep == {N_SEEDS}, n_ep
    D = {(m, a): {t: pp['%s-%s' % (m, a)]['tasks'][t]['success'] - off[m][t] for t in TASKS} for m, _ in MODELS for a in ARMS}
    # --- cross-checks of the block sums against summary.json totals and secondary.json split contrasts
    for m, _ in MODELS:
        for a in ARMS:
            for blk, ts in [('seen', SEEN), ('unseen', UNSEEN), ('all', TASKS)]:
                s = sum(D[(m, a)][t] for t in ts)
                assert s == pp['%s-%s' % (m, a)][blk][0] - pp['%s-original' % m][blk][0], (m, a, blk)
                if blk != 'all':
                    sc = secondary['split_contrasts']['%s:%s-original:%s' % (m, a, blk)]
                    assert s == sc['a_success'] - sc['b_success'] and abs(sc['diff_pp'] - s) < 1e-9, (m, a, blk)
    # --- episode logs: reproduce every count, then pair episodes per cell
    logs = load_logs(root / 'results/eval')
    cell = {}
    for m, _ in MODELS:
        keys0 = set(logs[(m, 'original')])
        assert keys0 == {(t, s) for t in TASKS for s in range(N_SEEDS)}, m
        for c in ['original'] + ARMS:
            assert set(logs[(m, c)]) == keys0, (m, c)
            for t in TASKS:
                assert sum(logs[(m, c)][(t, s)] for s in range(N_SEEDS)) == pp['%s-%s' % (m, c)]['tasks'][t]['success'], (m, c, t)
        for a in ARMS:
            for t in TASKS:
                w = sum(1 for s in range(N_SEEDS) if logs[(m, a)][(t, s)] and not logs[(m, 'original')][(t, s)])
                l = sum(1 for s in range(N_SEEDS) if logs[(m, 'original')][(t, s)] and not logs[(m, a)][(t, s)])
                assert w - l == D[(m, a)][t]
                cell[(m, a, t)] = (w, l, exact_mcnemar(w, l))
        resc = secondary['rescues'][m]
        for t in TASKS:  # F rescues/regressions per task in secondary.json
            assert cell[(m, 'F', t)][:2] == (resc['rescued'].get(t, 0), resc['regressed'].get(t, 0)), (m, t)
    return off, D, cell


def facts(off, D, cell):
    out = []

    def fact(text):
        out.append('[T%02d] %s' % (len(out) + 1, text))

    name = dict(MODELS)
    blk_sum = lambda m, a, ts: sum(D[(m, a)][t] for t in ts)
    # ---------- full matrix (also what the figure shows)
    print('Successes of the arm minus successes of the official checkpoint, per task (%d paired episodes per cell)' % N_SEEDS)
    print('%-11s %-8s ' % ('model', 'policy') + ' '.join('%7s' % SHORT[t] for t in TASKS) + '   seen unseen   all')
    for m, mn in MODELS:
        print('%-11s %-8s ' % (mn, 'official') + ' '.join('%7d' % off[m][t] for t in TASKS) +
              '  %5d %6d %5d   (official successes)' % (sum(off[m][t] for t in SEEN), sum(off[m][t] for t in UNSEEN), sum(off[m].values())))
        for a in ARMS:
            print('%-11s %-8s ' % ('', a) + ' '.join('%+7d' % D[(m, a)][t] for t in TASKS) +
                  '  %+5d %+6d %+5d' % (blk_sum(m, a, SEEN), blk_sum(m, a, UNSEEN), blk_sum(m, a, TASKS)))
    print('(seen/unseen/all sums are out of %d/%d/%d paired episodes; seen and unseen sums equal the split contrasts in pp)' % (
        len(SEEN) * N_SEEDS, len(UNSEEN) * N_SEEDS, len(TASKS) * N_SEEDS))
    print()
    print('Per-cell exact McNemar (arm wins / official wins, two-sided p, uncorrected)')
    for m, mn in MODELS:
        for a in ARMS:
            print('%-11s %-3s ' % (mn, a) + '  '.join('%s %d/%d p=%.3f' % (SHORT[t], *cell[(m, a, t)]) for t in TASKS))
    print()

    # ---------- generic facts used in caption / noise sentence
    n_cells = len(cell)
    fact('cells in the figure: %d (%d models x %d arms x %d tasks), each on %d paired episodes, so one success = %g pp; '
         'a block sum covers %d tasks x %d = %d paired episodes, so one success = %g pp' % (
             n_cells, len(MODELS), len(ARMS), len(TASKS), N_SEEDS, 100 / N_SEEDS,
             len(SEEN), N_SEEDS, len(SEEN) * N_SEEDS, 100 / (len(SEEN) * N_SEEDS)))
    dmin = min(abs(b - c) for b in range(N_SEEDS + 1) for c in range(N_SEEDS + 1 - b) if exact_mcnemar(b, c) < 0.05)
    fact('smallest per-task difference that an exact McNemar test on %d paired episodes can call p < 0.05: %d successes '
         '(differences of up to %d cannot reach it whatever the discordant split)' % (N_SEEDS, dmin, dmin - 1))
    big = [(m, a, t) for (m, a, t) in cell if abs(D[(m, a)][t]) >= dmin]
    fact('cells with |difference| >= %d: %d -- %s' % (dmin, len(big), '; '.join(
        '%s-%s %s %s (wins/losses %d/%d, p=%.3f)' % (name[m], a, SHORT[t], sgn(D[(m, a)][t]), *cell[(m, a, t)]) for m, a, t in big)))
    sig = [k for k, v in cell.items() if v[2] < 0.05]
    pmin_key = min(cell, key=lambda k: cell[k][2])
    fact('cells with uncorrected exact McNemar p < 0.05: %d of %d; smallest p = %.4f (%s-%s %s, %s)' % (
        len(sig), n_cells, cell[pmin_key][2], name[pmin_key[0]], pmin_key[1], SHORT[pmin_key[2]], sgn(D[pmin_key[:2]][pmin_key[2]])))
    rng = [D[(m, a)][t] for m, _ in MODELS for a in ARMS for t in TASKS]
    fact('range of all cells: %s to %s successes' % (sgn(min(rng)), sgn(max(rng))))

    # ---------- per model and arm: sign counts and block sums
    for m, mn in MODELS:
        for a in ARMS:
            d = D[(m, a)]
            pos, neg, zer = [sum(1 for t in TASKS if f(d[t])) for f in (lambda v: v > 0, lambda v: v < 0, lambda v: v == 0)]
            tmin = min(TASKS, key=lambda t: d[t]); tmax = max(TASKS, key=lambda t: d[t])
            fact('%s %s: %d tasks up, %d down, %d level; sums seen %s, unseen %s, all %s; largest loss %s (%s), largest gain %s (%s)' % (
                mn, a, pos, neg, zer, sgn(blk_sum(m, a, SEEN)), sgn(blk_sum(m, a, UNSEEN)), sgn(blk_sum(m, a, TASKS)),
                sgn(d[tmin]), SHORT[tmin], sgn(d[tmax]), SHORT[tmax]))

    # ---------- XR-1: where the gain sits
    m = 'xr1'
    allup = [t for t in SEEN if all(D[(m, a)][t] > 0 for a in ARMS)]
    rest = [t for t in SEEN if t not in allup]
    cells_up = [D[(m, a)][t] for a in ARMS for t in allup]
    fact('XR-1 seen tasks on which all three arms gain: %d of %d (%s); their %d cells range %s to %s; official successes on them %d-%d of %d' % (
        len(allup), len(SEEN), ', '.join(SHORT[t] for t in allup), len(cells_up), sgn(min(cells_up)), sgn(max(cells_up)),
        min(off[m][t] for t in allup), max(off[m][t] for t in allup), N_SEEDS))
    for t in rest:
        fact('XR-1 remaining seen task %s: official %d/%d (%s among the seen tasks), S/BL/F differences %s' % (
            SHORT[t], off[m][t], N_SEEDS, 'highest' if off[m][t] == max(off[m][x] for x in SEEN) else 'not highest',
            ', '.join(sgn(D[(m, a)][t]) for a in ARMS)))
    fact('XR-1 sum over those %d seen tasks per arm S/BL/F: %s (seen-block sums %s)' % (
        len(allup), ', '.join(sgn(sum(D[(m, a)][t] for t in allup)) for a in ARMS), ', '.join(sgn(blk_sum(m, a, SEEN)) for a in ARMS)))
    un = [D[(m, a)][t] for a in ARMS for t in UNSEEN]
    fact('XR-1 unseen cells range %s to %s; unseen sums S/BL/F %s (out of %d paired episodes = pp)' % (
        sgn(min(un)), sgn(max(un)), ', '.join(sgn(blk_sum(m, a, UNSEEN)) for a in ARMS), len(UNSEEN) * N_SEEDS))

    # ---------- Awomo-0.5I: broad unseen losses
    m = 'awomo_i'
    un = [D[(m, a)][t] for a in ARMS for t in UNSEEN]
    fact('Awomo-0.5I unseen cells below zero: %d of %d; range %s to %s; unseen sums S/BL/F %s; seen sums S/BL/F %s' % (
        sum(1 for v in un if v < 0), len(un), sgn(min(un)), sgn(max(un)),
        ', '.join(sgn(blk_sum(m, a, UNSEEN)) for a in ARMS), ', '.join(sgn(blk_sum(m, a, SEEN)) for a in ARMS)))
    fu = [D[(m, 'F')][t] for t in UNSEEN]
    fact('Awomo-0.5I F unseen cells: %s (every unseen task between %s and %s)' % (
        ', '.join('%s %s' % (SHORT[t], sgn(D[(m, 'F')][t])) for t in UNSEEN), sgn(min(fu)), sgn(max(fu))))
    worst = min(un)
    fact('Awomo-0.5I largest unseen losses (%s): %s' % (sgn(worst), ', '.join(
        '%s %s' % (a, SHORT[t]) for a in ARMS for t in UNSEEN if D[(m, a)][t] == worst)))

    # ---------- Awomo-0.5V: losses on both blocks, not one collapsed task
    m = 'awomo_v'
    d = D[(m, 'F')]
    tmin = min(TASKS, key=lambda t: d[t])
    fact('Awomo-0.5V F: %d of %d tasks below zero, total %s; largest single loss %s (%s: official %d/%d -> F %d/%d); other %d tasks sum to %s' % (
        sum(1 for t in TASKS if d[t] < 0), len(TASKS), sgn(sum(d.values())), sgn(d[tmin]), SHORT[tmin], off[m][tmin], N_SEEDS,
        off[m][tmin] + d[tmin], N_SEEDS, len(TASKS) - 1, sgn(sum(d[t] for t in TASKS if t != tmin))))
    fact('Awomo-0.5V seen sums S/BL/F %s; unseen sums S/BL/F %s' % (
        ', '.join(sgn(blk_sum(m, a, SEEN)) for a in ARMS), ', '.join(sgn(blk_sum(m, a, UNSEEN)) for a in ARMS)))
    for t in SEEN[:2]:
        fact('Awomo-0.5V %s: official %d/%d (rank %d of %d among seen tasks), S/BL/F differences %s' % (
            SHORT[t], off[m][t], N_SEEDS, 1 + sorted([off[m][x] for x in SEEN], reverse=True).index(off[m][t]), len(SEEN),
            ', '.join(sgn(D[(m, a)][t]) for a in ARMS)))

    # ---------- across models
    for m, mn in MODELS:
        cells = [D[(m, a)][t] for a in ARMS for t in TASKS]
        fact('%s cells above / below / at zero (all arms, all tasks): %d / %d / %d of %d' % (
            mn, sum(v > 0 for v in cells), sum(v < 0 for v in cells), sum(v == 0 for v in cells), len(cells)))

    # ---------- shared baseline (not quoted in the text; supports its last sentence qualitatively)
    def pearson(x, y):
        mx, my = sum(x) / len(x), sum(y) / len(y)
        sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
        return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    rs, X, Y = [], [], []
    for m, mn in MODELS:
        o = [off[m][t] for t in TASKS]; md = [sum(D[(m, a)][t] for a in ARMS) / len(ARMS) for t in TASKS]
        rs.append('%s %.2f' % (mn, pearson(o, md)))
        X += [v - sum(o) / len(o) for v in o]; Y += [v - sum(md) / len(md) for v in md]
    fact('Pearson r between a task\'s official count and its mean S/BL/F difference (10 tasks per model): %s; pooled after '
         'centring within model: %.2f (a noisy shared baseline alone produces a negative r)' % ('; '.join(rs), pearson(X, Y)))
    for line in out:
        print(line)
    return out


def draw(off, D, out_base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize, to_rgb
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': True})
    FS, FS_HEAD = 7, 7.5                     # pt; nothing smaller than 7 pt
    W = 3.25                                 # in: one column of a two-column paper, drawn at final size
    x_lab = 0.585                            # right edge of the task labels
    x0 = 0.64                                # left edge of the first cell column
    gm = 0.075                               # extra space between model groups
    cw = (W - 0.015 - x0 - 2 * gm) / 9       # cell pitch
    rh = 0.140                               # row pitch
    g = 0.014                                # white gutter between cells
    top = 0.315                              # header height
    gb = 0.065                               # space between the seen and unseen blocks
    H = top + 12 * rh + gb + 0.02
    cmap = plt.get_cmap('RdBu'); norm = Normalize(-8, 8)   # blue = more successes than official, red = fewer; white = 0

    def ink(rgb):  # black or white text, whichever contrasts more with the cell
        lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
        lin = [((v / 12.92) if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) for v in rgb]
        return 'black' if (lum(lin) + 0.05) / 0.05 >= 1.05 / (lum(lin) + 0.05) else 'white'

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    colx = lambda mi, ai: x0 + mi * (3 * cw + gm) + ai * cw
    # row layout: 5 seen tasks, seen sum, gap, 5 unseen tasks, unseen sum
    rows, y = [], top
    for blk, ts in [('seen', SEEN), ('unseen', UNSEEN)]:
        y_blk0 = y
        for t in ts:
            rows.append(('task', t, y)); y += rh
        rows.append(('sum', ts, y)); y_blk1 = y; y += rh
        # block label with a thin bracket spanning the task rows
        ax.plot([0.165, 0.165], [y_blk0 + g, y_blk1 - g], color='black', lw=0.5, solid_capstyle='butt')
        ax.text(0.10, (y_blk0 + y_blk1) / 2, blk, rotation=90, ha='center', va='center', fontsize=FS)
        y += gb
    for mi, (m, mn) in enumerate(MODELS):
        xa, xb = colx(mi, 0), colx(mi, 2) + cw
        ax.text((xa + xb) / 2, 0.068, mn, ha='center', va='center', fontsize=FS_HEAD)
        ax.plot([xa + 0.02, xb - 0.02], [0.145, 0.145], color='black', lw=0.5)
        for ai, a in enumerate(ARMS):
            ax.text(colx(mi, ai) + cw / 2, 0.228, a, ha='center', va='center', fontsize=FS)
    for kind, t, y in rows:
        if kind == 'task':
            ax.text(x_lab, y + rh / 2, SHORT[t], ha='right', va='center', fontsize=FS)
        else:
            ax.text(x_lab, y + rh / 2, '\u03a3', ha='right', va='center', fontsize=FS)
        for mi, (m, mn) in enumerate(MODELS):
            if kind == 'sum':
                ax.plot([colx(mi, 0) + g / 2, colx(mi, 2) + cw - g / 2], [y + 0.004, y + 0.004], color='black', lw=0.4)
            for ai, a in enumerate(ARMS):
                xc = colx(mi, ai)
                if kind == 'task':
                    v = D[(m, a)][t]; rgb = to_rgb(cmap(norm(v)))
                    ax.add_patch(Rectangle((xc + g / 2, y + g / 2), cw - g, rh - g, facecolor=rgb, edgecolor='none'))
                    ax.text(xc + cw / 2, y + rh / 2, sgn(v), ha='center', va='center', fontsize=FS, color=ink(rgb))
                else:
                    v = sum(D[(m, a)][x] for x in t)
                    ax.text(xc + cw / 2, y + rh / 2 + 0.006, sgn(v), ha='center', va='center', fontsize=FS, fontweight='bold')
    out_base = Path(out_base); out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix('.pdf'))
    fig.savefig(out_base.with_suffix('.png'), dpi=300)
    plt.close(fig)
    print('figure: %s.{pdf,png}  size %.3f x %.3f in (%.2f x %.2f cm)' % (out_base, W, H, 2.54 * W, 2.54 * H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--fig-out', default=None, help='output base path (default <root>/paper/figures/task_heatmap)')
    ap.add_argument('--no-fig', action='store_true')
    a = ap.parse_args(); root = Path(a.root)
    off, D, cell = compute(root)
    print('cross-checks passed: summary.json totals, secondary.json split contrasts and F rescues, 2,400 logged episodes\n')
    facts(off, D, cell)
    if not a.no_fig:
        draw(off, D, a.fig_out or root / 'paper/figures/task_heatmap')


if __name__ == '__main__':
    main()
