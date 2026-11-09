"""Pre-registered analysis of the SBL study.

Inputs : eval/<model>-<condition>[-seen|-unseen]/results.jsonl (one JSON per episode)
Outputs: analysis/summary.json, analysis/tables.md  (per-task success counts, paired contrasts with
         task-stratified paired bootstrap CIs, exact McNemar tests, Holm over the two primary contrasts per model).
"""
from pathlib import Path
import argparse, collections, json, math, random
MODELS = ['xr1', 'awomo_i', 'awomo_v']
CONDS = ['original', 'S', 'BL', 'F']
SEEN = ['DeliverStraw', 'GetToastedBread', 'PackIdenticalLunches', 'SearingMeat', 'StirVegetables']
UNSEEN = ['ArrangeTea', 'CuttingToolSelection', 'RecycleBottlesByType', 'WaffleReheat', 'WeighIngredients']
TASKS = SEEN + UNSEEN
PRIMARY = [('F', 'BL'), ('F', 'S')]
SECONDARY = [('F', 'original'), ('S', 'original'), ('BL', 'original'), ('S', 'BL')]


def load(root):
    data = collections.defaultdict(dict)  # (model, cond) -> {(task, seed): record}
    for d in sorted(Path(root).glob('*-*')):
        if d.name.startswith('smoke'):
            continue
        parts = d.name.split('-'); model = parts[0]; cond = parts[1]
        if model not in MODELS or cond not in CONDS:
            continue
        f = d / 'results.jsonl'
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            key = (r['task'], r['seed'])
            if key in data[(model, cond)]:
                raise ValueError('duplicate episode %s %s %s' % (model, cond, key))
            data[(model, cond)][key] = r
    return data


def exact_mcnemar(b, c):
    """Two-sided exact binomial test on discordant pairs b (A wins) and c (B wins)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_contrast(recs_a, recs_b, n_boot=10000, seed=20261108):
    keys = sorted(set(recs_a) & set(recs_b))
    keys = [k for k in keys if recs_a[k].get('success') is not None and recs_b[k].get('success') is not None]
    if not keys:
        return None
    by_task = collections.defaultdict(list)
    for k in keys:
        by_task[k[0]].append((int(bool(recs_a[k]['success'])), int(bool(recs_b[k]['success']))))
    n = len(keys); sa = sum(a for t in by_task.values() for a, b in t); sb = sum(b for t in by_task.values() for a, b in t)
    wins = sum(1 for t in by_task.values() for a, b in t if a > b); losses = sum(1 for t in by_task.values() for a, b in t if a < b)
    diff = 100 * (sa - sb) / n
    rng = random.Random(seed); boots = []
    tasks = list(by_task)
    for _ in range(n_boot):
        tot = 0; cnt = 0
        for t in tasks:
            pairs = by_task[t]; m = len(pairs)
            for _ in range(m):
                a, b = pairs[rng.randrange(m)]; tot += a - b; cnt += 1
        boots.append(100 * tot / cnt)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]; hi = boots[int(0.975 * n_boot) - 1]
    lo99 = boots[int(0.0125 * n_boot)]; hi99 = boots[int(0.9875 * n_boot) - 1]  # 97.5% interval for Holm (2 contrasts)
    p = exact_mcnemar(wins, losses)
    per_task = {t: {'n': len(v), 'a': sum(a for a, b in v), 'b': sum(b for a, b in v)} for t, v in by_task.items()}
    return {'n_pairs': n, 'a_success': sa, 'b_success': sb, 'diff_pp': round(diff, 2), 'ci95': [round(lo, 2), round(hi, 2)],
            'ci975': [round(lo99, 2), round(hi99, 2)], 'wins': wins, 'losses': losses, 'mcnemar_p': round(p, 4), 'per_task': per_task}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/dev/shm/qinzhen/sbl/eval'); ap.add_argument('--out', default='/dev/shm/qinzhen/sbl/analysis')
    ap.add_argument('--n-boot', type=int, default=10000)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = load(a.root); summary = {'per_policy': {}, 'contrasts': {}}
    lines = ['# SBL study — results (pre-registered analysis)', '']
    lines.append('## Success counts per policy (successes / episodes)'); lines.append('')
    header = '| model | condition | ' + ' | '.join(TASKS) + ' | seen | unseen | all |'
    lines += [header, '|' + '---|' * (len(TASKS) + 5)]
    for model in MODELS:
        for cond in CONDS:
            recs = data.get((model, cond), {})
            if not recs:
                continue
            row = {'tasks': {}, 'errors': sum(1 for r in recs.values() if r.get('status') != 'ok')}
            cells = []
            for t in TASKS:
                rs = [r for (tt, s), r in recs.items() if tt == t]
                succ = sum(1 for r in rs if r.get('success')); row['tasks'][t] = {'n': len(rs), 'success': succ}
                cells.append('%d/%d' % (succ, len(rs)) if rs else '-')
            def agg(ts):
                n = sum(row['tasks'][t]['n'] for t in ts); s = sum(row['tasks'][t]['success'] for t in ts)
                return s, n
            s1, n1 = agg(SEEN); s2, n2 = agg(UNSEEN); s3, n3 = agg(TASKS)
            row.update(seen=[s1, n1], unseen=[s2, n2], all=[s3, n3], rate=round(100 * s3 / n3, 1) if n3 else None)
            summary['per_policy']['%s-%s' % (model, cond)] = row
            lines.append('| %s | %s | %s | %d/%d | %d/%d | %d/%d (%.1f%%) |' % (model, cond, ' | '.join(cells), s1, n1, s2, n2, s3, n3, 100 * s3 / n3 if n3 else 0))
    lines.append(''); lines.append('## Paired contrasts (pooled over tasks; task-stratified paired bootstrap; exact McNemar)'); lines.append('')
    lines += ['| model | contrast | pairs | A | B | diff (pp) | 95% CI | 97.5% CI (Holm, primary) | wins/losses | McNemar p |', '|---|---|---|---|---|---|---|---|---|---|']
    for model in MODELS:
        for (x, y) in PRIMARY + SECONDARY:
            ra, rb = data.get((model, x)), data.get((model, y))
            if not ra or not rb:
                continue
            c = paired_contrast(ra, rb, n_boot=a.n_boot)
            if c is None:
                continue
            c['primary'] = (x, y) in PRIMARY
            summary['contrasts']['%s:%s-%s' % (model, x, y)] = c
            lines.append('| %s | %s − %s%s | %d | %d | %d | %+.2f | [%+.2f, %+.2f] | [%+.2f, %+.2f] | %d/%d | %.4f |' % (
                model, x, y, ' (primary)' if c['primary'] else '', c['n_pairs'], c['a_success'], c['b_success'], c['diff_pp'],
                c['ci95'][0], c['ci95'][1], c['ci975'][0], c['ci975'][1], c['wins'], c['losses'], c['mcnemar_p']))
    # Holm over the two primary contrasts per model
    lines.append(''); lines.append('## Primary verdicts (Holm over F−BL and F−S within each model; supported = corrected CI excludes 0 and diff ≥ +5 pp)'); lines.append('')
    for model in MODELS:
        ps = [(k, v) for k, v in summary['contrasts'].items() if k.startswith(model + ':') and v['primary']]
        if len(ps) < 2:
            continue
        ps.sort(key=lambda kv: kv[1]['mcnemar_p'])
        # Holm step-down over the two contrasts: the one with the smaller p is judged on its 97.5% interval and doubled p; the
        # other only if the first is rejected, on its 95% interval. Adjusted p-values are monotone: p_adj(2) = max(p_adj(1), p(2)).
        prev_holm, prev_rejected = 0.0, True
        for rank, (k, v) in enumerate(ps):
            holm_p = max(prev_holm, min(1.0, v['mcnemar_p'] * (2 - rank)))
            v['holm_p'] = round(holm_p, 4)
            ci = v['ci975'] if rank == 0 else v['ci95']
            rejected = prev_rejected and (ci[0] > 0 or ci[1] < 0)
            supported = rejected and (ci[0] > 0) and v['diff_pp'] >= 5
            v['supported'] = bool(supported)
            prev_holm, prev_rejected = holm_p, rejected
            lines.append('- %s: diff %+.2f pp, Holm-adjusted interval [%+.2f, %+.2f], Holm p = %.4f → %s' % (k, v['diff_pp'], ci[0], ci[1], holm_p, 'SUPPORTED' if supported else 'not supported'))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1)); (out / 'tables.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
