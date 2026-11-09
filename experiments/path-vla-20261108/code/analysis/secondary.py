"""Secondary, descriptive analyses from the episode logs: seen/unseen split of contrasts, rescues/regressions,
episode length, and candidate qualitative pairs. Reads the local mirror results/eval/*/results.jsonl."""
from pathlib import Path
import collections, json, math, random, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load, paired_contrast, SEEN, UNSEEN, MODELS

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'D:/codex/sbl-study-20261108/results/eval')
out = Path(sys.argv[2] if len(sys.argv) > 2 else 'D:/codex/sbl-study-20261108/analysis/out')
data = load(root)
res = {'split_contrasts': {}, 'steps': {}, 'rescues': {}, 'qual_pairs': {}}
pairs = [('F', 'BL'), ('F', 'S'), ('F', 'original'), ('S', 'original'), ('BL', 'original')]
for model in MODELS:
    for x, y in pairs:
        for split, tasks in [('seen', SEEN), ('unseen', UNSEEN)]:
            ra = {k: v for k, v in data[(model, x)].items() if k[0] in tasks}
            rb = {k: v for k, v in data[(model, y)].items() if k[0] in tasks}
            c = paired_contrast(ra, rb, n_boot=4000)
            res['split_contrasts']['%s:%s-%s:%s' % (model, x, y, split)] = {k: c[k] for k in ['n_pairs', 'a_success', 'b_success', 'diff_pp', 'ci95', 'wins', 'losses', 'mcnemar_p']}
    for cond in ['original', 'S', 'BL', 'F']:
        recs = data[(model, cond)].values()
        succ = [r['steps'] for r in recs if r['success']]; fail = [r['steps'] for r in recs if not r['success']]
        res['steps']['%s-%s' % (model, cond)] = {'median_steps_success': sorted(succ)[len(succ) // 2] if succ else None,
                                                 'timeouts': sum(1 for r in recs if not r['success'] and r['steps'] >= r['horizon']),
                                                 'early_terminations': sum(1 for r in recs if not r['success'] and r['steps'] < r['horizon']),
                                                 'mean_elapsed_s': round(sum(r['elapsed_s'] for r in recs) / len(recs), 1)}
    # rescues / regressions of F vs original with task breakdown
    ro, rf = data[(model, 'original')], data[(model, 'F')]
    resc = collections.Counter(); regr = collections.Counter()
    for k in ro:
        if rf[k]['success'] and not ro[k]['success']:
            resc[k[0]] += 1
        if ro[k]['success'] and not rf[k]['success']:
            regr[k[0]] += 1
    res['rescues'][model] = {'rescued': dict(resc), 'regressed': dict(regr)}
    # qualitative candidates: shortest rescue and a regression
    rescues = [(rf[k]['steps'], k) for k in ro if rf[k]['success'] and not ro[k]['success']]
    regressions = [(ro[k]['steps'], k) for k in ro if ro[k]['success'] and not rf[k]['success']]
    res['qual_pairs'][model] = {'rescue': sorted(rescues)[:3], 'regression': sorted(regressions)[:3]}
# instruction examples per task
instr = {}
for (m, c), recs in data.items():
    for k, r in recs.items():
        instr.setdefault(k[0], set()).add(r['instruction'])
res['instructions_per_task'] = {t: sorted(v)[:3] for t, v in instr.items()}
res['n_distinct_instructions'] = {t: len(v) for t, v in instr.items()}
out.mkdir(parents=True, exist_ok=True)
(out / 'secondary.json').write_text(json.dumps(res, indent=1, default=str))
for k, v in res['split_contrasts'].items():
    print(k, v['diff_pp'], v['ci95'], 'W/L %d/%d' % (v['wins'], v['losses']))
print(json.dumps(res['steps'], indent=0))
print(json.dumps(res['rescues'], indent=0))
print(json.dumps(res['qual_pairs'], indent=0, default=str))
print(res['n_distinct_instructions'])
