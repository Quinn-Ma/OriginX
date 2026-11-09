"""Post-hoc sensitivity for two deviations in the official-checkpoint evaluation.

1. Eighteen Awomo-0.5V official-checkpoint episodes crashed with a CUDA out-of-memory error. Their error rows were
   removed and the episodes rerun from scratch after partial results had been viewed. The pre-registration says every
   episode is reported (success / failure / error / timeout); here the crashed attempts are counted as failures and every
   contrast against that checkpoint is recomputed.
2. Nine official-checkpoint episodes (one smoke test, eight from a first launch that was aborted and relaunched with a
   different shard split) are not part of the 2,400; they are compared with their released counterparts.

Inputs : results/eval (released episodes), results/eval_crashed/*.with_errors.jsonl (the original rows incl. the 18 errors),
         results/eval_unreleased/raw_rows.txt (the nine rows, copied from the cluster)
Outputs: analysis/out/crash_sensitivity.json, paper/tables/macros_crash.tex
"""
from pathlib import Path
import json, sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from analyze import load, paired_contrast

data = load(ROOT / 'results/eval')
off = data[('awomo_v', 'original')]
crashed, ok_checked = [], 0
for f in sorted((ROOT / 'results/eval_crashed').glob('awomo_v-original-*.with_errors.jsonl')):
    for line in open(f, encoding='utf-8'):
        r = json.loads(line); k = (r['task'], r['seed'])
        if r.get('status') == 'error':
            assert 'OutOfMemoryError' in r.get('error', '') or 'out of memory' in r.get('error', '').lower(), r.get('error', '')[-300:]
            crashed.append(k)
        else:  # every completed row of the first attempt must be unchanged in the released file
            assert all(off[k][x] == r[x] for x in ('success', 'steps', 'first_frame_sha256')), k
            ok_checked += 1
assert len(crashed) == len(set(crashed)) == 18, len(crashed)
rerun_success = sorted(k for k in crashed if off[k]['success'])
strict = {k: (dict(v, success=False) if k in crashed else v) for k, v in off.items()}


def sg(x):
    s = '%.1f' % abs(x)
    return s if float(s) == 0 else ('+' if x > 0 else '-') + s


out = {'n_crashed': len(crashed), 'crashed': [list(k) for k in sorted(crashed)], 'rerun_successes': [list(k) for k in rerun_success],
       'completed_rows_unchanged': ok_checked, 'official_successes_released': sum(bool(v['success']) for v in off.values()),
       'official_successes_errors_as_failures': sum(bool(v['success']) for v in strict.values()), 'contrasts': {}}
macros = [r'\newcommand{\crashN}{%d}' % len(crashed), r'\newcommand{\crashRerunSucc}{%d}' % len(rerun_success),
          r'\newcommand{\crashAVoffReleased}{%d}' % out['official_successes_released'],
          r'\newcommand{\crashAVoffStrict}{%d}' % out['official_successes_errors_as_failures']]
for c in ['F', 'S', 'BL']:
    for tag, ref in [('released', off), ('strict', strict)]:
        r = paired_contrast(data[('awomo_v', c)], ref, n_boot=10000)
        out['contrasts']['%s-official:%s' % (c, tag)] = {k: r[k] for k in ['n_pairs', 'a_success', 'b_success', 'diff_pp', 'ci95', 'wins', 'losses', 'mcnemar_p']}
    r = out['contrasts']['%s-official:strict' % c]
    macros += [r'\newcommand{\crashAV%soffdiff}{%s}' % (c, sg(r['diff_pp'])),
               r'\newcommand{\crashAV%soffci}{[%s, %s]}' % (c, sg(r['ci95'][0]), sg(r['ci95'][1])),
               r'\newcommand{\crashAV%soffp}{%.3f}' % (c, r['mcnemar_p'])]

# where the crashes fell, and every strict-count value the paper quotes: seen/unseen contrasts, per-task official counts,
# F rescues/regressions (Table S5), F-official on the identical seeds (Table S6) and the per-task sum quoted with Figure 4
SEEN = ['DeliverStraw', 'GetToastedBread', 'PackIdenticalLunches', 'SearingMeat', 'StirVegetables']
UNSEEN = ['ArrangeTea', 'CuttingToolSelection', 'RecycleBottlesByType', 'WaffleReheat', 'WeighIngredients']
assert all(k[0] in UNSEEN for k in crashed)
out['all_crashes_unseen'] = True


def isg(x):
    """integer-valued difference (pp on 100 pairs) for math mode: signed, zero unsigned"""
    v = int(round(x))
    assert abs(v - x) < 1e-6, x
    return '0' if v == 0 else ('+%d' % v if v > 0 else '-%d' % -v)


def part(d, tasks):
    return {k: v for k, v in d.items() if k[0] in tasks}


split = {}
for c in ['S', 'BL', 'F']:
    for name, tasks in [('seen', SEEN), ('unseen', UNSEEN)]:
        for tag, ref in [('released', off), ('strict', strict)]:
            r = paired_contrast(part(data[('awomo_v', c)], tasks), part(ref, tasks), n_boot=10000)
            split['%s-official:%s:%s' % (c, name, tag)] = {k: r[k] for k in ['n_pairs', 'diff_pp', 'ci95', 'wins', 'losses', 'mcnemar_p']}
    assert split['%s-official:seen:released' % c] == split['%s-official:seen:strict' % c]  # the crashes are all unseen
    macros += [r'\newcommand{\crashAV%soffSeen}{%s}' % (c, isg(split['%s-official:seen:released' % c]['diff_pp'])),
               r'\newcommand{\crashAV%soffUnRel}{%s}' % (c, isg(split['%s-official:unseen:released' % c]['diff_pp'])),
               r'\newcommand{\crashAV%soffUnStrict}{%s}' % (c, isg(split['%s-official:unseen:strict' % c]['diff_pp']))]
out['split_contrasts'] = split
n_un = [sum(bool(v['success']) for k, v in part(d, UNSEEN).items()) for d in (off, strict)]
out['official_unseen_successes'] = {'released': n_un[0], 'strict': n_un[1]}
macros += [r'\newcommand{\crashAVoffUnseenRel}{%d}' % n_un[0], r'\newcommand{\crashAVoffUnseenStrict}{%d}' % n_un[1]]
SHORT = {'CuttingToolSelection': 'Cut', 'RecycleBottlesByType': 'Recycle', 'WaffleReheat': 'Waffle', 'WeighIngredients': 'Weigh'}
out['official_per_task'] = {}
for t in sorted({k[0] for k in crashed}):
    a = [sum(bool(v['success']) for k, v in d.items() if k[0] == t) for d in (off, strict)]
    out['official_per_task'][t] = {'released': a[0], 'strict': a[1]}
    macros += [r'\newcommand{\crashOff%sRel}{%d}' % (SHORT[t], a[0]), r'\newcommand{\crashOff%sStrict}{%d}' % (SHORT[t], a[1])]
for tag in ['released', 'strict']:
    r = out['contrasts']['F-official:%s' % tag]
    nm = 'Rel' if tag == 'released' else 'Strict'
    macros += [r'\newcommand{\crashAVFoffWins%s}{%d}' % (nm, r['wins']), r'\newcommand{\crashAVFoffLosses%s}{%d}' % (nm, r['losses'])]
sens = json.loads((HERE / 'out/sensitivity.json').read_text())
variant = {tuple(x) for x in sens['frame_variant_seeds']} | {tuple(x) for x in sens['state_variant_seeds']}
ident = [k for k in off if k not in variant]
assert len(ident) == sens['n_identical'], (len(ident), sens['n_identical'])
r = paired_contrast({k: data[('awomo_v', 'F')][k] for k in ident}, {k: strict[k] for k in ident}, n_boot=10000)
out['F-official:strict:identical_seeds'] = {k: r[k] for k in ['n_pairs', 'diff_pp', 'ci95', 'wins', 'losses', 'mcnemar_p']}
macros += [r'\newcommand{\crashAVFoffSensDiff}{%s}' % sg(r['diff_pp']), r'\newcommand{\crashAVFoffSensP}{%.3f}' % r['mcnemar_p']]
per_task = {}
for tag, ref in [('released', off), ('strict', strict)]:
    per_task[tag] = {t: sum(bool(v['success']) for k, v in data[('awomo_v', 'F')].items() if k[0] == t)
                     - sum(bool(v['success']) for k, v in ref.items() if k[0] == t) for t in SEEN + UNSEEN}
largest = min(per_task['released'], key=per_task['released'].get)
nine = {tag: sum(v for t, v in d.items() if t != largest) for tag, d in per_task.items()}
out['F-official_per_task'] = per_task
out['F-official_sum_without_largest_loss'] = dict(nine, largest_loss_task=largest)
macros += [r'\newcommand{\crashAVFnineRel}{%s}' % isg(nine['released']), r'\newcommand{\crashAVFnineStrict}{%s}' % isg(nine['strict'])]

# the nine unreleased official-checkpoint episodes
rows, cur = [], None
for line in open(ROOT / 'results/eval_unreleased/raw_rows.txt', encoding='utf-8'):
    line = line.strip()
    if line.startswith('###'):
        cur = line[4:]
    elif line:
        rows.append((cur, json.loads(line)))
same = 0
for src, r in rows:
    rel = data[(r['model'], r['condition'])][(r['task'], r['seed'])]
    if all(rel[x] == r[x] for x in ('success', 'steps', 'first_frame_sha256')):
        same += 1
out['unreleased'] = {'n': len(rows), 'identical_to_released': same, 'sources': sorted({s for s, _ in rows})}
macros += [r'\newcommand{\unrelN}{%d}' % len(rows), r'\newcommand{\unrelSame}{%d}' % same]
macros.append(r'\newcommand{\crashOKrows}{%d}' % ok_checked)
(HERE / 'out').mkdir(exist_ok=True)
(HERE / 'out/crash_sensitivity.json').write_bytes((json.dumps(out, indent=1) + '\n').encode('utf-8'))
(ROOT / 'paper/tables/macros_crash.tex').write_bytes(('\n'.join(macros) + '\n').encode('utf-8'))
print(json.dumps({k: out[k] for k in ['n_crashed', 'rerun_successes', 'completed_rows_unchanged', 'official_successes_released', 'official_successes_errors_as_failures', 'unreleased']}, indent=1))
for k, v in out['contrasts'].items():
    print(k, v['a_success'], v['b_success'], v['diff_pp'], v['ci95'], 'p=%.4f' % v['mcnemar_p'], '%d/%d' % (v['wins'], v['losses']))
