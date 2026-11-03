#!/usr/bin/env python3
"""Reproduce the bounded-target-cache pilot from its 40 paired scenes.

Requires Python 3 and NumPy. Does not run a model or simulator.
Run: python reproduce_memory_cache.py
"""
from pathlib import Path
import csv
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent
TASKS = (0, 3, 4, 5)
MODES = ('off', 'hold1')


def read_pairs(path):
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['task', 'initial_state', 'off', 'hold1']:
            raise ValueError('Unexpected CSV columns')
        rows = [{k: int(v) for k, v in r.items()} for r in reader]
    keys = [(r['task'], r['initial_state']) for r in rows]
    expected = {(t, i) for t in TASKS for i in range(20, 30)}
    if len(rows) != 40 or len(set(keys)) != 40 or set(keys) != expected:
        raise ValueError('Expected exactly 40 unique prespecified paired scenes')
    if any(r[m] not in (0, 1) for r in rows for m in MODES):
        raise ValueError('Outcomes must be binary')
    return sorted(rows, key=lambda r: (r['task'], r['initial_state']))


def summarize(rows):
    tasks = sorted({r['task'] for r in rows})
    diffs = [np.array([r['hold1']-r['off'] for r in rows if r['task']==t], dtype=np.int8) for t in tasks]
    rng = np.random.default_rng(1)
    boot = np.zeros(100000)
    for d in diffs:
        boot += d[rng.integers(0, len(d), (100000, len(d)))].mean(axis=1) * (100.0 / len(tasks))
    delta = np.concatenate(diffs)
    wins = int(np.count_nonzero(delta == 1))
    losses = int(np.count_nonzero(delta == -1))
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses)+1)) / 2**discordant) if discordant else 1.0
    successes = {m: sum(r[m] for r in rows) for m in MODES}
    return {
        'pairs': len(rows),
        'successes': successes,
        'success_percent': {m: 100.0*successes[m]/len(rows) for m in MODES},
        'difference_pp': 100.0*(wins-losses)/len(rows),
        'ci95_pp': np.quantile(boot, [0.025, 0.975]).tolist(),
        'rescues': wins,
        'regressions': losses,
        'concordant_successes': sum(r['off']==r['hold1']==1 for r in rows),
        'concordant_failures': sum(r['off']==r['hold1']==0 for r in rows),
        'mcnemar_exact_two_sided_p': p,
    }


def analyze(rows):
    result = summarize(rows)
    result['per_task'] = {str(t): summarize([r for r in rows if r['task']==t]) for t in TASKS}
    result['nonnegative_tasks'] = sum(v['difference_pp'] >= 0 for v in result['per_task'].values())
    result['net_additional_successes'] = result['rescues'] - result['regressions']
    result['screening_go'] = result['net_additional_successes'] >= 4 and result['nonnegative_tasks'] >= 3
    return result


if __name__ == '__main__':
    result = analyze(read_pairs(ROOT/'paired_outcomes.csv'))
    expected = json.loads((ROOT/'statistics.json').read_text(encoding='utf-8'))
    if result != expected:
        raise RuntimeError('Reproduced statistics do not exactly match statistics.json')
    print(json.dumps({'status': 'PASS', 'statistics': result}, indent=2))
