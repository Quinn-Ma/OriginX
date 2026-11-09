"""Evaluation facts quoted in the protocol (official horizons, wall-clock per episode and in total), from the episode logs.
Writes paper/tables/macros_evalfacts.tex."""
from pathlib import Path
import collections, statistics, sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from analyze import load

data = load(ROOT / 'results/eval')
hz = collections.defaultdict(set); el = []; pol_mean = []
for (m, c), recs in data.items():
    v = [r['elapsed_s'] for r in recs.values()]
    el += v; pol_mean.append(statistics.mean(v) / 60)
    for (t, s), r in recs.items():
        hz[t].add(r['horizon'])
assert all(len(v) == 1 for v in hz.values())
h = sorted(next(iter(v)) for v in hz.values())
fmt = lambda x: '{:,}'.format(x).replace(',', '{,}')
macros = [r'\newcommand{\evalHorizonMin}{%s}' % fmt(h[0]), r'\newcommand{\evalHorizonMax}{%s}' % fmt(h[-1]),
          r'\newcommand{\evalHoursTotal}{%d}' % round(sum(el) / 3600),
          r'\newcommand{\evalEpisodeMinutes}{%.1f--%d}' % (min(el) / 60, round(max(el) / 60)),
          r'\newcommand{\evalPolicyMeanMinutes}{%.1f--%.1f}' % (min(pol_mean), max(pol_mean))]
(ROOT / 'paper/tables/macros_evalfacts.tex').write_text('\n'.join(macros) + '\n')
print('\n'.join(macros))
