"""Mechanism diagnostic: do the auxiliary heads actually learn?

Uses only the nine existing training logs and the frozen window panel (no GPU, no checkpoints).

Inputs
  train/runs/<model>-<arm>/training.jsonl   one line per micro-batch (8,000 per run = 1,000 updates x 8)
  results/panel-train.json                  the 8,174-window panel the trainer sampled from (sha256 checked)
Outputs
  analysis/out/head_diagnostics.json        every number, with checks and definitions
  paper/tables/heads_table.tex              the paper table (generated from the JSON values; do not edit by hand)

Loss definitions (lerobot-training-ready-20261106/scripts/training_core.py, training_data.py)
  stage    = F.cross_entropy(stage_head(h), official subtask index at the anchor)       16 classes, nats
  location = F.smooth_l1_loss(sigmoid(location_head(h))[visible], centre[visible])      beta = 1; mean over the
             visible cameras' (x, y); centre = ((x1+x2)/2, (y1+y2)/2) of box_xyxy. All values lie in [0, 1], so the
             loss equals 0.5 * mean squared error.
  total    = action + video + 0.1 stage + 0.1 location        (switched-off terms are logged as exactly 0)
Sampler (train/train_fast.py): rng = random.Random(20261106); micro-batch m draws rng.choice(windows of task m mod 5),
  tasks sorted alphabetically, windows in panel order. Every 5 consecutive micro-batches cover each task once, so the
  means over the first 50 updates (400 draws) and the last 100 updates (800 draws) are task-balanced.
The worker reseeds torch with 20261106 + micro-batch before each forward, so within a model the three arms see the
same window AND the same flow-matching time and noise at every micro-batch: per-draw differences are paired.

References use the same task-equal weighting and loss. They are fitted on the training panel (in-sample, like the
logged losses) and reported both panel-wide and on exactly the draws they are compared with. The elapsed-time
reference is fitted leave-one-demonstration-out. Run:  python -X utf8 analysis/head_diagnostics.py
"""
from pathlib import Path
import hashlib
import json
import math
import random
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'results' / 'panel-train.json'
RUNS = ROOT / 'train' / 'runs'
OUT = ROOT / 'analysis' / 'out' / 'head_diagnostics.json'
TABLE = ROOT / 'paper' / 'tables' / 'heads_table.tex'
TRAINER_PANEL_SHA256 = 'a1ef1941093390efeeeb85e7b6f908fd2075bdb7fd13b9584cabd5cf2a201289'  # results/panel-summary.json
MODELS = ['xr1', 'awomo_i', 'awomo_v']
ARMS = ['S', 'BL', 'F']
SEED, UPDATES, ACCUM, N_STAGES = 20261106, 1000, 8, 16
FIRST = (0, 50)      # updates [0, 50)    -> micro-batches 0..399
LATE = (900, 1000)   # updates [900, 1000) -> micro-batches 7200..7999
TERMS = ['action', 'video', 'stage', 'location']
CLOCK_BINS = [64, 128, 256]
CLOCK_TABLE = 128    # bin width (frames) of the elapsed-time reference shown in the table


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def r(x, nd=6):
    return None if x is None else float(round(float(x), nd))


def mean_se(x):
    x = np.asarray(x, dtype=float)
    se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float('nan')
    return {'mean': r(x.mean(), 7), 'se': r(se, 7), 'n': int(len(x))}


def diff_stats(d):
    """Mean of paired per-draw differences, SE (draws treated as independent) and normal 95% interval."""
    s = mean_se(d)
    s['ci95'] = [r(s['mean'] - 1.96 * s['se'], 7), r(s['mean'] + 1.96 * s['se'], 7)]
    return s


def smooth_l1(d):
    a = np.abs(d)
    return np.where(a < 1.0, 0.5 * d * d, a - 0.5)


# ================================================================================================ panel and logs
def load_panel():
    raw = PANEL.read_bytes()
    rows = json.loads(raw)['windows']
    tasks = sorted({w['task'] for w in rows})
    n = len(rows)
    P = dict(rows=rows, tasks=tasks, n=n, sha256=hashlib.sha256(raw).hexdigest(),
             task=np.array([tasks.index(w['task']) for w in rows]), episode=np.array([w['episode'] for w in rows]),
             anchor=np.array([w['anchor'] for w in rows]), stage=np.array([w['stage'] for w in rows]))
    centre = np.full((n, 3, 2), np.nan)
    visible = np.zeros((n, 3), bool)
    for i, w in enumerate(rows):
        ann = w['annotation']
        assert (ann['task'], ann['episode'], ann['anchor'], ann['stage']) == (w['task'], w['episode'], w['anchor'], w['stage'])
        for c, e in enumerate(ann['frames']['0']):
            if e['visible']:
                x1, y1, x2, y2 = e['box_xyxy']
                centre[i, c] = [(x1 + x2) / 2, (y1 + y2) / 2]   # the trainer's exact definition
                visible[i, c] = True
    assert visible.any(1).all() and P['stage'].max() < N_STAGES
    P['centre'], P['visible'] = centre, visible
    P['key'] = {(w['task'], w['episode'], w['anchor']): i for i, w in enumerate(rows)}
    assert len(P['key']) == n, 'duplicate window in panel'
    P['n_task'] = np.bincount(P['task'], minlength=len(tasks))
    P['pi'] = 1.0 / (len(tasks) * P['n_task'][P['task']])   # probability that one draw is this window
    return P


def reproduce_schedule(P):
    """Exact re-draw of the trainer's sampling sequence (train_fast.execute_prefetch)."""
    rng = random.Random(SEED)
    by_task = {t: [i for i in range(P['n']) if P['rows'][i]['task'] == t] for t in P['tasks']}
    return np.array([rng.choice(by_task[P['tasks'][m % len(P['tasks'])]]) for m in range(UPDATES * ACCUM)])


def load_run(model, arm, P):
    path = RUNS / f'{model}-{arm}' / 'training.jsonl'
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == UPDATES * ACCUM, (path, len(rows))
    micro = np.array([x['microbatch'] for x in rows])
    update = np.array([x['update'] for x in rows])
    assert (micro == np.arange(len(rows))).all() and (update == micro // ACCUM).all()
    win = np.array([P['key'][(x['task'], x['episode'], x['anchor'])] for x in rows])
    terms = {k: np.array([x['terms'][k] for x in rows], dtype=float) for k in TERMS}
    loss = np.array([x['loss'] for x in rows], dtype=float)
    recon = terms['action'] + terms['video'] + 0.1 * terms['stage'] + 0.1 * terms['location']
    return dict(path=str(path.relative_to(ROOT)).replace('\\', '/'), sha256=sha256(path), win=win, update=update,
                terms=terms, stage_ok=bool((np.array([x['stage'] for x in rows]) == P['stage'][win]).all()),
                total_recon_max_rel_err=float((np.abs(loss - recon) / np.maximum(np.abs(loss), 1e-12)).max()),
                status=json.loads((path.parent / 'status.json').read_text()))


def in_updates(update, lo_hi):
    return (update >= lo_hi[0]) & (update < lo_hi[1])


def task_equal(P, per_window):
    return float(np.mean([per_window[P['task'] == t].mean() for t in range(len(P['tasks']))]))


# ================================================================================================ stage references
def stage_references(P):
    """Per-window cross-entropy of reference predictors (nats)."""
    T = len(P['tasks'])
    ref = {'uniform_16_classes': np.full(P['n'], math.log(N_STAGES)),
           'uniform_over_stages_in_task': np.zeros(P['n']), 'task_prior': np.zeros(P['n'])}
    per_task = {}
    for t, name in enumerate(P['tasks']):
        idx = np.where(P['task'] == t)[0]
        cnt = Counter(P['stage'][idx].tolist())
        prob = {s: c / len(idx) for s, c in cnt.items()}
        ref['uniform_over_stages_in_task'][idx] = math.log(len(cnt))
        ref['task_prior'][idx] = [-math.log(prob[s]) for s in P['stage'][idx]]     # mean = H(stage | task)
        per_task[name] = {'windows': int(len(idx)), 'stages_present': sorted(int(s) for s in cnt),
                          'ln_n_stages': r(math.log(len(cnt))), 'H_stage_given_task': r(-sum(p * math.log(p) for p in prob.values()))}
    # Extra calibration reference (not one of the three requested baselines): a "clock" that knows the task and the
    # elapsed time (anchor frame, binned). p(stage | task, bin) is estimated from the other 19 demonstrations of the
    # task (leave-one-demonstration-out, add-0.5 smoothing over the task's stages; empty bin -> task frequencies).
    for width in CLOCK_BINS:
        ce = np.zeros(P['n'])
        for t in range(T):
            idx = np.where(P['task'] == t)[0]
            stages = sorted(set(P['stage'][idx].tolist())); pos = {s: j for j, s in enumerate(stages)}
            for e in np.unique(P['episode'][idx]):
                test, train = idx[P['episode'][idx] == e], idx[P['episode'][idx] != e]
                train_bin = P['anchor'][train] // width
                for i in test:
                    m = train_bin == P['anchor'][i] // width
                    if not m.any():
                        m = np.ones(len(train), bool)
                    cnt = np.bincount([pos[s] for s in P['stage'][train][m]], minlength=len(stages)) + 0.5
                    ce[i] = -math.log(cnt[pos[P['stage'][i]]] / cnt.sum())
        ref[f'task_plus_elapsed_time_{width}f_LOEO'] = ce
    summary = {k: r(task_equal(P, v)) for k, v in ref.items()}
    summary['note'] = ('panel-wide, task-equal weights. task_prior = H(stage | task) = CE of the best predictor that '
                       'knows only the task; task_plus_elapsed_time_* are leave-one-demonstration-out (extra reference)')
    summary['per_task'] = per_task
    return ref, summary


# ================================================================================================ location references
def location_window_loss(P, pred):
    """Per-window location loss exactly as the trainer computes it: mean smooth-L1 over the visible (x, y)."""
    v = np.repeat(P['visible'][:, :, None], 2, axis=2)
    l = np.where(v, smooth_l1(np.nan_to_num(pred - P['centre'])), 0.0)
    return l.sum((1, 2)) / v.sum((1, 2))


def group_mean_centres(P, groups, weights, fit_mask=None):
    """Weighted mean centre per (group, camera, coordinate), fitted on fit_mask, predicted for every window."""
    pred = np.full((P['n'], 3, 2), np.nan)
    fit = np.ones(P['n'], bool) if fit_mask is None else fit_mask
    for g in np.unique(groups):
        gi = groups == g
        for c in range(3):
            m = gi & fit & P['visible'][:, c]
            if m.any():
                pred[gi, c] = (weights[m][:, None] * P['centre'][m, c]).sum(0) / weights[m].sum()
    return pred


def location_references(P):
    nvis = P['visible'].sum(1).astype(float)
    w_opt = P['pi'] / nvis       # exact minimiser of the task-equal expected loss (the loss is quadratic here)
    task_stage = P['task'] * 100 + P['stage']
    one = np.zeros(P['n'], int)
    preds = {'image_centre': np.full((P['n'], 3, 2), 0.5),
             'global_mean_centre': group_mean_centres(P, one, w_opt),
             'per_task_mean_centre': group_mean_centres(P, P['task'], w_opt),
             'per_task_stage_mean_centre': group_mean_centres(P, task_stage, w_opt)}
    for name, groups in [('per_task_mean_centre', P['task']), ('per_task_stage_mean_centre', task_stage)]:
        pred = np.full((P['n'], 3, 2), np.nan)                 # leave-one-demonstration-out variants
        for t in range(len(P['tasks'])):
            for e in np.unique(P['episode'][P['task'] == t]):
                test = (P['task'] == t) & (P['episode'] == e)
                fit = (P['task'] == t) & (P['episode'] != e)
                gp = group_mean_centres(P, groups, w_opt, fit)
                fb = group_mean_centres(P, P['task'], w_opt, fit)   # (task, stage) never visible elsewhere -> task mean
                pred[test] = np.where(np.isnan(gp[test]), fb[test], gp[test])
        preds[name + '_LOEO'] = pred
    ref = {k: location_window_loss(P, np.nan_to_num(v, nan=0.5)) for k, v in preds.items()}
    summary = {k: r(task_equal(P, v), 8) for k, v in ref.items()}
    plain = {k: group_mean_centres(P, g, P['pi']) for k, g in
             [('global_mean_centre', one), ('per_task_mean_centre', P['task']), ('per_task_stage_mean_centre', task_stage)]}
    summary['plain_mean_variant'] = {k: r(task_equal(P, location_window_loss(P, np.nan_to_num(v, nan=0.5))), 8) for k, v in plain.items()}
    summary['note'] = ('panel-wide, task-equal weights; mean centres weighted by 1/(n_task * visible cameras), the exact '
                       'loss minimiser; plain_mean_variant weights every visible camera-window equally (within task)')
    summary['per_task'] = {P['tasks'][t]: {k: r(v[P['task'] == t].mean(), 8) for k, v in ref.items()} for t in range(len(P['tasks']))}
    return ref, summary


# ================================================================================================ novelty
def novelty(P, sched):
    T = len(P['tasks']); M = len(sched)
    seen, seen_keys = set(), set()
    novel, adjacent = np.zeros(M, bool), np.zeros(M, bool)
    for m, i in enumerate(sched):
        t, e, a = P['rows'][i]['task'], P['rows'][i]['episode'], P['rows'][i]['anchor']
        novel[m] = i not in seen
        adjacent[m] = (t, e, a - 16) in seen_keys or (t, e, a + 16) in seen_keys
        seen.add(i); seen_keys.add((t, e, a))
    update = np.arange(M) // ACCUM
    late, first = in_updates(update, LATE), in_updates(update, FIRST)
    task_of_draw = np.arange(M) % T
    # draw j (0-based) of task t is a first visit with probability (1 - 1/n_t)^j under the sampler
    expected = np.mean([(1 - 1 / P['n_task'][m % T]) ** (m // T) for m in np.where(late)[0]])
    demo = np.array([P['task'][i] * 1000 + P['episode'][i] for i in sched])
    out = {
        'late_novel_fraction': r(novel[late].mean(), 4), 'late_novel_count': int(novel[late].sum()), 'late_draws': int(late.sum()),
        'late_novel_fraction_expected_under_sampler': r(expected, 4),
        'first50_novel_fraction': r(novel[first].mean(), 4),
        'late_novel_with_adjacent_window_drawn_before': r(adjacent[late & novel].mean(), 4),
        'late_draws_from_demonstrations_drawn_before': r(np.mean(np.isin(demo[late], demo[:LATE[0] * ACCUM])), 4),
        'distinct_windows_drawn': int(len(set(sched.tolist()))), 'panel_windows': int(P['n']),
        'distinct_fraction_of_panel': r(len(set(sched.tolist())) / P['n'], 4), 'draws_per_panel_window': r(M / P['n'], 4),
        'per_task': {P['tasks'][t]: {'panel_windows': int(P['n_task'][t]), 'draws': int((task_of_draw == t).sum()),
                                     'late_novel_fraction': r(novel[late & (task_of_draw == t)].mean(), 4)} for t in range(T)},
        'note': 'novel = the window had not been drawn at any earlier micro-batch; all nine runs share this schedule',
    }
    return out, novel, task_of_draw


def novel_minus_repeated(values, novel, mask, task_of_draw, T):
    """Task-stratified (equal task weights) mean difference novel - repeated among the masked draws, with SE."""
    ds, vs = [], []
    for t in range(T):
        a = values[mask & novel & (task_of_draw == t)]; b = values[mask & ~novel & (task_of_draw == t)]
        ds.append(a.mean() - b.mean()); vs.append(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    d, se = float(np.mean(ds)), math.sqrt(sum(vs)) / T
    return {'mean': r(d, 7), 'se': r(se, 7), 'ci95': [r(d - 1.96 * se, 7), r(d + 1.96 * se, 7)]}


def head_vs_refs(head, refs, win, mask):
    return {k: {'reference_mean': r(v[win][mask].mean(), 7), 'head_minus_reference': diff_stats(head[mask] - v[win][mask])}
            for k, v in refs.items()}


# ================================================================================================ main
def main():
    P = load_panel()
    sched = reproduce_schedule(P)
    runs = {(m, a): load_run(m, a, P) for m in MODELS for a in ARMS}
    T = len(P['tasks'])
    update = runs[MODELS[0], 'S']['update']
    first, late, u0 = in_updates(update, FIRST), in_updates(update, LATE), update == 0

    checks = {
        'panel_sha256': P['sha256'], 'panel_matches_trainer_panel_hash': P['sha256'] == TRAINER_PANEL_SHA256,
        'schedule_reproduced_from_seed_matches_every_log': all((run['win'] == sched).all() for run in runs.values()),
        'logged_stage_equals_panel_stage': all(run['stage_ok'] for run in runs.values()),
        'all_runs_completed_1000_updates': all(run['status']['status'] == 'completed' and run['status']['optimizer_updates'] == UPDATES
                                               for run in runs.values()),
        'switched_off_terms_are_zero': all((a != 'S' or not run['terms']['location'].any()) and (a != 'BL' or not run['terms']['stage'].any())
                                           and (m != 'xr1' or not run['terms']['video'].any()) for (m, a), run in runs.items()),
        'logged_total_equals_action_video_plus_0.1_aux_max_rel_err': r(max(run['total_recon_max_rel_err'] for run in runs.values()), 9),
        'first50_draws_per_task': np.bincount((np.arange(len(sched)) % T)[first]).tolist(),
        'late_draws_per_task': np.bincount((np.arange(len(sched)) % T)[late]).tolist(),
    }
    # Update 0 runs at the initial parameters. BL and F see identical inputs (box drawn), so identical location losses
    # <=> identical location-head initialisation; S and F differ only by the drawn box, so their stage CE should be close.
    init = {}
    for m in MODELS:
        tm = {a: runs[m, a]['terms'] for a in ARMS}
        init[m] = {'action_BL_vs_F_max_abs_diff': r(np.abs(tm['BL']['action'][u0] - tm['F']['action'][u0]).max(), 9),
                   'location_BL_vs_F_max_abs_diff': r(np.abs(tm['BL']['location'][u0] - tm['F']['location'][u0]).max(), 6),
                   'stage_S_vs_F_max_abs_diff': r(np.abs(tm['S']['stage'][u0] - tm['F']['stage'][u0]).max(), 4)}
        init[m]['head_init_identical_across_arms'] = bool(init[m]['location_BL_vs_F_max_abs_diff'] < 1e-6 and init[m]['stage_S_vs_F_max_abs_diff'] < 0.2)
    checks['update0_head_initialisation'] = init

    nov, novel, task_of_draw = novelty(P, sched)
    sref, ssum = stage_references(P)
    lref, lsum = location_references(P)

    # ---------------- stage heads (S and F)
    stage_runs = {}
    for m in MODELS:
        stage_runs[m] = {}
        for a in ['S', 'F']:
            ce, win = runs[m, a]['terms']['stage'], runs[m, a]['win']
            stage_runs[m][a] = {
                'update0': r(ce[u0].mean(), 5), 'first50': mean_se(ce[first]), 'last100': mean_se(ce[late]),
                'blocks_of_100_updates': [r(ce[in_updates(update, (b, b + 100))].mean(), 5) for b in range(0, UPDATES, 100)],
                'last100_vs_references_same_draws': head_vs_refs(ce, sref, win, late),
                'last100_first_visits': mean_se(ce[late & novel]),
                'last100_first_visits_vs_references_same_draws': head_vs_refs(ce, sref, win, late & novel),
                'last100_repeat_visits': mean_se(ce[late & ~novel]),
                'last100_first_minus_repeat_visits_task_stratified': novel_minus_repeated(ce, novel, late, task_of_draw, T),
                'per_task_last100': {P['tasks'][t]: {'head': r(ce[late & (task_of_draw == t)].mean(), 5),
                                                     'task_prior_same_draws': r(sref['task_prior'][win][late & (task_of_draw == t)].mean(), 5)}
                                     for t in range(T)},
            }
        stage_runs[m]['F_minus_S_last100_paired'] = diff_stats(runs[m, 'F']['terms']['stage'][late] - runs[m, 'S']['terms']['stage'][late])

    # ---------------- location heads (BL and F)
    loc_runs = {}
    for m in MODELS:
        loc_runs[m] = {}
        for a in ['BL', 'F']:
            lo, win = runs[m, a]['terms']['location'], runs[m, a]['win']
            loc_runs[m][a] = {
                'update0': r(lo[u0].mean(), 7), 'first50': mean_se(lo[first]), 'last100': mean_se(lo[late]),
                'blocks_of_100_updates': [r(lo[in_updates(update, (b, b + 100))].mean(), 7) for b in range(0, UPDATES, 100)],
                'first50_references_same_draws': {k: r(v[win][first].mean(), 7) for k, v in lref.items()},
                'last100_vs_references_same_draws': head_vs_refs(lo, lref, win, late),
                'last100_first_visits': mean_se(lo[late & novel]),
                'last100_first_visits_vs_references_same_draws': head_vs_refs(lo, lref, win, late & novel),
                'last100_repeat_visits': mean_se(lo[late & ~novel]),
                'last100_first_minus_repeat_visits_task_stratified': novel_minus_repeated(lo, novel, late, task_of_draw, T),
            }
        loc_runs[m]['F_minus_BL_last100_paired'] = diff_stats(runs[m, 'F']['terms']['location'][late] - runs[m, 'BL']['terms']['location'][late])
    # Interpretation aid: loss = 0.5 * mean squared error of normalised coordinates, so 256 * sqrt(2 * loss) is the
    # (window-weighted) RMS centre error in pixels of a 256-px camera image.
    px = lambda loss: r(256 * math.sqrt(2 * loss), 1)
    late_refs = loc_runs['xr1']['BL']['last100_vs_references_same_draws']
    rms_px = {'heads_last100': {f'{m}-{a}': px(loc_runs[m][a]['last100']['mean']) for m in MODELS for a in ['BL', 'F']},
              'references_on_late_draws': {k: px(v['reference_mean']) for k, v in late_refs.items()},
              'note': '256 * sqrt(2 * mean loss); approximate RMS error of the predicted box centre in pixels'}

    # ---------------- action and future-latent ("video") losses, paired across arms
    av = {}
    for m in MODELS:
        av[m] = {}
        for k in ['action'] + (['video'] if m != 'xr1' else []):
            per_arm = {}
            for a in ARMS:
                x = runs[m, a]['terms'][k]
                per_arm[a] = {'update0': r(x[u0].mean(), 7), 'first50': mean_se(x[first]), 'last100': mean_se(x[late]),
                              'first50_median': r(np.median(x[first]), 7), 'last100_median': r(np.median(x[late]), 7),
                              'blocks_of_100_updates': [r(x[in_updates(update, (b, b + 100))].mean(), 7) for b in range(0, UPDATES, 100)],
                              'last100_over_first50_minus_1_unpaired': r(x[late].mean() / x[first].mean() - 1, 4),
                              'last100_first_minus_repeat_visits_task_stratified': novel_minus_repeated(x, novel, late, task_of_draw, T)}
            pairs = {}
            for a, b in [('BL', 'S'), ('F', 'S'), ('F', 'BL')]:
                xa, xb = runs[m, a]['terms'][k], runs[m, b]['terms'][k]
                for name, msk in [('last100', late), ('first50', first), ('update0', u0)]:
                    d = diff_stats(xa[msk] - xb[msk])
                    d['relative'] = r(xa[msk].mean() / xb[msk].mean() - 1, 5)
                    d['relative_se'] = r(d['se'] / xb[msk].mean(), 5)
                    pairs.setdefault(f'{a}-{b}', {})[name] = d
            av[m][k] = {'per_arm': per_arm, 'paired_differences': pairs}

    out = {
        'script': 'analysis/head_diagnostics.py',
        'inputs': {'panel': {'path': 'results/panel-train.json', 'sha256': P['sha256'], 'windows': P['n'],
                             'per_task_windows': dict(zip(P['tasks'], P['n_task'].tolist()))},
                   'logs': {f'{m}-{a}': {'path': run['path'], 'sha256': run['sha256']} for (m, a), run in runs.items()}},
        'definitions': {
            'first50': 'updates 0-49 = micro-batches 0-399 (80 draws per task); update 0 runs at the initial parameters',
            'last100': 'updates 900-999 = micro-batches 7200-7999 (160 draws per task)',
            'weighting': 'tasks equally weighted (the sampler draws task m mod 5), windows uniform within a task',
            'units': 'stage CE in nats; location smooth-L1 and action/future-latent losses as logged (unitless)',
            'references': 'fitted on the training panel; *_same_draws = mean of the reference loss on exactly the draws '
                          'the head is compared with (the location loss varies strongly between windows)',
            'se': 'sd/sqrt(n) over draws, treating draws as independent; ci95 = mean +/- 1.96 se',
            'paired': 'per-draw differences on identical windows; across arms also identical flow-matching time and noise',
            'first_visit': 'a late draw whose window had not been drawn at any earlier micro-batch of the run',
        },
        'checks': checks,
        'novelty': nov,
        'stage': {'references_panel': ssum, 'runs': stage_runs},
        'location': {'references_panel': lsum, 'runs': loc_runs, 'rms_centre_error_px_equivalent': rms_px},
        'action_and_future_latent': av,
    }
    out['table_values'] = table_values(out)
    out['paragraph_values'] = paragraph_values(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    TABLE.write_text(render_table(out['table_values']))
    summarize(out)


# ================================================================================================ paragraph numbers
PARAGRAPH = ROOT / 'paper' / 'sections' / 'results_heads.tex'


def paragraph_values(o):
    """Every number quoted in paper/sections/results_heads.tex, formatted as printed, with source and support."""
    S, L, A = o['stage']['runs'], o['location']['runs'], o['action_and_future_latent']
    tv = o['table_values']
    st = [(m, a) for m in MODELS for a in ['S', 'F']]
    lt = [(m, a) for m in MODELS for a in ['BL', 'F']]
    rng = lambda xs, f: f'{f(min(xs))}--{f(max(xs))}'
    first = [S[m][a]['first50']['mean'] for m, a in st]
    late = [S[m][a]['last100']['mean'] for m, a in st]
    loc_late = [1e3 * L[m][a]['last100']['mean'] for m, a in lt]
    vs = lambda run, key, ref: run[key][ref]['head_minus_reference']
    fbl = [100 * A[m]['action']['paired_differences']['F-BL']['last100']['relative'] for m in MODELS]
    fs = [100 * A[m]['action']['paired_differences']['F-S']['last100']['relative'] for m in MODELS]
    awv_first = [-vs(S['awomo_v'][a], 'last100_first_visits_vs_references_same_draws', 'task_prior')['mean'] for a in ['S', 'F']]
    pv = {
        'stage_first50_range': {'text': rng(first, lambda x: f'{x:.2f}'), 'source': 'stage.runs.*.{S,F}.first50.mean (min--max of 6)'},
        'stage_last100_range': {'text': rng(late, lambda x: f'{x:.2f}'), 'source': 'stage.runs.*.{S,F}.last100.mean (min--max of 6)'},
        'task_prior': {'text': f"{tv['stage_references_late_draws']['task_prior']:.2f}",
                       'panel_wide': f"{o['stage']['references_panel']['task_prior']:.2f}",
                       'source': 'table_values.stage_references_late_draws.task_prior (= H(stage|task) on the late draws)'},
        'task_plus_elapsed_time': {'text': f"{tv['stage_references_late_draws'][CLK]:.2f}",
                                   'source': f'table_values.stage_references_late_draws.{CLK}'},
        'support_all_stage_heads_below_task_prior_late': all(vs(S[m][a], 'last100_vs_references_same_draws', 'task_prior')['ci95'][1] < 0 for m, a in st),
        'support_all_stage_heads_above_elapsed_time_late': all(vs(S[m][a], 'last100_vs_references_same_draws', CLK)['ci95'][0] > 0 for m, a in st),
        'location_last100_range_x1e3_table_only': rng(loc_late, lambda x: f'{x:.1f}'),
        'location_gain_over_task_centre_xr1_awv_x1e3': {
            'text': f"{max(-1e3 * vs(L[m][a], 'last100_vs_references_same_draws', 'per_task_mean_centre')['mean'] for m in ['xr1', 'awomo_v'] for a in ['BL', 'F']):.1f}",
            'values': {f'{m}-{a}': r(-1e3 * vs(L[m][a], 'last100_vs_references_same_draws', 'per_task_mean_centre')['mean'], 3) for m, a in lt},
            'source': 'location.runs.<model>.<BL|F>.last100_vs_references_same_draws.per_task_mean_centre.head_minus_reference.mean x -1e3 (max over XR-1, Awomo-V)'},
        'support_xr1_awv_location_below_task_centre_late': all(
            vs(L[m][a], 'last100_vs_references_same_draws', 'per_task_mean_centre')['ci95'][1] < 0 for m in ['xr1', 'awomo_v'] for a in ['BL', 'F']),
        'support_awi_location_above_task_centre_late': all(
            vs(L['awomo_i'][a], 'last100_vs_references_same_draws', 'per_task_mean_centre')['ci95'][0] > 0 for a in ['BL', 'F']),
        'task_stage_centre_x1e3_table_only': {
            'in_sample': f"{tv['location_references_late_draws_x1e3']['per_task_stage_mean_centre']:.1f}",
            'held_out_demonstrations': f"{tv['location_references_late_draws_x1e3']['per_task_stage_mean_centre_LOEO']:.1f}"},
        # paragraph says "none beats a constant per task and stage": must hold for the in-sample AND the held-out fit
        'support_no_location_head_beats_task_stage_centre_late_in_sample': all(
            vs(L[m][a], 'last100_vs_references_same_draws', 'per_task_stage_mean_centre')['ci95'][0] > 0 for m, a in lt),
        'support_no_location_head_beats_task_stage_centre_late_held_out': all(   # no CI entirely below zero
            vs(L[m][a], 'last100_vs_references_same_draws', 'per_task_stage_mean_centre_LOEO')['ci95'][1] > 0 for m, a in lt),
        'support_xr1_awv_below_task_centre_late_held_out_fit_too': all(
            vs(L[m][a], 'last100_vs_references_same_draws', 'per_task_mean_centre_LOEO')['ci95'][1] < 0 for m in ['xr1', 'awomo_v'] for a in ['BL', 'F']),
        'support_no_location_head_reliably_beats_task_centre_first_visits_held_out_fit_too': all(
            vs(L[m][a], 'last100_first_visits_vs_references_same_draws', 'per_task_mean_centre_LOEO')['ci95'][1] > 0 for m, a in lt),
        'awomo_i_location_minus_global_centre_late_x1e3': {
            'values': [r(1e3 * vs(L['awomo_i'][a], 'last100_vs_references_same_draws', 'global_mean_centre')['mean'], 3) for a in ['BL', 'F']],
            'global_centre_x1e3': r(tv['location_references_late_draws_x1e3']['global_mean_centre'], 3),
            'source': 'location.runs.awomo_i.{BL,F}.last100_vs_references_same_draws.global_mean_centre.head_minus_reference.mean x 1e3',
            'text_claim': 'stays close to a single global constant'},
        'late_first_visit_pct': {'text': f"{round(100 * o['novelty']['late_novel_fraction'])}", 'source': 'novelty.late_novel_fraction'},
        'support_all_stage_heads_below_task_prior_first_visits': all(
            vs(S[m][a], 'last100_first_visits_vs_references_same_draws', 'task_prior')['ci95'][1] < 0 for m, a in st),
        'awomo_v_first_visit_margin_below_task_prior': {
            'text': rng(awv_first, lambda x: f'{x:.2f}'),
            'source': 'stage.runs.awomo_v.{S,F}.last100_first_visits_vs_references_same_draws.task_prior.head_minus_reference.mean (negated)'},
        'support_no_location_head_reliably_beats_task_centre_first_visits': all(
            vs(L[m][a], 'last100_first_visits_vs_references_same_draws', 'per_task_mean_centre')['ci95'][1] > 0 for m, a in lt),
        'action_F_minus_BL_pct_range': {'text': rng(fbl, lambda x: f'+{x:.1f}').replace('--+', ' to +'),
                                        'source': 'action_and_future_latent.*.action.paired_differences.F-BL.last100.relative x 100'},
        'action_F_minus_S_pct_range': {'text': rng(fs, lambda x: f'+{x:.1f}').replace('--+', ' to +'),
                                       'source': 'action_and_future_latent.*.action.paired_differences.F-S.last100.relative x 100'},
        'support_F_raises_action_loss_every_model': all(
            A[m]['action']['paired_differences'][p]['last100']['ci95'][0] > 0 for m in MODELS for p in ['F-BL', 'F-S']),
    }
    if PARAGRAPH.exists():   # consistency check: every formatted number must appear in the paragraph source
        text = PARAGRAPH.read_text(encoding='utf8').replace('$', '').replace('\\%', '%')
        pv['found_in_paragraph'] = {k: (v['text'] in text) for k, v in pv.items() if isinstance(v, dict) and 'text' in v}
    return pv


# ================================================================================================ table
CLK = f'task_plus_elapsed_time_{CLOCK_TABLE}f_LOEO'


def table_values(o):
    """Exactly the rounded values printed in paper/tables/heads_table.tex, with their JSON sources."""
    S, L, A = o['stage']['runs'], o['location']['runs'], o['action_and_future_latent']
    v = {'stage': {}, 'location': {}, 'action_rel_pct': {}, 'future_latent_rel_pct': {}}
    for m in MODELS:
        v['stage'][m] = {row: [S[m][a][row]['mean'] for a in ['S', 'F']] for row in ['first50', 'last100']}
        v['stage'][m]['first_visits_minus_task_prior'] = [
            S[m][a]['last100_first_visits_vs_references_same_draws']['task_prior']['head_minus_reference']['mean'] for a in ['S', 'F']]
        v['location'][m] = {row: [L[m][a][row]['mean'] * 1e3 for a in ['BL', 'F']] for row in ['first50', 'last100']}
        v['location'][m]['first_visits_minus_task_centre'] = [
            1e3 * L[m][a]['last100_first_visits_vs_references_same_draws']['per_task_mean_centre']['head_minus_reference']['mean']
            for a in ['BL', 'F']]
        v['action_rel_pct'][m] = [100 * A[m]['action']['paired_differences'][p]['last100']['relative'] for p in ['BL-S', 'F-S']]
        v['future_latent_rel_pct'][m] = (None if m == 'xr1' else
                                         [100 * A[m]['video']['paired_differences'][p]['last100']['relative'] for p in ['BL-S', 'F-S']])
    late_ref = S['xr1']['S']['last100_vs_references_same_draws']          # identical draws in every run
    late_lref = L['xr1']['BL']['last100_vs_references_same_draws']
    v['stage_references_late_draws'] = {k: late_ref[k]['reference_mean'] for k in
                                        ['uniform_16_classes', 'uniform_over_stages_in_task', 'task_prior', CLK]}
    v['location_references_late_draws_x1e3'] = {k: late_lref[k]['reference_mean'] * 1e3 for k in
                                                ['global_mean_centre', 'per_task_mean_centre', 'per_task_stage_mean_centre',
                                                 'per_task_stage_mean_centre_LOEO']}
    v['late_first_visit_fraction'] = o['novelty']['late_novel_fraction']
    v['late_first_visit_count'] = o['novelty']['late_novel_count']
    # Upper bounds (rounded up at display precision) of the SEs of the paired rows, quoted in the caption.
    up = lambda x, nd: math.ceil(x * 10 ** nd - 1e-9) / 10 ** nd
    v['paired_se_bounds'] = {
        'stage_first_visits_nats': up(max(S[m][a]['last100_first_visits_vs_references_same_draws']['task_prior']['head_minus_reference']['se']
                                          for m in MODELS for a in ['S', 'F']), 2),
        'location_first_visits_x1e3': up(max(1e3 * L[m][a]['last100_first_visits_vs_references_same_draws']['per_task_mean_centre']['head_minus_reference']['se']
                                             for m in MODELS for a in ['BL', 'F']), 1),
        'action_rel_pct': up(max(100 * A[m]['action']['paired_differences'][p]['last100']['relative_se'] for m in MODELS for p in ['BL-S', 'F-S']), 1),
        'future_latent_rel_pct': up(max(100 * A[m]['video']['paired_differences'][p]['last100']['relative_se']
                                        for m in MODELS if m != 'xr1' for p in ['BL-S', 'F-S']), 1),
    }
    v['sources'] = {'stage': 'stage.runs.<model>.<S|F>.{first50,last100}.mean; first_visits_minus_task_prior = '
                             'stage.runs.<model>.<S|F>.last100_first_visits_vs_references_same_draws.task_prior.head_minus_reference.mean',
                    'location': 'location.runs.<model>.<BL|F>.{first50,last100}.mean x 1e3; first_visits_minus_task_centre = location.runs.'
                                '<model>.<BL|F>.last100_first_visits_vs_references_same_draws.per_task_mean_centre.head_minus_reference.mean x 1e3',
                    'action_rel_pct': 'action_and_future_latent.<model>.action.paired_differences.{BL-S,F-S}.last100.relative x 100',
                    'future_latent_rel_pct': 'action_and_future_latent.<model>.video.paired_differences.{BL-S,F-S}.last100.relative x 100',
                    'stage_references_late_draws': 'stage.runs.xr1.S.last100_vs_references_same_draws.<ref>.reference_mean',
                    'location_references_late_draws_x1e3': 'location.runs.xr1.BL.last100_vs_references_same_draws.<ref>.reference_mean x 1e3'}
    return v


def render_table(v):
    f2 = lambda x: f'{x:.2f}'
    f1 = lambda x: f'{x:.1f}'

    def signed(x, nd):
        s = f'{abs(x):.{nd}f}'
        return s if float(s) == 0 else ('$+$' if x > 0 else '$-$') + s
    pair = lambda xs, f: r'\,/\,'.join(f(x) for x in xs)
    row = lambda label, cells: f'{label} & ' + ' & '.join(cells) + r' \\'
    # reference values are shared by the three models (same panel, same draws): say so in the spanning cell
    span = lambda label, text: f'{label} & \\multicolumn{{3}}{{c}}{{\\emph{{all models:}} {text}}} \\\\'
    sr, lr, se = v['stage_references_late_draws'], v['location_references_late_draws_x1e3'], v['paired_se_bounds']
    pct = round(100 * v['late_first_visit_fraction'])
    lines = [
        '% Generated by analysis/head_diagnostics.py from analysis/out/head_diagnostics.json (table_values); do not edit by hand.',
        r'\begin{table}[t]',
        r'\centering',
        r'\footnotesize',
        r'\setlength{\tabcolsep}{2.2pt}',
        r'\begin{tabular}{@{}lccc@{}}',
        r'\toprule',
        r' & \xr{} & \awi{} & \awv{} \\',
        r'\midrule',
        r'\multicolumn{4}{@{}l}{\emph{Stage cross-entropy (nats), S\,/\,F}} \\',
        row(r'First 50 updates$^\ddagger$', [pair(v['stage'][m]['first50'], f2) for m in MODELS]),
        row('Last 100 updates', [pair(v['stage'][m]['last100'], f2) for m in MODELS]),
        span(r"Uniform: 16\,/\,task's stages", f2(sr['uniform_16_classes']) + r'\,/\,' + f2(sr['uniform_over_stages_in_task'])),
        span(r'Prior: task\,/\,task\,+\,time', f2(sr['task_prior']) + r'\,/\,' + f2(sr[CLK]) + r'$^\dagger$'),
        row(r'First visits vs.\ task prior', [pair(v['stage'][m]['first_visits_minus_task_prior'], lambda x: signed(x, 2)) for m in MODELS]),
        r'\midrule',
        r'\multicolumn{4}{@{}l}{\emph{Location smooth-L1 ($\times10^{-3}$), BL\,/\,F}} \\',
        row(r'First 50 updates$^\ddagger$', [pair(v['location'][m]['first50'], f1) for m in MODELS]),
        row('Last 100 updates', [pair(v['location'][m]['last100'], f1) for m in MODELS]),
        span(r'Mean centre: global\,/\,task', f1(lr['global_mean_centre']) + r'\,/\,' + f1(lr['per_task_mean_centre'])),
        span(r'Mean centre: task\,+\,stage', f1(lr['per_task_stage_mean_centre']) + r'\,/\,'
             + f1(lr['per_task_stage_mean_centre_LOEO']) + r'$^\dagger$'),
        row(r'First visits vs.\ task centre', [pair(v['location'][m]['first_visits_minus_task_centre'], lambda x: signed(x, 1)) for m in MODELS]),
        r'\midrule',
        r'\multicolumn{4}{@{}l}{\emph{Last 100 updates, change vs.\ S (\%), BL\,/\,F}} \\',
        row('Action loss', [pair(v['action_rel_pct'][m], lambda x: signed(x, 1)) for m in MODELS]),
        row('Future-latent loss', ['--' if v['future_latent_rel_pct'][m] is None else
                                   pair(v['future_latent_rel_pct'][m], lambda x: signed(x, 1)) for m in MODELS]),
        r'\bottomrule',
        r'\end{tabular}',
        (r'\caption{\textbf{Do the auxiliary heads learn?} Means of the logged training losses (1{,}000 updates). '
         r'References are fitted on the training panel and scored on the same draws. First visits: head minus reference '
         r'on the ' + f'{v["late_first_visit_count"]}' + r' late draws (' + f'{pct}' + r'\%) whose window had not been '
         r'drawn before. Bottom: paired change vs.\ S. Paired standard errors: stage ${\le}' + f'{se["stage_first_visits_nats"]:.2f}'
         + r'$, location ${\le}' + f'{se["location_first_visits_x1e3"]:.1f}' + r'$, action ${\le}' + f'{se["action_rel_pct"]:.1f}'
         + r'$\%, future latent ${\le}' + f'{se["future_latent_rel_pct"]:.1f}' + r'$\%. '
         r'$^\dagger$Fitted on held-out demonstrations; elapsed time in ' + f'{CLOCK_TABLE}' + r'-frame bins. '
         r'$^\ddagger$\xr{}'"'"r's heads were initialised differently in each arm.}'),
        r'\label{tab:heads}',
        r'\end{table}',
        '',
    ]
    return '\n'.join(lines)


def summarize(o):
    c = o['checks']
    print('checks:', {k: v for k, v in c.items() if k != 'update0_head_initialisation'})
    print('head init identical:', {m: v['head_init_identical_across_arms'] for m, v in c['update0_head_initialisation'].items()})
    n = o['novelty']
    print('late first-visit fraction', n['late_novel_fraction'], 'expected', n['late_novel_fraction_expected_under_sampler'],
          'adjacent drawn before', n['late_novel_with_adjacent_window_drawn_before'])
    print('stage refs (panel):', {k: v for k, v in o['stage']['references_panel'].items() if k not in ('per_task', 'note')})
    for m in MODELS:
        for a in ['S', 'F']:
            s = o['stage']['runs'][m][a]
            lr, nr = s['last100_vs_references_same_draws'], s['last100_first_visits_vs_references_same_draws']
            print(f'  stage {m:7s} {a:2s} first {s["first50"]["mean"]:.3f} late {s["last100"]["mean"]:.3f}+-{s["last100"]["se"]:.3f} '
                  f'| late-prior {lr["task_prior"]["head_minus_reference"]["mean"]:+.3f}+-{lr["task_prior"]["head_minus_reference"]["se"]:.3f} '
                  f'late-clock {lr[CLK]["head_minus_reference"]["mean"]:+.3f} '
                  f'| first-visit {s["last100_first_visits"]["mean"]:.3f} (prior {nr["task_prior"]["reference_mean"]:.3f}, '
                  f'diff {nr["task_prior"]["head_minus_reference"]["mean"]:+.3f}+-{nr["task_prior"]["head_minus_reference"]["se"]:.3f}) '
                  f'| first-repeat {s["last100_first_minus_repeat_visits_task_stratified"]["mean"]:+.3f}')
    print('location refs (panel):', {k: v for k, v in o['location']['references_panel'].items() if k not in ('per_task', 'note')})
    for m in MODELS:
        for a in ['BL', 'F']:
            s = o['location']['runs'][m][a]
            lr, nr = s['last100_vs_references_same_draws'], s['last100_first_visits_vs_references_same_draws']
            fmt = lambda d: ' '.join(f'{k[:10]}:{d[k]["reference_mean"]*1e3:.2f}({d[k]["head_minus_reference"]["mean"]*1e3:+.2f}+-{d[k]["head_minus_reference"]["se"]*1e3:.2f})'
                                     for k in ['global_mean_centre', 'per_task_mean_centre', 'per_task_stage_mean_centre'])
            print(f'  loc {m:7s} {a:2s} first {s["first50"]["mean"]*1e3:.2f} late {s["last100"]["mean"]*1e3:.2f} | {fmt(lr)}')
            print(f'      first-visit {s["last100_first_visits"]["mean"]*1e3:.2f} | {fmt(nr)}')
    for m in MODELS:
        for k, v in o['action_and_future_latent'][m].items():
            print(f'  {k:6s} {m:7s} ' + ' '.join(f'{a}:{v["per_arm"][a]["first50"]["mean"]*1e3:.2f}->{v["per_arm"][a]["last100"]["mean"]*1e3:.2f}' for a in ARMS)
                  + ' | ' + ' '.join(f'{p}:{d["last100"]["relative"]*100:+.2f}%+-{d["last100"]["relative_se"]*100:.2f} (first50 {d["first50"]["relative"]*100:+.2f}%)'
                                     for p, d in v['paired_differences'].items()))
    print('paragraph values:', json.dumps({k: (v['text'] if isinstance(v, dict) and 'text' in v else v)
                                           for k, v in o['paragraph_values'].items() if k != 'found_in_paragraph'}))
    print('found in paragraph:', o['paragraph_values'].get('found_in_paragraph'))
    print('wrote', OUT, 'and', TABLE)


if __name__ == '__main__':
    main()
