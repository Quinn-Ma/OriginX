"""Secondary feedback study (condition FB), analysed against the ordinary-input runs of the same checkpoints.

Protocol (fixed before the run): the PaTH-VLA (F) checkpoint of each model plays the same 200 seeded episodes as the primary
evaluation, but at every policy query (every 16 environment steps) the location head's prediction from the PREVIOUS query is
drawn as a fixed 32x32-px, 2-px red box centred on the predicted point on the current frame of all three cameras (no box at
the first query; instruction unchanged; stage head unused).  Training instead drew tight mask boxes from labels, only in
cameras where the target is visible.

Steps
  (a) sync    pull <model>-FB-<shard>/{results.jsonl,meta.json} from hkust-cluster into results/eval_fb/ with scp
              (skipped with a message if the cluster is unreachable; local copies are then used as they are);
  (b) paired  on every (task, seed) with both an FB and an F result: successes, FB-F and FB-official paired differences
              (task-stratified paired bootstrap, 10,000 draws; exact McNemar; code reused from analyze.py), per-task
              successes, n pairs, status complete (200 pairs) / provisional (0 < n < 200) / pending (n = 0);
  (c) traces  fraction of queries with a location prediction and with a drawn box, within-episode spread of the predicted
              centres, query-to-query displacement, boxes clipped by the frame border;
  (d) writes  analysis/out/feedback.json, paper/tables/macros_feedback.tex, paper/tables/feedback_table.tex.

The discarded Awomo-0.5V run (73 episodes whose location head read the last denoising-step feature instead of the
observation-only warm-up feature used in training; kept on the cluster under eval/discarded-awomo_v-FB-wrong-feature/) is
never pulled, read or analysed: its shard names are excluded below and any shard found under a remote discarded-* directory
is excluded as well.

Re-run (from any directory):
  python -X utf8 D:/codex/sbl-study-20261108/analysis/feedback.py            # sync + analyse
  python -X utf8 D:/codex/sbl-study-20261108/analysis/feedback.py --no-sync  # analyse local copies only
"""
from pathlib import Path
import argparse, collections, datetime, json, math, os, re, shutil, subprocess, sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
from analyze import load, paired_contrast, exact_mcnemar, SEEN, UNSEEN, TASKS  # noqa: E402  (the pre-registered primary analysis code)

CLUSTER = 'hkust-cluster'
REMOTE_EVAL = '/dev/shm/qinzhen/sbl/eval'
FB_ROOT = STUDY / 'results' / 'eval_fb'
PRIMARY_ROOT = STUDY / 'results' / 'eval'
PANEL_JSON = STUDY / 'results' / 'panel-train.json'  # training-window labels; only used for a reference spread of labelled centres
OUT_JSON = STUDY / 'analysis' / 'out' / 'feedback.json'
MACROS_TEX = STUDY / 'paper' / 'tables' / 'macros_feedback.tex'
TABLE_TEX = STUDY / 'paper' / 'tables' / 'feedback_table.tex'

MODELS = [('xr1', 'XR', 'XR-1'), ('awomo_i', 'AI', 'Awomo-0.5I'), ('awomo_v', 'AV', 'Awomo-0.5V')]
N_DESIGN = len(TASKS) * 20  # 10 tasks x 20 seeds = 200 paired episodes per model
# Standard shard names of the evaluation chain (q = quarter, h = half). Anything else (smoke-*, discarded-*) is ignored.
SHARD_RE = re.compile(r'^(xr1|awomo_i|awomo_v)-FB-([qh]\d+)$')
# Shard names of the discarded wrong-feature Awomo-0.5V run (now under eval/discarded-awomo_v-FB-wrong-feature/ on the
# cluster). Never analysed, even if a stale local copy with this name exists. The restarted run uses q0..q3.
DISCARDED_SHARDS = {'awomo_v-FB-h0', 'awomo_v-FB-h1'}
DISCARDED_AV_EPISODES = 73  # wc -l of the two discarded results.jsonl files (38 + 35), checked 2026-11-09 03:50 UTC; not analysed
FRAME_PX = 255.0  # feedback_worker.paint(): centre pixel = round(x * (w - 1)) on 256 x 256 frames
BOX_HALF = 16     # feedback_worker.HALF: the box edges are drawn at centre +-16 px (nominally 32 x 32 px)


# ----------------------------------------------------------------------------------------------------------------- (a) sync
def _ssh(cmd, timeout=60):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', CLUSTER, cmd],
                          capture_output=True, text=True, timeout=timeout)


def sync():
    """Mirror results.jsonl and meta.json of every FB shard from the cluster. Never raises: on failure, local copies are kept."""
    info = {'attempted': True, 'reachable': False, 'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            'remote_shards': {}, 'pulled': [], 'not_pulled': [], 'ignored_remote_dirs': [], 'discarded_remote_shards': []}
    listing = ('cd %s && for d in *-FB-*; do [ -d "$d" ] || continue; if [ -f "$d/results.jsonl" ]; then echo "SHARD $d $(wc -l < "$d/results.jsonl")"; '
               'else echo "SHARD $d -"; fi; done; for d in discarded-*/*-FB-*; do [ -d "$d" ] && echo "DISCARDED $(basename "$d")"; done; true') % REMOTE_EVAL
    try:
        r = _ssh(listing)
    except (OSError, subprocess.TimeoutExpired) as e:
        print('[sync] skipped, cluster not reachable (%s); using local copies' % type(e).__name__); info['error'] = repr(e); return info
    if r.returncode != 0:
        print('[sync] skipped, ssh exit %d: %s; using local copies' % (r.returncode, r.stderr.strip()[:200])); info['error'] = r.stderr.strip()[:500]; return info
    info['reachable'] = True
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == 'DISCARDED':
            info['discarded_remote_shards'].append(parts[1])
        elif len(parts) == 3 and parts[0] == 'SHARD':
            name, n = parts[1], parts[2]
            if not SHARD_RE.match(name) or name in DISCARDED_SHARDS:
                info['ignored_remote_dirs'].append(name); continue
            info['remote_shards'][name] = None if n == '-' else int(n)
    for name in sorted(info['remote_shards']):
        if name in info['discarded_remote_shards']:
            print('[sync] WARNING: %s also exists under a discarded-* directory on the cluster; not pulled' % name)
            info['not_pulled'].append(name); continue
        if info['remote_shards'][name] is None:
            print('[sync] %s: no results.jsonl on the cluster yet' % name); info['not_pulled'].append(name); continue
        local = FB_ROOT / name; part = local / '.part'
        shutil.rmtree(part, ignore_errors=True); part.mkdir(parents=True)
        src = ['%s:%s/%s/%s' % (CLUSTER, REMOTE_EVAL, name, f) for f in ('results.jsonl', 'meta.json')]
        try:  # relative destination + cwd: avoids scp reading the drive letter of a Windows path as a host name
            s = subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15'] + src + ['.'], cwd=str(part),
                               capture_output=True, text=True, timeout=300)
            ok = s.returncode == 0 and (part / 'results.jsonl').exists()
        except (OSError, subprocess.TimeoutExpired) as e:
            ok = False; s = None; print('[sync] %s: scp failed (%s)' % (name, type(e).__name__))
        if ok:
            for f in ('results.jsonl', 'meta.json'):
                if (part / f).exists():
                    os.replace(part / f, local / f)
            info['pulled'].append(name)
        else:
            info['not_pulled'].append(name)
            if s is not None:
                print('[sync] %s: scp exit %d %s; keeping the local copy' % (name, s.returncode, s.stderr.strip()[:200]))
        shutil.rmtree(part, ignore_errors=True)
    print('[sync] pulled %d shard(s): %s' % (len(info['pulled']), ', '.join('%s (%s lines)' % (n, info['remote_shards'][n]) for n in info['pulled'])))
    return info


# ---------------------------------------------------------------------------------------------------------------- loading
def load_fb(root, exclude):
    """FB episodes per model, keyed by (task, seed); mirrors analyze.load (duplicates are an error) but tolerates a partially
    written last line of a shard that is still running."""
    data = collections.defaultdict(dict); shards = collections.defaultdict(list); meta = {}; notes = []
    if not Path(root).exists():
        return data, shards, meta, notes
    for d in sorted(Path(root).iterdir()):
        m = SHARD_RE.match(d.name)
        if not d.is_dir() or not m:
            continue
        if d.name in DISCARDED_SHARDS or d.name in exclude:
            notes.append('excluded discarded shard %s' % d.name); continue
        model = m.group(1); f = d / 'results.jsonl'
        if not f.exists():
            continue
        if (d / 'meta.json').exists():
            meta[d.name] = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
        lines = f.read_text(encoding='utf-8').splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    notes.append('skipped a partially written last line in %s' % d.name); continue
                raise
            if r.get('model') != model or r.get('condition') != 'FB':
                raise ValueError('%s holds a record of %s/%s' % (d.name, r.get('model'), r.get('condition')))
            key = (r['task'], r['seed'])
            if key in data[model]:
                raise ValueError('duplicate FB episode %s %s (shards overlap; resolve before analysing)' % (model, key))
            data[model][key] = r
        shards[model].append(d.name)
    return data, shards, meta, notes


# ------------------------------------------------------------------------------------------------------------- statistics
def _q(xs, p):
    """Linear-interpolated percentile (numpy's default) of a non-empty list."""
    xs = sorted(xs); k = (len(xs) - 1) * p; lo = math.floor(k); hi = math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _summ(xs):
    return {'median': round(_q(xs, .5), 1), 'q25': round(_q(xs, .25), 1), 'q75': round(_q(xs, .75), 1), 'n_episodes': len(xs)} if xs else None


def trace_stats(recs):
    """Location-trace statistics over FB episodes. Spread = RMS distance of an episode's predicted centres from their own
    mean (pixels of the 256-px frame), averaged over the three cameras; jump = median distance between the predictions of
    consecutive queries (i.e. how far the drawn box lags the current prediction)."""
    n_q = n_pred = n_boxed = n_mismatch = n_pts = n_clip = 0
    spread, jump = [], []; spread_cam = [[], [], []]
    for r in recs:
        tr = r.get('location_trace') or []
        q = int(r.get('policy_queries') or len(tr))
        n_mismatch += len(tr) != q
        n_q += q
        ok = [x is not None and len(x) == 3 for x in tr]
        n_pred += sum(ok)
        n_boxed += sum(1 for k in range(1, q) if k - 1 < len(tr) and ok[k - 1])  # query k shows the prediction of query k-1
        pts = [x for x, o in zip(tr, ok) if o]
        for p in pts:
            for c in range(3):
                x, y = p[c][0] * FRAME_PX, p[c][1] * FRAME_PX; n_pts += 1
                n_clip += min(x, y, FRAME_PX - x, FRAME_PX - y) < BOX_HALF
        if len(pts) >= 2:
            per_cam = []
            for c in range(3):
                xs = [p[c][0] * FRAME_PX for p in pts]; ys = [p[c][1] * FRAME_PX for p in pts]
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                v = math.sqrt(sum((x - mx) ** 2 + (y - my) ** 2 for x, y in zip(xs, ys)) / len(xs))
                per_cam.append(v); spread_cam[c].append(v)
            spread.append(sum(per_cam) / 3)
            js = [sum(math.hypot((tr[k][c][0] - tr[k - 1][c][0]) * FRAME_PX, (tr[k][c][1] - tr[k - 1][c][1]) * FRAME_PX) for c in range(3)) / 3
                  for k in range(1, len(tr)) if ok[k] and ok[k - 1]]
            if js:
                jump.append(_q(js, .5))
    return {'n_episodes': len(recs), 'n_queries': n_q, 'frac_queries_with_prediction': round(n_pred / n_q, 4) if n_q else None,
            'frac_queries_with_box_drawn': round(n_boxed / n_q, 4) if n_q else None, 'episodes_trace_length_mismatch': n_mismatch,
            'spread_px': _summ(spread), 'spread_px_per_camera': {c: _summ(v) for c, v in zip(['left', 'right', 'wrist'], spread_cam)},
            'jump_px': _summ(jump), 'frac_boxes_clipped_by_border': round(n_clip / n_pts, 4) if n_pts else None}


def label_reference():
    """Reference for the trace statistics: how much the LABELLED target centre moves within a training demonstration
    (same definitions as trace_stats; windows are 16 frames apart like policy queries; labels exist only where visible)."""
    if not PANEL_JSON.exists():
        return None
    eps = collections.defaultdict(dict)
    for w in json.loads(PANEL_JSON.read_text(encoding='utf-8'))['windows']:
        fr = w['annotation']['frames']; fr = fr.get(str(w['anchor'])) or next(iter(fr.values()))
        eps[(w['task'], w['episode'])][w['anchor']] = [f['location_xy'] if f.get('visible') else None for f in fr]
    spread, jump = [], []
    for an in eps.values():
        a = sorted(an); per_cam = []
        for c in range(3):
            pts = [an[x][c] for x in a if an[x][c]]
            if len(pts) >= 2:
                xs = [p[0] * FRAME_PX for p in pts]; ys = [p[1] * FRAME_PX for p in pts]; mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                per_cam.append(math.sqrt(sum((x - mx) ** 2 + (y - my) ** 2 for x, y in zip(xs, ys)) / len(xs)))
        if per_cam:
            spread.append(sum(per_cam) / len(per_cam))
        js = []
        for x0, x1 in zip(a, a[1:]):
            if x1 - x0 == 16:
                d = [math.hypot((an[x1][c][0] - an[x0][c][0]) * FRAME_PX, (an[x1][c][1] - an[x0][c][1]) * FRAME_PX) for c in range(3) if an[x0][c] and an[x1][c]]
                if d:
                    js.append(sum(d) / len(d))
        if js:
            jump.append(_q(js, .5))
    return {'source': str(PANEL_JSON.relative_to(STUDY)), 'n_demonstrations': len(eps), 'n_windows': sum(len(v) for v in eps.values()),
            'spread_px': _summ(spread), 'jump_px': _summ(jump),
            'note': 'labelled centres exist only in cameras where the target is visible; demonstrations are shorter and more direct than rollouts'}


def _slim(c):
    if c is None:
        return None
    out = {k: c[k] for k in ['n_pairs', 'a_success', 'b_success', 'diff_pp', 'ci95', 'wins', 'losses', 'mcnemar_p']}
    out['mcnemar_p_exact'] = exact_mcnemar(c['wins'], c['losses'])  # unrounded, for 3-decimal formatting without double rounding
    return out


def analyse_model(model, fb, prim, primary_hashes, n_boot):
    ok = {k: r for k, r in fb.items() if r.get('status') == 'ok' and r.get('success') is not None}
    F, O = prim[(model, 'F')], prim[(model, 'original')]
    keys = sorted(set(ok) & set(F) & set(O))
    res = {'n_fb_episodes': len(fb), 'n_fb_errors': len(fb) - len(ok), 'n_pairs': len(keys),
           'fb_without_primary_partner': sorted('%s/%d' % k for k in set(ok) - set(F))}
    res['status'] = 'complete' if len(keys) == N_DESIGN else ('provisional' if keys else 'pending')
    # Scene check. For 9 of the 200 seeds (5 SearingMeat, 4 StirVegetables) the 12 primary policies themselves start from one
    # of two first-frame variants (2 of the 9 also differ in the initial base pose), so a hash difference to F is expected there.
    # Such pairs are kept, as in the primary analysis; a sensitivity contrast without them is reported. An FB first frame that
    # matches none of the variants the primary runs produced is flagged loudly.
    unknown = sorted('%s/%d' % k for k in keys if ok[k].get('first_frame_sha256') not in primary_hashes.get(k, set()))
    res['first_frame_unknown_variant'] = unknown
    if unknown:
        print('[check] WARNING: %d FB episode(s) of %s start from a first frame no primary run produced: %s' % (len(unknown), model, unknown[:5]))
    mism = [k for k in keys if ok[k].get('first_frame_sha256') != F[k].get('first_frame_sha256')]
    res['first_frame_differs_from_F'] = sorted('%s/%d' % k for k in mism)
    res['first_frame_differs_from_official'] = sorted('%s/%d' % k for k in keys if ok[k].get('first_frame_sha256') != O[k].get('first_frame_sha256'))
    sub = {k: ok[k] for k in keys}
    res['instruction_or_horizon_differs_from_F'] = sorted('%s/%d' % k for k in keys if (sub[k].get('instruction'), sub[k].get('horizon')) != (F[k].get('instruction'), F[k].get('horizon')))
    if res['instruction_or_horizon_differs_from_F']:
        print('[check] WARNING: %s FB episodes differ from F in instruction or horizon: %s' % (model, res['instruction_or_horizon_differs_from_F'][:5]))
    res['tasks_covered'] = {t: sum(1 for k in keys if k[0] == t) for t in TASKS if any(k[0] == t for k in keys)}
    res['n_seen_pairs'] = sum(1 for k in keys if k[0] in SEEN); res['n_unseen_pairs'] = len(keys) - res['n_seen_pairs']
    res['fb_success'] = sum(1 for k in keys if sub[k]['success'])
    res['f_success'] = sum(1 for k in keys if F[k]['success']); res['official_success'] = sum(1 for k in keys if O[k]['success'])
    res['f_success_all200'] = sum(1 for r in F.values() if r['success']); res['official_success_all200'] = sum(1 for r in O.values() if r['success'])
    res['per_task'] = {t: {'n': v, 'FB': sum(1 for k in keys if k[0] == t and sub[k]['success']),
                           'F': sum(1 for k in keys if k[0] == t and F[k]['success']),
                           'official': sum(1 for k in keys if k[0] == t and O[k]['success'])} for t, v in res['tasks_covered'].items()}
    res['fb_minus_F'] = _slim(paired_contrast(sub, F, n_boot=n_boot))
    res['fb_minus_official'] = _slim(paired_contrast(sub, O, n_boot=n_boot))
    same = {k: v for k, v in sub.items() if k not in set(mism)}  # sensitivity: only pairs whose first frames are identical
    res['fb_minus_F_identical_first_frame'] = _slim(paired_contrast(same, F, n_boot=n_boot))
    res['split'] = {}
    for split, ts in [('seen', SEEN), ('unseen', UNSEEN)]:
        s = {k: v for k, v in sub.items() if k[0] in ts}
        res['split'][split] = {'fb_minus_F': _slim(paired_contrast(s, F, n_boot=n_boot)), 'fb_minus_official': _slim(paired_contrast(s, O, n_boot=n_boot))}
    res['steps'] = {'median_steps_success_FB': _q([sub[k]['steps'] for k in keys if sub[k]['success']], .5) if res['fb_success'] else None,
                    'early_terminations_FB': sum(1 for k in keys if not sub[k]['success'] and sub[k]['steps'] < sub[k]['horizon'])}
    res['trace'] = trace_stats(list(sub.values()))
    return res


# ---------------------------------------------------------------------------------------------------------------- outputs
def _num(x):
    x = round(x, 1) + 0.0  # no negative zero
    return '%+.1f' % x


def _p(p):
    return '<0.001' if p < 0.001 else '%.3f' % p


def _diff(c):
    """Exact paired difference in pp (analyze.paired_contrast rounds it to 2 decimals; avoid rounding twice)."""
    return 100.0 * (c['a_success'] - c['b_success']) / c['n_pairs']


def write_macros(res):
    L = ['%% Generated by analysis/feedback.py on %s UTC -- do not edit by hand; re-run the script instead.' % res['generated_utc'],
         '% Secondary feedback study (FB). Load once (guarded). Numbers carry their sign; use them in text or math mode.',
         '% \\fbStatusXX is empty when the model has all 200 pairs and " (provisional, $n{=}..$)" otherwise (leading space',
         '% included: write \\awi{}\\fbStatusAI{}). Pending models (n = 0) render "[pending]" in every number macro.',
         '\\ifdefined\\fbMacrosLoaded\\else', '\\def\\fbMacrosLoaded{}']
    def m(name, value):
        assert re.fullmatch(r'[A-Za-z]+', name), name
        L.append('\\newcommand{\\%s}{%s}' % (name, value))
    pend = '[pending]'
    all_complete = True
    for model, mk, label in MODELS:
        r = res['models'][model]; n = r['n_pairs']; c = r['fb_minus_F']; o = r['fb_minus_official']
        m('fb%sn' % mk, '%d' % n)
        m('fb%sstatus' % mk, r['status'])
        if r['status'] == 'complete':
            m('fbStatus%s' % mk, '')
        elif r['status'] == 'provisional':
            m('fbStatus%s' % mk, ' (provisional, $n{=}%d$)' % n); all_complete = False
        else:
            m('fbStatus%s' % mk, ' (provisional, $n{=}0$)'); all_complete = False
        if c is None:
            for nm in ['succ', 'Fsucc', 'offsucc', 'rate', 'Frate', 'diff', 'ci', 'p', 'pfull', 'wl', 'cizero',
                       'offdiff', 'offci', 'offp', 'offpfull', 'offwl', 'offcizero', 'spread', 'jump', 'predfrac']:
                m('fb%s%s' % (mk, nm), pend)
            continue
        m('fb%ssucc' % mk, '%d/%d' % (r['fb_success'], n)); m('fb%sFsucc' % mk, '%d/%d' % (r['f_success'], n))
        m('fb%soffsucc' % mk, '%d/%d' % (r['official_success'], n))
        m('fb%srate' % mk, '%.1f' % (100 * r['fb_success'] / n)); m('fb%sFrate' % mk, '%.1f' % (100 * r['f_success'] / n))
        for pre, cc in [('', c), ('off', o)]:
            m('fb%s%sdiff' % (mk, pre), '\\ensuremath{%s}' % _num(_diff(cc)))
            m('fb%s%sci' % (mk, pre), '\\ensuremath{[%s, %s]}' % (_num(cc['ci95'][0]), _num(cc['ci95'][1])))
            m('fb%s%sp' % (mk, pre), _p(cc['mcnemar_p_exact']))
            m('fb%s%spfull' % (mk, pre), '\\ensuremath{p%s}' % ('<\\nobreak 0.001' if cc['mcnemar_p_exact'] < 0.001 else '=\\nobreak %.3f' % cc['mcnemar_p_exact']))
            m('fb%s%swl' % (mk, pre), '%d/%d' % (cc['wins'], cc['losses']))
            m('fb%s%scizero' % (mk, pre), 'covers zero' if cc['ci95'][0] <= 0 <= cc['ci95'][1] else 'excludes zero')
        # FB episodes whose first frame is the other rendering variant than in the F run, and FB-F without them
        m('fb%sframediffn' % mk, '%d' % len(r.get('first_frame_differs_from_F') or []))
        e = r.get('fb_minus_F_identical_first_frame')
        if e:
            m('fb%sidentn' % mk, '%d' % e['n_pairs'])
            m('fb%sidentdiff' % mk, '\\ensuremath{%s}' % _num(_diff(e)))
            m('fb%sidentp' % mk, _p(e['mcnemar_p_exact']))
        t = r['trace']
        m('fb%sspread' % mk, '%.0f' % t['spread_px']['median'] if t['spread_px'] else pend)
        m('fb%sjump' % mk, '%.0f' % t['jump_px']['median'] if t['jump_px'] else pend)
        m('fb%spredfrac' % mk, '%.0f' % (100 * t['frac_queries_with_prediction']) if t['frac_queries_with_prediction'] is not None else pend)
    ref = res.get('label_reference') or {}
    m('fbLabelSpread', '%.0f' % ref['spread_px']['median'] if ref.get('spread_px') else pend)
    m('fbLabelJump', '%.0f' % ref['jump_px']['median'] if ref.get('jump_px') else pend)
    m('fbDiscardedAV', '%d' % DISCARDED_AV_EPISODES)
    m('fbAllComplete', 'yes' if all_complete else 'no')
    L.append('\\fi')
    MACROS_TEX.write_text('\n'.join(L) + '\n', encoding='utf-8')
    return len([x for x in L if x.startswith('\\newcommand')])


def write_table(res):
    prov = [label for model, mk, label in MODELS if res['models'][model]['status'] != 'complete']
    # The tabular is set at its natural size; only if wide provisional intervals make it wider than the column is it scaled
    # down to the column width (factor close to 1). Complete results fit without scaling.
    # The interval is set in two right-aligned columns ("[lo," and "hi]") so that brackets and commas line up.
    L = ['%% Generated by analysis/feedback.py on %s UTC -- do not edit by hand.' % res['generated_utc'],
         '\\begin{table}[t]\\centering\\footnotesize', '\\setlength{\\tabcolsep}{3.5pt}', '\\setbox0=\\hbox{\\begin{tabular}{@{}lrrrrrr@{\\,}rr@{}}', '\\toprule',
         'Model & $n$ & Off. & F & FB & $\\Delta$ & \\multicolumn{2}{c}{95\\% CI} & $p$ \\\\', '\\midrule']
    for model, mk, label in MODELS:
        r = res['models'][model]; c = r['fb_minus_F']; mark = '$^\\dagger$' if r['status'] != 'complete' else ''
        if c is None:
            L.append('%s%s & 0 & -- & -- & -- & -- & \\multicolumn{2}{c}{pending} & -- \\\\' % (label, mark)); continue
        L.append('%s%s & %d & %d & %d & %d & $%s$ & $[%s,$ & $%s]$ & %s \\\\' % (
            label, mark, r['n_pairs'], r['official_success'], r['f_success'], r['fb_success'], _num(_diff(c)),
            _num(c['ci95'][0]), _num(c['ci95'][1]), _p(c['mcnemar_p_exact'])))
    cap = ('\\caption{\\textbf{Feeding the policy\'s own target back (secondary).} Successes of the official checkpoint (Off.), of the '
           '\\pathvla{} checkpoint with ordinary inputs (F) and of the same checkpoint with its location-head prediction from the '
           'previous query drawn as a fixed $32{\\times}32$-pixel box on the current frame of every camera (FB; none at the first query, '
           'instruction unchanged), on the same $n$ seeded episodes. $\\Delta$: FB$-$F success difference (pp) with task-stratified paired-bootstrap 95\\% interval and exact McNemar $p$ '
           '(uncorrected).')
    spreads = [(label, res['models'][model]['trace']['spread_px']) for model, mk, label in MODELS if res['models'][model]['n_pairs']]
    ref = (res.get('label_reference') or {}).get('spread_px')
    if spreads and ref and all(s for _, s in spreads):
        items = ['%.0f\\,px (%s)' % (s['median'], lab) for lab, s in spreads]
        items = items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]
        cap += (' Within an episode the predicted centre has a median RMS spread of %s, against %.0f\\,px for the labelled target '
                'within a training demonstration.' % (items, ref['median']))
    if prov:
        cap += ' $^\\dagger$Provisional: evaluation still running; the episodes so far are not a balanced sample of the ten tasks.'
    L += ['\\bottomrule', '\\end{tabular}}%', '\\ifdim\\wd0>\\columnwidth\\resizebox{\\columnwidth}{!}{\\box0}\\else\\box0\\fi',
          cap + '}', '\\label{tab:feedback}', '\\end{table}']
    TABLE_TEX.write_text('\n'.join(L) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--no-sync', action='store_true', help='do not contact the cluster; analyse the local copies')
    ap.add_argument('--n-boot', type=int, default=10000)
    a = ap.parse_args()
    info = sync() if not a.no_sync else {'attempted': False}
    exclude = set(info.get('discarded_remote_shards', []))
    fb, shards, meta, notes = load_fb(FB_ROOT, exclude)
    for n in notes:
        print('[load] ' + n)
    prim = load(PRIMARY_ROOT)
    primary_hashes = collections.defaultdict(set)  # (task, seed) -> first-frame hashes seen across the 12 primary policies
    for recs in prim.values():
        for k, r in recs.items():
            primary_hashes[k].add(r.get('first_frame_sha256'))
    n_var = sum(1 for v in primary_hashes.values() if len(v) > 1)
    res = {'generated_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M'), 'sync': info, 'load_notes': notes,
           'protocol': 'F checkpoint; at every policy query (every 16 env steps) the location-head prediction of the previous query is drawn '
                       'as a fixed 32x32-px 2-px red box centred on the predicted point on the current frame of all three cameras; no box at the '
                       'first query; original instruction; stage head unused. Training drew tight label mask boxes only where the target was visible.',
           'comparison': 'paired by (task, seed) with the primary ordinary-input F and official runs (results/eval); task-stratified paired '
                         'bootstrap (%d draws, seed 20261108) and exact two-sided McNemar from analyze.py; p uncorrected' % a.n_boot,
           'discarded': {'awomo_v': {'episodes': DISCARDED_AV_EPISODES, 'where': REMOTE_EVAL + '/discarded-awomo_v-FB-wrong-feature/',
                                     'reason': 'location head read the last denoising-step feature instead of the observation-only warm-up feature used in training',
                                     'analysed': False}},
           'primary_seeds_with_two_first_frame_variants': n_var,
           'models': {}}
    for model, mk, label in MODELS:
        F_sha = json.loads((PRIMARY_ROOT / ('%s-F-q0' % model) / 'meta.json').read_text(encoding='utf-8'))['adapter_sha256']
        shas = {meta[s].get('adapter_sha256') for s in shards.get(model, []) if s in meta}
        if shas and shas != {F_sha}:
            raise SystemExit('FB shards of %s used adapter(s) %s, not the F adapter %s' % (model, shas, F_sha))
        r = analyse_model(model, fb.get(model, {}), prim, primary_hashes, a.n_boot)
        r.update(label=label, shards=shards.get(model, []), adapter_sha256=F_sha, adapter_matches_F=(shas == {F_sha}) if shas else None)
        res['models'][model] = r
    res['label_reference'] = label_reference()
    # The prose in paper/sections/results_feedback.tex states facts about XR-1 (complete) in words. Re-check them on every run.
    x = res['models']['xr1']; checks = []
    if x['fb_minus_F'] is not None:
        F_off = paired_contrast(prim[('xr1', 'F')], prim[('xr1', 'original')], n_boot=a.n_boot)
        checks = [('XR-1 FB run is complete (200 pairs)', x['status'] == 'complete'),
                  ('XR-1 FB-F interval lies below zero ("erases F\'s gain")', x['fb_minus_F']['ci95'][1] < 0),
                  ('XR-1 F-official is positive ("F\'s gain over the official checkpoint")', F_off['diff_pp'] > 0),
                  ('XR-1 FB-official interval covers zero ("erases", not "reverses")', x['fb_minus_official']['ci95'][0] <= 0 <= x['fb_minus_official']['ci95'][1])]
    res['text_checks'] = {k: bool(v) for k, v in checks}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding='utf-8')
    n_mac = write_macros(res); write_table(res)
    for k, v in checks:
        if not v:
            print('[text check] FAILED: %s -- revise the XR-1 sentence in paper/sections/results_feedback.tex' % k)
    print('\nFB (own location prediction fed back as a 32-px box) vs ordinary-input F and official, paired by (task, seed)')
    for model, mk, label in MODELS:
        r = res['models'][model]; c = r['fb_minus_F']; o = r['fb_minus_official']; t = r['trace']
        if c is None:
            print('  %-11s %-11s n=0' % (label, r['status'])); continue
        print('  %-11s %-11s n=%3d  FB %3d  F %3d  off %3d | FB-F %+5.1f pp [%+5.1f, %+5.1f] W/L %d/%d p=%.4f | FB-off %+5.1f pp [%+5.1f, %+5.1f] W/L %d/%d p=%.4f' % (
            label, r['status'], r['n_pairs'], r['fb_success'], r['f_success'], r['official_success'], c['diff_pp'], c['ci95'][0], c['ci95'][1],
            c['wins'], c['losses'], c['mcnemar_p'], o['diff_pp'], o['ci95'][0], o['ci95'][1], o['wins'], o['losses'], o['mcnemar_p']))
        print('  %-23s tasks %s' % ('', ', '.join('%s %d/%d' % (k, v['FB'], v['n']) for k, v in r['per_task'].items())))
        e = r['fb_minus_F_identical_first_frame'] or {'diff_pp': float('nan'), 'ci95': [float('nan')] * 2, 'n_pairs': 0}
        print('  %-23s %d pair(s) start from the other first-frame variant than F; FB-F on identical first frames: %+.1f pp [%+.1f, %+.1f] (n=%d)' % (
            '', len(r['first_frame_differs_from_F']), e['diff_pp'], e['ci95'][0], e['ci95'][1], e['n_pairs']))
        sp = t['spread_px'] or {'median': '-', 'q25': '-', 'q75': '-'}
        print('  %-23s trace: %.1f%% queries with prediction, %.1f%% with a box; spread median %s px (IQR %s-%s), jump median %s px, clipped %.1f%%' % (
            '', 100 * t['frac_queries_with_prediction'], 100 * t['frac_queries_with_box_drawn'], sp['median'], sp['q25'], sp['q75'],
            t['jump_px']['median'] if t['jump_px'] else '-', 100 * (t['frac_boxes_clipped_by_border'] or 0)))
    ref = res['label_reference']
    if ref:
        print('  reference: labelled target centre within a training demonstration: spread median %s px, jump median %s px (%d demos)' % (
            ref['spread_px']['median'], ref['jump_px']['median'], ref['n_demonstrations']))
    print('  text checks: %s' % ('all passed' if all(res['text_checks'].values()) else res['text_checks']))
    print('\nwrote %s\n      %s (%d macros)\n      %s' % (OUT_JSON, MACROS_TEX, n_mac, TABLE_TEX))


if __name__ == '__main__':
    main()
