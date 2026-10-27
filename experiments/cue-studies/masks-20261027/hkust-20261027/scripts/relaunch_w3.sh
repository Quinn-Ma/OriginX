#!/usr/bin/env bash
# Pre-execution worker-count amendment (user, 2026-10-27): stop the 2-worker chain that had completed
# no case, archive it, re-freeze the pilot and formal manifests with 3 workers (one per healthy GPU 5/6/7),
# record the amendment with the old/new hashes and an invariance check, dry-run, and relaunch the chain.
set -uo pipefail
cd /ephemeral/qinzhen/originx-harness-v2 || exit 1
W=work/hkust; S=/dev/shm/qinzhen/originx-envs/sim/bin/python; P=/usr/bin/python3
export PYTHONPATH=$PWD MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
TS=$(date -u +%Y%m%dT%H%M%SZ)

echo "== stop my 2-worker chain (only processes under this project path)"
pkill -f "$PWD/work/hkust/chain_hkust.sh" ; pkill -f "work/hkust/chain_hkust.sh"
pkill -f "scripts/run_id60_masks_pilot_local.py"
pkill -f "$PWD/scripts/serve_id60_hierarchical"
sleep 5; pkill -9 -f "$PWD/scripts/serve_id60_hierarchical" 2>/dev/null; pkill -9 -f "scripts/run_id60_masks_pilot_local.py" 2>/dev/null
pgrep -af "originx-harness-v2/scripts|chain_hkust" | cut -c1-120 || true
N=$(find runs/id60-masks-pilot-hkust-v1 -path '*/results/*.json' 2>/dev/null | wc -l)
$P - "$TS" "$N" <<'PYEOF'
import json, sys
rec = {'schema': 'id60-masks-pilot-interruption-v1', 'recorded_utc': sys.argv[1], 'run': 'runs/id60-masks-pilot-hkust-v1',
       'reason': 'stopped by the operator 3 minutes after launch, before any case completed, to apply the user-requested '
                 'pre-execution worker-count amendment (2 workers -> 3 workers, one per GPU 5/6/7)',
       'result_files_present': int(sys.argv[2]),
       'treatment': 'archived unchanged; engineering-only; no outcome from it is used'}
json.dump(rec, open('runs/id60-masks-pilot-hkust-v1/interruption-record-%s.json' % sys.argv[1], 'w'), indent=2)
PYEOF
mv runs/id60-masks-pilot-hkust-v1 runs/id60-masks-pilot-hkust-v1.interrupted-$TS
[ -f runs/id60-masks-pilot-hkust-v1.launcher.log ] && mv runs/id60-masks-pilot-hkust-v1.launcher.log runs/id60-masks-pilot-hkust-v1.interrupted-$TS.launcher.log
mv $W/chain.log $W/chain.2workers-stopped-$TS.log 2>/dev/null; mv $W/chain.out $W/chain.2workers-stopped-$TS.out 2>/dev/null

echo "== re-freeze with 3 workers"
$S scripts/run_id60_masks_pilot_local.py --freeze --manifest work/id60-masks-validation/pilot-manifest-hkust-w3.json --workers 3 2>&1 | tail -2
$S scripts/run_id60_masks_validation_local.py --freeze --manifest work/id60-masks-validation/manifest-hkust-w3.json --pairs 12 --workers 3 2>&1 | tail -2
$P - "$TS" <<'PYEOF'
import json, sys, hashlib
def load(p): return json.load(open(p))
def fh(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
out = {'schema': 'id60-masks-preexecution-worker-amendment-v1', 'recorded_utc': sys.argv[1],
       'instruction': 'user 2026-10-27 (relayed from the forked session): use GPU 5/6/7, as many workers as fit, break the 2-worker setting; '
                      'one ~20 GiB replica fits per GPU beside other users\' jobs (~28-30 GB free each) -> 3 workers',
       'phase': 'before any pilot or formal case completed on this site (the 2-worker pilot ran 3 min, 0 results, archived)',
       'pairs': {}}
for name, old, new in (('pilot', 'work/id60-masks-validation/pilot-manifest-v1.json', 'work/id60-masks-validation/pilot-manifest-hkust-w3.json'),
                       ('formal', 'work/id60-masks-validation/manifest.json', 'work/id60-masks-validation/manifest-hkust-w3.json')):
    a, b = load(old), load(new)
    inv = ['seed', 'pair_index', 'pair_position', 'comparison_arm', 'arm', 'task', 'model', 'max_steps', 'case_id', 'pair_key']
    same_cases = [{k: c[k] for k in inv} for c in a['cases']] == [{k: c[k] for k in inv} for c in b['cases']]
    changed_top = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    out['pairs'][name] = {'old': {'path': old, 'file_sha256': fh(old), 'manifest_sha256': a['manifest_sha256'], 'workers': a.get('workers'), 'max_local_seconds': a.get('max_local_seconds')},
                          'new': {'path': new, 'file_sha256': fh(new), 'manifest_sha256': b['manifest_sha256'], 'workers': b.get('workers'), 'max_local_seconds': b.get('max_local_seconds')},
                          'cases_invariant_fields_identical': same_cases, 'top_level_keys_changed': changed_top,
                          'execution_budget_new': b.get('execution_budget')}
    print(name, 'old', a['manifest_sha256'][:12], 'workers', a.get('workers'), '-> new', b['manifest_sha256'][:12], 'workers', b.get('workers'),
          'cases invariant:', same_cases, 'changed keys:', changed_top, 'max_s', b.get('max_local_seconds'))
json.dump(out, open('work/hkust/preexecution-worker-amendment-v1.json', 'w'), indent=2)
PYEOF

echo "== dry-run (3 workers on GPUs 5,6,7)"
rm -rf $W/pilot-dry-run-w3 $W/formal-dry-run-w3
$S scripts/run_id60_masks_pilot_local.py --dry-run --manifest work/id60-masks-validation/pilot-manifest-hkust-w3.json --output $W/pilot-dry-run-w3 --site linux_egl --site-overrides $W/site-overrides-hkust.json --workers 3 --gpus 5,6,7 --workers-per-gpu 1 2>&1 | tail -2
$S scripts/run_id60_masks_validation_local.py --dry-run --manifest work/id60-masks-validation/manifest-hkust-w3.json --output $W/formal-dry-run-w3 --site linux_egl --site-overrides $W/site-overrides-hkust.json --workers 3 --gpus 5,6,7 --workers-per-gpu 1 2>&1 | tail -2
$P -c "import json;e=json.load(open('$W/formal-dry-run-w3/local-execution.json'));print('layout',[(r['worker'],r['gpu']) for r in e['layout']],'watchdog',e.get('watchdog_seconds'))"
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader -i 5,6,7
echo "== launch"
(setsid nohup bash $W/chain_hkust.sh > $W/chain.out 2>&1 < /dev/null &)
sleep 50; cat $W/chain.log; pgrep -af 'run_id60_masks|serve_id60' | cut -c1-110 | head -12
