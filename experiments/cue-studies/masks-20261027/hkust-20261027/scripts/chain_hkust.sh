#!/usr/bin/env bash
# HKUST cluster chain for the mask study (adapted from cloud/chain_after_boot.sh):
# prerequisites -> pilot (6 cases) -> pilot engineering analysis -> gate decision
# -> formal run (12 pairs) -> prospective analysis (best effort; the local WSL
# analysis is authoritative) -> results bundle.  Every step appends a marker line
# to work/hkust/chain.log.  Two workers as frozen, one per healthy GPU (5 and 7):
# GPU 2/3 carry "GPU Recovery Action: Reset" and corrupt or segfault EGL depth readback.
set -uo pipefail
cd /ephemeral/qinzhen/originx-harness-v2 || exit 1
W=work/hkust
LOG=$PWD/$W/chain.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
E=/dev/shm/qinzhen/originx-envs
S=$E/sim/bin/python
P=/usr/bin/python3
GPUS="5 6 7"
WORKERS=3            # user 2026-10-27: fill GPU 5/6/7 with as many workers as fit; ~20 GiB per replica, ~28-30 GB free per GPU -> one each
MIN_FREE_MIB=21500   # one policy+planner replica measured ~20 GiB on the H800 smoke
PILOT=id60-masks-pilot-hkust-w3-v1
FORMAL=id60-masks-validation-hkust-w3-v1
PILOT_MANIFEST=work/id60-masks-validation/pilot-manifest-hkust-w3.json
FORMAL_MANIFEST=work/id60-masks-validation/manifest-hkust-w3.json
SITE="--site linux_egl --site-overrides $W/site-overrides-hkust.json --workers $WORKERS --gpus ${GPUS// /,} --workers-per-gpu 1"
gpu_headroom() {  # every execution GPU must have MIN_FREE_MIB free; prints the free list
  local ok=0 free list=""
  for g in $GPUS; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g | tr -d ' '); list="$list gpu$g=${free}MiB"; [ "$free" -ge $MIN_FREE_MIB ] || ok=1; done
  echo "$list"; return $ok
}
export PYTHONPATH=$PWD MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

log "CHAIN_START pid=$$ host=$(hostname)"

# ---- prerequisites: envs, weights, pinned renderer, T3 result, GPU headroom ----
$E/policy/bin/python -c "import torch, flash_attn, transformers; print('POLICY_OK', torch.__version__, flash_attn.__version__)" >> "$LOG" 2>&1 || { log "PREREQ_FAIL policy env"; exit 2; }
$E/planner/bin/python -c "import torch, transformers; print('PLANNER_OK', transformers.__version__)" >> "$LOG" 2>&1 || { log "PREREQ_FAIL planner env"; exit 2; }
[ "$($P -c 'import json;d=json.load(open("/dev/shm/qinzhen/originx-models/download_receipt.json"));print(d["status"], d["verified_files"])')" = "complete 30" ] || { log "PREREQ_FAIL models not verified"; exit 2; }
R=$(MUJOCO_GL=egl $S $W/native_egl_probe.py 2>/dev/null | $P -c 'import json,sys;d=json.load(sys.stdin);print([x for x in d["devices"] if x.get("stage")=="ready"][0]["renderer"])')
PIN=$($P -c "import json;print(json.load(open('$W/site-overrides-hkust.json'))['gl_renderer'])")
[ "$R" = "$PIN" ] || { log "PREREQ_FAIL renderer '$R' != pinned '$PIN'"; exit 2; }
[ -f work/id60-targeted-parser/analyze_targeted.py ] || { log "PREREQ_FAIL analyzer gate module missing"; exit 2; }
[ -f $W/t3/exactness-checks-hkust-h100-v1.json ] || { log "PREREQ_FAIL T3 result missing"; exit 2; }
[ -f $PILOT_MANIFEST ] && [ -f $FORMAL_MANIFEST ] || { log "PREREQ_FAIL re-frozen manifests missing"; exit 2; }
pgrep -f check_mask_exactness.py >/dev/null && { log "PREREQ_FAIL T3 still running"; exit 2; }
FREE=$(gpu_headroom) || { log "PREREQ_FAIL GPU headroom below ${MIN_FREE_MIB} MiB:$FREE"; exit 2; }
log "PREREQ_OK renderer=$R$FREE"

# ---- a formal directory from an interrupted attempt needs a human decision ----
if [ -e runs/$FORMAL ] && [ ! -f runs/$FORMAL-analysis/validation-results-server-v1.json ]; then
  log "FORMAL_EXISTS results=$(find runs/$FORMAL -path '*/results/*.json' | wc -l) manual decision required (resume driver for zero-step pairs)"
  exit 4
fi

# ---- pilot ------------------------------------------------------------------
if [ ! -f runs/$PILOT-analysis/pilot-engineering-v1.json ]; then
  if [ -e runs/$PILOT ]; then log "PILOT_EXISTS without analysis; manual decision required"; exit 4; fi
  log "PILOT_START"
  $S scripts/run_id60_masks_pilot_local.py --manifest $PILOT_MANIFEST \
    --output runs/$PILOT $SITE > runs/$PILOT.launcher.log 2>&1
  log "PILOT_EXIT code=$? results=$(find runs/$PILOT -path '*/results/*.json' | wc -l)"
  mkdir -p runs/$PILOT-analysis
  $S work/id60-masks-validation/analyze_masks_validation.py --run runs/$PILOT --pilot-engineering \
    --output runs/$PILOT-analysis/pilot-engineering-v1.json > runs/$PILOT-analysis/analyzer.log 2>&1
  log "PILOT_ANALYSIS_EXIT code=$?"
fi

# ---- gate decision (pre-declared: G9 complete traces, G5 inside-mask zero, sources match)
$P - "$PILOT" <<'PYEOF' >> "$LOG" 2>&1
import json, sys
r = json.load(open('runs/%s-analysis/pilot-engineering-v1.json' % sys.argv[1]))
g9 = r['g9_complete_trace_gate']; g5 = r['fidelity']['inside_mask_gate']
checks = {'G9_complete_valid_traces': bool(g9['passed']),
          'G5_inside_mask_all_reported_zero': bool(g5.get('all_reported_zero')) and g5.get('events_nonzero') == 0 and g5.get('invalid_values') == 0,
          'sources_match': bool(r['source_files_match'])}
print('PILOT_GATE_CHECKS', json.dumps(checks), 'g9', json.dumps({k: g9[k] for k in ('complete_valid_traces', 'invalid_or_missing_case_ids', 'mask_event_cadence_issues')}),
      'counts', json.dumps(r['descriptive_outcome_counts']))
sys.exit(0 if all(checks.values()) else 3)
PYEOF
if [ $? -ne 0 ]; then log "PILOT_GATES_FAIL manual decision required"; exit 3; fi
log "PILOT_GATES_PASS"

# ---- formal -----------------------------------------------------------------
if [ ! -f runs/$FORMAL-analysis/validation-results-server-v1.json ]; then
  # the pilot's policy/planner services may still be winding down: wait until none of ours is alive
  # and no listener remains on the worker port range (the driver refuses an occupied port and marks
  # every case unattempted; it never touches existing processes)
  for i in $(seq 1 60); do
    busy=$(pgrep -fc "$PWD/scripts/serve_id60_hierarchical"); ports=$(ss -ltn 2>/dev/null | grep -cE ':20(2[7-9][0-9]|30[0-9]) ')
    [ "$busy" = 0 ] && [ "$ports" = 0 ] && break; sleep 5
  done
  log "PORTS_READY after $((i*5))s busy_services=$busy listeners=$ports"
  FREE=$(gpu_headroom) || { log "FORMAL_BLOCKED GPU headroom below ${MIN_FREE_MIB} MiB:$FREE"; exit 2; }
  log "FORMAL_START$FREE"
  $S scripts/run_id60_masks_validation_local.py --manifest $FORMAL_MANIFEST \
    --output runs/$FORMAL $SITE > runs/$FORMAL.launcher.log 2>&1
  log "FORMAL_EXIT code=$? results=$(find runs/$FORMAL -path '*/results/*.json' | wc -l)"
  mkdir -p runs/$FORMAL-analysis
  if $S work/id60-masks-validation/analyze_masks_validation.py --run runs/$FORMAL \
       --output runs/$FORMAL-analysis/validation-results-server-v1.json > runs/$FORMAL-analysis/analyzer.log 2>&1; then
    log "FORMAL_ANALYSIS_OK"
  else
    log "FORMAL_ANALYSIS_FAIL (server-side analysis is best effort; the local WSL analysis is authoritative)"
  fi
fi

# ---- bundle without the NPZ arrays (pulled separately) ----------------------
( cd runs && tar czf ../$W/results-bundle.tar.gz --exclude='*.npz' \
    $PILOT $PILOT-analysis $FORMAL $FORMAL-analysis *.launcher.log 2>/dev/null )
log "BUNDLE_DONE $(du -h $W/results-bundle.tar.gz | cut -f1)"
log "CHAIN_DONE"
