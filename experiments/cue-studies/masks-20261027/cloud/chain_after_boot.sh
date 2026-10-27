#!/bin/bash
# Runs on the H800 instance after a reboot (detached via setsid nohup):
# prerequisites -> archive the interrupted pilot -> pilot (6 cases) -> pilot
# engineering analysis -> gate decision -> formal run (12 pairs) -> prospective
# analysis (best effort; the local WSL analysis is authoritative) -> results bundle.
# Every step appends a marker line to /root/chain.log; the local watcher reads it.
set -uo pipefail
cd /root/autodl-tmp/originx-harness-v2 || exit 1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
LOG=/root/chain.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
S=work/autodl/sim-env/bin/python
P=/root/miniconda3/bin/python
PILOT=id60-masks-pilot-h800-v1
FORMAL=id60-masks-validation-h800-v1
SITE="--site linux_egl --site-overrides work/autodl/site-overrides-autodl.json --workers 2 --gpus 0 --workers-per-gpu 2"
export PYTHONPATH=$PWD MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

log "CHAIN_START pid=$$"

# ---- prerequisites: envs, weights, pinned renderer -------------------------
/root/originx-policy-env/bin/python -c "import torch, flash_attn, transformers; print('POLICY_OK', torch.__version__, flash_attn.__version__)" >> "$LOG" 2>&1 || { log "PREREQ_FAIL policy env"; exit 2; }
/root/originx-planner-env/bin/python -c "import torch, transformers; print('PLANNER_OK', transformers.__version__)" >> "$LOG" 2>&1 || { log "PREREQ_FAIL planner env"; exit 2; }
[ "$(grep -c verified_ /root/originx-models/download.log)" -ge 30 ] || { log "PREREQ_FAIL models not verified"; exit 2; }
R=$(MUJOCO_GL=egl $S work/autodl/native_egl_probe.py 2>/dev/null | $P -c 'import json,sys;d=json.load(sys.stdin);print([x for x in d["devices"] if x.get("stage")=="ready"][0]["renderer"])')
PIN=$($P -c 'import json;print(json.load(open("work/autodl/site-overrides-autodl.json"))["gl_renderer"])')
[ "$R" = "$PIN" ] || { log "PREREQ_FAIL renderer '$R' != pinned '$PIN'"; exit 2; }
[ -f work/id60-targeted-parser/analyze_targeted.py ] || { log "PREREQ_FAIL analyzer gate module missing"; exit 2; }
log "PREREQ_OK renderer=$R"

# ---- a formal directory from an interrupted attempt needs a human decision --
if [ -e runs/$FORMAL ] && [ ! -f runs/$FORMAL-analysis/validation-results-server-v1.json ]; then
  log "FORMAL_EXISTS results=$(find runs/$FORMAL -path '*/results/*.json' | wc -l) manual decision required (resume driver for zero-step pairs)"
  exit 4
fi

# ---- archive an interrupted pilot unchanged; it is engineering-only ----------
if [ -d runs/$PILOT ] && [ ! -f runs/$PILOT-analysis/pilot-engineering-v1.json ]; then
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  N=$(find runs/$PILOT -path '*/results/*.json' | wc -l)
  $P - "$TS" "$N" <<'PYEOF'
import json, sys
rec = {'schema': 'id60-masks-pilot-interruption-v1', 'recorded_utc': sys.argv[1],
       'run': 'runs/id60-masks-pilot-h800-v1',
       'reason': 'AutoDL instance shut down (balance exhausted / shutdown) while the engineering pilot was running; '
                 'no process survived the reboot',
       'result_files_present': int(sys.argv[2]),
       'treatment': 'pilot is engineering-only; the interrupted attempt is archived unchanged and the pilot is re-run '
                    'from scratch; no outcome from the interrupted attempt is used'}
json.dump(rec, open('runs/id60-masks-pilot-h800-v1/interruption-record-%s.json' % sys.argv[1], 'w'), indent=2)
PYEOF
  mv runs/$PILOT runs/$PILOT.interrupted-$TS
  [ -f runs/$PILOT.launcher.log ] && mv runs/$PILOT.launcher.log runs/$PILOT.interrupted-$TS.launcher.log
  log "PILOT_ARCHIVED runs/$PILOT.interrupted-$TS results_present=$N"
fi

# ---- pilot ------------------------------------------------------------------
if [ ! -f runs/$PILOT-analysis/pilot-engineering-v1.json ]; then
  log "PILOT_START"
  $S scripts/run_id60_masks_pilot_local.py --manifest work/id60-masks-validation/pilot-manifest-v1.json \
    --output runs/$PILOT $SITE > runs/$PILOT.launcher.log 2>&1
  log "PILOT_EXIT code=$? results=$(find runs/$PILOT -path '*/results/*.json' | wc -l)"
  mkdir -p runs/$PILOT-analysis
  $S work/id60-masks-validation/analyze_masks_validation.py --run runs/$PILOT --pilot-engineering \
    --output runs/$PILOT-analysis/pilot-engineering-v1.json > runs/$PILOT-analysis/analyzer.log 2>&1
  log "PILOT_ANALYSIS_EXIT code=$?"
fi

# ---- gate decision (pre-declared: G9 complete traces, G5 inside-mask zero, sources match)
$P - <<'PYEOF' >> "$LOG" 2>&1
import json, sys
r = json.load(open('runs/id60-masks-pilot-h800-v1-analysis/pilot-engineering-v1.json'))
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
  log "FORMAL_START"
  $S scripts/run_id60_masks_validation_local.py --manifest work/id60-masks-validation/manifest.json \
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

# ---- bundle without the NPZ arrays (pulled separately if time allows) --------
( cd runs && tar czf /root/autodl-tmp/results-bundle.tar.gz --exclude='*.npz' \
    $PILOT $PILOT-analysis $FORMAL $FORMAL-analysis *.launcher.log $PILOT.interrupted-* 2>/dev/null )
log "BUNDLE_DONE $(du -h /root/autodl-tmp/results-bundle.tar.gz | cut -f1)"
log "CHAIN_DONE"
