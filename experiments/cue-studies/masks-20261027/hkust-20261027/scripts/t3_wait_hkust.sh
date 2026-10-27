#!/usr/bin/env bash
# Wait for the T3 exactness check on the cluster to finish, then print its gate summary.
T=/ephemeral/qinzhen/originx-harness-v2
for i in $(seq 1 90); do
  OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=20 hkust-cluster "cd $T/work/hkust && if [ -f t3/exactness-checks-hkust-h100-v1.json ] && ! pgrep -f check_mask_exactness.py >/dev/null; then echo T3_DONE; else echo WAIT lines=\$(wc -l < t3/t3.log) scenes=\$(grep -c 'Creating CoffeeSetupMug' t3/t3.log) running=\$(pgrep -fc check_mask_exactness.py); fi" 2>&1)
  echo "$(date -u +%H:%M) $OUT"
  if echo "$OUT" | grep -q T3_DONE; then
    ssh -o BatchMode=yes hkust-cluster "cd $T/work/hkust && python3 - <<'PY'
import json
r = json.load(open('t3/exactness-checks-hkust-h100-v1.json'))
print('all_passed', r.get('all_passed'), 'predeclared', r.get('all_gates_are_predeclared'))
for k, v in r.items():
    if isinstance(v, bool): print('  ', k, v)
    elif isinstance(v, dict) and 'passed' in v: print('  ', k, 'passed=', v.get('passed'))
for k in ('augmenter_extra_ms_median_per_seed', 'depth_convention_selected', 'site_label', 'renderer', 'gl_renderer', 'overhead_ms_median_per_seed'):
    if k in r: print('  ', k, r[k])
def walk(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ('iou_min', 'iou_median', 'min_iou', 'median_iou', 'pixels_outside_band', 'outside_band_px', 'masks_evaluated', 'mismatched_pixels'):
                print('  ', p + k, v)
            walk(v, p + k + '.')
    elif isinstance(o, list) and o and isinstance(o[0], dict) and len(o) <= 5:
        for i, v in enumerate(o): walk(v, p + str(i) + '.')
walk(r)
PY
tail -5 t3/t3.log | cut -c1-200; ls t3/contact | head -3"
    exit 0
  fi
  if echo "$OUT" | grep -q "running=0" && ! echo "$OUT" | grep -q T3_DONE; then
    echo "T3_PROCESS_GONE_WITHOUT_RESULT"; ssh -o BatchMode=yes hkust-cluster "tail -30 $T/work/hkust/t3/t3.log | cut -c1-200"; exit 2
  fi
  sleep 60
done
echo T3_WAIT_TIMEOUT; exit 3
