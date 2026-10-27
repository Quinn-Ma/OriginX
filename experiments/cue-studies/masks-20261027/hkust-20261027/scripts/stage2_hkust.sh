#!/usr/bin/env bash
# HKUST stage 2 (adapted from cloud/server_stage2_h800.sh): pin the probed renderer
# string, declare the registered zero-sample policy-RGB cohort for both arms, run T3
# on GPU 3 in the background, dry-run the frozen pilot (6 cases) and formal (12 pairs)
# manifests for 2 workers on one GPU, and print status.
set -uo pipefail
cd /ephemeral/qinzhen/originx-harness-v2 || exit 1
W=work/hkust
S=/dev/shm/qinzhen/originx-envs/sim/bin/python
P=/usr/bin/python3
GPU=5          # T3 site GPU (healthy; 2/3 are in "Recovery Action: Reset")
GPUS=5,7       # one worker per healthy GPU (each has only ~25 GB free)
export PYTHONPATH=$PWD MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
mkdir -p $W/tmp $W/hf-cache $W/mpl-cache

echo "== renderer"
R=$(MUJOCO_GL=egl $S $W/native_egl_probe.py 2>/dev/null | $P -c 'import json,sys;d=json.load(sys.stdin);print([x for x in d["devices"] if x.get("stage")=="ready"][0]["renderer"])')
echo "RENDERER=$R"
$P - "$R" <<'PYEOF'
import json, sys
for p, key in (("work/hkust/site-t3.json", None), ("work/hkust/site-overrides-hkust.json", "gl_renderer")):
    s = json.load(open(p))
    if key is None:
        s["renderer"]["gl_renderer"] = sys.argv[1]
        s["policy_rgb_samples"] = 0
    else:
        s[key] = sys.argv[1]
        s["policy_rgb_samples"] = 0
    json.dump(s, open(p, "w"), indent=2)
    print("updated", p)
PYEOF

echo "== t3 (GPU $GPU, zero-sample policy RGB cohort)"
if [ ! -f $W/t3/exactness-checks-hkust-h100-v1.json ] && ! pgrep -f check_mask_exactness.py >/dev/null; then
  mkdir -p $W/t3
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  PYTHONPATH=$PWD:$PWD/vendor/robocasa:$PWD/vendor/robosuite \
  setsid nohup $S work/id60-masks-validation/check_mask_exactness.py --site-json $W/site-t3.json \
    --site-label hkust_h100_egl_rgb0 --output $W/t3/exactness-checks-hkust-h100-v1.json --contact-dir $W/t3/contact > $W/t3/t3.log 2>&1 < /dev/null &
  echo "T3_LAUNCHED $(date -u +%FT%TZ)"
fi

echo "== manifests"
$S -c "import json;m=json.load(open('work/id60-masks-validation/manifest.json'));print('formal',len(m['cases']),'cases',m['planned_pairs'],'pairs workers',m.get('workers'),'max_s',m.get('max_local_seconds'),'sha',m['manifest_sha256'][:12])"
$S -c "import json;m=json.load(open('work/id60-masks-validation/pilot-manifest-v1.json'));print('pilot',len(m['cases']),'cases',m['planned_pairs'],'pairs workers',m.get('workers'),'max_s',m.get('max_local_seconds'),'sha',m['manifest_sha256'][:12])"

echo "== dry-run"
rm -rf $W/pilot-dry-run $W/formal-dry-run
$S scripts/run_id60_masks_pilot_local.py --dry-run --manifest work/id60-masks-validation/pilot-manifest-v1.json --output $W/pilot-dry-run --site linux_egl --site-overrides $W/site-overrides-hkust.json --workers 2 --gpus $GPUS --workers-per-gpu 1 2>&1 | tail -3
$S scripts/run_id60_masks_validation_local.py --dry-run --manifest work/id60-masks-validation/manifest.json --output $W/formal-dry-run --site linux_egl --site-overrides $W/site-overrides-hkust.json --workers 2 --gpus $GPUS --workers-per-gpu 1 2>&1 | tail -3
$P -c "import json;e=json.load(open('$W/pilot-dry-run/local-execution.json'));print('layout',e.get('layout'),'gpus',e.get('gpus'))"
ls $W/pilot-dry-run $W/formal-dry-run 2>/dev/null | wc -l
echo "== status"
tail -2 $W/t3/t3.log 2>/dev/null | cut -c1-160
df -h /ephemeral | tail -1; du -sh /dev/shm/qinzhen
