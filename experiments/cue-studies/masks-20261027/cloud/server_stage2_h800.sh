#!/usr/bin/env bash
# Stage 2 on the H800 instance: pin the probed renderer string, declare the
# registered zero-sample policy-RGB cohort for both arms, run T3 on GPU 0,
# freeze the pilot (6 cases) and formal (12 pairs) manifests for 2 workers on
# one GPU, dry-run both, and print status.
set -uo pipefail
cd /root/autodl-tmp/originx-harness-v2
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
P=/root/miniconda3/bin/python
W=work/autodl
S=$W/sim-env/bin/python
export PYTHONPATH=$PWD MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
mkdir -p $W/tmp $W/hf-cache $W/mpl-cache

echo "== renderer"
R=$(MUJOCO_GL=egl $S $W/native_egl_probe.py 2>/dev/null | $P -c 'import json,sys;d=json.load(sys.stdin);print([x for x in d["devices"] if x.get("stage")=="ready"][0]["renderer"])')
echo "RENDERER=$R"
$P - "$R" <<'PYEOF'
import json, sys
for p, key in (("work/autodl/site-t3.json", None), ("work/autodl/site-overrides-autodl.json", "gl_renderer")):
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

echo "== t3 (GPU 0, zero-sample policy RGB cohort)"
if [ ! -f $W/t3/exactness-checks-autodl-h800-v1.json ] && ! pgrep -f check_mask_exactness.py >/dev/null; then
  mkdir -p $W/t3
  CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  PYTHONPATH=$PWD:$PWD/vendor/robocasa:$PWD/vendor/robosuite \
  nohup $S work/id60-masks-validation/check_mask_exactness.py --site-json $W/site-t3.json \
    --site-label autodl_h800_egl_rgb0 --output $W/t3/exactness-checks-autodl-h800-v1.json --contact-dir $W/t3/contact > $W/t3/t3.log 2>&1 &
fi

echo "== freeze + dry-run"
[ -f work/id60-masks-validation/pilot-manifest-v1.json ] || $S scripts/run_id60_masks_pilot_local.py --freeze --manifest work/id60-masks-validation/pilot-manifest-v1.json --workers 2 2>&1 | tail -2
[ -f work/id60-masks-validation/manifest.json ] || $S scripts/run_id60_masks_validation_local.py --freeze --manifest work/id60-masks-validation/manifest.json --pairs 12 --workers 2 2>&1 | tail -2
rm -rf $W/pilot-dry-run $W/formal-dry-run
$S scripts/run_id60_masks_pilot_local.py --dry-run --manifest work/id60-masks-validation/pilot-manifest-v1.json --output $W/pilot-dry-run --site linux_egl --site-overrides $W/site-overrides-autodl.json --workers 2 --gpus 0 --workers-per-gpu 2 2>&1 | tail -3
$S scripts/run_id60_masks_validation_local.py --dry-run --manifest work/id60-masks-validation/manifest.json --output $W/formal-dry-run --site linux_egl --site-overrides $W/site-overrides-autodl.json --workers 2 --gpus 0 --workers-per-gpu 2 2>&1 | tail -3
$S -c "import json;m=json.load(open('work/id60-masks-validation/manifest.json'));print('formal',len(m['cases']),'cases',m['planned_pairs'],'pairs workers',m.get('workers'),'max_s',m.get('max_local_seconds'),'sha',m['manifest_sha256'][:12])"
ls $W/pilot-dry-run $W/formal-dry-run | wc -l

echo "== status"
tail -1 /root/originx-models/download.log | cut -c1-160
echo "verified: $(grep -c verified_ /root/originx-models/download.log)"
grep -E "ENV_DONE|_OK|ALL_ENVS|ERROR" /root/build_model_envs.log | tail -3
tail -2 $W/t3/t3.log | cut -c1-160
df -h /root/autodl-tmp | tail -1
