#!/usr/bin/env bash
# Stage 2 on the AutoDL instance: fix the site renderer name, restart the model
# download through the HF mirror, launch T3 on GPU 1, and print status.
set -uo pipefail
cd /root/autodl-tmp/originx-harness-v2
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
P=/root/miniconda3/bin/python
W=work/autodl

echo "== renderer"
R=$(MUJOCO_GL=egl $W/sim-env/bin/python $W/native_egl_probe.py 2>/dev/null | $P -c 'import json,sys;d=json.load(sys.stdin);print([x for x in d["devices"] if x.get("stage")=="ready"][0]["renderer"])')
echo "RENDERER=$R"
$P - "$R" <<'PYEOF'
import json, sys
p = "work/autodl/site-t3.json"
s = json.load(open(p)); s["renderer"]["gl_renderer"] = sys.argv[1]
json.dump(s, open(p, "w"), indent=2); print("site_updated", sys.argv[1])
PYEOF

echo "== models"
if ! pgrep -f cloud_download_models_mirror.py >/dev/null; then
  pkill -f cloud_download_models.py; sleep 1
  sed 's#https://huggingface.co/#https://hf-mirror.com/#' $W/cloud_download_models.py > $W/cloud_download_models_mirror.py
  (cd /root && nohup $P /root/autodl-tmp/originx-harness-v2/$W/cloud_download_models_mirror.py >> /root/originx-models/download2.log 2>&1 &)
  sleep 20
fi
tail -1 /root/originx-models/download2.log | cut -c1-200
echo "verified_so_far: $(cat /root/originx-models/download.log /root/originx-models/download2.log 2>/dev/null | grep -c verified_)"

echo "== envs"
pgrep -f build_model_envs.sh >/dev/null && echo build_running || echo build_not_running
grep -E "ENV_DONE|_OK|ALL_ENVS|Error|error:" /root/build_model_envs.log | tail -4

echo "== t3"
if ! pgrep -f check_mask_exactness.py >/dev/null; then
  rm -rf $W/t3; mkdir -p $W/t3
  CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLBACKEND=Agg PYTHONHASHSEED=0 \
  PYTHONPATH=$PWD:$PWD/vendor/robocasa:$PWD/vendor/robosuite \
  nohup $W/sim-env/bin/python work/id60-masks-validation/check_mask_exactness.py --site-json $W/site-t3.json \
    --site-label autodl_4090_egl --output $W/t3/exactness-checks-autodl-4090-v1.json --contact-dir $W/t3/contact > $W/t3/t3.log 2>&1 &
  sleep 60
fi
tail -4 $W/t3/t3.log | cut -c1-220
df -h /root/autodl-tmp | tail -1
