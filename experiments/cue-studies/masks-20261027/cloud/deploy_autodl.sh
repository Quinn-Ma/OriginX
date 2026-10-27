#!/usr/bin/env bash
# One-shot AutoDL deployment: upload staged archives, build the pinned sim env,
# extract vendor assets, write the linux_egl site, probe native EGL.
# Usage: deploy_autodl.sh <host> <port>
set -euo pipefail
HOST="$1"; PORT="$2"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -p $PORT root@$HOST"
SCP="scp -o BatchMode=yes -P $PORT"
L=/d/codex/originx-harness-v2
R=/d/codex/2026-09-27/review/work/masks-repair-20261025
T=/root/autodl-tmp/originx-harness-v2

$SSH "mkdir -p $T/work/autodl $T/vendor $T/work/id60-masks-validation/deps/wheels; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; df -h / /root/autodl-tmp | tail -2; ls $T/work/autodl 2>/dev/null | head"
# Small, order-independent uploads (resume-safe: scp overwrites; archives are hash-checked remotely).
$SCP $L/work/id60-masks-validation/cloud/autodl-source-v2.tar.gz $L/work/id60-masks-validation/cloud/autodl-source-v2.json \
     $R/autodl-sim-requirements.lock $R/prepare-autodl-sim.sh $R/prepare-cloud-project.py $R/cloud_download_models.py $R/models_manifest.json \
     $R/native_egl_probe.py $R/policy-cloud.requirements.txt $R/planner-cloud.requirements.txt $R/backend-xr1-cloud.json \
     $L/work/id60-revision/asset-pack/stage-a-upstreams.tar.gz $L/work/id60-revision/asset-pack/stage-a-assets.tar.gz \
     $R/dev-assets/dev-seeds-assets-delta-v1.tar.gz $L/work/id60-masks-validation/seed-assets/formal-seeds-assets-delta-v1.tar.gz \
     root@$HOST:$T/work/autodl/
$SCP $L/work/id60-masks-validation/deps/wheels/*.whl root@$HOST:$T/work/id60-masks-validation/deps/wheels/
$SSH "cd $T && tar xzf work/autodl/autodl-source-v2.tar.gz && bash work/autodl/prepare-autodl-sim.sh && \
      work/autodl/sim-env/bin/python -m pip install --no-deps --no-index $T/work/id60-masks-validation/deps/wheels/*.whl && \
      work/autodl/sim-env/bin/python work/autodl/prepare-cloud-project.py && \
      tar xzf work/autodl/dev-seeds-assets-delta-v1.tar.gz -C vendor && tar xzf work/autodl/formal-seeds-assets-delta-v1.tar.gz -C vendor && \
      MUJOCO_GL=egl work/autodl/sim-env/bin/python work/autodl/native_egl_probe.py | tail -5; df -h /root/autodl-tmp | tail -1"
