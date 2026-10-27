#!/usr/bin/env bash
set -uo pipefail
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 hkust-cluster"
SCP="scp -o BatchMode=yes"
L=/d/codex/originx-harness-v2
R=/d/codex/2026-09-27/review/work/masks-repair-20261025
O=/d/codex/OriginX-repo/experiments/cue-studies/masks-20261027
SC="$HOME/AppData/Local/Temp/claude/D--paper-prep-2026-09-21/09f1707e-5452-4c1f-b446-178b7b1f1d99/scratchpad"
T=/ephemeral/qinzhen/originx-harness-v2
M=/dev/shm/qinzhen/originx-models
echo "START $(date -u +%FT%TZ)"
$SSH "mkdir -p $T/work/hkust $T/vendor $T/work/id60-masks-validation/deps/wheels $T/work/id60-targeted-parser $M/planner $M/policy && df -h /ephemeral | tail -1"
# 1) small: model manifest + downloader, launch the policy download immediately (planner reused from disk)
$SCP $R/models_manifest.json $SC/cloud_download_models_hkust.py hkust-cluster:$M/ && \
$SSH "cd $M && cp -n /ephemeral/qinzhen/originx-harness-pilot-20261022/weights/awomo/qwen/* planner/ && ls planner | wc -l && (setsid nohup python3 $M/cloud_download_models_hkust.py > $M/download.log 2>&1 < /dev/null &) ; sleep 5; tail -2 $M/download.log | cut -c1-200"
echo "DOWNLOAD_LAUNCHED $(date -u +%FT%TZ)"
# 2) source + scripts + wheels (small)
$SCP $L/work/id60-masks-validation/cloud/autodl-source-v2.tar.gz $L/work/id60-masks-validation/cloud/autodl-source-v2.json \
     $R/autodl-sim-requirements.lock $R/prepare-cloud-project.py $R/native_egl_probe.py $R/policy-cloud.requirements.txt $R/planner-cloud.requirements.txt $R/backend-xr1-cloud.json \
     $O/fidelity/site-t3.json $O/cloud/site-overrides-autodl.json \
     hkust-cluster:$T/work/hkust/
$SCP $L/work/id60-masks-validation/deps/wheels/*.whl hkust-cluster:$T/work/id60-masks-validation/deps/wheels/
$SCP $L/work/id60-targeted-parser/analyze_targeted.py hkust-cluster:$T/work/id60-targeted-parser/
echo "SMALL_DONE $(date -u +%FT%TZ)"
# 3) asset archives (~520 MB)
$SCP $L/work/id60-revision/asset-pack/stage-a-assets.tar.gz $R/dev-assets/dev-seeds-assets-delta-v1.tar.gz $L/work/id60-masks-validation/seed-assets/formal-seeds-assets-delta-v1.tar.gz hkust-cluster:$T/work/hkust/
echo "ASSETS1_DONE $(date -u +%FT%TZ)"
$SCP $L/work/id60-revision/asset-pack/stage-a-upstreams.tar.gz hkust-cluster:$T/work/hkust/
echo "UPSTREAMS_DONE $(date -u +%FT%TZ)"
$SSH "ls -la $T/work/hkust/ | awk '{print \$5, \$9}'; tail -2 $M/download.log | cut -c1-200"
echo "END $(date -u +%FT%TZ)"
