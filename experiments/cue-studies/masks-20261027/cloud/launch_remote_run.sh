#!/usr/bin/env bash
# Launch a pilot or formal run detached on the H800 (2 workers on GPU 0), then exit.
# Usage: launch_remote_run.sh pilot|formal <run-dir-name>
set -euo pipefail
KIND="$1"; NAME="$2"
HOST=connect.westb.seetacloud.com; PORT=40137
if [ "$KIND" = pilot ]; then DRV=scripts/run_id60_masks_pilot_local.py; MAN=work/id60-masks-validation/pilot-manifest-v1.json
else DRV=scripts/run_id60_masks_validation_local.py; MAN=work/id60-masks-validation/manifest.json; fi
ssh -o BatchMode=yes -p $PORT root@$HOST "cd /root/autodl-tmp/originx-harness-v2 && test ! -e runs/$NAME && \
  env PYTHONPATH=\$PWD MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  setsid nohup work/autodl/sim-env/bin/python $DRV --manifest $MAN --output runs/$NAME \
    --site linux_egl --site-overrides work/autodl/site-overrides-autodl.json --workers 2 --gpus 0 --workers-per-gpu 2 \
    > runs/$NAME.launcher.log 2>&1 < /dev/null & sleep 3; pgrep -af run_id60_masks | head -2"
