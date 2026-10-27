#!/usr/bin/env bash
# Pull run outputs (results, receipts, logs, traces metadata) from the AutoDL instance to the local mirror.
# Usage: pull_results.sh <remote-run-dir-name> [--traces]   (--traces also pulls the NPZ arrays)
set -euo pipefail
NAME="$1"; TRACES="${2:-}"
HOST=connect.westb.seetacloud.com; PORT=40137
R=/root/autodl-tmp/originx-harness-v2/runs
L=/d/codex/originx-harness-v2/runs/autodl-mirror
mkdir -p "$L"
EXCL="--exclude=*.npz"
[ "$TRACES" = "--traces" ] && EXCL=""
ssh -o BatchMode=yes -p $PORT root@$HOST "cd $R && tar czf - $EXCL $NAME $NAME.launcher.log 2>/dev/null" | tar xzf - -C "$L"
ls "$L/$NAME"
