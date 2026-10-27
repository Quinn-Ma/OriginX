#!/usr/bin/env bash
# Pull run outputs (results, receipts, logs, trace metadata) from the cluster to the local mirror.
# Usage: pull_results_hkust.sh <remote-run-dir-name> [--traces]   (--traces also pulls the NPZ arrays)
set -euo pipefail
NAME="$1"; TRACES="${2:-}"
R=/ephemeral/qinzhen/originx-harness-v2/runs
L=/d/codex/originx-harness-v2/runs/hkust-mirror
mkdir -p "$L"
EXCL="--exclude=*.npz"
[ "$TRACES" = "--traces" ] && EXCL=""
ssh -o BatchMode=yes hkust-cluster "cd $R && tar czf - $EXCL $NAME $NAME.launcher.log 2>/dev/null" | tar xzf - -C "$L"
ls "$L/$NAME"
