#!/usr/bin/env bash
# After the AutoDL instance is powered on again: ship the analyzer gate module and
# the chain script, launch the chain detached, print its first log lines.
# Usage: resume_after_boot.sh [port]
set -euo pipefail
HOST=connect.westb.seetacloud.com; PORT="${1:-40137}"
L=/d/codex/originx-harness-v2; T=/root/autodl-tmp/originx-harness-v2
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -p $PORT root@$HOST"
SCP="scp -o BatchMode=yes -P $PORT"
$SSH "mkdir -p $T/work/id60-targeted-parser; hostname; date -u; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; ls $T/runs; pgrep -af chain_after_boot || true"
$SCP "$L/work/id60-targeted-parser/analyze_targeted.py" root@$HOST:$T/work/id60-targeted-parser/
$SCP "$L/work/id60-masks-validation/cloud/chain_after_boot.sh" root@$HOST:/root/chain_after_boot.sh
$SSH "chmod +x /root/chain_after_boot.sh; pgrep -f chain_after_boot.sh >/dev/null && echo ALREADY_RUNNING || (setsid nohup /root/chain_after_boot.sh > /root/chain.out 2>&1 < /dev/null & sleep 8; tail -5 /root/chain.log)"
