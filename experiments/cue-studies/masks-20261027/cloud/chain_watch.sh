#!/usr/bin/env bash
# Poll the chain log on the instance every 3 minutes; exit on a terminal marker.
# Usage: chain_watch.sh [port]
HOST=connect.westb.seetacloud.com; PORT="${1:-40137}"
FAILS=0
for i in $(seq 1 200); do
  OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PORT" root@$HOST 'cd /root/autodl-tmp/originx-harness-v2/runs 2>/dev/null; echo "$(date -u +%H:%M) pilot=$(find id60-masks-pilot-h800-v1 -path "*/results/*.json" 2>/dev/null | wc -l)/6 formal=$(find id60-masks-validation-h800-v1 -path "*/results/*.json" 2>/dev/null | wc -l)/24 gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d " ") | $(tail -1 /root/chain.log 2>/dev/null | cut -c1-150)"' 2>&1)
  if [ $? -ne 0 ]; then FAILS=$((FAILS+1)); echo "$(date -u +%H:%M) SSH_FAIL $FAILS: $(echo "$OUT" | tail -1)"; [ $FAILS -ge 6 ] && { echo "WATCH_ABORT_UNREACHABLE"; exit 2; }
  else FAILS=0; echo "$OUT"; fi
  echo "$OUT" | grep -qE "CHAIN_DONE|_FAIL|FORMAL_EXISTS" && { echo "WATCH_END"; exit 0; }
  sleep 180
done
echo WATCH_TIMEOUT
