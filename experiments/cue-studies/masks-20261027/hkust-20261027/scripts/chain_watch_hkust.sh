#!/usr/bin/env bash
# Poll the chain log on the cluster every 5 minutes; exit on a terminal marker.
T=/ephemeral/qinzhen/originx-harness-v2
FAILS=0
for i in $(seq 1 150); do
  OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=20 hkust-cluster "cd $T/runs 2>/dev/null; echo \"\$(date -u +%H:%M) pilot=\$(find id60-masks-pilot-hkust-w3-v1 -path '*/results/*.json' 2>/dev/null | wc -l)/6 formal=\$(find id60-masks-validation-hkust-w3-v1 -path '*/results/*.json' 2>/dev/null | wc -l)/24 free5/6/7=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 5,6,7 | tr -d ' ' | paste -sd/) disk=\$(df --output=avail -BG /ephemeral | tail -1 | tr -d ' ') | \$(tail -1 $T/work/hkust/chain.log 2>/dev/null | cut -c1-160)\"" 2>&1)
  if [ $? -ne 0 ]; then FAILS=$((FAILS+1)); echo "$(date -u +%H:%M) SSH_FAIL $FAILS: $(echo "$OUT" | tail -1)"; [ $FAILS -ge 6 ] && { echo "WATCH_ABORT_UNREACHABLE"; exit 2; }
  else FAILS=0; echo "$OUT"; fi
  echo "$OUT" | grep -qE "CHAIN_DONE|_FAIL|FORMAL_EXISTS|PILOT_EXISTS|FORMAL_BLOCKED" && { echo "WATCH_END"; exit 0; }
  sleep 300
done
echo WATCH_TIMEOUT
