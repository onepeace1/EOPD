#!/usr/bin/env bash
# Exits (=> notifies the caller) on the first event: a queue-log line, a crash signature, a stall, or a heartbeat.
#   bash scripts/watch_runs.sh [heartbeat_seconds]
cd "$(dirname "$0")/.."
HB=${1:-7200}
QUEUES="runs/queue_4b.log runs/post_gpu3.log"
TRAINS="runs/train_4b_opd.log runs/train_4b_eopd.log runs/train_eopd.log"
sig() { for f in $QUEUES; do [ -f $f ] && wc -l < $f; done | paste -sd,; }
s0=$(sig); t0=$(date +%s); reason=""
while :; do
  sleep 60
  [ "$(sig)" != "$s0" ] && { reason="queue log changed"; break; }
  for f in $TRAINS; do
    [ -f $f ] || continue
    age=$(( $(date +%s) - $(stat -c %Y $f) ))
    if [ $age -lt 900 ] && grep -qE "Traceback|OutOfMemory|CUDA out of memory" <(tail -c 200000 $f); then
      reason="crash signature in $f"; break 2; fi
    if [ $age -gt 1200 ] && [ $age -lt 3600 ]; then reason="stall: $f silent for ${age}s"; break 2; fi
  done
  [ $(( $(date +%s) - t0 )) -ge $HB ] && { reason="heartbeat"; break; }
done
echo "### EVENT: $reason  ($(date +%T))"
for f in $QUEUES; do echo "--- $f"; tail -4 $f 2>/dev/null; done
for f in $TRAINS; do
  [ -f $f ] || continue
  echo "--- $f (modified $(( $(date +%s) - $(stat -c %Y $f) ))s ago)"
  grep -E "step:[0-9]+ -" $f | tail -1 | grep -oE "(training/global_step|timing_s/step|train/accuracy|response_length/mean|response_length/clip_ratio|actor/rev_kl|actor/entropy)[:][0-9.]+" | paste -sd' '
  grep -oE "val-core/[A-Za-z0-9-]+/acc/mean@[0-9]+:[0-9.]+" $f | tail -3 | paste -sd' '
  grep -nE "Traceback|OutOfMemory" $f | tail -2 | cut -c1-200
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 0,2,3,5,6 | paste -sd'|'
df -h ~ | tail -1 | awk '{print "disk free:",$4}'
