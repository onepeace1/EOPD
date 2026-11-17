#!/usr/bin/env bash
# Disk guard: once 4B OPD has finished (final step-270 checkpoint written), delete its one non-final
# checkpoint (redundant: evals use the final one) so the EOPD run has room for its own 2 x 63 GB checkpoints.
cd "$(dirname "$0")/.."
D=runs/qwen3_4b_dapo17k_opd/ckpt
until [ "$(cat $D/latest_checkpointed_iteration.txt 2>/dev/null)" = "270" ] && grep -q "EVAL\|TRAIN eopd" runs/queue_4b.log; do sleep 120; done
for d in $D/global_step_*; do
  [ "$d" = "$D/global_step_270" ] && continue
  echo "[$(date)] removing $d"; rm -rf "$d"
done
df -h ~ | tail -1
