#!/usr/bin/env bash
# 4B queue on GPUs 0,2,5,6: (wait for difficulty scoring) -> build data -> OPD -> EOPD -> eval base/OPD/EOPD.
# Launch fully detached:
#   (setsid nohup bash ~/EOPD/run_4b_queue.sh > ~/EOPD/runs/queue_4b.log 2>&1 < /dev/null &)
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
export GPUS=${GPUS:-0,2,5,6}
mkdir -p runs
exec 9> runs/queue_4b.lock
flock -n 9 || { echo "another 4B queue is running - not starting"; exit 1; }
source "$HOME/Rethinking-OPD/.venv/bin/activate"
export PYTHONPATH="$ROOT"

echo "== [$(date)] waiting for difficulty scoring"
while pgrep -f "prep_dapo_4[b].py (score|shard)" >/dev/null; do sleep 30; done
[ -s data/dapo17k/pool_scores.jsonl ] || { echo "== no pool_scores.jsonl - scoring failed"; exit 1; }
echo "== [$(date)] BUILD"
python data/prep_dapo_4b.py build --steps ${STEPS:-50} || { echo "== BUILD FAILED"; exit 1; }

latest_hf() { d=$(ls -d "$ROOT/runs/$1/ckpt"/global_step_* 2>/dev/null | sort -V | tail -1); echo "$d/actor/huggingface"; }

for MODE in opd eopd; do
  echo "== [$(date)] TRAIN $MODE"
  bash run_4b.sh $MODE > runs/train_4b_$MODE.log 2>&1
  rc=$?
  echo "== [$(date)] TRAIN $MODE exited rc=$rc"
  if [ $rc -ne 0 ] || [ ! -d "$ROOT/runs/qwen3_4b_dapo17k_$MODE/ckpt/global_step_${STEPS:-50}" ]; then
    echo "== [$(date)] TRAIN $MODE FAILED (no final checkpoint) - stopping queue"; exit 1
  fi
done

echo "== [$(date)] EVAL base / opd / eopd in parallel"
IFS=, read -r G0 G1 G2 _ <<<"$GPUS"
CUDA_VISIBLE_DEVICES=$G0 python eval_math.py --model Qwen/Qwen3-4B-Base --out runs/eval_4b_base.json > runs/eval_4b_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G1 python eval_math.py --model "$(latest_hf qwen3_4b_dapo17k_opd)" --out runs/eval_4b_opd.json > runs/eval_4b_opd.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G2 python eval_math.py --model "$(latest_hf qwen3_4b_dapo17k_eopd)" --out runs/eval_4b_eopd.json > runs/eval_4b_eopd.log 2>&1 &
wait
echo "== [$(date)] REPORT"
python scripts/behavior_report.py runs/qwen3_4b_dapo17k_opd runs/qwen3_4b_dapo17k_eopd --out runs/report_4b.md
echo "== [$(date)] QUEUE_DONE"
