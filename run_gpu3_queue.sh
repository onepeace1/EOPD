#!/usr/bin/env bash
# GPU 3 queue: naive OPD -> EOPD -> paper-style eval of both final checkpoints (+ the untrained base).
# Launch: setsid nohup bash ~/EOPD/run_gpu3_queue.sh > ~/EOPD/runs/queue.log 2>&1 < /dev/null & disown
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
export GPU=${GPU:-3}

latest_hf() {  # HF weights of the last saved step
  local d; d=$(ls -d "$ROOT/runs/$1/ckpt"/global_step_* 2>/dev/null | sort -V | tail -1)
  echo "$d/actor/huggingface"
}

for MODE in opd eopd; do
  echo "== [$(date)] TRAIN $MODE"
  bash run_1p7b.sh $MODE > runs/train_$MODE.log 2>&1
  rc=$?
  echo "== [$(date)] TRAIN $MODE exited rc=$rc"
  # verl can exit 0 after a Ray task error, so also require the final (step 174) checkpoint
  if [ $rc -ne 0 ] || [ ! -d "$ROOT/runs/qwen3_1p7b_math_$MODE/ckpt/global_step_174" ]; then
    echo "== [$(date)] TRAIN $MODE FAILED (no final checkpoint) - stopping queue"; exit 1
  fi
done

source "$HOME/Rethinking-OPD/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$ROOT"
for MODE in opd eopd; do
  echo "== [$(date)] EVAL $MODE ($(latest_hf qwen3_1p7b_math_$MODE))"
  python eval_math.py --model "$(latest_hf qwen3_1p7b_math_$MODE)" --out runs/eval_$MODE.json > runs/eval_$MODE.log 2>&1
  echo "== [$(date)] EVAL $MODE exited rc=$?"
done
echo "== [$(date)] QUEUE_DONE"
