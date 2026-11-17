#!/usr/bin/env bash
# GPU 3 follow-up after the resumed 1.7B EOPD run: wait for it, verify the final checkpoint,
# then paper-style eval (6 benchmarks, 8 samples, 8192 tokens) of OPD / EOPD / base, then the behaviour report.
#   (setsid nohup bash ~/EOPD/run_gpu3_post.sh > ~/EOPD/runs/post_gpu3.log 2>&1 < /dev/null &)
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
source "$HOME/Rethinking-OPD/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=${GPU:-3} PYTHONPATH="$ROOT"

echo "== [$(date)] waiting for 1.7B EOPD training to exit"
while pgrep -f "run_1p7[b].sh eopd" >/dev/null; do sleep 60; done
last=$(cat runs/qwen3_1p7b_math_eopd/ckpt/latest_checkpointed_iteration.txt 2>/dev/null)
echo "== [$(date)] EOPD exited; latest checkpoint = $last"
[ "$last" = "174" ] || { echo "== EOPD did not reach step 174 - not evaluating"; exit 1; }

for M in opd eopd; do
  echo "== [$(date)] EVAL $M"
  python eval_math.py --model runs/qwen3_1p7b_math_$M/ckpt/global_step_174/actor/huggingface \
    --out runs/eval_$M.json > runs/eval_$M.log 2>&1
  echo "== [$(date)] EVAL $M exited rc=$?"
done
echo "== [$(date)] EVAL base"
python eval_math.py --model Qwen/Qwen3-1.7B-Base --out runs/eval_1p7b_base.json > runs/eval_1p7b_base.log 2>&1
echo "== [$(date)] EVAL base exited rc=$?"
python scripts/behavior_report.py runs/qwen3_1p7b_math_opd runs/qwen3_1p7b_math_eopd --out runs/report_1p7b.md \
  > runs/report_1p7b.log 2>&1
echo "== [$(date)] POST_DONE"
