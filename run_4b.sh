#!/usr/bin/env bash
# OPD / EOPD with a Qwen3-4B-Base student <- Qwen3-8B (non-thinking) teacher on DAPO-Math-17k, 4 GPUs.
#
#   bash run_4b.sh opd     # naive on-policy distillation: reverse KL only
#   bash run_4b.sh eopd    # EOPD: + forward KL (teacher top-16) where teacher entropy >= 0.8
#
# Same objective/hyper-parameters as run_1p7b.sh (paper Table 9: batch 128, mini-batch 32, 1 sample/prompt,
# lr 3e-6 cosine, T=1.0, top-p 1.0, max response 4096, top-k 16). Differences:
#   data   data/dapo17k/train.parquet from data/prep_dapo_4b.py: 50 x 128 = 6,400 prompts, kept IN ORDER
#          (data.shuffle=False) so every step's batch holds the same hard/medium/easy quota
#   steps  STEPS (50) instead of 3 epochs
#   val    every TEST_FREQ steps, 4 samples per question, on the SAME fixed questions:
#          MATH-500, AIME24, and DAPO-probe (60 held-out DAPO problems, 20 per difficulty bin)
#          -> scripts/behavior_report.py tracks per-question changes across steps
# Env overrides: GPUS (default 0,2,5,6), TEACHER, STUDENT, STEPS, SAVE_FREQ, TEST_FREQ, EARLY_TOKENS.
set -euo pipefail
MODE=${1:?usage: run_4b.sh opd|eopd}
case "$MODE" in
  opd)  SOFT_KD_COEF=0.0 ;;
  eopd) SOFT_KD_COEF=1.0 ;;
  *) echo "MODE must be opd or eopd"; exit 1 ;;
esac

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
source "$HOME/Rethinking-OPD/.venv/bin/activate"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=${GPUS:-0,2,5,6}
NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
STUDENT=${STUDENT:-Qwen/Qwen3-4B-Base}
TEACHER=${TEACHER:-Qwen/Qwen3-8B}
STEPS=${STEPS:-50}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-10}
TB=$HOME/Rethinking-OPD/datasets/test_data

EXP=qwen3_4b_dapo17k_${MODE}
RUN_DIR=$ROOT/runs/$EXP
mkdir -p "$RUN_DIR"
# verl's file logger opens metrics.jsonl with "w": keep the previous one on a resume
[ -s "$RUN_DIR/metrics.jsonl" ] && mv "$RUN_DIR/metrics.jsonl" "$RUN_DIR/metrics.$(date +%s).jsonl"

export RAY_TMPDIR=/tmp/ray_4b_${MODE}_$(date +%s)
export RAY_USAGE_STATS_ENABLED=0 RAY_memory_usage_threshold=0.99
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true HYDRA_FULL_ERROR=1 NCCL_DEBUG=WARN
export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.jsonl
export TENSORBOARD_DIR=$RUN_DIR/tb
export VERL_ON_POLICY_SOFT_KD_DEBUG_DIR=$RUN_DIR/high_entropy_tokens

MAX_PROMPT=2048
MAX_RESP=4096

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=on_policy \
  algorithm.use_kl_in_reward=False \
  +trainer.trainer_class=OnPolicyDistillTrainer \
  data.train_files="$ROOT/data/dapo17k/train.parquet" \
  data.val_files="['$TB/MATH-500/test.parquet','$TB/AIME24/test.parquet','$ROOT/data/dapo17k/probe.parquet']" \
  data.shuffle=False \
  data.train_batch_size=128 \
  data.max_prompt_length=$MAX_PROMPT \
  data.max_response_length=$MAX_RESP \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path=$STUDENT \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.teacher_model.path=$TEACHER \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.policy_loss.loss_mode=on_policy_distill \
  +actor_rollout_ref.actor.policy_loss.soft_kd_student_full_vocab=True \
  +actor_rollout_ref.actor.soft_kd_loss_coef=$SOFT_KD_COEF \
  +actor_rollout_ref.actor.soft_kd_entropy_threshold=0.8 \
  +actor_rollout_ref.actor.high_entropy_threshold=0.8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
  +actor_rollout_ref.ref.topk_logits=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT + MAX_RESP)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
  custom_reward_function.path="$ROOT/data/reward.py" \
  custom_reward_function.name=compute_score \
  trainer.logger="['console','file','tensorboard']" \
  trainer.project_name=EOPD_repro \
  trainer.experiment_name=$EXP \
  trainer.default_local_dir="$RUN_DIR/ckpt" \
  trainer.n_gpus_per_node=$NGPU \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=$STEPS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.val_before_train=True \
  trainer.log_val_generations=2 \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  +trainer.rollout_early_tokens=${EARLY_TOKENS:-64} \
  trainer.validation_data_dir="$RUN_DIR/val_generations" \
  trainer.resume_mode=auto \
  "${@:2}"
