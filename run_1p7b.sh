#!/usr/bin/env bash
# EOPD paper reproduction, Qwen3-1.7B-Base student <- Qwen3-8B (non-thinking) teacher, MATH train, 1 GPU.
#
#   bash run_1p7b.sh opd    # naive on-policy distillation: reverse KL only (soft_kd_loss_coef=0)
#   bash run_1p7b.sh eopd   # EOPD: reverse KL + forward KL (teacher top-16) where teacher entropy >= 0.8
#
# Hyperparameters follow the paper's Table 9 (OPD/EOPD): batch 128, mini-batch 32 (4 updates/step),
# 1 sample/prompt, lr 3e-6 cosine, temp 1.0, top-p 1.0, max response 4096, top-k 16, 3 epochs on MATH.
# Env overrides: GPU (default 3), TEACHER, STUDENT, EPOCHS, SAVE_FREQ, TEST_FREQ, EARLY_TOKENS.
#
# Saved per run under runs/qwen3_1p7b_math_<mode>/:
#   rollouts/<step>.jsonl         every training step: all 128 prompts + responses + MATH score, and for the
#                                 first EARLY_TOKENS (64) response tokens: token, p_student, p_teacher,
#                                 teacher entropy, teacher top-5 alternatives
#   val_generations/<step>.jsonl  MATH-500 validation responses + scores (every TEST_FREQ steps)
#   metrics.jsonl, tb/            scalar metrics;  ckpt/global_step_*/actor/huggingface  HF weights
set -euo pipefail
MODE=${1:?usage: run_1p7b.sh opd|eopd}
case "$MODE" in
  opd)  SOFT_KD_COEF=0.0 ;;
  eopd) SOFT_KD_COEF=1.0 ;;
  *) echo "MODE must be opd or eopd"; exit 1 ;;
esac

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
source "$HOME/Rethinking-OPD/.venv/bin/activate"   # torch 2.8 / vllm 0.11 env, already working on Blackwell
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"          # use EOPD's verl fork, not the venv's verl

export CUDA_VISIBLE_DEVICES=${GPU:-3}
STUDENT=${STUDENT:-Qwen/Qwen3-1.7B-Base}
TEACHER=${TEACHER:-Qwen/Qwen3-8B}
EPOCHS=${EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-58}     # 7500 // 128 = 58 steps per epoch -> one checkpoint per epoch
TEST_FREQ=${TEST_FREQ:-29}

EXP=qwen3_1p7b_math_${MODE}
RUN_DIR=$ROOT/runs/$EXP
mkdir -p "$RUN_DIR"

# fresh Ray dir per launch so a dying earlier launch can't take this one down
export RAY_TMPDIR=/tmp/ray_eopd_gpu${CUDA_VISIBLE_DEVICES}_${MODE}_$(date +%s)
export RAY_USAGE_STATS_ENABLED=0 RAY_memory_usage_threshold=0.99
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true HYDRA_FULL_ERROR=1 NCCL_DEBUG=WARN
export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.jsonl
export TENSORBOARD_DIR=$RUN_DIR/tb
export VERL_ON_POLICY_SOFT_KD_DEBUG_DIR=$RUN_DIR/high_entropy_tokens

MAX_PROMPT=1024
MAX_RESP=4096

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=on_policy \
  algorithm.use_kl_in_reward=False \
  +trainer.trainer_class=OnPolicyDistillTrainer \
  data.train_files="$ROOT/data/math/train.parquet" \
  data.val_files="$HOME/Rethinking-OPD/datasets/test_data/MATH-500/test.parquet" \
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
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144 \
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
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
  custom_reward_function.path="$ROOT/data/reward.py" \
  custom_reward_function.name=compute_score \
  trainer.logger="['console','file','tensorboard']" \
  trainer.project_name=EOPD_repro \
  trainer.experiment_name=$EXP \
  trainer.default_local_dir="$RUN_DIR/ckpt" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=$EPOCHS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.val_before_train=True \
  trainer.log_val_generations=2 \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  +trainer.rollout_early_tokens=${EARLY_TOKENS:-64} \
  trainer.validation_data_dir="$RUN_DIR/val_generations" \
  trainer.resume_mode=auto \
  "${@:2}"
