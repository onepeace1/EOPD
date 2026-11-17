#!/usr/bin/env bash
# DAPO RL baseline: Qwen3-4B-Base on MATH train, 4 GPUs, 50 steps (~10h budget).
#
#   setsid nohup bash run_dapo_4b.sh > runs/train_dapo_4b.log 2>&1 < /dev/null & disown
#
# DAPO recipe (recipe/dapo): GRPO advantages, no KL, clip-higher (0.2 / 0.28), token-mean loss,
# dynamic sampling (drop prompts whose 8 samples are all right or all wrong), overlong soft penalty.
# Scaled down from the paper (512 prompts x 16 samples, 20k tokens) to fit 50 steps on 4 GPUs.
#
# Batch math:
#   train batch  128 prompts x 8 samples = 1,024 responses per update step (4 mini-batches of 32 prompts)
#   gen batch    250 prompts drawn per generation round; groups with zero reward variance are dropped,
#                and another round of 250 is generated if fewer than 128 prompts survive
#   50 steps  -> 6,400 trained prompts; >= 12,500 prompts drawn (>= 1.67 epochs of the 7,500)
# Difficulty coverage: the sampler shuffles without replacement and 250 divides 7,500, so epoch 1 is
# exactly 30 generation rounds -> every problem, incl. all 2,304 Level-5 ones, is drawn by step 30.
# (A Level-5 problem the model gets 0/8 on is drawn but yields no gradient - that is DAPO by design.)
#
# Env overrides: GPUS (default 0,2,5,6), MODEL, STEPS, SAVE_FREQ, TEST_FREQ.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
source "$HOME/Rethinking-OPD/.venv/bin/activate"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=${GPUS:-0,2,5,6}
NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
MODEL=${MODEL:-Qwen/Qwen3-4B-Base}
STEPS=${STEPS:-50}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-10}

EXP=qwen3_4b_math_dapo
RUN_DIR=$ROOT/runs/$EXP
mkdir -p "$RUN_DIR"
# verl's file logger opens metrics.jsonl with "w": keep the previous one on a resume
[ -s "$RUN_DIR/metrics.jsonl" ] && mv "$RUN_DIR/metrics.jsonl" "$RUN_DIR/metrics.$(date +%s).jsonl"

export RAY_TMPDIR=/tmp/ray_dapo_gpu${CUDA_VISIBLE_DEVICES//,/}_$(date +%s)
export RAY_USAGE_STATS_ENABLED=0 RAY_memory_usage_threshold=0.99
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true HYDRA_FULL_ERROR=1 NCCL_DEBUG=WARN
export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.jsonl
export TENSORBOARD_DIR=$RUN_DIR/tb

MAX_PROMPT=2048   # longest MATH-train prompt is 1,681 tokens: keep all 7,500 so 250 | 7,500
MAX_RESP=4096

python3 -m recipe.dapo.main_dapo ${DRY:+--cfg job} \
  data.train_files="$ROOT/data/math/train.parquet" \
  data.val_files="$HOME/Rethinking-OPD/datasets/test_data/MATH-500/test.parquet" \
  data.prompt_key=prompt \
  data.train_batch_size=128 \
  data.gen_batch_size=250 \
  data.max_prompt_length=$MAX_PROMPT \
  data.max_response_length=$MAX_RESP \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.metric=acc \
  algorithm.filter_groups.max_num_gen_batches=4 \
  actor_rollout_ref.model.path=$MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT + MAX_RESP)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
  reward_model.reward_manager=dapo \
  reward_model.overlong_buffer.enable=True \
  reward_model.overlong_buffer.len=1024 \
  reward_model.overlong_buffer.penalty_factor=1.0 \
  reward_model.overlong_buffer.log=True \
  custom_reward_function.path="$ROOT/data/reward.py" \
  custom_reward_function.name=compute_score \
  trainer.logger="['console','file','tensorboard']" \
  trainer.project_name=EOPD_repro \
  trainer.experiment_name=$EXP \
  trainer.default_local_dir="$RUN_DIR/ckpt" \
  trainer.n_gpus_per_node=$NGPU \
  trainer.nnodes=1 \
  trainer.total_epochs=10 \
  trainer.total_training_steps=$STEPS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.val_before_train=True \
  trainer.log_val_generations=2 \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  trainer.validation_data_dir="$RUN_DIR/val_generations" \
  trainer.resume_mode=auto \
  "$@"
