#!/bin/bash
# Train the DeepHit10 head on a 50/50 mix of discrete and continuous synthetic priors.
#
# Usage:
#   1. Edit DATA_ROOT, CHECKPOINT_ROOT, and CHECKPOINT_PATH below (or export them).
#   2. Run examples/run_genload_continuous.sh and examples/run_genload_discrete.sh first.
#   3. Activate your sic conda environment.
#   4. bash examples/run_train_deephit10.sh
#
# DeepHit with 10 time bins placed at quantiles of each task's training durations.
# Loss: DeepHit likelihood + alpha * ranking loss (alpha 0.2, sigma 0.1).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints}"
# TabICL classifier backbone used to initialize the ICL predictor (edit path):
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/tabicl-classifier-v1.1-0506.ckpt}"

# activate your sic conda environment

python -m sic.train.run \
    --disable_ddp True \
    --wandb_log True \
    --wandb_project SIC \
    --wandb_name SIC_mix_DeepHit10 \
    --wandb_mode online \
    --device cuda \
    --dtype float32 \
    --np_seed 42 \
    --torch_seed 42 \
    --max_steps 10000 \
    --batch_size 512 \
    --micro_batch_size 8 \
    --lr 1e-4 \
    --scheduler cosine_warmup \
    --warmup_proportion 0.02 \
    --gradient_clipping 1.0 \
    --prior_dirs "${DATA_ROOT}/prior_discrete" "${DATA_ROOT}/prior_continuous" \
    --prior_mix_probs 0.5 0.5 \
    --load_prior_start 0 \
    --delete_after_load False \
    --prior_device cpu \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --norm_first True \
    --only_load_model True \
    --checkpoint_dir "${CHECKPOINT_ROOT}/DeepHit10_mix" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --save_temp_every 1000 \
    --save_perm_every 1000 \
    --surv_model_type deephit \
    --surv_loss deephit \
    --num_bins 10 \
    --time_binning_scheme quantiles \
    --deephit_loss_alpha 0.2 \
    --deephit_loss_sigma 0.1
