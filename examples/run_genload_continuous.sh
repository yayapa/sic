#!/bin/bash
# Generate continuous (Reg2Surv) synthetic prior batches.
#
# Usage:
#   1. Edit DATA_ROOT below (or export it before running).
#   2. Activate your sic conda environment.
#   3. bash examples/run_genload_continuous.sh
#
# Prior hyperparameters: configs/gen/reg2surv_params.yaml
# 5300 batches x 512 datasets is the size used for the released checkpoints;
# lower --num_batches for a quick test.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
# activate your sic conda environment

python -m sic.prior.genload \
    --save_dir "${DATA_ROOT}/prior_continuous" \
    --reg2surv_config "${REPO_ROOT}/configs/gen/reg2surv_params.yaml" \
    --np_seed 42 \
    --torch_seed 42 \
    --num_batches 10 \
    --resume_from 0 \
    --batch_size 512 \
    --batch_size_per_gp 1 \
    --prior_type mix_scm \
    --min_features 2 \
    --max_features 100 \
    --max_classes 10 \
    --max_seq_len 1024 \
    --min_train_size 0.1 \
    --max_train_size 0.9 \
    --n_jobs 1 \
    --num_threads_per_generate 1 \
    --device cuda
