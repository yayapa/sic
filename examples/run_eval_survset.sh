#!/bin/bash
# Evaluate SIC on the whole benchmark (sic.list_datasets()): all SurvSet datasets + METABRIC.
# SEER / UNOS are added only if their CSVs (seernfg.csv, unos.csv) are in DATA_DIR;
# they are restricted-access and not distributed with SIC.
#
# Usage:
#   1. pip install -e ".[survset]"
#   2. Optionally set MODEL_PATH (default: download the released SuMoPP weights)
#      and DATA_DIR (folder with the restricted CSVs).
#   3. bash examples/run_eval_survset.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/eval_results}"
HEAD="${HEAD:-SuMoPP}"

# activate your sic conda environment

python "${REPO_ROOT}/examples/run_eval_survset.py" \
    --survival_head "${HEAD}" \
    ${MODEL_PATH:+--model_path "${MODEL_PATH}"} \
    ${DATA_DIR:+--data_dir "${DATA_DIR}"} \
    --k_folds 5 \
    --seed 42 \
    --out "${OUT_DIR}/eval_survset_${HEAD}.xlsx" \
    --json-out "${OUT_DIR}/eval_survset_${HEAD}.json"
