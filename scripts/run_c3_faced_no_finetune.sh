#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_BIN="${CONDA_BIN:-/home/eshuranov/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-cbramod}"
CUDA_ID="${CUDA_ID:-0}"

PERM_SEEDS="${PERM_SEEDS:-0,1,2,3,4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-16}"

DATASETS_DIR="${DATASETS_DIR:-/media/public/Datasets/cbramod_data/FACED/processed/}"
BASELINE_CKPT="${BASELINE_CKPT:-/media/public/ckpts/CBR_chkpnts_for_shufle_track/FACED_baseline/epoch38_acc_0.56347_kappa_0.50726_f1_0.56972.pth}"
CHANNEL_NAMES_FILE="${CHANNEL_NAMES_FILE:-${PROJECT_ROOT}/configs/channel_names/FACED.txt}"
OUTPUT_DIR="${PROJECT_ROOT}/results/channel"
RESULT_CSV="${OUTPUT_DIR}/c3_no_finetune_immediate_drop.csv"
SHUFFLE_PLAN_CSV="${OUTPUT_DIR}/c3_no_finetune_FACED_shuffle_plan.csv"
LOG_FILE="${OUTPUT_DIR}/c3_no_finetune_FACED.log"

mkdir -p "${OUTPUT_DIR}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output \
  python -m experiments.export_channel_shuffle_plan \
  --dataset FACED \
  --n-channels 32 \
  --channel-names "${CHANNEL_NAMES_FILE}" \
  --perm-seeds "${PERM_SEEDS}" \
  --output "${SHUFFLE_PLAN_CSV}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output \
  python -m experiments.channel_c3_shuffle_sft \
  --model CBraMod \
  --dataset FACED \
  --datasets-dir "${DATASETS_DIR}" \
  --baseline-checkpoint "${BASELINE_CKPT}" \
  --shuffled-sft-checkpoint "${BASELINE_CKPT}" \
  --num-of-classes 9 \
  --n-channels 32 \
  --channel-names "${CHANNEL_NAMES_FILE}" \
  --split test \
  --perm-seeds "${PERM_SEEDS}" \
  --seed 0 \
  --cuda "${CUDA_ID}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --classifier all_patch_reps \
  --finetune-epochs 0 \
  --output "${RESULT_CSV}" \
  2>&1 | tee "${LOG_FILE}"
