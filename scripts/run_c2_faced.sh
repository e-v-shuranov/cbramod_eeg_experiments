#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_BIN="${CONDA_BIN:-/home/eshuranov/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-cbramod}"
CUDA_ID="${CUDA_ID:-0}"

OUTPUT_DIR="${PROJECT_ROOT}/results/channel"
OUTPUT_CSV="${OUTPUT_DIR}/c2_joint_permutation.csv"
LOG_FILE="${OUTPUT_DIR}/c2_FACED.log"
mkdir -p "${OUTPUT_DIR}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output \
  python -m experiments.channel_c2_joint_perm \
  --model CBraMod \
  --dataset FACED \
  --datasets-dir /media/public/Datasets/cbramod_data/FACED/processed/ \
  --checkpoint /media/public/ckpts/CBR_chkpnts_for_shufle_track/FACED_baseline/epoch38_acc_0.56347_kappa_0.50726_f1_0.56972.pth \
  --num-of-classes 9 \
  --n-channels 32 \
  --split test \
  --perm-seeds 0,1,2,3,4 \
  --seed 0 \
  --cuda "${CUDA_ID}" \
  --batch-size 64 \
  --num-workers 16 \
  --classifier all_patch_reps \
  --output "${OUTPUT_CSV}" \
  2>&1 | tee "${LOG_FILE}"
