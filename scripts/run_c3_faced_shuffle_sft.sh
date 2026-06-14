#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_BIN="${CONDA_BIN:-/home/eshuranov/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-cbramod}"
CUDA_ID="${CUDA_ID:-0}"

# Conservative first run. Full run: PERM_SEEDS="0,1,2,3,4" bash scripts/run_c3_faced_shuffle_sft.sh
PERM_SEEDS="${PERM_SEEDS:-0}"
PERM_SEEDS="${PERM_SEEDS//,/ }"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-16}"

DATASETS_DIR="${DATASETS_DIR:-/media/public/Datasets/cbramod_data/FACED/processed/}"
BASELINE_CKPT="${BASELINE_CKPT:-/media/public/ckpts/CBR_chkpnts_for_shufle_track/FACED_baseline/epoch38_acc_0.56347_kappa_0.50726_f1_0.56972.pth}"
MODEL_ROOT="${MODEL_ROOT:-/media/public/ckpts/CBR_chkpnts_for_shufle_track/FACED_c3_shuffle_sft}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/results/channel/c3_faced_shuffle_sft_$(date +%Y%m%d_%H%M%S)}"
RESULT_CSV="${RESULT_CSV:-${PROJECT_ROOT}/results/channel/c3_shuffle_sft_recovery.csv}"
CHANNEL_NAMES_FILE="${CHANNEL_NAMES_FILE:-${PROJECT_ROOT}/configs/channel_names/FACED.txt}"
SHUFFLE_PLAN_CSV="${RUN_DIR}/shuffle_plan.csv"
CHANNEL_NAMES_ARG=()
if [[ -n "${CHANNEL_NAMES_FILE:-}" ]]; then
  CHANNEL_NAMES_ARG=(--channel-names "${CHANNEL_NAMES_FILE}")
fi

mkdir -p "${RUN_DIR}" "$(dirname "${RESULT_CSV}")"
LOG_FILE="${RUN_DIR}/run.log"
STATUS_FILE="${RUN_DIR}/status.txt"

exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

write_status() {
  printf '%s\n' "$*" > "${STATUS_FILE}"
}

make_perm() {
  local seed="$1"
  "${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output \
    python -c "from experiments.channel_permutation import make_permutation; print(make_permutation(32, ${seed}))"
}

latest_checkpoint() {
  local model_dir="$1"
  find "${model_dir}" -maxdepth 1 -type f -name '*.pth' -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

estimate_remaining() {
  local started_at="$1"
  local done_count="$2"
  local total_count="$3"
  if [[ "${done_count}" -eq 0 ]]; then
    echo "unknown"
    return
  fi
  local now elapsed avg remaining
  now="$(date +%s)"
  elapsed=$((now - started_at))
  avg=$((elapsed / done_count))
  remaining=$(((total_count - done_count) * avg))
  printf '%02dh:%02dm:%02ds' $((remaining / 3600)) $(((remaining % 3600) / 60)) $((remaining % 60))
}

IFS=' ' read -r -a SEEDS <<< "${PERM_SEEDS}"
TOTAL="${#SEEDS[@]}"
STARTED_AT="$(date +%s)"

log "Starting FACED C3 shuffle-then-SFT diagnostic"
log "Project root: ${PROJECT_ROOT}"
log "Run dir: ${RUN_DIR}"
log "Result CSV: ${RESULT_CSV}"
log "Baseline checkpoint: ${BASELINE_CKPT}"
log "Seeds: ${PERM_SEEDS}"
log "CUDA_ID=${CUDA_ID}, EPOCHS=${EPOCHS}, BATCH_SIZE=${BATCH_SIZE}"
write_status "started: 0/${TOTAL} seeds complete; current=initializing; eta=unknown; log=${LOG_FILE}"

rm -f "${RESULT_CSV}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output \
  python -m experiments.export_channel_shuffle_plan \
  --dataset FACED \
  --n-channels 32 \
  --channel-names "${CHANNEL_NAMES_FILE}" \
  --perm-seeds "${PERM_SEEDS// /,}" \
  --output "${SHUFFLE_PLAN_CSV}"
log "Shuffle plan with indices and names: ${SHUFFLE_PLAN_CSV}"

DONE=0
for PERM_SEED in "${SEEDS[@]}"; do
  PERM="$(make_perm "${PERM_SEED}")"
  MODEL_DIR="${MODEL_ROOT}/perm_seed_${PERM_SEED}"

  log "Seed ${PERM_SEED}: C3 corrupted assignment permutation=${PERM}"
  log "Seed ${PERM_SEED}: training shuffled-SFT checkpoint into ${MODEL_DIR}"
  mkdir -p "${MODEL_DIR}"

  ETA="$(estimate_remaining "${STARTED_AT}" "${DONE}" "${TOTAL}")"
  write_status "running: ${DONE}/${TOTAL} seeds complete; current=train shuffled-SFT perm_seed=${PERM_SEED}; eta=${ETA}; log=${LOG_FILE}"

  "${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output \
    python finetune_main.py \
      --cuda "${CUDA_ID}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr 0.0001 \
      --weight_decay 0.05 \
      --optimizer AdamW \
      --clip_value 1 \
      --dropout 0.1 \
      --classifier all_patch_reps \
      --downstream_dataset FACED \
      --datasets_dir "${DATASETS_DIR}" \
      --num_of_classes 9 \
      --model_dir "${MODEL_DIR}" \
      --num_workers "${NUM_WORKERS}" \
      --label_smoothing 0.1 \
      --multi_lr True \
      --use_pretrained_weights True \
      --use_scheduler False \
      --is_chanle_shafle True \
      --new_order "${PERM}"

  SFT_CKPT="$(latest_checkpoint "${MODEL_DIR}")"
  if [[ -z "${SFT_CKPT}" ]]; then
    log "Seed ${PERM_SEED}: no checkpoint found in ${MODEL_DIR}"
    exit 1
  fi

  log "Seed ${PERM_SEED}: evaluating baseline/immediate-drop/recovery with ${SFT_CKPT}"
  write_status "running: ${DONE}/${TOTAL} seeds complete; current=evaluate C3 perm_seed=${PERM_SEED}; eta=${ETA}; log=${LOG_FILE}"

  "${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output \
    python -m experiments.channel_c3_shuffle_sft \
      --model CBraMod \
      --dataset FACED \
      --datasets-dir "${DATASETS_DIR}" \
      --baseline-checkpoint "${BASELINE_CKPT}" \
      --shuffled-sft-checkpoint "${SFT_CKPT}" \
      --num-of-classes 9 \
      --n-channels 32 \
      "${CHANNEL_NAMES_ARG[@]}" \
      --split test \
      --perm-seeds "${PERM_SEED}" \
      --seed 0 \
      --cuda "${CUDA_ID}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --classifier all_patch_reps \
      --finetune-epochs "${EPOCHS}" \
      --output "${RESULT_CSV}" \
      --append

  DONE=$((DONE + 1))
  ETA="$(estimate_remaining "${STARTED_AT}" "${DONE}" "${TOTAL}")"
  log "Seed ${PERM_SEED}: complete (${DONE}/${TOTAL}); estimated remaining ${ETA}"
  write_status "running: ${DONE}/${TOTAL} seeds complete; current=between-seeds; eta=${ETA}; log=${LOG_FILE}; result=${RESULT_CSV}"
done

log "FACED C3 shuffle-then-SFT diagnostic complete"
write_status "complete: ${DONE}/${TOTAL} seeds complete; eta=00h:00m:00s; log=${LOG_FILE}; result=${RESULT_CSV}"
