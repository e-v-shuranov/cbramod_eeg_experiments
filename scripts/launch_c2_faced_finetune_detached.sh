#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/results/channel/c2_faced_finetune_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_DIR}"

export RUN_DIR

nohup bash "${PROJECT_ROOT}/scripts/run_c2_faced_finetune.sh" \
  > "${RUN_DIR}/nohup.out" 2>&1 &

PID="$!"
echo "${PID}" > "${RUN_DIR}/pid.txt"

cat <<EOF
Started FACED C2 fine-tune diagnostic in background.
PID: ${PID}
Run dir: ${RUN_DIR}
Status: ${RUN_DIR}/status.txt
Log: ${RUN_DIR}/run.log
Nohup: ${RUN_DIR}/nohup.out

Watch progress:
  tail -f "${RUN_DIR}/run.log"
  cat "${RUN_DIR}/status.txt"
EOF
