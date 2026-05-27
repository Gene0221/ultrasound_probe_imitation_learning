#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config/move_ee_local_linear.yaml}"

CANDIDATE_EXECUTABLES=(
  "${SCRIPT_DIR}/build/move_ee_local_linear"
  "${SCRIPT_DIR}/build/Release/move_ee_local_linear"
  "${SCRIPT_DIR}/src/move_ee_local_linear"
)

BINARY_PATH=""
for candidate in "${CANDIDATE_EXECUTABLES[@]}"; do
  if [[ -f "${candidate}" ]]; then
    BINARY_PATH="${candidate}"
    break
  fi
done

if [[ -z "${BINARY_PATH}" ]]; then
  echo "[ERROR] Cannot find move_ee_local_linear binary. Checked build/, build/Release/, and src/." >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Cannot find config file: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[INFO] Launching: ${BINARY_PATH}"
echo "[INFO] Config: ${CONFIG_PATH}"

exec "${BINARY_PATH}" "${CONFIG_PATH}"
