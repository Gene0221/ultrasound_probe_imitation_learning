#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config/default.yaml"

CANDIDATE_EXECUTABLES=(
  "${SCRIPT_DIR}/build/read_franka_ee_pose"
  "${SCRIPT_DIR}/build/Release/read_franka_ee_pose"
  "${SCRIPT_DIR}/src/read_franka_ee_pose"
)

BINARY_PATH=""
for candidate in "${CANDIDATE_EXECUTABLES[@]}"; do
  if [[ -f "${candidate}" ]]; then
    BINARY_PATH="${candidate}"
    break
  fi
done

if [[ -z "${BINARY_PATH}" ]]; then
  echo "[ERROR] Cannot find read_franka_ee_pose binary. Checked build/, build/Release/, and src/." >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Cannot find default config file: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[INFO] Launching: ${BINARY_PATH}"
echo "[INFO] Config: ${CONFIG_PATH}"

exec "${BINARY_PATH}" "${CONFIG_PATH}"
