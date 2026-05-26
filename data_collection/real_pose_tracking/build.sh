#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "[INFO] Source directory: ${SCRIPT_DIR}"
echo "[INFO] Build directory: ${BUILD_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j

echo "[DONE] Build finished."
