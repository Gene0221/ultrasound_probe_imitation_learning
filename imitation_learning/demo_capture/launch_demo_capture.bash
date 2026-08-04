#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  --robot-ip <ip>          Override franka.robot_ip from YAML.
  --config <file>          YAML config. Default: ../config/infer.yaml.
  --trial-id <name>        Trial output folder name.
  --output-root <dir>      Output root. Default: demo_capture/trials.
  --image-dir <dir>        Override config and use an image-folder source.
  --save-every <n>         Save one frame every n inference requests.
  --capture-settle-s <sec> Wait before each snapshot. Default: 0.15.
  --flush-frames <n>       Discard live-camera frames before each snapshot. Default: 8.
  --build                  Build the C++ controller before launching.
  --print-only             Print commands without executing them.
  -h, --help               Show this help.

Example:
  $(basename "$0") --build --trial-id trial_001
EOF
}

CONFIG_FILE="${PROJECT_ROOT}/config/infer.yaml"
ROBOT_IP="${ROBOT_IP:-}"
TRIAL_ID=""
OUTPUT_ROOT="${PROJECT_ROOT}/demo_capture/trials"
IMAGE_DIR=""
SAVE_EVERY="1"
CAPTURE_SETTLE_S="0.15"
FLUSH_FRAMES="8"
DO_BUILD="0"
PRINT_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-ip)
      ROBOT_IP="${2:?Missing value for --robot-ip}"
      shift 2
      ;;
    --config)
      CONFIG_FILE="${2:?Missing value for --config}"
      shift 2
      ;;
    --trial-id)
      TRIAL_ID="${2:?Missing value for --trial-id}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?Missing value for --output-root}"
      shift 2
      ;;
    --image-dir)
      IMAGE_DIR="${2:?Missing value for --image-dir}"
      shift 2
      ;;
    --save-every)
      SAVE_EVERY="${2:?Missing value for --save-every}"
      shift 2
      ;;
    --capture-settle-s)
      CAPTURE_SETTLE_S="${2:?Missing value for --capture-settle-s}"
      shift 2
      ;;
    --flush-frames)
      FLUSH_FRAMES="${2:?Missing value for --flush-frames}"
      shift 2
      ;;
    --build)
      DO_BUILD="1"
      shift
      ;;
    --print-only)
      PRINT_ONLY="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${CONFIG_FILE}" != /* ]]; then
  CONFIG_FILE="${PWD}/${CONFIG_FILE}"
fi

read_yaml_value() {
  "${PYTHON:-python}" - "$1" "$2" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1]).resolve()
keys = sys.argv[2].split(".")
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
value = data
for key in keys:
    if not isinstance(value, dict) or key not in value:
        raise SystemExit(1)
    value = value[key]
print(value)
PY
}

resolve_config_path() {
  "${PYTHON:-python}" - "${PROJECT_ROOT}" "$1" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
value = Path(sys.argv[2])
print(value.resolve() if value.is_absolute() else (root / value).resolve())
PY
}

START_SIGNAL_VALUE="$(read_yaml_value "${CONFIG_FILE}" runtime.start_signal_file 2>/dev/null || true)"
if [[ -z "${START_SIGNAL_VALUE}" ]]; then
  START_SIGNAL_VALUE="runtime/start_policy_stream.flag"
fi
START_SIGNAL_FILE="$(resolve_config_path "${START_SIGNAL_VALUE}")"

CONTROLLER_BINARY="${PROJECT_ROOT}/cpp_controller/build/rolling_policy_controller"
CONTROLLER_LAUNCHER_CMD=(
  "${PYTHON:-python}"
  "${PROJECT_ROOT}/cpp_controller/launch_controller.py"
  --config "${CONFIG_FILE}"
  --binary "${CONTROLLER_BINARY}"
  --print-only
)
if [[ -n "${ROBOT_IP}" ]]; then
  CONTROLLER_LAUNCHER_CMD+=(--robot-ip "${ROBOT_IP}")
fi

CALIBRATION_INITIALIZER_CMD=(
  "${PYTHON:-python}"
  "${PROJECT_ROOT}/scripts/initialize_calibration.py"
  --config "${CONFIG_FILE}"
)

SENDER_CMD=(
  "${PYTHON:-python}"
  "${SCRIPT_DIR}/online_demo_capture.py"
  --config "${CONFIG_FILE}"
  --output-root "${OUTPUT_ROOT}"
  --save-every "${SAVE_EVERY}"
  --capture-settle-s "${CAPTURE_SETTLE_S}"
  --flush-frames "${FLUSH_FRAMES}"
  --wait-for-start
  --start-signal-file "${START_SIGNAL_FILE}"
)
if [[ -n "${TRIAL_ID}" ]]; then
  SENDER_CMD+=(--trial-id "${TRIAL_ID}")
fi
if [[ -n "${IMAGE_DIR}" ]]; then
  SENDER_CMD+=(--image-dir "${IMAGE_DIR}")
fi

if [[ "${PRINT_ONLY}" == "1" ]]; then
  printf '[INFO] Calibration initializer command:'
  printf ' %q' "${CALIBRATION_INITIALIZER_CMD[@]}"
  printf '\n'
  echo "[INFO] Controller command:"
  "${CONTROLLER_LAUNCHER_CMD[@]}"
  printf '[INFO] Demo sender command:'
  printf ' %q' "${SENDER_CMD[@]}"
  printf '\n'
  exit 0
fi

SENDER_PID=""
CONTROLLER_PID=""

cleanup() {
  if [[ -n "${SENDER_PID}" ]] && kill -0 "${SENDER_PID}" 2>/dev/null; then
    echo "[INFO] Stopping demo capture sender..."
    kill "${SENDER_PID}" 2>/dev/null || true
    wait "${SENDER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${CONTROLLER_PID}" ]] && kill -0 "${CONTROLLER_PID}" 2>/dev/null; then
    echo "[INFO] Stopping C++ controller..."
    kill "${CONTROLLER_PID}" 2>/dev/null || true
    wait "${CONTROLLER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[1/7] Config: ${CONFIG_FILE}"
echo "[1/7] Start signal file: ${START_SIGNAL_FILE}"
rm -f "${START_SIGNAL_FILE}"

echo "[2/7] Running guided calibration initialization..."
"${CALIBRATION_INITIALIZER_CMD[@]}"

echo "[3/7] Starting Python demo capture sender..."
"${SENDER_CMD[@]}" &
SENDER_PID="$!"

sleep 1
if ! kill -0 "${SENDER_PID}" 2>/dev/null; then
  echo "[ERROR] Python demo capture sender exited during startup." >&2
  wait "${SENDER_PID}" || true
  exit 1
fi

echo "[4/7] Starting C++ controller..."
if [[ "${DO_BUILD}" == "1" ]]; then
  "${PROJECT_ROOT}/cpp_controller/build.bash"
fi
COMMAND_TEXT="$("${CONTROLLER_LAUNCHER_CMD[@]}")"
CONTROLLER_CMD=()
while IFS= read -r token; do
  CONTROLLER_CMD+=("${token}")
done < <("${PYTHON:-python}" - "${COMMAND_TEXT}" <<'PY'
import shlex
import sys
for item in shlex.split(sys.argv[1]):
    print(item)
PY
)
"${CONTROLLER_CMD[@]}" &
CONTROLLER_PID="$!"

sleep 1
if ! kill -0 "${CONTROLLER_PID}" 2>/dev/null; then
  echo "[ERROR] C++ controller exited during startup." >&2
  wait "${CONTROLLER_PID}" || true
  exit 1
fi

echo "[5/7] Wait for these records:"
echo "      - Python: Policy loaded and ultrasound stream ready"
echo "      - Python: Calibration force reference captured"
echo "      - C++: Initial EE orientation captured"
echo "      - C++: Python policy client connected"
echo
read -r -p "After force/contact and initial pose are ready, press Enter to save the initial frame and start inference..."
mkdir -p "$(dirname "${START_SIGNAL_FILE}")"
touch "${START_SIGNAL_FILE}"
echo "[6/7] Start signal sent. The sender will save initial_frame.png before sending ready."
echo "[7/7] Stop manually with Ctrl+C; the sender will save final_frame.png."

wait "${SENDER_PID}"
