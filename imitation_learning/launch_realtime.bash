#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  --robot-ip <ip>        Override franka.robot_ip from YAML.
  --config <file>        YAML config. Default: config/infer.yaml.
  --build                Build the C++ controller before launching.
  --image-dir <dir>      Override config and use an image-folder source.
  --controller-only      Debug: launch only the C++ controller.
  --sender-only          Debug: launch only the Python inference sender.
  --print-only           Print commands without executing them.
  -h, --help             Show this help.

Examples:
  $(basename "$0") --build
  $(basename "$0") --image-dir /path/to/images
  $(basename "$0") --sender-only --image-dir /path/to/images
EOF
}

CONFIG_FILE="${SCRIPT_DIR}/config/infer.yaml"
ROBOT_IP="${ROBOT_IP:-}"
DO_BUILD="0"
PRINT_ONLY="0"
CONTROLLER_ONLY="0"
SENDER_ONLY="0"
IMAGE_DIR=""
START_SIGNAL_FILE=""

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
    --build)
      DO_BUILD="1"
      shift
      ;;
    --image-dir)
      IMAGE_DIR="${2:?Missing value for --image-dir}"
      shift 2
      ;;
    --controller-only)
      CONTROLLER_ONLY="1"
      shift
      ;;
    --sender-only)
      SENDER_ONLY="1"
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

if [[ "${CONTROLLER_ONLY}" == "1" && "${SENDER_ONLY}" == "1" ]]; then
  echo "[ERROR] --controller-only and --sender-only cannot be used together." >&2
  exit 1
fi

if [[ "${CONFIG_FILE}" != /* ]]; then
  CONFIG_FILE="${PWD}/${CONFIG_FILE}"
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_FILE}" >&2
  exit 1
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
  "${PYTHON:-python}" - "${SCRIPT_DIR}" "$1" <<'PY'
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

CONTROLLER_BINARY="${SCRIPT_DIR}/cpp_controller/build/rolling_policy_controller"
CONTROLLER_LAUNCHER_CMD=(
  "${PYTHON:-python}"
  "${SCRIPT_DIR}/cpp_controller/launch_controller.py"
  --config "${CONFIG_FILE}"
  --binary "${CONTROLLER_BINARY}"
  --print-only
)
if [[ -n "${ROBOT_IP}" ]]; then
  CONTROLLER_LAUNCHER_CMD+=(--robot-ip "${ROBOT_IP}")
fi

SENDER_CMD=(
  "${PYTHON:-python}"
  "${SCRIPT_DIR}/scripts/infer_sender.py"
  --config "${CONFIG_FILE}"
)
if [[ "${SENDER_ONLY}" != "1" ]]; then
  SENDER_CMD+=(--wait-for-start --start-signal-file "${START_SIGNAL_FILE}")
fi
if [[ -n "${IMAGE_DIR}" ]]; then
  SENDER_CMD+=(--image-dir "${IMAGE_DIR}")
fi

if [[ "${PRINT_ONLY}" == "1" ]]; then
  if [[ "${SENDER_ONLY}" != "1" ]]; then
    echo "[INFO] Controller command:"
    "${CONTROLLER_LAUNCHER_CMD[@]}"
  fi
  if [[ "${CONTROLLER_ONLY}" != "1" ]]; then
    printf '[INFO] Sender command:'
    printf ' %q' "${SENDER_CMD[@]}"
    printf '\n'
  fi
  exit 0
fi

if [[ "${SENDER_ONLY}" == "1" ]]; then
  exec "${SENDER_CMD[@]}"
fi

if [[ "${CONTROLLER_ONLY}" == "1" ]]; then
  if [[ "${DO_BUILD}" == "1" ]]; then
    "${SCRIPT_DIR}/cpp_controller/build.bash"
  fi
  if [[ ! -x "${CONTROLLER_BINARY}" && ! -f "${CONTROLLER_BINARY}" ]]; then
    echo "[ERROR] Cannot find controller binary: ${CONTROLLER_BINARY}" >&2
    echo "[HINT] Run: ${SCRIPT_DIR}/launch_realtime.bash --build --controller-only" >&2
    exit 1
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
  exec "${CONTROLLER_CMD[@]}"
fi

CONTROLLER_PID=""
SENDER_PID=""

cleanup() {
  if [[ -n "${SENDER_PID}" ]] && kill -0 "${SENDER_PID}" 2>/dev/null; then
    echo "[INFO] Stopping Python inference sender..."
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

echo "[1/6] Config: ${CONFIG_FILE}"
echo "[1/6] Start signal file: ${START_SIGNAL_FILE}"
rm -f "${START_SIGNAL_FILE}"

echo "[2/6] Starting Python inference sender..."
"${SENDER_CMD[@]}" &
SENDER_PID="$!"

sleep 1
if ! kill -0 "${SENDER_PID}" 2>/dev/null; then
  echo "[ERROR] Python inference sender exited during startup." >&2
  wait "${SENDER_PID}" || true
  exit 1
fi
echo "[2/6] Python inference sender process started."
echo "      Watch for: Policy loaded, Ultrasound video stream ready, Waiting for start signal."

echo "[3/6] Starting C++ controller..."
if [[ "${DO_BUILD}" == "1" ]]; then
  "${SCRIPT_DIR}/cpp_controller/build.bash"
fi
if [[ ! -x "${CONTROLLER_BINARY}" && ! -f "${CONTROLLER_BINARY}" ]]; then
  echo "[ERROR] Cannot find controller binary: ${CONTROLLER_BINARY}" >&2
  echo "[HINT] Run: ${SCRIPT_DIR}/launch_realtime.bash --build" >&2
  exit 1
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

echo "[4/6] C++ controller process started. It should now connect to Franka and open the TCP server."

echo "[5/6] Waiting for readiness messages:"
echo "      - Python: Policy loaded and set to eval mode"
echo "      - Python: Ultrasound video stream ready"
echo "      - C++: Waiting for Python policy stream"
echo "      - Python: Connected to controller"
echo "      - C++: Python policy client connected"
echo
read -r -p "Press Enter to start policy streaming and robot motion..."
mkdir -p "$(dirname "${START_SIGNAL_FILE}")"
touch "${START_SIGNAL_FILE}"
echo "[6/6] Start signal sent. Policy actions are now allowed to stream."

wait "${SENDER_PID}"
