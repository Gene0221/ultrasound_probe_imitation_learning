#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REALTIME_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --robot-ip <ip> [options] [-- extra controller args]

Options:
  --robot-ip <ip>        Franka robot IP address. Required unless ROBOT_IP is set.
  --config <file>        YAML config. Default: ../config/default.yaml, or INFER_DEPLOY_CONFIG.
  --host <ip>            Override runtime.host from YAML.
  --port <port>          Override runtime.port from YAML.
  --binary <path>        Controller binary path. Default: build/rolling_policy_controller.
  --build                Run build.bash before launching.
  --print-only           Print the generated command without executing it.
  -h, --help             Show this help.

Environment overrides:
  ROBOT_IP
  INFER_DEPLOY_CONFIG
  BUILD_DIR
  BUILD_TYPE

Examples:
  $(basename "$0") --robot-ip 172.16.0.2 --build
  $(basename "$0") --robot-ip 172.16.0.2 --print-only
  $(basename "$0") --robot-ip 172.16.0.2 --host 127.0.0.1 --port 50555
EOF
}

CONFIG_FILE="${INFER_DEPLOY_CONFIG:-${REALTIME_DIR}/config/default.yaml}"
ROBOT_IP="${ROBOT_IP:-}"
BINARY_PATH="${SCRIPT_DIR}/build/rolling_policy_controller"
DO_BUILD="0"
PRINT_ONLY="0"
EXTRA_ARGS=()
OVERRIDE_HOST=""
OVERRIDE_PORT=""

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
    --host)
      OVERRIDE_HOST="${2:?Missing value for --host}"
      shift 2
      ;;
    --port)
      OVERRIDE_PORT="${2:?Missing value for --port}"
      shift 2
      ;;
    --binary)
      BINARY_PATH="${2:?Missing value for --binary}"
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
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${ROBOT_IP}" ]]; then
  echo "[ERROR] Robot IP is required. Pass --robot-ip or set ROBOT_IP." >&2
  usage >&2
  exit 1
fi

if [[ "${CONFIG_FILE}" != /* ]]; then
  CONFIG_FILE="${PWD}/${CONFIG_FILE}"
fi

if [[ "${BINARY_PATH}" != /* ]]; then
  BINARY_PATH="${PWD}/${BINARY_PATH}"
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if [[ "${DO_BUILD}" == "1" ]]; then
  "${SCRIPT_DIR}/build.bash"
fi

if [[ ! -x "${BINARY_PATH}" && ! -f "${BINARY_PATH}" ]]; then
  echo "[ERROR] Cannot find controller binary: ${BINARY_PATH}" >&2
  echo "[HINT] Run: ${SCRIPT_DIR}/build.bash" >&2
  exit 1
fi

LAUNCHER_ARGS=(
  "${SCRIPT_DIR}/launch_controller.py"
  --config "${CONFIG_FILE}"
  --robot-ip "${ROBOT_IP}"
  --binary "${BINARY_PATH}"
  --print-only
)

COMMAND_TEXT="$(python "${LAUNCHER_ARGS[@]}")"

COMMAND=()
while IFS= read -r token; do
  COMMAND+=("${token}")
done < <(python - "${COMMAND_TEXT}" <<'PY'
import shlex
import sys
for item in shlex.split(sys.argv[1]):
    print(item)
PY
)

if [[ -n "${OVERRIDE_HOST}" ]]; then
  COMMAND+=(--host "${OVERRIDE_HOST}")
fi
if [[ -n "${OVERRIDE_PORT}" ]]; then
  COMMAND+=(--port "${OVERRIDE_PORT}")
fi
COMMAND+=("${EXTRA_ARGS[@]}")

echo "[INFO] Launching realtime rolling controller:"
echo "[INFO] Config: ${CONFIG_FILE}"
printf '  %q' "${COMMAND[@]}"
printf '\n'

if [[ "${PRINT_ONLY}" == "1" ]]; then
  exit 0
fi

exec "${COMMAND[@]}"
