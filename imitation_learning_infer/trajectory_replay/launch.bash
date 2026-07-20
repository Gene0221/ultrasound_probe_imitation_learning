#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --robot-ip <ip> [options] [-- extra replay_trajectory args]

Options:
  --robot-ip <ip>                 Franka robot IP address. Can also use ROBOT_IP.
  --trajectory <csv>              Replay CSV path. Default: config/replay_example.csv, or TRAJECTORY.
  --mode <relative|absolute>      Replay mode. Default: relative, or REPLAY_MODE.
  --speed-scale <value>           Speed scale in (0, 1]. Default: 0.2, or SPEED_SCALE.
  --enable-force-correction       Enable experimental force correction.
  -h, --help                      Show this help.

Environment overrides:
  BUILD_DIR                       Build directory. Default: <trajectory_replay>/build
  ROBOT_IP                        Franka robot IP address
  TRAJECTORY                      Replay CSV path
  REPLAY_MODE                     relative or absolute
  SPEED_SCALE                     Replay speed scale
  ENABLE_FORCE_CORRECTION         1 to enable force correction

Examples:
  $(basename "$0") --robot-ip 172.16.0.2
  TRAJECTORY=/data/session_0001/franka_replay/replay_trajectory.csv $(basename "$0") --robot-ip 172.16.0.2 --speed-scale 0.1
EOF
}

BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
ROBOT_IP="${ROBOT_IP:-}"
TRAJECTORY="${TRAJECTORY:-${SCRIPT_DIR}/config/replay_example.csv}"
REPLAY_MODE="${REPLAY_MODE:-relative}"
SPEED_SCALE="${SPEED_SCALE:-0.2}"
ENABLE_FORCE_CORRECTION="${ENABLE_FORCE_CORRECTION:-0}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-ip)
      ROBOT_IP="${2:?Missing value for --robot-ip}"
      shift 2
      ;;
    --trajectory)
      TRAJECTORY="${2:?Missing value for --trajectory}"
      shift 2
      ;;
    --mode)
      REPLAY_MODE="${2:?Missing value for --mode}"
      shift 2
      ;;
    --speed-scale)
      SPEED_SCALE="${2:?Missing value for --speed-scale}"
      shift 2
      ;;
    --enable-force-correction)
      ENABLE_FORCE_CORRECTION="1"
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

CANDIDATE_EXECUTABLES=(
  "${BUILD_DIR}/replay_trajectory"
  "${BUILD_DIR}/Release/replay_trajectory"
  "${SCRIPT_DIR}/build/replay_trajectory"
  "${SCRIPT_DIR}/build/Release/replay_trajectory"
)

BINARY_PATH=""
for candidate in "${CANDIDATE_EXECUTABLES[@]}"; do
  if [[ -x "${candidate}" || -f "${candidate}" ]]; then
    BINARY_PATH="${candidate}"
    break
  fi
done

if [[ -z "${BINARY_PATH}" ]]; then
  echo "[ERROR] Cannot find replay_trajectory binary. Run ./build.bash first." >&2
  exit 1
fi

if [[ -z "${ROBOT_IP}" ]]; then
  echo "[ERROR] Robot IP is required. Pass --robot-ip or set ROBOT_IP." >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${TRAJECTORY}" ]]; then
  echo "[ERROR] Cannot find trajectory CSV: ${TRAJECTORY}" >&2
  exit 1
fi

COMMAND=(
  "${BINARY_PATH}"
  --robot-ip "${ROBOT_IP}"
  --trajectory "${TRAJECTORY}"
  --mode "${REPLAY_MODE}"
  --speed-scale "${SPEED_SCALE}"
)

if [[ "${ENABLE_FORCE_CORRECTION}" == "1" ]]; then
  COMMAND+=(--enable-force-correction)
fi

COMMAND+=("${EXTRA_ARGS[@]}")

echo "[INFO] Launching replay controller:"
printf '  %q' "${COMMAND[@]}"
printf '\n'

exec "${COMMAND[@]}"
