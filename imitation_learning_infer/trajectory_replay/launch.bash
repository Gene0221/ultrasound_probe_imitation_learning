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
  --max-translation-speed <m/s>   Max Cartesian translation speed. Default: 0.03, or MAX_TRANSLATION_SPEED.
  --max-translation-acceleration <m/s^2> Max Cartesian translation acceleration. Default: 0.01, or MAX_TRANSLATION_ACCELERATION.
  --max-rotation-speed <rad/s>    Max Cartesian rotation speed. Default: 0.35, or MAX_ROTATION_SPEED.
  --ramp-time <s>                 Startup ramp time. Default: 2.0, or RAMP_TIME.
  --enable-force-correction       Enable experimental force correction.
  -h, --help                      Show this help.

Environment overrides:
  BUILD_DIR                       Build directory. Default: <trajectory_replay>/build
  ROBOT_IP                        Franka robot IP address
  TRAJECTORY                      Replay CSV path
  REPLAY_MODE                     relative or absolute
  SPEED_SCALE                     Replay speed scale
  MAX_TRANSLATION_SPEED           Max Cartesian translation speed
  MAX_TRANSLATION_ACCELERATION    Max Cartesian translation acceleration
  MAX_ROTATION_SPEED              Max Cartesian rotation speed
  RAMP_TIME                       Startup ramp time
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
MAX_TRANSLATION_SPEED="${MAX_TRANSLATION_SPEED:-0.03}"
MAX_TRANSLATION_ACCELERATION="${MAX_TRANSLATION_ACCELERATION:-0.01}"
MAX_ROTATION_SPEED="${MAX_ROTATION_SPEED:-0.35}"
RAMP_TIME="${RAMP_TIME:-2.0}"
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
    --max-translation-speed)
      MAX_TRANSLATION_SPEED="${2:?Missing value for --max-translation-speed}"
      shift 2
      ;;
    --max-translation-acceleration)
      MAX_TRANSLATION_ACCELERATION="${2:?Missing value for --max-translation-acceleration}"
      shift 2
      ;;
    --max-rotation-speed)
      MAX_ROTATION_SPEED="${2:?Missing value for --max-rotation-speed}"
      shift 2
      ;;
    --ramp-time)
      RAMP_TIME="${2:?Missing value for --ramp-time}"
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
  --max-translation-speed "${MAX_TRANSLATION_SPEED}"
  --max-translation-acceleration "${MAX_TRANSLATION_ACCELERATION}"
  --max-rotation-speed "${MAX_ROTATION_SPEED}"
  --ramp-time "${RAMP_TIME}"
)

if [[ "${ENABLE_FORCE_CORRECTION}" == "1" ]]; then
  COMMAND+=(--enable-force-correction)
fi

COMMAND+=("${EXTRA_ARGS[@]}")

echo "[INFO] Launching replay controller:"
printf '  %q' "${COMMAND[@]}"
printf '\n'

exec "${COMMAND[@]}"
