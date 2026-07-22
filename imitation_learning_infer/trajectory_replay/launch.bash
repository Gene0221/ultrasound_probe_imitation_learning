#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --robot-ip <ip> [options] [-- extra replay args]

Options:
  --robot-ip <ip>                 Franka robot IP address. Can also use ROBOT_IP.
  --config <file>                 Launch config file. Default: config/replay_launch.conf, or REPLAY_LAUNCH_CONFIG.
  --interpolator <quintic|bspline> Replay executable to launch. Default: quintic, or REPLAY_INTERPOLATOR.
  --trajectory <csv>              Replay CSV path. Default: config/replay_example.csv, or TRAJECTORY.
  --mode <relative|absolute>      Replay mode. Default: relative, or REPLAY_MODE.
  --speed-scale <value>           Speed scale in (0, 1]. Default: 0.2, or SPEED_SCALE.
  --max-translation-speed <m/s>   Max Cartesian translation speed. Default: 0.03, or MAX_TRANSLATION_SPEED.
  --max-translation-acceleration <m/s^2> Max Cartesian translation acceleration. Default: 0.01, or MAX_TRANSLATION_ACCELERATION.
  --max-rotation-speed <rad/s>    Max Cartesian rotation speed. Default: 0.35, or MAX_ROTATION_SPEED.
  --max-rotation-acceleration <rad/s^2> Max Cartesian rotation acceleration. Default: 0.1, or MAX_ROTATION_ACCELERATION.
  --ramp-time <s>                 Startup ramp time. Default: 2.0, or RAMP_TIME.
  --bspline-smoothing-factor <value> B-spline waypoint smoothing factor. Default: 0.0016, or BSPLINE_SMOOTHING_FACTOR.
  --hold-at-end                   Keep commanding the final pose instead of exiting.
  --enable-force-correction       Enable experimental force correction.
  -h, --help                      Show this help.

Environment overrides:
  BUILD_DIR                       Build directory. Default: <trajectory_replay>/build
  REPLAY_LAUNCH_CONFIG            Launch config path
  ROBOT_IP                        Franka robot IP address
  REPLAY_INTERPOLATOR              quintic or bspline
  TRAJECTORY                      Replay CSV path
  REPLAY_MODE                     relative or absolute
  SPEED_SCALE                     Replay speed scale
  MAX_TRANSLATION_SPEED           Max Cartesian translation speed
  MAX_TRANSLATION_ACCELERATION    Max Cartesian translation acceleration
  MAX_ROTATION_SPEED              Max Cartesian rotation speed
  MAX_ROTATION_ACCELERATION       Max Cartesian rotation acceleration
  RAMP_TIME                       Startup ramp time
  BSPLINE_SMOOTHING_FACTOR        B-spline waypoint smoothing factor
  HOLD_AT_END                     1 to keep commanding the final pose
  ENABLE_FORCE_CORRECTION         1 to enable force correction

Examples:
  $(basename "$0") --robot-ip 172.16.0.2
  $(basename "$0") --robot-ip 172.16.0.2 --interpolator bspline
  TRAJECTORY=/data/session_0001/franka_replay/replay_trajectory.csv $(basename "$0") --robot-ip 172.16.0.2 --speed-scale 0.1
EOF
}

CONFIG_FILE="${REPLAY_LAUNCH_CONFIG:-${SCRIPT_DIR}/config/replay_launch.conf}"
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; ++i)); do
  if [[ "${ARGS[$i]}" == "--config" ]]; then
    if (( i + 1 >= ${#ARGS[@]} )); then
      echo "[ERROR] Missing value for --config" >&2
      exit 1
    fi
    CONFIG_FILE="${ARGS[$((i + 1))]}"
    break
  fi
done

if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
fi

BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
REPLAY_INTERPOLATOR="${REPLAY_INTERPOLATOR:-bspline}"
TRAJECTORY="${TRAJECTORY:-${SCRIPT_DIR}/config/replay_example.csv}"
REPLAY_MODE="${REPLAY_MODE:-relative}"
SPEED_SCALE="${SPEED_SCALE:-1.0}"
MAX_TRANSLATION_SPEED="${MAX_TRANSLATION_SPEED:-0.2}"
MAX_TRANSLATION_ACCELERATION="${MAX_TRANSLATION_ACCELERATION:-0.07}"
MAX_ROTATION_SPEED="${MAX_ROTATION_SPEED:-0.35}"
MAX_ROTATION_ACCELERATION="${MAX_ROTATION_ACCELERATION:-0.5}"
RAMP_TIME="${RAMP_TIME:-3.0}"
BSPLINE_SMOOTHING_FACTOR="${BSPLINE_SMOOTHING_FACTOR:-0.0016}"
HOLD_AT_END="${HOLD_AT_END:-1}"
ENABLE_FORCE_CORRECTION="${ENABLE_FORCE_CORRECTION:-0}"
EXTRA_ARGS=()

if [[ "${TRAJECTORY}" != /* ]]; then
  TRAJECTORY="${SCRIPT_DIR}/${TRAJECTORY}"
fi

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
    --interpolator)
      REPLAY_INTERPOLATOR="${2:?Missing value for --interpolator}"
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
    --max-rotation-acceleration)
      MAX_ROTATION_ACCELERATION="${2:?Missing value for --max-rotation-acceleration}"
      shift 2
      ;;
    --ramp-time)
      RAMP_TIME="${2:?Missing value for --ramp-time}"
      shift 2
      ;;
    --bspline-smoothing-factor)
      BSPLINE_SMOOTHING_FACTOR="${2:?Missing value for --bspline-smoothing-factor}"
      shift 2
      ;;
    --hold-at-end)
      HOLD_AT_END="1"
      shift
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

if [[ "${TRAJECTORY}" != /* ]]; then
  TRAJECTORY="${SCRIPT_DIR}/${TRAJECTORY}"
fi

case "${REPLAY_INTERPOLATOR}" in
  quintic)
    BINARY_NAME="replay_trajectory_quintic"
    ;;
  bspline)
    BINARY_NAME="replay_trajectory_bspline"
    ;;
  *)
    echo "[ERROR] --interpolator must be quintic or bspline. Got: ${REPLAY_INTERPOLATOR}" >&2
    exit 1
    ;;
esac

CANDIDATE_EXECUTABLES=(
  "${BUILD_DIR}/${BINARY_NAME}"
  "${BUILD_DIR}/Release/${BINARY_NAME}"
  "${SCRIPT_DIR}/build/${BINARY_NAME}"
  "${SCRIPT_DIR}/build/Release/${BINARY_NAME}"
)

BINARY_PATH=""
for candidate in "${CANDIDATE_EXECUTABLES[@]}"; do
  if [[ -x "${candidate}" || -f "${candidate}" ]]; then
    BINARY_PATH="${candidate}"
    break
  fi
done

if [[ -z "${BINARY_PATH}" ]]; then
  echo "[ERROR] Cannot find ${BINARY_NAME} binary. Run ./build.bash first." >&2
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
  --max-rotation-acceleration "${MAX_ROTATION_ACCELERATION}"
  --ramp-time "${RAMP_TIME}"
)

if [[ "${REPLAY_INTERPOLATOR}" == "bspline" ]]; then
  COMMAND+=(--bspline-smoothing-factor "${BSPLINE_SMOOTHING_FACTOR}")
fi

if [[ "${ENABLE_FORCE_CORRECTION}" == "1" ]]; then
  COMMAND+=(--enable-force-correction)
fi

if [[ "${HOLD_AT_END}" == "1" ]]; then
  COMMAND+=(--hold-at-end)
fi

COMMAND+=("${EXTRA_ARGS[@]}")

echo "[INFO] Launching replay controller:"
echo "[INFO] Config: ${CONFIG_FILE}"
echo "[INFO] Interpolator: ${REPLAY_INTERPOLATOR} (${BINARY_NAME})"
if [[ "${REPLAY_INTERPOLATOR}" == "bspline" ]]; then
  echo "[INFO] B-spline smoothing factor: ${BSPLINE_SMOOTHING_FACTOR}"
fi
printf '  %q' "${COMMAND[@]}"
printf '\n'

exec "${COMMAND[@]}"
