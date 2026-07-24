from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


ReplayRow = dict[str, float]


def normalize_quaternion(q: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in q))
    if n < 1e-12:
        raise ValueError("Quaternion norm is zero.")
    return [v / n for v in q]


def quat_conjugate(q: list[float]) -> list[float]:
    x, y, z, w = q
    return [-x, -y, -z, w]


def quat_multiply(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return normalize_quaternion(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_to_rotvec(q: list[float]) -> list[float]:
    x, y, z, w = normalize_quaternion(q)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, max(-1.0, min(1.0, w)))
    scale = angle / vector_norm
    return [x * scale, y * scale, z * scale]


def rotvec_to_quat(rotvec: list[float]) -> list[float]:
    angle = math.sqrt(sum(v * v for v in rotvec))
    if angle < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    axis = [v / angle for v in rotvec]
    half = 0.5 * angle
    s = math.sin(half)
    return normalize_quaternion([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)])


def quat_to_matrix(q: list[float]) -> list[list[float]]:
    x, y, z, w = normalize_quaternion(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def make_transform(translation: list[float], quaternion: list[float]) -> list[list[float]]:
    m = quat_to_matrix(quaternion)
    m[0][3], m[1][3], m[2][3] = translation
    return m


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def matrix_to_quat(m: list[list[float]]) -> list[float]:
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    return normalize_quaternion([qx, qy, qz, qw])


def identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def accumulate_action_chunks(chunks: list[list[list[float]]], action_dt_s: float) -> list[ReplayRow]:
    cumulative = identity()
    rows: list[ReplayRow] = [
        {
            "time_s": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "target_fz": 0.0,
        }
    ]
    step_index = 0
    for actions in chunks:
        for action in actions:
            if len(action) != 7:
                raise ValueError(f"Expected 7D action delta, got {len(action)} values.")
            translation = [float(v) for v in action[:3]]
            quaternion = normalize_quaternion([float(v) for v in action[3:7]])
            cumulative = matmul(cumulative, make_transform(translation, quaternion))
            q = matrix_to_quat(cumulative)
            step_index += 1
            rows.append(
                {
                    "time_s": step_index * action_dt_s,
                    "x": cumulative[0][3],
                    "y": cumulative[1][3],
                    "z": cumulative[2][3],
                    "qx": q[0],
                    "qy": q[1],
                    "qz": q[2],
                    "qw": q[3],
                    "target_fz": 0.0,
                }
            )
    return rows


def lowpass_series(times: list[float], values: list[float], cutoff_hz: float) -> list[float]:
    if cutoff_hz <= 0.0:
        raise ValueError("filter cutoff must be positive.")
    if not values:
        return []
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    filtered = [values[0]]
    for idx in range(1, len(values)):
        dt = max(times[idx] - times[idx - 1], 1e-9)
        alpha = dt / (tau + dt)
        filtered.append(filtered[-1] + alpha * (values[idx] - filtered[-1]))
    return filtered


def lowpass_zero_phase(times: list[float], values: list[float], cutoff_hz: float) -> list[float]:
    forward = lowpass_series(times, values, cutoff_hz)
    reversed_times = [times[-1] - t for t in reversed(times)]
    backward = list(reversed(lowpass_series(reversed_times, list(reversed(forward)), cutoff_hz)))
    backward[0] = values[0]
    backward[-1] = values[-1]
    return backward


def apply_replay_filter(rows: list[ReplayRow], filter_cfg: dict[str, Any]) -> list[ReplayRow]:
    if not rows:
        return []
    output = [dict(row) for row in rows]
    times = [row["time_s"] for row in rows]
    zero_phase = bool(filter_cfg.get("zero_phase", False))

    if bool(filter_cfg.get("enabled", False)):
        cutoff_hz = float(filter_cfg.get("cutoff_hz", 1.0))
        for key in ("x", "y", "z"):
            values = [row[key] for row in rows]
            filtered = lowpass_zero_phase(times, values, cutoff_hz) if zero_phase else lowpass_series(times, values, cutoff_hz)
            for row, value in zip(output, filtered):
                row[key] = value

    if bool(filter_cfg.get("orientation_enabled", filter_cfg.get("enabled", False))):
        cutoff_hz = float(filter_cfg.get("orientation_cutoff_hz", filter_cfg.get("cutoff_hz", 1.0)))
        q0 = normalize_quaternion([rows[0]["qx"], rows[0]["qy"], rows[0]["qz"], rows[0]["qw"]])
        rotvecs: list[list[float]] = []
        previous_q = q0
        for row in rows:
            q = normalize_quaternion([row["qx"], row["qy"], row["qz"], row["qw"]])
            if sum(a * b for a, b in zip(q, previous_q)) < 0.0:
                q = [-v for v in q]
            rotvecs.append(quat_to_rotvec(quat_multiply(quat_conjugate(q0), q)))
            previous_q = q

        filtered_axes: list[list[float]] = []
        for axis in range(3):
            values = [rotvec[axis] for rotvec in rotvecs]
            filtered = lowpass_zero_phase(times, values, cutoff_hz) if zero_phase else lowpass_series(times, values, cutoff_hz)
            filtered_axes.append(filtered)

        for idx, row in enumerate(output):
            filtered_rotvec = [filtered_axes[axis][idx] for axis in range(3)]
            filtered_q = quat_multiply(q0, rotvec_to_quat(filtered_rotvec))
            row["qx"], row["qy"], row["qz"], row["qw"] = filtered_q
    return output


def write_replay_csv(path: Path, rows: list[ReplayRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "x", "y", "z", "qx", "qy", "qz", "qw", "target_fz"])
        for row in rows:
            writer.writerow(
                [
                    f"{row['time_s']:.6f}",
                    f"{row['x']:.9f}",
                    f"{row['y']:.9f}",
                    f"{row['z']:.9f}",
                    f"{row['qx']:.12f}",
                    f"{row['qy']:.12f}",
                    f"{row['qz']:.12f}",
                    f"{row['qw']:.12f}",
                    f"{row['target_fz']:.6f}",
                ]
            )
