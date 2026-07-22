#!/usr/bin/env python3
"""Compare quintic Hermite interpolation with a smoothing spline trajectory.

The default smoothing factor is 0.022, which is the 10x-lower setting used in
the latest comparison plot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


DEFAULT_CSV = Path(
    r"C:\Users\zhj80\xwechat_files\wxid_5kihrfy1233y22_a8e2\msg\file\2026-07\replay_example.csv"
)


def load_translation(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    t = np.array([float(r["time_s"]) for r in rows], dtype=float)
    p = np.array(
        [[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows],
        dtype=float,
    )
    keep = np.r_[True, np.diff(t) > 1e-12]
    return t[keep], p[keep]


def quintic_hermite_eval(
    t_waypoints: np.ndarray,
    p_waypoints: np.ndarray,
    t_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_waypoints = np.gradient(p_waypoints, t_waypoints, axis=0, edge_order=2)
    a_waypoints = np.gradient(v_waypoints, t_waypoints, axis=0, edge_order=2)

    positions = np.zeros((len(t_query), 3))
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)

    segments = np.searchsorted(t_waypoints, t_query, side="right") - 1
    segments = np.clip(segments, 0, len(t_waypoints) - 2)

    for i in range(len(t_waypoints) - 1):
        mask = segments == i
        if not np.any(mask):
            continue

        h = t_waypoints[i + 1] - t_waypoints[i]
        s = ((t_query[mask] - t_waypoints[i]) / h)[:, None]

        p0, p1 = p_waypoints[i], p_waypoints[i + 1]
        v0, v1 = v_waypoints[i], v_waypoints[i + 1]
        a0, a1 = a_waypoints[i], a_waypoints[i + 1]

        c0 = p0
        c1 = h * v0
        c2 = 0.5 * h * h * a0
        c3 = (
            10 * (p1 - p0)
            - h * (6 * v0 + 4 * v1)
            - h * h * (1.5 * a0 - 0.5 * a1)
        )
        c4 = (
            -15 * (p1 - p0)
            + h * (8 * v0 + 7 * v1)
            + h * h * (1.5 * a0 - a1)
        )
        c5 = 6 * (p1 - p0) - 3 * h * (v0 + v1) + 0.5 * h * h * (a1 - a0)

        positions[mask] = c0 + c1 * s + c2 * s**2 + c3 * s**3 + c4 * s**4 + c5 * s**5
        velocities[mask] = (
            c1 + 2 * c2 * s + 3 * c3 * s**2 + 4 * c4 * s**3 + 5 * c5 * s**4
        ) / h
        accelerations[mask] = (
            2 * c2 + 6 * c3 * s + 12 * c4 * s**2 + 20 * c5 * s**3
        ) / (h * h)

    return positions, velocities, accelerations


def smoothing_spline_eval(
    t_waypoints: np.ndarray,
    p_waypoints: np.ndarray,
    t_query: np.ndarray,
    smoothing_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    positions = np.zeros((len(t_query), 3))
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    smoothing_s: list[float] = []

    for axis in range(3):
        variance = float(np.var(p_waypoints[:, axis]))
        smooth = max(len(t_waypoints) * variance * smoothing_factor, 1e-10)
        spline = UnivariateSpline(t_waypoints, p_waypoints[:, axis], k=5, s=smooth)
        smoothing_s.append(smooth)
        positions[:, axis] = spline(t_query)
        velocities[:, axis] = spline.derivative(1)(t_query)
        accelerations[:, axis] = spline.derivative(2)(t_query)

    return positions, velocities, accelerations, smoothing_s


def write_interpolated_csv(
    output_path: Path,
    t: np.ndarray,
    hermite: tuple[np.ndarray, np.ndarray, np.ndarray],
    spline: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    p_h, v_h, a_h = hermite
    p_s, v_s, a_s = spline
    speed_h = np.linalg.norm(v_h, axis=1)
    acc_h = np.linalg.norm(a_h, axis=1)
    speed_s = np.linalg.norm(v_s, axis=1)
    acc_s = np.linalg.norm(a_s, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "x_hermite",
                "y_hermite",
                "z_hermite",
                "vx_hermite",
                "vy_hermite",
                "vz_hermite",
                "ax_hermite",
                "ay_hermite",
                "az_hermite",
                "speed_hermite",
                "acc_hermite",
                "x_spline",
                "y_spline",
                "z_spline",
                "vx_spline",
                "vy_spline",
                "vz_spline",
                "ax_spline",
                "ay_spline",
                "az_spline",
                "speed_spline",
                "acc_spline",
            ]
        )
        for i in range(len(t)):
            row = [
                t[i],
                *p_h[i],
                *v_h[i],
                *a_h[i],
                speed_h[i],
                acc_h[i],
                *p_s[i],
                *v_s[i],
                *a_s[i],
                speed_s[i],
                acc_s[i],
            ]
            writer.writerow([f"{value:.9g}" for value in row])


def plot_comparison(
    output_path: Path,
    title: str,
    t_query: np.ndarray,
    t_waypoints: np.ndarray,
    p_waypoints: np.ndarray,
    hermite: tuple[np.ndarray, np.ndarray, np.ndarray],
    spline: tuple[np.ndarray, np.ndarray, np.ndarray],
    smoothing_factor: float,
    mask: np.ndarray | None = None,
) -> None:
    if mask is None:
        mask = np.ones_like(t_query, dtype=bool)

    p_h, v_h, a_h = hermite
    p_s, v_s, a_s = spline
    speed_h = np.linalg.norm(v_h, axis=1)
    acc_h = np.linalg.norm(a_h, axis=1)
    speed_s = np.linalg.norm(v_s, axis=1)
    acc_s = np.linalg.norm(a_s, axis=1)

    # Nearest t_query samples for fitting error at original waypoints.
    nearest = np.searchsorted(t_query, t_waypoints)
    nearest = np.clip(nearest, 0, len(t_query) - 1)
    fit_error = p_s[nearest] - p_waypoints
    rmse_mm = np.sqrt(np.mean(fit_error**2, axis=0)) * 1000.0
    max_error_mm = np.max(np.abs(fit_error), axis=0) * 1000.0

    colors = {"x": "#2563eb", "y": "#059669", "z": "#dc2626", "norm": "#111827"}
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True, constrained_layout=True)

    t_min, t_max = t_query[mask][0], t_query[mask][-1]
    waypoint_mask = (t_waypoints >= t_min) & (t_waypoints <= t_max)

    for axis, label in enumerate(labels):
        axes[0].plot(
            t_query[mask],
            p_h[mask, axis],
            color=colors[label],
            lw=1.0,
            alpha=0.32,
            linestyle="--",
            label=f"{label} Hermite",
        )
        axes[0].plot(
            t_query[mask],
            p_s[mask, axis],
            color=colors[label],
            lw=1.7,
            label=f"{label} spline",
        )
        axes[0].scatter(
            t_waypoints[waypoint_mask],
            p_waypoints[waypoint_mask, axis],
            color=colors[label],
            s=8,
            alpha=0.28,
        )
        axes[1].plot(
            t_query[mask],
            v_h[mask, axis],
            color=colors[label],
            lw=0.9,
            alpha=0.28,
            linestyle="--",
        )
        axes[1].plot(t_query[mask], v_s[mask, axis], color=colors[label], lw=1.5)
        axes[2].plot(
            t_query[mask],
            a_h[mask, axis],
            color=colors[label],
            lw=0.8,
            alpha=0.25,
            linestyle="--",
        )
        axes[2].plot(t_query[mask], a_s[mask, axis], color=colors[label], lw=1.4)

    axes[1].plot(
        t_query[mask],
        speed_h[mask],
        color=colors["norm"],
        lw=1.0,
        alpha=0.30,
        linestyle="--",
        label="|v| Hermite",
    )
    axes[1].plot(t_query[mask], speed_s[mask], color=colors["norm"], lw=1.7, label="|v| spline")
    axes[2].plot(
        t_query[mask],
        acc_h[mask],
        color=colors["norm"],
        lw=1.0,
        alpha=0.30,
        linestyle="--",
        label="|a| Hermite",
    )
    axes[2].plot(t_query[mask], acc_s[mask], color=colors["norm"], lw=1.7, label="|a| spline")

    axes[0].set_title(title)
    axes[0].set_ylabel("position delta (m)")
    axes[1].set_ylabel("velocity (m/s)")
    axes[2].set_ylabel("acceleration (m/s^2)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.28)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)
    axes[2].legend(loc="upper right", ncol=2, fontsize=8)

    fig.text(
        0.01,
        0.01,
        (
            f"smoothing_factor={smoothing_factor:g}; "
            f"Hermite max |v|={speed_h.max():.4f}, |a|={acc_h.max():.4f}; "
            f"Spline max |v|={speed_s.max():.4f}, |a|={acc_s.max():.4f}; "
            f"RMSE xyz={rmse_mm[0]:.2f},{rmse_mm[1]:.2f},{rmse_mm[2]:.2f} mm; "
            f"max err xyz={max_error_mm[0]:.2f},{max_error_mm[1]:.2f},{max_error_mm[2]:.2f} mm"
        ),
        fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot quintic Hermite vs smoothing spline position/velocity/acceleration."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input replay CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "analysis",
        help="Directory for PNG and interpolated CSV outputs.",
    )
    parser.add_argument(
        "--smoothing-factor",
        type=float,
        default=0.0016,
        help="Spline smoothing factor. Current comparison default is 0.022.",
    )
    parser.add_argument("--dt", type=float, default=0.001, help="Interpolation sample period in seconds.")
    parser.add_argument("--zoom-end", type=float, default=5.0, help="Zoom plot end time in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_waypoints, p_waypoints = load_translation(args.csv)

    t_query = np.arange(t_waypoints[0], t_waypoints[-1] + 1e-12, args.dt)
    if t_query[-1] < t_waypoints[-1]:
        t_query = np.r_[t_query, t_waypoints[-1]]

    hermite = quintic_hermite_eval(t_waypoints, p_waypoints, t_query)
    p_spline, v_spline, a_spline, smoothing_s = smoothing_spline_eval(
        t_waypoints,
        p_waypoints,
        t_query,
        args.smoothing_factor,
    )
    spline = (p_spline, v_spline, a_spline)

    stem = f"{args.csv.stem}_spline_{args.smoothing_factor:g}_vs_hermite"
    full_png = args.output_dir / f"{stem}_pva.png"
    zoom_png = args.output_dir / f"{stem}_zoom_0_{args.zoom_end:g}s.png"
    out_csv = args.output_dir / f"{stem}_1khz.csv"

    plot_comparison(
        full_png,
        "Smoothing spline vs quintic Hermite",
        t_query,
        t_waypoints,
        p_waypoints,
        hermite,
        spline,
        args.smoothing_factor,
    )
    plot_comparison(
        zoom_png,
        f"Zoom 0-{args.zoom_end:g} s: smoothing spline vs quintic Hermite",
        t_query,
        t_waypoints,
        p_waypoints,
        hermite,
        spline,
        args.smoothing_factor,
        mask=t_query <= min(args.zoom_end, t_query[-1]),
    )
    write_interpolated_csv(out_csv, t_query, hermite, spline)

    speed_h = np.linalg.norm(hermite[1], axis=1)
    acc_h = np.linalg.norm(hermite[2], axis=1)
    speed_s = np.linalg.norm(spline[1], axis=1)
    acc_s = np.linalg.norm(spline[2], axis=1)

    print(f"smoothing_factor: {args.smoothing_factor:g}")
    print(f"smoothing_s_xyz: {smoothing_s}")
    print(f"Hermite max speed: {speed_h.max():.6f} m/s")
    print(f"Hermite max acc: {acc_h.max():.6f} m/s^2")
    print(f"Spline max speed: {speed_s.max():.6f} m/s")
    print(f"Spline max acc: {acc_s.max():.6f} m/s^2")
    print("outputs:")
    print(full_png)
    print(zoom_png)
    print(out_csv)


if __name__ == "__main__":
    main()
