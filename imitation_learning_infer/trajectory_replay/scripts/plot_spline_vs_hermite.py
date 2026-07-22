#!/usr/bin/env python3
"""Plot replay trajectory interpolation kinematics from a YAML config."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config" / "plot_trajectory.yaml"


@dataclass
class EvalResult:
    name: str
    label: str
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    smoothing_s: list[float] | None = None


@dataclass
class PlotConfig:
    csv_path: Path
    output_dir: Path
    output_prefix: str | None
    mode: str
    dt: float
    zoom_start_s: float
    zoom_end_s: float
    make_full_plot: bool
    make_zoom_plot: bool
    write_interpolated_csv: bool
    smoothing_factor: float
    title: str | None


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_path(path_value: str | Path, base_dir: Path = PROJECT_DIR) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "paths": {
            "input_csv": "./config/replay_trajectory.csv",
            "output_dir": "./analysis",
            "output_prefix": None,
        },
        "plot": {
            "mode": "comparison",
            "dt": 0.001,
            "zoom_start_s": 0.0,
            "zoom_end_s": 5.0,
            "make_full_plot": True,
            "make_zoom_plot": True,
            "write_interpolated_csv": True,
            "title": None,
        },
        "bspline": {
            "smoothing_factor": 0.0016,
        },
    }

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a YAML mapping: {config_path}")
        deep_update(defaults, loaded)
    else:
        print(f"[WARN] Plot config not found, using built-in defaults: {config_path}")

    return defaults


def build_plot_config(args: argparse.Namespace) -> PlotConfig:
    raw = load_yaml_config(args.config)

    paths = raw["paths"]
    plot = raw["plot"]
    bspline = raw["bspline"]

    csv_path = resolve_path(args.csv or paths["input_csv"])
    output_dir = resolve_path(args.output_dir or paths["output_dir"])
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = paths.get("output_prefix")

    mode = args.mode or plot["mode"]
    mode = str(mode).lower()
    aliases = {"hermite": "polynomial", "quintic": "polynomial", "spline": "bspline"}
    mode = aliases.get(mode, mode)
    if mode not in {"comparison", "polynomial", "bspline"}:
        raise ValueError(f"plot.mode must be comparison, polynomial, or bspline. Got: {mode}")

    return PlotConfig(
        csv_path=csv_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        mode=mode,
        dt=float(args.dt if args.dt is not None else plot["dt"]),
        zoom_start_s=float(args.zoom_start if args.zoom_start is not None else plot["zoom_start_s"]),
        zoom_end_s=float(args.zoom_end if args.zoom_end is not None else plot["zoom_end_s"]),
        make_full_plot=bool(plot["make_full_plot"]),
        make_zoom_plot=bool(plot["make_zoom_plot"]),
        write_interpolated_csv=bool(plot["write_interpolated_csv"]),
        smoothing_factor=float(
            args.smoothing_factor
            if args.smoothing_factor is not None
            else bspline["smoothing_factor"]
        ),
        title=args.title if args.title is not None else plot.get("title"),
    )


def load_translation(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    t = np.array([float(r["time_s"]) for r in rows], dtype=float)
    p = np.array(
        [[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows],
        dtype=float,
    )
    keep = np.r_[True, np.diff(t) > 1e-12]
    return t[keep], p[keep]


def polynomial_eval(
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


def cubic_bspline_basis(u: float) -> np.ndarray:
    u = float(np.clip(u, 0.0, 1.0))
    u2 = u * u
    u3 = u2 * u
    return np.array(
        [
            (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0,
            (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
            (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
            u3 / 6.0,
        ],
        dtype=float,
    )


def smooth_waypoint_positions(p_waypoints: np.ndarray, smoothing_factor: float) -> np.ndarray:
    if smoothing_factor <= 0.0 or len(p_waypoints) < 3:
        return p_waypoints.copy()

    smoothed = p_waypoints.copy()
    denom = 1.0 + 2.0 * smoothing_factor
    for i in range(1, len(p_waypoints) - 1):
        smoothed[i] = (
            p_waypoints[i]
            + smoothing_factor * (p_waypoints[i - 1] + p_waypoints[i + 1])
        ) / denom
    return smoothed


def local_cubic_bspline_eval(
    t_waypoints: np.ndarray,
    p_waypoints: np.ndarray,
    t_query: np.ndarray,
    smoothing_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    controls = smooth_waypoint_positions(p_waypoints, smoothing_factor)
    positions = np.zeros((len(t_query), 3), dtype=float)

    segments = np.searchsorted(t_waypoints, t_query, side="right") - 1
    segments = np.clip(segments, 0, len(t_waypoints) - 2)
    last = len(t_waypoints) - 1

    for j, t in enumerate(t_query):
        left = int(segments[j])
        right = left + 1
        h = t_waypoints[right] - t_waypoints[left]
        u = (t - t_waypoints[left]) / h
        basis = cubic_bspline_basis(u)
        idx = np.array(
            [
                max(0, min(left - 1, last)),
                max(0, min(left, last)),
                max(0, min(left + 1, last)),
                max(0, min(left + 2, last)),
            ],
            dtype=int,
        )
        positions[j] = np.sum(controls[idx] * basis[:, None], axis=0)

    velocities = np.gradient(positions, t_query, axis=0, edge_order=2)
    accelerations = np.gradient(velocities, t_query, axis=0, edge_order=2)
    return positions, velocities, accelerations


def evaluate_methods(
    mode: str,
    t_waypoints: np.ndarray,
    p_waypoints: np.ndarray,
    t_query: np.ndarray,
    smoothing_factor: float,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    if mode in {"comparison", "polynomial"}:
        p, v, a = polynomial_eval(t_waypoints, p_waypoints, t_query)
        results.append(EvalResult("polynomial", "polynomial", p, v, a))
    if mode in {"comparison", "bspline"}:
        p, v, a = local_cubic_bspline_eval(
            t_waypoints,
            p_waypoints,
            t_query,
            smoothing_factor,
        )
        results.append(EvalResult("bspline", "B-spline", p, v, a))
    return results


def write_interpolated_csv(output_path: Path, t: np.ndarray, results: list[EvalResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["time_s"]
    for result in results:
        name = result.name
        header.extend(
            [
                f"x_{name}",
                f"y_{name}",
                f"z_{name}",
                f"vx_{name}",
                f"vy_{name}",
                f"vz_{name}",
                f"ax_{name}",
                f"ay_{name}",
                f"az_{name}",
                f"speed_{name}",
                f"acc_{name}",
            ]
        )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(t)):
            row: list[float] = [float(t[i])]
            for result in results:
                speed = float(np.linalg.norm(result.velocities[i]))
                acc = float(np.linalg.norm(result.accelerations[i]))
                row.extend(
                    [
                        *result.positions[i],
                        *result.velocities[i],
                        *result.accelerations[i],
                        speed,
                        acc,
                    ]
                )
            writer.writerow([f"{value:.9g}" for value in row])


def fitting_error_mm(
    t_query: np.ndarray,
    t_waypoints: np.ndarray,
    p_waypoints: np.ndarray,
    result: EvalResult,
) -> tuple[np.ndarray, np.ndarray]:
    nearest = np.searchsorted(t_query, t_waypoints)
    nearest = np.clip(nearest, 0, len(t_query) - 1)
    fit_error = result.positions[nearest] - p_waypoints
    rmse_mm = np.sqrt(np.mean(fit_error**2, axis=0)) * 1000.0
    max_error_mm = np.max(np.abs(fit_error), axis=0) * 1000.0
    return rmse_mm, max_error_mm


def plot_pva(
    output_path: Path,
    title: str,
    t_query: np.ndarray,
    t_waypoints: np.ndarray,
    p_waypoints: np.ndarray,
    results: list[EvalResult],
    smoothing_factor: float,
    mask: np.ndarray | None = None,
) -> None:
    if mask is None:
        mask = np.ones_like(t_query, dtype=bool)

    colors = {
        ("polynomial", "x"): "#93c5fd",
        ("polynomial", "y"): "#86efac",
        ("polynomial", "z"): "#fca5a5",
        ("polynomial", "norm"): "#9ca3af",
        ("bspline", "x"): "#2563eb",
        ("bspline", "y"): "#059669",
        ("bspline", "z"): "#dc2626",
        ("bspline", "norm"): "#111827",
    }
    linestyles = {"polynomial": "--", "bspline": "-"}
    linewidths = {"polynomial": 1.0, "bspline": 1.6}
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True, constrained_layout=True)

    t_min, t_max = t_query[mask][0], t_query[mask][-1]
    waypoint_mask = (t_waypoints >= t_min) & (t_waypoints <= t_max)

    for result in results:
        speed = np.linalg.norm(result.velocities, axis=1)
        acc = np.linalg.norm(result.accelerations, axis=1)
        for axis, label in enumerate(labels):
            color = colors[(result.name, label)]
            axes[0].plot(
                t_query[mask],
                result.positions[mask, axis],
                color=color,
                lw=linewidths[result.name],
                linestyle=linestyles[result.name],
                label=f"{label} {result.label}",
            )
            axes[1].plot(
                t_query[mask],
                result.velocities[mask, axis],
                color=color,
                lw=linewidths[result.name],
                linestyle=linestyles[result.name],
            )
            axes[2].plot(
                t_query[mask],
                result.accelerations[mask, axis],
                color=color,
                lw=linewidths[result.name],
                linestyle=linestyles[result.name],
            )
        norm_color = colors[(result.name, "norm")]
        axes[1].plot(
            t_query[mask],
            speed[mask],
            color=norm_color,
            lw=linewidths[result.name],
            linestyle=linestyles[result.name],
            label=f"|v| {result.label}",
        )
        axes[2].plot(
            t_query[mask],
            acc[mask],
            color=norm_color,
            lw=linewidths[result.name],
            linestyle=linestyles[result.name],
            label=f"|a| {result.label}",
        )

    for axis, label in enumerate(labels):
        axes[0].scatter(
            t_waypoints[waypoint_mask],
            p_waypoints[waypoint_mask, axis],
            color=colors[("bspline", label)],
            s=8,
            alpha=0.25,
        )

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

    summaries = [f"smoothing_factor={smoothing_factor:g}"]
    for result in results:
        speed = np.linalg.norm(result.velocities, axis=1)
        acc = np.linalg.norm(result.accelerations, axis=1)
        rmse_mm, max_error_mm = fitting_error_mm(t_query, t_waypoints, p_waypoints, result)
        summaries.append(
            (
                f"{result.name}: max |v|={speed.max():.4f}, |a|={acc.max():.4f}, "
                f"RMSE xyz={rmse_mm[0]:.2f},{rmse_mm[1]:.2f},{rmse_mm[2]:.2f} mm, "
                f"max err xyz={max_error_mm[0]:.2f},{max_error_mm[1]:.2f},{max_error_mm[2]:.2f} mm"
            )
        )
    fig.text(0.01, 0.01, "; ".join(summaries), fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot replay trajectory position/velocity/acceleration."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML plot config.")
    parser.add_argument("--csv", type=Path, default=None, help="Input replay CSV override.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory override.")
    parser.add_argument("--output-prefix", default=None, help="Output filename prefix override.")
    parser.add_argument(
        "--mode",
        choices=["comparison", "polynomial", "bspline"],
        default=None,
        help="Plot comparison or one interpolation method.",
    )
    parser.add_argument("--smoothing-factor", type=float, default=None, help="B-spline smoothing override.")
    parser.add_argument("--dt", type=float, default=None, help="Interpolation sample period override.")
    parser.add_argument("--zoom-start", type=float, default=None, help="Zoom plot start time override.")
    parser.add_argument("--zoom-end", type=float, default=None, help="Zoom plot end time override.")
    parser.add_argument("--title", default=None, help="Plot title override.")
    return parser.parse_args()


def default_title(mode: str) -> str:
    if mode == "comparison":
        return "B-spline vs polynomial interpolation"
    if mode == "bspline":
        return "B-spline interpolation"
    return "Polynomial interpolation"


def output_stem(cfg: PlotConfig) -> str:
    if cfg.output_prefix:
        return cfg.output_prefix
    if cfg.mode == "bspline":
        return f"{cfg.csv_path.stem}_bspline_{cfg.smoothing_factor:g}"
    if cfg.mode == "polynomial":
        return f"{cfg.csv_path.stem}_polynomial"
    return f"{cfg.csv_path.stem}_bspline_{cfg.smoothing_factor:g}_vs_polynomial"


def main() -> None:
    args = parse_args()
    cfg = build_plot_config(args)
    t_waypoints, p_waypoints = load_translation(cfg.csv_path)

    t_query = np.arange(t_waypoints[0], t_waypoints[-1] + 1e-12, cfg.dt)
    if t_query[-1] < t_waypoints[-1]:
        t_query = np.r_[t_query, t_waypoints[-1]]

    results = evaluate_methods(
        cfg.mode,
        t_waypoints,
        p_waypoints,
        t_query,
        cfg.smoothing_factor,
    )

    stem = output_stem(cfg)
    title = cfg.title or default_title(cfg.mode)

    outputs: list[Path] = []
    if cfg.make_full_plot:
        full_png = cfg.output_dir / f"{stem}_pva.png"
        plot_pva(
            full_png,
            title,
            t_query,
            t_waypoints,
            p_waypoints,
            results,
            cfg.smoothing_factor,
        )
        outputs.append(full_png)

    if cfg.make_zoom_plot:
        zoom_png = cfg.output_dir / f"{stem}_zoom_{cfg.zoom_start_s:g}_{cfg.zoom_end_s:g}s.png"
        zoom_mask = (t_query >= cfg.zoom_start_s) & (t_query <= min(cfg.zoom_end_s, t_query[-1]))
        if np.any(zoom_mask):
            plot_pva(
                zoom_png,
                f"Zoom {cfg.zoom_start_s:g}-{cfg.zoom_end_s:g} s: {title}",
                t_query,
                t_waypoints,
                p_waypoints,
                results,
                cfg.smoothing_factor,
                mask=zoom_mask,
            )
            outputs.append(zoom_png)

    if cfg.write_interpolated_csv:
        out_csv = cfg.output_dir / f"{stem}_1khz.csv"
        write_interpolated_csv(out_csv, t_query, results)
        outputs.append(out_csv)

    print(f"config: {args.config}")
    print(f"input_csv: {cfg.csv_path}")
    print(f"mode: {cfg.mode}")
    print(f"smoothing_factor: {cfg.smoothing_factor:g}")
    for result in results:
        speed = np.linalg.norm(result.velocities, axis=1)
        acc = np.linalg.norm(result.accelerations, axis=1)
        print(f"{result.name} max speed: {speed.max():.6f} m/s")
        print(f"{result.name} max acc: {acc.max():.6f} m/s^2")
        if result.smoothing_s is not None:
            print(f"{result.name} smoothing_s_xyz: {result.smoothing_s}")
    print("outputs:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
