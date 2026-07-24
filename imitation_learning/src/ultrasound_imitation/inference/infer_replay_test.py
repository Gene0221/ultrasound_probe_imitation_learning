from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from ultrasound_imitation.inference.infer_sender import build_source
from ultrasound_imitation.inference.policy_loader import PolicyRunner
from ultrasound_imitation.inference.replay_export import accumulate_action_chunks, apply_replay_filter, write_replay_csv
from ultrasound_imitation.paths import PROJECT_ROOT, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run policy inference without a robot and export replay-readable CSV.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "infer.yaml"))
    parser.add_argument("--image-dir", default=None, help="Override config and use an image-folder source.")
    parser.add_argument("--chunks", type=int, default=None, help="Number of image frames to infer. Default: dry_run.chunks or 1.")
    parser.add_argument("--output-dir", default=None, help="Directory for replay CSV files. Default: dry_run.output_dir.")
    parser.add_argument("--output-file", default=None, help="Filtered replay CSV name. Default: dry_run.output_file.")
    parser.add_argument("--raw-output-file", default=None, help="Raw replay CSV name. Default: dry_run.raw_output_file.")
    parser.add_argument("--no-raw", action="store_true", help="Do not write the unfiltered raw replay CSV.")
    parser.add_argument("--no-wait", action="store_true", help="Start inference immediately without waiting for Enter.")
    parser.add_argument("--quiet", action="store_true", help="Do not print every filtered pose row.")
    return parser.parse_args()


def merged_filter_config(config: dict[str, Any]) -> dict[str, Any]:
    motion_filter = dict(config.get("motion", {}).get("online_filter", {}))
    dry_filter = dict(config.get("dry_run", {}).get("filter", {}))
    filter_cfg: dict[str, Any] = {
        "enabled": bool(motion_filter.get("enabled", True)),
        "cutoff_hz": float(motion_filter.get("cutoff_hz", 1.0)),
        "orientation_enabled": bool(motion_filter.get("orientation_enabled", motion_filter.get("enabled", True))),
        "orientation_cutoff_hz": float(motion_filter.get("orientation_cutoff_hz", motion_filter.get("cutoff_hz", 1.0))),
        "zero_phase": False,
    }
    filter_cfg.update(dry_filter)
    return filter_cfg


def output_paths(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Path, Path, bool]:
    dry_run = config.get("dry_run", {})
    output_dir_value = args.output_dir or dry_run.get("output_dir", "runtime/inference_test")
    output_dir = resolve_path(str(output_dir_value))
    output_file = args.output_file or dry_run.get("output_file", "replay_trajectory.csv")
    raw_output_file = args.raw_output_file or dry_run.get("raw_output_file", "replay_trajectory_raw.csv")
    write_raw = bool(dry_run.get("write_raw_copy", True)) and not args.no_raw
    return output_dir / str(output_file), output_dir / str(raw_output_file), write_raw


def print_rows(rows: list[dict[str, float]]) -> None:
    print("[INFO] Filtered replay pose deltas relative to start:")
    print("step,time_s,x,y,z,qx,qy,qz,qw")
    for index, row in enumerate(rows[1:], start=1):
        print(
            f"{index},{row['time_s']:.6f},{row['x']:.9f},{row['y']:.9f},{row['z']:.9f},"
            f"{row['qx']:.12f},{row['qy']:.12f},{row['qz']:.12f},{row['qw']:.12f}"
        )


def wait_for_enter() -> None:
    input("[READY] Press Enter to start inference and export replay CSV...")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dry_run = config.get("dry_run", {})
    policy_cfg = config["policy"]
    motion = config["motion"]

    chunks = args.chunks if args.chunks is not None else int(dry_run.get("chunks", 1))
    if chunks <= 0:
        raise ValueError("--chunks must be positive.")

    action_dt_s = float(motion.get("action_dt_s", 0.03)) / max(float(motion.get("speed_scale", 1.0)), 1e-6)
    output_path, raw_output_path, write_raw = output_paths(args, config)
    filter_cfg = merged_filter_config(config)

    print("[1/5] Loading realtime inference test config.", flush=True)
    print(f"[INFO] Config: {Path(args.config).resolve()}", flush=True)
    print(
        f"[INFO] Policy: type={policy_cfg.get('type')} model_dir={policy_cfg.get('model_dir')} "
        f"checkpoint={policy_cfg.get('checkpoint_name')}",
        flush=True,
    )
    print(f"[INFO] Dry-run chunks: {chunks}", flush=True)
    print(f"[INFO] Replay action dt after speed_scale: {action_dt_s:.6f}s", flush=True)
    print(f"[INFO] Replay filter: {filter_cfg}", flush=True)

    print("[2/5] Loading policy model.", flush=True)
    policy = PolicyRunner(config)
    print("[INFO] Policy loaded and set to eval mode.", flush=True)

    print("[3/5] Opening ultrasound image source and validating video stream.", flush=True)
    source = build_source(args, config)
    frame_iter = source.frames()
    first_image = next(frame_iter)
    print(f"[INFO] Ultrasound video stream ready: first_frame_size={first_image.size}", flush=True)
    print("[INFO] No C++ controller is launched. No command will be sent to the robot.", flush=True)
    if not args.no_wait:
        wait_for_enter()

    print("[4/5] Running policy inference.", flush=True)
    action_chunks: list[list[list[float]]] = []
    for chunk_index in range(chunks):
        image = first_image if chunk_index == 0 else next(frame_iter)
        start = time.monotonic()
        actions = policy.predict(image)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        action_chunks.append(actions)
        print(
            f"[INFO] Inference chunk {chunk_index + 1}/{chunks}: actions={len(actions)} "
            f"elapsed_ms={elapsed_ms:.2f}",
            flush=True,
        )

    print("[5/5] Applying first-order low-pass filter and writing replay CSV.", flush=True)
    raw_rows = accumulate_action_chunks(action_chunks, action_dt_s)
    filtered_rows = apply_replay_filter(raw_rows, filter_cfg)

    if write_raw:
        write_replay_csv(raw_output_path, raw_rows)
        print(f"[INFO] Wrote raw replay CSV: {raw_output_path}", flush=True)
    write_replay_csv(output_path, filtered_rows)
    print(f"[DONE] Wrote filtered replay CSV: {output_path}", flush=True)
    print(f"[INFO] Output rows: {len(filtered_rows)} including the identity start row.", flush=True)

    if not args.quiet:
        print_rows(filtered_rows)


if __name__ == "__main__":
    main()
