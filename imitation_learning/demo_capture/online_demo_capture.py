from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ultrasound_imitation.inference.force6d_monitor import ForceSafetyMonitor
from ultrasound_imitation.inference.infer_sender import (
    JsonLineClient,
    build_source,
    capture_initial_force,
    read_force_axis,
    wait_for_start_signal,
)
from ultrasound_imitation.inference.policy_loader import PolicyRunner, active_motion_config, active_policy_config
from ultrasound_imitation.paths import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run robot-side online inference while saving ultrasound frames for the paper demo table."
        )
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "infer.yaml"))
    parser.add_argument("--image-dir", default=None, help="Override config and use an image-folder source for dry runs.")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "demo_capture" / "trials"))
    parser.add_argument("--trial-id", default=None, help="Folder name for this trial. Defaults to timestamp.")
    parser.add_argument("--initial-position", default="", help="Optional label or pose text for the initial position.")
    parser.add_argument("--final-position", default="", help="Optional label or pose text for the final position.")
    parser.add_argument("--wait-for-start", action="store_true", help="Wait for a start-signal file before streaming actions.")
    parser.add_argument("--start-signal-file", default=None, help="Path to the start-signal file.")
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save one scanning frame every N inference requests.",
    )
    parser.add_argument(
        "--capture-settle-s",
        type=float,
        default=0.15,
        help="Wait this many seconds before reading a demo snapshot after the controller requests inference.",
    )
    parser.add_argument(
        "--flush-frames",
        type=int,
        default=8,
        help="Discard this many live-camera frames before saving each demo snapshot.",
    )
    return parser.parse_args()


def make_trial_dir(output_root: str | Path, trial_id: str | None) -> Path:
    root = resolve_path(output_root)
    if trial_id is None:
        trial_id = datetime.now().strftime("trial_%Y%m%d_%H%M%S")
    trial_dir = root / trial_id
    frame_dir = trial_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=False)
    return trial_dir


def save_image(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "step",
        "request_id",
        "timestamp_s",
        "inference_input_frame",
        "frame_path",
        "action_count",
        "force_safety_ok",
        "calibration_Fz_N",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1.")

    config = load_config(args.config)
    runtime = config["runtime"]
    policy_type, policy_cfg = active_policy_config(config)
    motion = active_motion_config(config, policy_type)
    force_cfg = config.get("force_safety", {})
    calibration_cfg = config.get("calibration", {})
    calibration_force_cfg = dict(calibration_cfg.get("force", {}))
    calibration_force_enabled = bool(calibration_force_cfg.get("enabled", False))

    trial_dir = make_trial_dir(args.output_root, args.trial_id)
    frame_dir = trial_dir / "frames"
    inference_input_dir = trial_dir / "inference_inputs"
    inference_input_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = trial_dir / "inference_log.jsonl"
    csv_path = trial_dir / "frame_summary.csv"
    metadata_path = trial_dir / "metadata.json"
    summary_rows: list[dict[str, Any]] = []

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(Path(args.config).resolve()),
        "policy_type": policy_type,
        "model_dir": policy_cfg.get("model_dir"),
        "checkpoint_name": policy_cfg.get("checkpoint_name"),
        "dataset_version": policy_cfg.get("dataset_version"),
        "initial_position": args.initial_position,
        "final_position": args.final_position,
        "output_dir": str(trial_dir.resolve()),
    }
    write_metadata(metadata_path, metadata)

    print(f"[INFO] Trial output: {trial_dir.resolve()}", flush=True)
    print(
        f"[INFO] Policy: type={policy_type} model_dir={policy_cfg.get('model_dir')} "
        f"checkpoint={policy_cfg.get('checkpoint_name')}",
        flush=True,
    )
    policy = PolicyRunner(config)
    print("[INFO] Policy loaded and set to eval mode.", flush=True)

    source = build_source(args, config)
    first_image = source.snapshot(settle_s=0.0, flush_frames=args.flush_frames)
    print(f"[INFO] Ultrasound stream ready: first_frame_size={first_image.size}", flush=True)

    force_monitor = ForceSafetyMonitor(force_cfg)
    calibration_force_monitor: ForceSafetyMonitor | None = None
    initial_calibration_force: float | None = None
    if calibration_force_enabled:
        calibration_reader_cfg = dict(calibration_force_cfg.get("reader", {}))
        calibration_reader = str(calibration_reader_cfg.get("reader", "placeholder")).lower()
        if not bool(calibration_reader_cfg.get("enabled", False)) or calibration_reader == "placeholder":
            raise RuntimeError("calibration.force.enabled=true requires calibration.force.reader.enabled=true with a real reader.")
        calibration_force_monitor = ForceSafetyMonitor(calibration_reader_cfg)
        initial_calibration_force = capture_initial_force(calibration_force_monitor, calibration_force_cfg)

    client = JsonLineClient(
        str(runtime.get("host", "127.0.0.1")),
        int(runtime.get("port", 50555)),
        float(runtime.get("connect_timeout_s", 0.2)),
        float(runtime.get("reconnect_delay_s", 1.0)),
    )
    client.connect()

    if args.wait_for_start:
        start_signal_value = args.start_signal_file or runtime.get("start_signal_file", "runtime/start_policy_stream.flag")
        wait_for_start_signal(resolve_path(str(start_signal_value)))
    else:
        input("[PROMPT] Finish robot initialization, then press Enter to save the initial ultrasound frame.")

    initial_image = source.snapshot(settle_s=args.capture_settle_s, flush_frames=args.flush_frames)
    initial_frame = frame_dir / "initial_frame.png"
    save_image(initial_image, initial_frame)
    metadata["initial_frame"] = str(initial_frame.resolve())
    write_metadata(metadata_path, metadata)
    print(f"[INFO] Saved initial frame: {initial_frame}", flush=True)

    if not args.wait_for_start:
        input("[PROMPT] Press Enter again to start online inference and per-step frame capture.")

    seq = 0
    client.send({"mode": "ready", "timestamp_s": time.time()})
    print("[INFO] Demo capture service ready. Stop manually with Ctrl+C when the scan is finished.", flush=True)

    try:
        while True:
            command = client.receive()
            command_type = str(command.get("command", "")).lower()
            request_id = int(command.get("request_id", -1))
            if command_type == "infer":
                image = source.snapshot(settle_s=args.capture_settle_s, flush_frames=args.flush_frames)
                inference_input_path = inference_input_dir / f"input_{seq + 1:04d}.png"
                save_image(image, inference_input_path)
                frame_path: Path | None = None
                if seq % args.save_every == 0:
                    frame_path = frame_dir / f"step_{seq + 1:04d}.png"
                    save_image(image, frame_path)

                actions = policy.predict(image)
                force_sample = force_monitor.read()
                force_ok = force_monitor.check(force_sample)
                calibration_force_sample = calibration_force_monitor.read() if calibration_force_monitor is not None else force_sample
                payload = {
                    "seq": request_id if request_id >= 0 else seq,
                    "request_id": request_id,
                    "timestamp_s": time.time(),
                    "mode": "relative_delta_chunk",
                    "action_dt_s": float(motion.get("action_dt_s", 0.03)),
                    "speed_scale": float(motion.get("speed_scale", 0.4)),
                    "execute_steps": int(motion.get("execute_steps_per_inference", 1)),
                    "actions": actions,
                    "force_safety_ok": force_ok,
                    "force": force_sample.as_dict(),
                    "calibration_Fz_N": read_force_axis(
                        calibration_force_sample, str(calibration_force_cfg.get("axis", "Fz_N"))
                    ),
                    "inference_input_frame": str(inference_input_path.resolve()),
                    "saved_frame": str(frame_path.resolve()) if frame_path is not None else "",
                }
                if initial_calibration_force is not None:
                    payload["calibration_initial_force_N"] = initial_calibration_force
                client.send(payload)
                append_jsonl(jsonl_path, payload)
                summary_rows.append(
                    {
                        "step": seq + 1,
                        "request_id": request_id,
                        "timestamp_s": payload["timestamp_s"],
                        "inference_input_frame": payload["inference_input_frame"],
                        "frame_path": payload["saved_frame"],
                        "action_count": len(actions),
                        "force_safety_ok": force_ok,
                        "calibration_Fz_N": payload["calibration_Fz_N"],
                    }
                )
                write_summary_csv(csv_path, summary_rows)
                print(
                    f"[INFO] step={seq + 1} request_id={request_id} "
                    f"actions={len(actions)} frame={frame_path.name if frame_path else 'skipped'}",
                    flush=True,
                )
                seq += 1
            elif command_type == "force_sample":
                force_sample = calibration_force_monitor.read() if calibration_force_monitor is not None else force_monitor.read()
                payload = {
                    "mode": "force_sample",
                    "request_id": request_id,
                    "timestamp_s": time.time(),
                    "calibration_Fz_N": read_force_axis(force_sample, str(calibration_force_cfg.get("axis", "Fz_N"))),
                }
                if initial_calibration_force is not None:
                    payload["calibration_initial_force_N"] = initial_calibration_force
                client.send(payload)
                append_jsonl(jsonl_path, payload)
            else:
                raise ValueError(f"Unsupported controller command: {command_type!r}")
    except KeyboardInterrupt:
        final_image = source.snapshot(settle_s=args.capture_settle_s, flush_frames=args.flush_frames)
        final_frame = frame_dir / "final_frame.png"
        save_image(final_image, final_frame)
        metadata["final_frame"] = str(final_frame.resolve())
        metadata["final_step_count"] = seq
        metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_metadata(metadata_path, metadata)
        write_summary_csv(csv_path, summary_rows)
        print(f"\n[INFO] Saved final frame: {final_frame}", flush=True)
        print(f"[INFO] Wrote metadata/logs under: {trial_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
