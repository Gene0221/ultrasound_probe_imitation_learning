from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
from typing import Any

from ultrasound_imitation.inference.force6d_monitor import ForceSafetyMonitor
from ultrasound_imitation.inference.policy_loader import PolicyRunner
from ultrasound_imitation.inference.ultrasound_source import ImageFolderUltrasoundSource, LiveCameraUltrasoundSource
from ultrasound_imitation.paths import PROJECT_ROOT, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime ultrasound policy inference and stream action chunks.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "infer.yaml"))
    parser.add_argument("--image-dir", default=None, help="Override config and use an image-folder source for dry runs.")
    return parser.parse_args()


class JsonLineClient:
    def __init__(self, host: str, port: int, timeout_s: float, reconnect_delay_s: float) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.reconnect_delay_s = reconnect_delay_s
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        while self.sock is None:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
                sock.settimeout(self.timeout_s)
                self.sock = sock
                print(f"[INFO] Connected to controller at {self.host}:{self.port}")
            except OSError as exc:
                print(f"[WARN] Controller unavailable: {exc}. Retrying in {self.reconnect_delay_s:.1f}s.")
                time.sleep(self.reconnect_delay_s)

    def send(self, payload: dict[str, Any]) -> None:
        data = (json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
        if self.sock is None:
            self.connect()
        assert self.sock is not None
        try:
            self.sock.sendall(data)
        except OSError:
            self.sock.close()
            self.sock = None
            self.connect()
            assert self.sock is not None
            self.sock.sendall(data)


def build_source(args: argparse.Namespace, config: dict[str, Any]):
    if args.image_dir is None:
        ultrasound_cfg = config.get("ultrasound", {})
        source_type = str(ultrasound_cfg.get("source", "image_folder")).lower()
        if source_type == "image_folder":
            image_dir = ultrasound_cfg.get("image_dir")
            if not image_dir:
                raise ValueError("Set ultrasound.image_dir or pass --image-dir for image-folder dry runs.")
            from ultrasound_imitation.paths import resolve_path

            return ImageFolderUltrasoundSource(resolve_path(str(image_dir)))
        if source_type == "live_camera":
            return LiveCameraUltrasoundSource(str(ultrasound_cfg.get("live_config")))
        raise ValueError(f"Unsupported ultrasound.source: {source_type}")
    return ImageFolderUltrasoundSource(Path(args.image_dir))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    policy = PolicyRunner(config)
    source = build_source(args, config)
    force_monitor = ForceSafetyMonitor(config.get("force_safety", {}))

    runtime = config["runtime"]
    motion = config["motion"]
    client = JsonLineClient(
        str(runtime.get("host", "127.0.0.1")),
        int(runtime.get("port", 50555)),
        float(runtime.get("send_timeout_s", 0.2)),
        float(runtime.get("reconnect_delay_s", 1.0)),
    )

    period_s = 1.0 / max(float(runtime.get("camera_hz", 30.0)), 1e-6)
    seq = 0
    for image in source.frames():
        loop_start = time.monotonic()
        actions = policy.predict(image)
        force_sample = force_monitor.read()
        force_ok = force_monitor.check(force_sample)
        payload = {
            "seq": seq,
            "timestamp_s": time.time(),
            "mode": "relative_delta_chunk",
            "action_dt_s": float(motion.get("action_dt_s", 0.03)),
            "speed_scale": float(motion.get("speed_scale", 0.4)),
            "execute_steps": int(motion.get("execute_steps_per_inference", 1)),
            "actions": actions,
            "force_safety_ok": force_ok,
            "force": force_sample.as_dict(),
        }
        client.send(payload)
        seq += 1
        sleep_s = period_s - (time.monotonic() - loop_start)
        if sleep_s > 0.0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
