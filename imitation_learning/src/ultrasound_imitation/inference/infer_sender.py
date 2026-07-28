from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
from typing import Any

from ultrasound_imitation.inference.force6d_monitor import ForceSafetyMonitor
from ultrasound_imitation.inference.policy_loader import PolicyRunner, active_motion_config, active_policy_config
from ultrasound_imitation.inference.ultrasound_source import ImageFolderUltrasoundSource, LiveCameraUltrasoundSource
from ultrasound_imitation.paths import PROJECT_ROOT, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime ultrasound policy inference and stream action chunks.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "infer.yaml"))
    parser.add_argument("--image-dir", default=None, help="Override config and use an image-folder source for dry runs.")
    parser.add_argument("--wait-for-start", action="store_true", help="Wait for a start-signal file before streaming actions.")
    parser.add_argument("--start-signal-file", default=None, help="Path to the start-signal file.")
    return parser.parse_args()


class JsonLineClient:
    def __init__(self, host: str, port: int, timeout_s: float, reconnect_delay_s: float) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.reconnect_delay_s = reconnect_delay_s
        self.sock: socket.socket | None = None
        self._receive_buffer = b""

    def connect(self) -> None:
        while self.sock is None:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
                sock.settimeout(None)
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

    def receive(self) -> dict[str, Any]:
        if self.sock is None:
            self.connect()
        assert self.sock is not None
        while b"\n" not in self._receive_buffer:
            received = self.sock.recv(4096)
            if not received:
                raise ConnectionError("Controller closed the command connection.")
            self._receive_buffer += received
        line, self._receive_buffer = self._receive_buffer.split(b"\n", 1)
        payload = json.loads(line.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Controller command must be a JSON object.")
        return payload


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


def wait_for_start_signal(path: Path) -> None:
    print(f"[INFO] Waiting for start signal: {path}", flush=True)
    while not path.exists():
        time.sleep(0.1)
    print("[INFO] Start signal received. Streaming policy actions.", flush=True)


def read_force_axis(sample, axis: str) -> float:
    values = sample.as_dict()
    if axis not in values:
        raise ValueError(f"Unsupported calibration.force_axis: {axis}")
    return float(values[axis])


def capture_initial_force(force_monitor: ForceSafetyMonitor, calibration_cfg: dict[str, Any]) -> float:
    axis = str(calibration_cfg.get("force_axis", "Fz_N"))
    samples_required = max(1, int(calibration_cfg.get("initial_force_samples", 5)))
    timeout_s = float(calibration_cfg.get("initial_force_timeout_s", 3.0))
    deadline = time.monotonic() + timeout_s
    values: list[float] = []
    while len(values) < samples_required and time.monotonic() < deadline:
        values.append(read_force_axis(force_monitor.read(), axis))
        time.sleep(0.01)
    if len(values) < samples_required:
        raise RuntimeError(
            f"Calibration needs {samples_required} initial force samples, got {len(values)} within {timeout_s:.2f}s."
        )
    initial_force = sum(values) / len(values)
    print(
        f"[INFO] Calibration force reference captured: axis={axis} value={initial_force:.4f} N "
        f"samples={len(values)}",
        flush=True,
    )
    return initial_force


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = config["runtime"]
    policy_type, policy_cfg = active_policy_config(config)
    motion = active_motion_config(config, policy_type)
    force_cfg = config.get("force_safety", {})
    calibration_cfg = config.get("calibration", {})
    calibration_enabled = bool(calibration_cfg.get("enabled", False))

    print(f"[INFO] Config: {Path(args.config).resolve()}", flush=True)
    print(
        f"[INFO] Policy: type={policy_type} model_dir={policy_cfg.get('model_dir')} "
        f"checkpoint={policy_cfg.get('checkpoint_name')} version={policy_cfg.get('dataset_version')}",
        flush=True,
    )
    policy = PolicyRunner(config)
    print("[INFO] Policy loaded and set to eval mode.", flush=True)

    source = build_source(args, config)
    frame_iter = source.frames()
    first_image = next(frame_iter)
    print(f"[INFO] Ultrasound video stream ready: first_frame_size={first_image.size}", flush=True)

    force_monitor = ForceSafetyMonitor(force_cfg)
    print(
        f"[INFO] Force safety: enabled={force_cfg.get('enabled', False)} reader={force_cfg.get('reader', 'placeholder')}",
        flush=True,
    )
    calibration_force_monitor: ForceSafetyMonitor | None = None
    initial_calibration_force: float | None = None
    if calibration_enabled:
        calibration_force_cfg = dict(calibration_cfg.get("force_reader", {}))
        calibration_reader = str(calibration_force_cfg.get("reader", "placeholder")).lower()
        if not bool(calibration_force_cfg.get("enabled", False)) or calibration_reader == "placeholder":
            raise RuntimeError("calibration.enabled=true requires calibration.force_reader.enabled=true with a real reader.")
        calibration_force_monitor = ForceSafetyMonitor(calibration_force_cfg)
        print(
            f"[INFO] Calibration force reader: enabled={calibration_force_cfg.get('enabled', False)} "
            f"reader={calibration_force_cfg.get('reader', 'placeholder')}",
            flush=True,
        )
        initial_calibration_force = capture_initial_force(calibration_force_monitor, calibration_cfg)
        print("[INFO] Calibration module enabled and calibration force reader loaded successfully.", flush=True)
    else:
        print("[INFO] Calibration module disabled.", flush=True)

    client = JsonLineClient(
        str(runtime.get("host", "127.0.0.1")),
        int(runtime.get("port", 50555)),
        float(runtime.get("connect_timeout_s", 0.2)),
        float(runtime.get("reconnect_delay_s", 1.0)),
    )
    client.connect()

    wait_enabled = bool(args.wait_for_start)
    start_signal_value = args.start_signal_file or runtime.get("start_signal_file", "runtime/start_policy_stream.flag")
    from ultrasound_imitation.paths import resolve_path

    start_signal_file = resolve_path(str(start_signal_value))
    if wait_enabled:
        wait_for_start_signal(start_signal_file)

    calibration_axis = str(calibration_cfg.get("force_axis", "Fz_N"))
    seq = 0
    pending_image = first_image
    client.send({"mode": "ready", "timestamp_s": time.time()})
    print("[INFO] Python inference service ready. Waiting for C++ requests.", flush=True)
    while True:
        command = client.receive()
        command_type = str(command.get("command", "")).lower()
        request_id = int(command.get("request_id", -1))
        if command_type == "infer":
            image = pending_image
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
                "calibration_Fz_N": read_force_axis(calibration_force_sample, calibration_axis),
            }
            if initial_calibration_force is not None:
                payload["calibration_initial_force_N"] = initial_calibration_force
            client.send(payload)
            print(f"[INFO] Responded to infer request {request_id}: actions={len(actions)}", flush=True)
            seq += 1
            pending_image = next(frame_iter)
        elif command_type == "force_sample":
            force_sample = calibration_force_monitor.read() if calibration_force_monitor is not None else force_monitor.read()
            payload = {
                "mode": "force_sample",
                "request_id": request_id,
                "timestamp_s": time.time(),
                "calibration_Fz_N": read_force_axis(force_sample, calibration_axis),
            }
            if initial_calibration_force is not None:
                payload["calibration_initial_force_N"] = initial_calibration_force
            client.send(payload)
        else:
            raise ValueError(f"Unsupported controller command: {command_type!r}")


if __name__ == "__main__":
    main()
