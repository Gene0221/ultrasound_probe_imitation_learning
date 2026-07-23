from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from config import PROJECT_ROOT, resolve_path


@dataclass
class ForceSample:
    Fx_N: float = 0.0
    Fy_N: float = 0.0
    Fz_N: float = 0.0
    Mx_Nm: float = 0.0
    My_Nm: float = 0.0
    Mz_Nm: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "Fx_N": self.Fx_N,
            "Fy_N": self.Fy_N,
            "Fz_N": self.Fz_N,
            "Mx_Nm": self.Mx_Nm,
            "My_Nm": self.My_Nm,
            "Mz_Nm": self.Mz_Nm,
        }


class ForceSafetyMonitor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.reader = str(config.get("reader", "placeholder")).lower()
        self.max_abs = dict(config.get("max_abs", {}))
        self.consecutive_violation_limit = int(config.get("consecutive_violation_limit", 3))
        self._violations = 0
        self._serial_context: dict[str, Any] | None = None
        if self.enabled and self.reader == "kwr75b_serial":
            self._serial_context = self._open_kwr75b(config.get("serial", {}))

    def read(self) -> ForceSample:
        if not self.enabled or self.reader == "placeholder":
            return ForceSample()
        if self.reader == "kwr75b_serial":
            return self._read_kwr75b()
        raise ValueError(f"Unsupported force_safety.reader: {self.reader}")

    def _open_kwr75b(self, serial_cfg: dict[str, Any]) -> dict[str, Any]:
        force_root = PROJECT_ROOT.parent / "data_collection" / "6D_force_grasping"
        scripts_root = force_root / "scripts"
        if str(scripts_root) not in sys.path:
            sys.path.insert(0, str(scripts_root))
        import serial
        from kwr75b_common import load_bias, read_one_sample, resolve_port

        config = {"serial": dict(serial_cfg)}
        port = resolve_port(config)
        ser = serial.Serial(
            port=port,
            baudrate=int(serial_cfg.get("baudrate", 460800)),
            bytesize=8,
            stopbits=1,
            parity="N",
            timeout=0.05,
        )
        ser.reset_input_buffer()
        bias_file = resolve_path(str(serial_cfg.get("bias_file", force_root / "config" / "zero_bias.json")))
        bias = load_bias(Path(bias_file))
        print(f"[INFO] Opened KWR75B force sensor on {port}; loaded bias from {bias_file}")
        return {
            "ser": ser,
            "buffer": bytearray(),
            "command": bytes([0x49, 0xAA, 0x0D, 0x0A]),
            "request_mode": bool(serial_cfg.get("request_mode", True)),
            "debug": bool(serial_cfg.get("debug", False)),
            "bias": bias,
            "read_one_sample": read_one_sample,
        }

    def _read_kwr75b(self) -> ForceSample:
        if self._serial_context is None:
            raise RuntimeError("KWR75B serial context was not initialized.")
        ctx = self._serial_context
        raw = ctx["read_one_sample"](
            ctx["ser"],
            ctx["buffer"],
            ctx["command"],
            ctx["request_mode"],
            debug=ctx["debug"],
        )
        bias = ctx["bias"]
        zeroed = tuple(raw[i] - bias[i] for i in range(6))
        gravity = 9.80665
        return ForceSample(
            Fx_N=zeroed[0] * gravity,
            Fy_N=zeroed[1] * gravity,
            Fz_N=zeroed[2] * gravity,
            Mx_Nm=zeroed[3] * gravity,
            My_Nm=zeroed[4] * gravity,
            Mz_Nm=zeroed[5] * gravity,
        )

    def close(self) -> None:
        if self._serial_context is not None:
            self._serial_context["ser"].close()
            self._serial_context = None

    def __del__(self) -> None:
        self.close()

    def check(self, sample: ForceSample) -> bool:
        if not self.enabled:
            self._violations = 0
            return True
        values = sample.as_dict()
        violated = any(
            key in self.max_abs and abs(float(values[key])) > float(limit)
            for key, limit in self.max_abs.items()
        )
        self._violations = self._violations + 1 if violated else 0
        return self._violations < self.consecutive_violation_limit
