from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ultrasound_imitation.inference.force6d_monitor import ForceSafetyMonitor
from ultrasound_imitation.paths import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guide force tare and contact setup before realtime inference.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    calibration_cfg = config.get("calibration", {})
    force_enabled = bool(calibration_cfg.get("enabled", False)) and bool(calibration_cfg.get("force_enabled", True))
    if not force_enabled:
        print("[INFO] Force calibration disabled; guided force initialization skipped.", flush=True)
        return

    reader_cfg = dict(calibration_cfg.get("force_reader", {}))
    reader = str(reader_cfg.get("reader", "placeholder")).lower()
    if not bool(reader_cfg.get("enabled", False)) or reader != "kwr75b_serial":
        raise RuntimeError(
            "Guided calibration initialization requires calibration.force_reader "
            "to be an enabled kwr75b_serial reader."
        )

    initialization_cfg = calibration_cfg.get("initialization", {})
    tare_enabled = bool(initialization_cfg.get("tare_enabled", True))
    tare_samples = max(1, int(initialization_cfg.get("tare_samples", 100)))
    tare_settle_s = max(0.0, float(initialization_cfg.get("tare_settle_s", 1.0)))
    prompt_for_contact = bool(initialization_cfg.get("prompt_for_contact", True))

    print("\n[CALIBRATION INIT] Force drift, contact force, and EE orientation setup.", flush=True)
    if tare_enabled:
        input(
            "[STEP 1/2] Keep the fully assembled probe still and clear of the skin or any external "
            "contact, then press Enter to tare the force sensor..."
        )
        if tare_settle_s > 0.0:
            print(f"[INFO] Waiting {tare_settle_s:.2f}s for the sensor to settle...", flush=True)
            time.sleep(tare_settle_s)
        monitor = ForceSafetyMonitor(reader_cfg)
        try:
            bias = monitor.tare(tare_samples)
        finally:
            monitor.close()
        print(
            "[INFO] Force drift compensation completed and saved: "
            + ", ".join(f"{value:.6f}" for value in bias),
            flush=True,
        )
    else:
        print("[STEP 1/2] Tare disabled; the configured bias file will be reused.", flush=True)

    if prompt_for_contact:
        input(
            "[STEP 2/2] Place the probe on the skin, adjust it to the desired contact pressure and "
            "EE orientation, keep it still, then press Enter..."
        )
    else:
        print("[STEP 2/2] Contact confirmation prompt disabled.", flush=True)
    print(
        "[INFO] Contact setup confirmed. The sender and controller will now record the contact Fz "
        "and EE orientation before inference starts.",
        flush=True,
    )


if __name__ == "__main__":
    main()
