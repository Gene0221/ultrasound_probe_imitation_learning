from __future__ import annotations

from pathlib import Path
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

runpy.run_module("ultrasound_imitation.inference.infer_replay_test", run_name="__main__")
