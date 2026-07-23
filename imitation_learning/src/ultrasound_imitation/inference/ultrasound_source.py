from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterator

from PIL import Image

from ultrasound_imitation.paths import RESEARCH_ROOT, resolve_path


class UltrasoundSource:
    def frames(self) -> Iterator[Image.Image]:
        raise NotImplementedError


class ImageFolderUltrasoundSource(UltrasoundSource):
    def __init__(self, image_dir: str | Path) -> None:
        self.image_dir = Path(image_dir).resolve()
        self.paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not self.paths:
            raise FileNotFoundError(f"No ultrasound images found in {self.image_dir}")

    def frames(self) -> Iterator[Image.Image]:
        while True:
            for path in self.paths:
                yield Image.open(path).convert("RGB")


class LiveCameraUltrasoundSource(UltrasoundSource):
    def __init__(self, config_path: str | Path) -> None:
        ultrasound_root = RESEARCH_ROOT / "data_collection" / "ultrasound_image_collection"
        scripts_root = ultrasound_root / "scripts"
        if str(scripts_root) not in sys.path:
            sys.path.insert(0, str(scripts_root))
        from capture_ultrasound_frames import load_config, open_capture, resolve_device

        self.config_path = resolve_path(config_path)
        self.config = load_config(self.config_path)
        self.device_info = resolve_device(self.config)
        self.capture = open_capture(self.device_info.device, self.config.get("capture", {}))
        print(f"[INFO] Opened ultrasound camera: {self.device_info.device}")

    def frames(self) -> Iterator[Image.Image]:
        try:
            while True:
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    continue
                import cv2

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield Image.fromarray(rgb)
        finally:
            self.capture.release()
