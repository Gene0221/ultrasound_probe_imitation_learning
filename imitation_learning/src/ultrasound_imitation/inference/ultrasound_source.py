from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterator

from PIL import Image

from ultrasound_imitation.paths import RESEARCH_ROOT, resolve_path


class UltrasoundSource:
    def frames(self) -> Iterator[Image.Image]:
        raise NotImplementedError

    def snapshot(self, *, settle_s: float = 0.0, flush_frames: int = 0) -> Image.Image:
        del settle_s, flush_frames
        return next(self.frames())


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
        self._snapshot_index = 0
        print(f"[INFO] Ultrasound source: image_folder dir={self.image_dir} frames={len(self.paths)}")

    def frames(self) -> Iterator[Image.Image]:
        while True:
            for path in self.paths:
                yield Image.open(path).convert("RGB")

    def snapshot(self, *, settle_s: float = 0.0, flush_frames: int = 0) -> Image.Image:
        del settle_s, flush_frames
        path = self.paths[self._snapshot_index % len(self.paths)]
        self._snapshot_index += 1
        return Image.open(path).convert("RGB")


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
        try:
            import cv2

            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        print(f"[INFO] Ultrasound source: live_camera device={self.device_info.device}")

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

    def snapshot(self, *, settle_s: float = 0.0, flush_frames: int = 8) -> Image.Image:
        if settle_s > 0.0:
            import time

            time.sleep(settle_s)

        for _ in range(max(0, int(flush_frames))):
            self.capture.grab()

        while True:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                continue
            import cv2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
