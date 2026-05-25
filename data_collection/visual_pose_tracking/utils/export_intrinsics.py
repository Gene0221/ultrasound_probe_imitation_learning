import json
from pathlib import Path

import pyrealsense2 as rs


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INTRINSICS_DIR = PROJECT_ROOT / "config" / "intrinsics"


def export_rgb_intrinsics(serial, output_path, width=1280, height=720, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)

    try:
        color_profile = profile.get_stream(rs.stream.color)
        video_profile = color_profile.as_video_stream_profile()
        intr = video_profile.get_intrinsics()

        camera_matrix = [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ]

        dist_coeffs = list(intr.coeffs)

        data = {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "width": intr.width,
            "height": intr.height,
            "distortion_model": str(intr.model),
            "serial_number": serial,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        print(f"[DONE] Saved intrinsics to: {output_path}")
        print(json.dumps(data, indent=2))

    finally:
        pipeline.stop()


def export_default_pair_intrinsics(
    camera_a_serial="213622073198",
    camera_b_serial="337122072369",
    intrinsics_dir=DEFAULT_INTRINSICS_DIR,
    width=1280,
    height=720,
    fps=30,
):
    intrinsics_dir = Path(intrinsics_dir)
    export_rgb_intrinsics(
        camera_a_serial,
        intrinsics_dir / "camera_a_calibration_result.json",
        width=width,
        height=height,
        fps=fps,
    )
    export_rgb_intrinsics(
        camera_b_serial,
        intrinsics_dir / "camera_b_calibration_result.json",
        width=width,
        height=height,
        fps=fps,
    )


if __name__ == "__main__":
    export_default_pair_intrinsics()
