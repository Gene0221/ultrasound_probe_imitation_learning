from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a checkerboard calibration board as a PNG image. "
            "The rows and cols arguments are interpreted as inner-corner counts "
            "to match OpenCV calibration settings."
        )
    )
    parser.add_argument("--rows", type=int, required=True, help="Number of inner corners along rows.")
    parser.add_argument("--cols", type=int, required=True, help="Number of inner corners along columns.")
    parser.add_argument(
        "--square-size",
        type=float,
        required=True,
        help="Square size in millimeters.",
    )
    return parser.parse_args()


def validate_args(rows: int, cols: int, square_size_mm: float) -> None:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers.")
    if square_size_mm <= 0:
        raise ValueError("square_size must be a positive number.")


def build_checkerboard_image(
    rows: int,
    cols: int,
    square_size_mm: float,
    margin_mm: float,
    pixels_per_mm: float,
) -> np.ndarray:
    square_rows = rows + 1
    square_cols = cols + 1

    square_px = max(1, int(round(square_size_mm * pixels_per_mm)))
    margin_px = max(1, int(round(margin_mm * pixels_per_mm)))

    board_width_px = square_cols * square_px
    board_height_px = square_rows * square_px
    image_width_px = board_width_px + 2 * margin_px
    image_height_px = board_height_px + 2 * margin_px

    image = np.full((image_height_px, image_width_px), 255, dtype=np.uint8)

    cv2.rectangle(
        image,
        (margin_px, margin_px),
        (margin_px + board_width_px - 1, margin_px + board_height_px - 1),
        color=0,
        thickness=1,
    )

    for row in range(square_rows):
        for col in range(square_cols):
            if (row + col) % 2 == 0:
                x0 = margin_px + col * square_px
                y0 = margin_px + row * square_px
                x1 = x0 + square_px
                y1 = y0 + square_px
                image[y0:y1, x0:x1] = 0

    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    label = f"checkerboard inner corners: {cols} x {rows}, square size: {square_size_mm:g} mm"
    cv2.putText(
        image_bgr,
        label,
        (margin_px, image_height_px - max(10, margin_px // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return image_bgr


def main() -> None:
    args = parse_args()
    validate_args(args.rows, args.cols, args.square_size)

    output_dir = Path("data") / "boards"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_stem = f"checkerboard_{args.cols}x{args.rows}_{args.square_size:g}mm"
    output_path = output_dir / f"{file_stem}.png"
    image = build_checkerboard_image(
        args.rows,
        args.cols,
        args.square_size,
        margin_mm=10.0,
        pixels_per_mm=10.0,
    )
    cv2.imwrite(str(output_path), image)

    square_rows = args.rows + 1
    square_cols = args.cols + 1
    board_width_mm = square_cols * args.square_size
    board_height_mm = square_rows * args.square_size

    print(f"[DONE] Checkerboard saved to: {output_path.resolve()}")
    print(f"[INFO] Inner corners (cols x rows): {args.cols} x {args.rows}")
    print(f"[INFO] Square size: {args.square_size:g} mm")
    print(f"[INFO] Board size: {board_width_mm:g} mm x {board_height_mm:g} mm")


if __name__ == "__main__":
    main()
