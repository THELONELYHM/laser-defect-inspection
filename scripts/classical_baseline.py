#!/usr/bin/env python
"""OpenCV-only black-hat/contour baseline for comparison with YOLO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from laser_inspection.pipeline import load_image, save_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "classical"
    )
    parser.add_argument("--kernel", type=int, default=15)
    parser.add_argument("--min-area", type=float, default=20.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.kernel < 3:
        raise ValueError("--kernel must be >= 3")
    kernel_size = args.kernel if args.kernel % 2 == 1 else args.kernel + 1
    image = load_image(args.source)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    canvas = image.copy()
    boxes: list[dict[str, float | list[int]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < args.min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        boxes.append({"bbox_xywh": [x, y, width, height], "contour_area_px2": area})
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 0, 255), 1)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(output_dir / f"{args.source.stem}_mask.png", mask)
    save_image(output_dir / f"{args.source.stem}_contours.jpg", canvas)
    (output_dir / f"{args.source.stem}_report.json").write_text(
        json.dumps({"candidate_count": len(boxes), "candidates": boxes}, indent=2),
        encoding="utf-8",
    )
    print(f"Found {len(boxes)} candidate regions. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
