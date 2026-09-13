#!/usr/bin/env python
"""Estimate millimeters per pixel from a known distance in an image."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from laser_inspection.pipeline import load_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--known-distance-mm", type=float, required=True)
    parser.add_argument(
        "--points",
        help="non-interactive points as x1,y1,x2,y2; otherwise click two points",
    )
    return parser


def parse_points(value: str) -> list[tuple[int, int]]:
    values = [int(part.strip()) for part in value.split(",")]
    if len(values) != 4:
        raise ValueError("--points must be x1,y1,x2,y2")
    return [(values[0], values[1]), (values[2], values[3])]


def select_points(image) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    window = "Calibration: click two points, ESC to cancel"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while len(points) < 2:
        canvas = image.copy()
        for point in points:
            cv2.circle(canvas, point, 4, (0, 0, 255), -1)
        cv2.imshow(window, canvas)
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
    return points


def main() -> int:
    args = build_parser().parse_args()
    if args.known_distance_mm <= 0:
        raise ValueError("--known-distance-mm must be positive")
    image = load_image(args.image)
    points = parse_points(args.points) if args.points else select_points(image)
    if len(points) != 2:
        print("Calibration cancelled", file=sys.stderr)
        return 1
    pixel_distance = math.dist(points[0], points[1])
    if pixel_distance <= 0:
        raise ValueError("Selected points must be different")
    mm_per_pixel = args.known_distance_mm / pixel_distance
    print(f"pixel_distance={pixel_distance:.4f}")
    print(f"mm_per_pixel={mm_per_pixel:.8f}")
    print(f"Use during inference: --mm-per-pixel {mm_per_pixel:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
