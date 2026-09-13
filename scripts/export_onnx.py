#!/usr/bin/env python
"""Export trained YOLO weights to ONNX for C++ OpenCV DNN deployment."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models" / "best.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "neu_det.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    weights = args.weights.resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=args.dynamic,
            simplify=args.simplify,
        )
    )
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != destination:
        shutil.copy2(exported, destination)
    print(f"ONNX model: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

