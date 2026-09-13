#!/usr/bin/env python
"""Evaluate trained weights on the held-out NEU-DET test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models" / "best.pt")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "neu_det" / "dataset.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    from ultralytics import YOLO

    output_dir = PROJECT_ROOT / "outputs" / "evaluation"
    model = YOLO(str(args.weights.resolve()))
    metrics = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        plots=True,
        project=str(output_dir),
        name="test",
        exist_ok=True,
    )
    class_names = metrics.names
    per_class_map = {
        str(class_names[index]): round(float(value), 6)
        for index, value in enumerate(metrics.box.maps)
    }
    summary = {
        "weights": str(args.weights.resolve()),
        "split": "test",
        "precision": round(float(metrics.box.mp), 6),
        "recall": round(float(metrics.box.mr), 6),
        "mAP50": round(float(metrics.box.map50), 6),
        "mAP50_95": round(float(metrics.box.map), 6),
        "per_class_mAP50_95": per_class_map,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Metrics saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

