#!/usr/bin/env python
"""Train a reproducible YOLO baseline on the prepared NEU-DET split."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "neu_det" / "dataset.yaml",
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="neu_det_yolov8n")
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction of training data")
    parser.add_argument("--mosaic", type=float, default=0.5, help="mosaic augmentation probability")
    parser.add_argument("--no-val", action="store_true", help="skip validation during smoke tests")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="1 epoch on 5%% of data with mosaic disabled",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}\n"
            "Run: python scripts/prepare_dataset.py"
        )
    from ultralytics import YOLO

    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")
    if not 0.0 <= args.mosaic <= 1.0:
        raise ValueError("--mosaic must be in [0, 1]")
    epochs = 1 if args.quick else args.epochs
    fraction = min(args.fraction, 0.05) if args.quick else args.fraction
    mosaic = 0.0 if args.quick else args.mosaic
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        project=str(PROJECT_ROOT / "outputs" / "train"),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        plots=True,
        val=not args.no_val,
        fraction=fraction,
        degrees=5.0,
        translate=0.08,
        scale=0.25,
        fliplr=0.5,
        mosaic=mosaic,
        close_mosaic=10,
    )
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    if not best.exists():
        raise RuntimeError(f"Training finished but no checkpoint was found in {save_dir}")
    destination_name = "smoke_test.pt" if args.quick else "best.pt"
    destination = PROJECT_ROOT / "models" / destination_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destination)
    print(f"Checkpoint copied to: {destination}")
    print(f"Training artifacts: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
