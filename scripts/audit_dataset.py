#!/usr/bin/env python
"""Audit YOLO labels, class balance and bounding-box geometry."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "neu_det" / "dataset.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dataset_audit.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = (PROJECT_ROOT / dataset_root).resolve()
    names = {int(key): value for key, value in config["names"].items()}

    problems: list[str] = []
    class_objects: Counter[str] = Counter()
    image_counts: dict[str, int] = {}
    box_areas: list[float] = []
    box_aspects: list[float] = []
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = sorted(path for path in image_dir.glob("*") if path.is_file())
        labels = sorted(label_dir.glob("*.txt"))
        image_counts[split] = len(images)
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        for stem in sorted(image_stems - label_stems):
            problems.append(f"{split}: image without label: {stem}")
        for stem in sorted(label_stems - image_stems):
            problems.append(f"{split}: label without image: {stem}")

        for label_path in labels:
            seen_lines: set[str] = set()
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                normalized_line = " ".join(line.split())
                if normalized_line in seen_lines:
                    problems.append(f"{label_path}:{line_number}: duplicate label")
                seen_lines.add(normalized_line)
                parts = line.split()
                if len(parts) != 5:
                    problems.append(f"{label_path}:{line_number}: expected 5 fields")
                    continue
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                except ValueError:
                    problems.append(f"{label_path}:{line_number}: non-numeric value")
                    continue
                if class_id not in names:
                    problems.append(f"{label_path}:{line_number}: invalid class {class_id}")
                if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
                    problems.append(f"{label_path}:{line_number}: coordinate out of [0,1]")
                if width <= 0.0 or height <= 0.0:
                    problems.append(f"{label_path}:{line_number}: non-positive box")
                    continue
                class_objects[names.get(class_id, f"unknown_{class_id}")] += 1
                box_areas.append(width * height)
                box_aspects.append(width / height)

    summary = {
        "dataset_root": str(dataset_root),
        "images": image_counts,
        "total_images": sum(image_counts.values()),
        "objects_per_class": dict(sorted(class_objects.items())),
        "total_objects": sum(class_objects.values()),
        "normalized_box_area": {
            "min": min(box_areas, default=0.0),
            "median": statistics.median(box_areas) if box_areas else 0.0,
            "max": max(box_areas, default=0.0),
        },
        "box_aspect_ratio": {
            "min": min(box_aspects, default=0.0),
            "median": statistics.median(box_aspects) if box_aspects else 0.0,
            "max": max(box_aspects, default=0.0),
        },
        "problem_count": len(problems),
        "problems": problems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
