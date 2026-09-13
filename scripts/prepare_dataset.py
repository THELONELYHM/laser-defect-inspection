#!/usr/bin/env python
"""Convert NEU-DET Pascal VOC annotations into a stratified YOLO dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from laser_inspection.constants import CLASS_NAMES, normalize_class_name

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def find_voc_directories(root: Path) -> tuple[Path, Path]:
    annotation_dirs = [
        path for path in root.rglob("*") if path.is_dir() and path.name.lower() == "annotations"
    ]
    for annotations in annotation_dirs:
        siblings = {path.name.lower(): path for path in annotations.parent.iterdir() if path.is_dir()}
        for candidate in ("images", "jpegimages"):
            if candidate in siblings:
                return annotations, siblings[candidate]
    raise FileNotFoundError(
        f"Could not find sibling ANNOTATIONS and IMAGES/JPEGImages directories under {root}"
    )


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
        candidate_upper = images_dir / f"{stem}{suffix.upper()}"
        if candidate_upper.exists():
            return candidate_upper
    return None


def parse_voc(xml_path: Path) -> tuple[int, int, list[tuple[int, float, float, float, float]]]:
    root = ET.parse(xml_path).getroot()
    width = int(root.findtext("size/width", default="0"))
    height = int(root.findtext("size/height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {xml_path}")

    labels: list[tuple[int, float, float, float, float]] = []
    seen_labels: set[tuple[int, float, float, float, float]] = set()
    for obj in root.findall("object"):
        class_name = normalize_class_name(obj.findtext("name", default=""))
        if class_name not in CLASS_NAMES:
            raise ValueError(f"Unknown class '{class_name}' in {xml_path}")
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = max(0.0, min(float(box.findtext("xmin", "0")), float(width)))
        ymin = max(0.0, min(float(box.findtext("ymin", "0")), float(height)))
        xmax = max(0.0, min(float(box.findtext("xmax", "0")), float(width)))
        ymax = max(0.0, min(float(box.findtext("ymax", "0")), float(height)))
        if xmax <= xmin or ymax <= ymin:
            continue
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        label = (CLASS_NAMES.index(class_name), x_center, y_center, box_width, box_height)
        rounded_label = tuple(round(value, 8) if isinstance(value, float) else value for value in label)
        if rounded_label not in seen_labels:
            labels.append(label)
            seen_labels.add(rounded_label)
    return width, height, labels


def stratified_split(
    records: list[tuple[Path, Path, str]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[tuple[Path, Path, str]]]:
    if abs(sum(ratios) - 1.0) > 1e-9 or any(value <= 0 for value in ratios):
        raise ValueError("Split ratios must be positive and sum to 1")
    grouped: dict[str, list[tuple[Path, Path, str]]] = defaultdict(list)
    for record in records:
        grouped[record[2]].append(record)

    randomizer = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for class_name in sorted(grouped):
        items = sorted(grouped[class_name], key=lambda item: item[0].name)
        randomizer.shuffle(items)
        train_end = int(len(items) * ratios[0])
        val_end = train_end + int(len(items) * ratios[1])
        splits["train"].extend(items[:train_end])
        splits["val"].extend(items[train_end:val_end])
        splits["test"].extend(items[val_end:])
    for items in splits.values():
        randomizer.shuffle(items)
    return splits


def _safe_clear_output(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved.name.lower() != "neu_det" or "datasets" not in {
        part.lower() for part in resolved.parts
    }:
        raise ValueError(f"Refusing to clear unexpected output directory: {resolved}")
    shutil.rmtree(resolved)


def prepare_neu_det(
    input_dir: Path,
    output_dir: Path,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    overwrite: bool = False,
) -> dict[str, object]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    existing_labels = list(output_dir.glob("labels/*/*.txt")) if output_dir.exists() else []
    if existing_labels and not overwrite:
        raise FileExistsError(
            f"Prepared data already exists in {output_dir}; pass --overwrite to rebuild"
        )
    if output_dir.exists() and overwrite:
        _safe_clear_output(output_dir)

    annotations_dir, images_dir = find_voc_directories(input_dir)
    records: list[tuple[Path, Path, str]] = []
    missing_images: list[str] = []
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        image_path = find_image(images_dir, xml_path.stem)
        if image_path is None:
            missing_images.append(xml_path.name)
            continue
        _, _, labels = parse_voc(xml_path)
        if not labels:
            continue
        primary_class = CLASS_NAMES[labels[0][0]]
        records.append((image_path, xml_path, primary_class))

    if not records:
        raise RuntimeError("No valid image/XML pairs were found")

    splits = stratified_split(records, ratios, seed)
    class_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    for split_name, items in splits.items():
        image_output = output_dir / "images" / split_name
        label_output = output_dir / "labels" / split_name
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        for image_path, xml_path, primary_class in items:
            shutil.copy2(image_path, image_output / image_path.name)
            _, _, labels = parse_voc(xml_path)
            yolo_lines = [
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                for class_id, x_center, y_center, width, height in labels
            ]
            (label_output / f"{image_path.stem}.txt").write_text(
                "\n".join(yolo_lines) + "\n", encoding="utf-8"
            )
            class_counts[f"{split_name}/{primary_class}"] += 1
            for class_id, *_ in labels:
                object_counts[CLASS_NAMES[class_id]] += 1

    dataset_yaml = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES))
    )
    (output_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    manifest: dict[str, object] = {
        "source": "NEU-DET",
        "input_directory": str(input_dir),
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "images": {name: len(items) for name, items in splits.items()},
        "objects_per_class": dict(sorted(object_counts.items())),
        "images_per_split_and_class": dict(sorted(class_counts.items())),
        "missing_images": missing_images,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"YOLO dataset written to: {output_dir}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw" / "neu_det",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "datasets" / "neu_det"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prepare_neu_det(
        args.input,
        args.output,
        seed=args.seed,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
