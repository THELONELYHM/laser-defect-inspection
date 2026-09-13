#!/usr/bin/env python
"""Run industrial inspection on images, a video, or a camera stream."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from laser_inspection.pipeline import InspectionPipeline, load_image, save_image
from laser_inspection.preprocess import parse_roi
from laser_inspection.reporting import save_report
from laser_inspection.visualize import draw_result

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="image, directory, video, or camera index")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models" / "best.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "inference")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu", help="cpu, 0, 0,1, ...")
    parser.add_argument("--roi", help="x,y,width,height in source-image pixels")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--mm-per-pixel", type=float)
    parser.add_argument("--min-area-mm2", type=float, default=0.0)
    parser.add_argument("--max-defects", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    return parser


def make_pipeline(args: argparse.Namespace) -> InspectionPipeline:
    return InspectionPipeline(
        weights=args.weights.resolve(),
        image_size=args.imgsz,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
        use_clahe=args.clahe,
    )


def inspect_image_file(
    pipeline: InspectionPipeline,
    source_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> str:
    image = load_image(source_path)
    roi = parse_roi(args.roi, image.shape)
    result = pipeline.inspect(
        image,
        source=str(source_path.resolve()),
        roi_xywh=roi,
        mm_per_pixel=args.mm_per_pixel,
        max_allowed_defects=args.max_defects,
        min_defect_area_mm2=args.min_area_mm2,
    )
    annotated = draw_result(image, result)
    output_stem = output_dir / source_path.stem
    save_image(output_stem.with_name(output_stem.name + "_annotated.jpg"), annotated)
    save_report(output_stem.with_name(output_stem.name + "_report.json"), result)
    if args.show:
        cv2.imshow("Laser Defect Inspection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return result.decision


def image_sources(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return sorted(
            path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
    return []


def inspect_stream(
    pipeline: InspectionPipeline,
    source: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> int:
    camera_source: int | str = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(camera_source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video/camera source: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "stream_annotated.mp4"
    output_jsonl = output_dir / "stream_report.jsonl"
    writer = None
    frame_index = 0
    roi = None
    with output_jsonl.open("w", encoding="utf-8") as report_file:
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if roi is None:
                    roi = parse_roi(args.roi, frame.shape)
                result = pipeline.inspect(
                    frame,
                    source=f"{source}#frame={frame_index}",
                    roi_xywh=roi,
                    mm_per_pixel=args.mm_per_pixel,
                    max_allowed_defects=args.max_defects,
                    min_defect_area_mm2=args.min_area_mm2,
                )
                annotated = draw_result(frame, result)
                if writer is None:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 1:
                        fps = 25.0
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create output video: {output_video}")
                writer.write(annotated)
                report_file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                if args.show:
                    cv2.imshow("Laser Defect Inspection - Q to quit", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                frame_index += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
    print(f"Processed {frame_index} frames")
    print(f"Video: {output_video}")
    print(f"Reports: {output_jsonl}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.mm_per_pixel is not None and args.mm_per_pixel <= 0:
        raise ValueError("--mm-per-pixel must be positive")
    if args.min_area_mm2 > 0 and args.mm_per_pixel is None:
        raise ValueError("--min-area-mm2 requires --mm-per-pixel")
    if args.max_defects < 0:
        raise ValueError("--max-defects cannot be negative")

    pipeline = make_pipeline(args)
    output_dir = args.output.resolve()
    source_path = Path(args.source)
    images = image_sources(source_path)
    if images:
        decisions = {"OK": 0, "NG": 0}
        for image_path in images:
            decision = inspect_image_file(pipeline, image_path, output_dir, args)
            decisions[decision] += 1
            print(f"{decision}: {image_path}")
        print(f"Summary: {decisions}; output={output_dir}")
        return 0

    if args.source.isdigit() or (
        source_path.is_file() and source_path.suffix.lower() in VIDEO_SUFFIXES
    ):
        return inspect_stream(pipeline, args.source, output_dir, args)
    raise FileNotFoundError(f"Unsupported or missing source: {args.source}")


if __name__ == "__main__":
    raise SystemExit(main())

