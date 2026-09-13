"""YOLO inference pipeline with OpenCV ROI and contrast preprocessing."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .models import Detection, InspectionResult
from .preprocess import apply_clahe, crop_roi


class InspectionPipeline:
    def __init__(
        self,
        weights: str | Path,
        image_size: int = 640,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
    ) -> None:
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {weights_path}. Train first or pass --weights."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

        self.model = YOLO(str(weights_path))
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

    def inspect(
        self,
        image: np.ndarray,
        source: str = "memory",
        roi_xywh: tuple[int, int, int, int] | None = None,
        mm_per_pixel: float | None = None,
        max_allowed_defects: int = 0,
        min_defect_area_mm2: float = 0.0,
    ) -> InspectionResult:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if mm_per_pixel is not None and mm_per_pixel <= 0:
            raise ValueError("mm_per_pixel must be positive")

        cropped, (offset_x, offset_y) = crop_roi(image, roi_xywh)
        model_input = (
            apply_clahe(cropped, self.clahe_clip_limit, self.clahe_grid_size)
            if self.use_clahe
            else cropped
        )

        started = perf_counter()
        predictions = self.model.predict(
            source=model_input,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        elapsed_ms = (perf_counter() - started) * 1000.0

        result = predictions[0]
        names: dict[int, str] = result.names
        detections: list[Detection] = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            for box, class_id, score in zip(xyxy, classes, confidences, strict=True):
                detection = Detection(
                    class_id=int(class_id),
                    class_name=str(names[int(class_id)]),
                    confidence=float(score),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                ).shifted(offset_x, offset_y)
                detections.append(detection)

        counted = detections
        if mm_per_pixel is not None and min_defect_area_mm2 > 0:
            counted = [
                detection
                for detection in detections
                if detection.area_px2 * mm_per_pixel**2 >= min_defect_area_mm2
            ]
        decision = "NG" if len(counted) > max_allowed_defects else "OK"
        height, width = image.shape[:2]
        return InspectionResult(
            source=source,
            width=width,
            height=height,
            detections=detections,
            inference_ms=elapsed_ms,
            decision=decision,
            decision_defect_count=len(counted),
            roi_xywh=roi_xywh,
            mm_per_pixel=mm_per_pixel,
        )


def load_image(path: str | Path) -> np.ndarray:
    """Read paths containing Chinese characters reliably on Windows."""
    path = Path(path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Write paths containing Chinese characters reliably on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"Failed to encode image as {suffix}")
    encoded.tofile(path)
