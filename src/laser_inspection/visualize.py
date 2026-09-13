"""OpenCV rendering for detections and OK/NG decisions."""

from __future__ import annotations

import cv2
import numpy as np

from .constants import CLASS_COLORS
from .models import InspectionResult


def draw_result(image: np.ndarray, result: InspectionResult) -> np.ndarray:
    canvas = image.copy()
    if result.roi_xywh:
        x, y, width, height = result.roi_xywh
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (255, 255, 0), 1)
        cv2.putText(
            canvas,
            "ROI",
            (x + 3, max(15, y + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    for detection in result.detections:
        color = CLASS_COLORS[detection.class_id % len(CLASS_COLORS)]
        x1, y1, x2, y2 = (int(round(value)) for value in (
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
        ))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        if result.mm_per_pixel is not None:
            label += (
                f" {detection.width_px * result.mm_per_pixel:.2f}x"
                f"{detection.height_px * result.mm_per_pixel:.2f}mm"
            )
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        text_y = max(text_height + 4, y1)
        cv2.rectangle(
            canvas,
            (x1, text_y - text_height - 4),
            (x1 + text_width + 4, text_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    decision_color = (0, 180, 0) if result.decision == "OK" else (0, 0, 220)
    decision_count = (
        result.decision_defect_count
        if result.decision_defect_count is not None
        else len(result.detections)
    )
    banner = f"{result.decision} | defects={decision_count} | {result.inference_ms:.1f} ms"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), decision_color, cv2.FILLED)
    cv2.putText(
        canvas,
        banner,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
