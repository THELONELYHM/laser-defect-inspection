"""Small data models shared by inference, reporting and visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width_px(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height_px(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area_px2(self) -> float:
        return self.width_px * self.height_px

    @property
    def center_px(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def shifted(self, dx: float, dy: float) -> "Detection":
        return Detection(
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            x1=self.x1 + dx,
            y1=self.y1 + dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
        )

    def to_dict(self, mm_per_pixel: float | None = None) -> dict[str, Any]:
        cx, cy = self.center_px
        item: dict[str, Any] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "bbox_xyxy_px": [round(v, 2) for v in (self.x1, self.y1, self.x2, self.y2)],
            "center_px": [round(cx, 2), round(cy, 2)],
            "width_px": round(self.width_px, 2),
            "height_px": round(self.height_px, 2),
            "area_px2": round(self.area_px2, 2),
        }
        if mm_per_pixel is not None:
            item.update(
                {
                    "center_mm": [round(cx * mm_per_pixel, 3), round(cy * mm_per_pixel, 3)],
                    "width_mm": round(self.width_px * mm_per_pixel, 3),
                    "height_mm": round(self.height_px * mm_per_pixel, 3),
                    "area_mm2": round(self.area_px2 * mm_per_pixel**2, 3),
                }
            )
        return item


@dataclass
class InspectionResult:
    source: str
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    decision: str = "OK"
    decision_defect_count: int | None = None
    roi_xywh: tuple[int, int, int, int] | None = None
    mm_per_pixel: float | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(Path(self.source)),
            "created_at": self.created_at,
            "image_size": {"width": self.width, "height": self.height},
            "roi_xywh": list(self.roi_xywh) if self.roi_xywh else None,
            "mm_per_pixel": self.mm_per_pixel,
            "decision": self.decision,
            "defect_count": len(self.detections),
            "decision_defect_count": (
                self.decision_defect_count
                if self.decision_defect_count is not None
                else len(self.detections)
            ),
            "inference_ms": round(self.inference_ms, 3),
            "detections": [d.to_dict(self.mm_per_pixel) for d in self.detections],
        }
