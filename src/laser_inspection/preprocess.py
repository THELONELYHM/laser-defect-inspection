"""OpenCV preprocessing and region-of-interest helpers."""

from __future__ import annotations

import cv2
import numpy as np


def parse_roi(value: str | None, image_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    """Parse ``x,y,w,h`` and clamp it to the image boundary."""
    if value is None:
        return None
    try:
        x, y, width, height = (int(part.strip()) for part in value.split(","))
    except (ValueError, TypeError) as exc:
        raise ValueError("ROI must use the format x,y,width,height") from exc

    image_height, image_width = image_shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")
    x1 = max(0, min(x, image_width - 1))
    y1 = max(0, min(y, image_height - 1))
    x2 = max(x1 + 1, min(x + width, image_width))
    y2 = max(y1 + 1, min(y + height, image_height))
    return x1, y1, x2 - x1, y2 - y1


def crop_roi(
    image: np.ndarray, roi_xywh: tuple[int, int, int, int] | None
) -> tuple[np.ndarray, tuple[int, int]]:
    if roi_xywh is None:
        return image, (0, 0)
    x, y, width, height = roi_xywh
    return image[y : y + height, x : x + width], (x, y)


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8
) -> np.ndarray:
    """Enhance local luminance contrast while preserving a 3-channel image."""
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=(int(grid_size), int(grid_size))
    )
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize with unchanged aspect ratio and symmetric padding."""
    src_height, src_width = image.shape[:2]
    dst_height, dst_width = new_shape
    scale = min(dst_width / src_width, dst_height / src_height)
    resized_width = int(round(src_width * scale))
    resized_height = int(round(src_height * scale))
    pad_width = dst_width - resized_width
    pad_height = dst_height - resized_height
    left = int(round(pad_width / 2.0 - 0.1))
    right = int(round(pad_width / 2.0 + 0.1))
    top = int(round(pad_height / 2.0 - 0.1))
    bottom = int(round(pad_height / 2.0 + 0.1))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    output = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return output, scale, (float(left), float(top))


def restore_xyxy(
    box: tuple[float, float, float, float],
    scale: float,
    padding: tuple[float, float],
    image_shape: tuple[int, ...],
) -> tuple[float, float, float, float]:
    """Map a letterboxed ``xyxy`` box back to original image coordinates."""
    pad_x, pad_y = padding
    x1, y1, x2, y2 = box
    image_height, image_width = image_shape[:2]
    restored = (
        max(0.0, min((x1 - pad_x) / scale, float(image_width))),
        max(0.0, min((y1 - pad_y) / scale, float(image_height))),
        max(0.0, min((x2 - pad_x) / scale, float(image_width))),
        max(0.0, min((y2 - pad_y) / scale, float(image_height))),
    )
    return restored

