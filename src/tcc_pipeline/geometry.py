from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def xywh_to_xyxy(box: Iterable[float]) -> np.ndarray:
    x, y, w, h = map(float, box)
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def xyxy_to_xywh(box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def box_iou_xywh(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def relative_area(box_xywh: Iterable[float], image_width: float, image_height: float) -> float:
    _, _, w, h = map(float, box_xywh)
    denom = float(image_width) * float(image_height)
    return (max(w, 0.0) * max(h, 0.0) / denom) if denom > 0 else 0.0


def size_bin(relative: float, small_max: float, medium_max: float) -> str:
    if relative < small_max:
        return "small"
    if relative < medium_max:
        return "medium"
    return "large"
