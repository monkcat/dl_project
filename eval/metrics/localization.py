"""Localization metrics: element-IoU, page-hit rate."""
from __future__ import annotations


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Standard IoU between two (x0, y0, x1, y1) boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def element_iou(retrieved_bboxes: list[tuple], gt_bboxes: list[tuple]) -> float:
    """Max IoU across all retrieved×gt pairs (same page only — caller filters)."""
    best = 0.0
    for r in retrieved_bboxes:
        for g in gt_bboxes:
            best = max(best, bbox_iou(r, g))
    return best


def page_hit(retrieved_pages: set[int], gt_pages: set[int]) -> float:
    return 1.0 if retrieved_pages & gt_pages else 0.0
