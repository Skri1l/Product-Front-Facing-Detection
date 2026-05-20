from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config


@dataclass
class ProductDetection:
    bbox: tuple[int, int, int, int]
    contour_area: float
    aspect_ratio: float
    rectangularity: float
    edge_density: float
    symmetry_score: float
    front_texture_score: float
    label: str = "UNCLASSIFIED"
    confidence: float = 0.0


def _edge_density(gray_roi: np.ndarray) -> float:
    edges = cv2.Canny(gray_roi, 60, 180)
    return float(np.count_nonzero(edges) / max(edges.size, 1))


def _symmetry_score(gray_roi: np.ndarray) -> float:
    _, w = gray_roi.shape
    if w < 6:
        return 0.0
    half = w // 2
    left = gray_roi[:, :half]
    right = gray_roi[:, w - half :]
    right_flip = cv2.flip(right, 1)
    diff = cv2.absdiff(left, right_flip)
    return float(1.0 - np.mean(diff) / 255.0)


def _front_texture_score(gray_roi: np.ndarray) -> float:
    lap = cv2.Laplacian(gray_roi, cv2.CV_32F)
    var_lap = float(lap.var())
    return min(var_lap / 1000.0, 1.0)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def _passes_geom(area: float, w: int, h: int, rect: float, min_area: int, config: Config) -> bool:
    if area < min_area or area > config.max_area:
        return False
    if w < config.min_width or h < config.min_height:
        return False
    ar = w / float(h)
    if not (config.min_aspect_ratio <= ar <= config.max_aspect_ratio):
        return False
    if rect < 0.18:
        return False
    return True


def _build_detection(contour: np.ndarray, gray: np.ndarray) -> ProductDetection | None:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return None
    bbox_area = max(w * h, 1)
    return ProductDetection(
        bbox=(x, y, w, h),
        contour_area=float(area),
        aspect_ratio=float(w / float(h)),
        rectangularity=float(area / bbox_area),
        edge_density=_edge_density(roi),
        symmetry_score=_symmetry_score(roi),
        front_texture_score=_front_texture_score(roi),
    )


def _deduplicate(dets: list[ProductDetection], iou_thr: float = 0.55) -> list[ProductDetection]:
    ordered = sorted(dets, key=lambda d: d.contour_area, reverse=True)
    kept: list[ProductDetection] = []
    for det in ordered:
        if all(_iou(det.bbox, k.bbox) < iou_thr for k in kept):
            kept.append(det)
    return kept


def detect_products(cleaned_mask: np.ndarray, enhanced_image: np.ndarray, config: Config) -> list[ProductDetection]:
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    image_area = gray.shape[0] * gray.shape[1]
    dynamic_min_area = max(int(image_area * 0.0008), 250)
    min_area = min(config.min_area, dynamic_min_area)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[ProductDetection] = []
    for contour in contours:
        det = _build_detection(contour, gray)
        if det and _passes_geom(det.contour_area, det.bbox[2], det.bbox[3], det.rectangularity, min_area, config):
            detections.append(det)

    edges = cv2.Canny(gray, 40, 140)
    fallback = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    fallback = cv2.dilate(fallback, np.ones((3, 3), np.uint8), iterations=1)
    fb_contours, _ = cv2.findContours(fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in fb_contours:
        det = _build_detection(contour, gray)
        if det and _passes_geom(det.contour_area, det.bbox[2], det.bbox[3], det.rectangularity, min_area, config):
            detections.append(det)

    detections = _deduplicate(detections)
    detections.sort(key=lambda d: d.contour_area, reverse=True)
    return detections[:60]
