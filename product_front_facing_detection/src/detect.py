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


def _passes_geom(area: float, w: int, h: int, rect: float, config: Config) -> bool:
    if area < config.min_area or area > config.max_area:
        return False
    if w < config.min_width or h < config.min_height:
        return False
    ar = w / float(h)
    if not (config.min_aspect_ratio <= ar <= config.max_aspect_ratio):
        return False
    if rect < 0.25:
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


def detect_products(cleaned_mask: np.ndarray, enhanced_image: np.ndarray, config: Config) -> list[ProductDetection]:
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[ProductDetection] = []
    for contour in contours:
        det = _build_detection(contour, gray)
        if det and _passes_geom(det.contour_area, det.bbox[2], det.bbox[3], det.rectangularity, config):
            detections.append(det)

    if not detections:
        edges = cv2.Canny(gray, 40, 140)
        fallback = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
        fallback = cv2.dilate(fallback, np.ones((3, 3), np.uint8), iterations=1)
        fb_contours, _ = cv2.findContours(fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in fb_contours:
            det = _build_detection(contour, gray)
            if det and _passes_geom(det.contour_area, det.bbox[2], det.bbox[3], det.rectangularity, config):
                detections.append(det)

    detections.sort(key=lambda d: d.contour_area, reverse=True)
    return detections[:40]
