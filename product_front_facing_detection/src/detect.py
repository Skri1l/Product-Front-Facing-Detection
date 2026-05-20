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
    return float(np.count_nonzero(edges) / edges.size)


def _symmetry_score(gray_roi: np.ndarray) -> float:
    h, w = gray_roi.shape
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
    return min(var_lap / 1200.0, 1.0)


def detect_products(cleaned_mask: np.ndarray, enhanced_image: np.ndarray, config: Config) -> list[ProductDetection]:
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    detections: list[ProductDetection] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.min_area or area > config.max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < config.min_width or h < config.min_height:
            continue

        aspect_ratio = w / float(h)
        if not (config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio):
            continue

        bbox_area = w * h
        rectangularity = area / float(bbox_area + 1e-6)
        if rectangularity < 0.35:
            continue

        roi = gray[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        detections.append(
            ProductDetection(
                bbox=(x, y, w, h),
                contour_area=float(area),
                aspect_ratio=float(aspect_ratio),
                rectangularity=float(rectangularity),
                edge_density=_edge_density(roi),
                symmetry_score=_symmetry_score(roi),
                front_texture_score=_front_texture_score(roi),
            )
        )

    detections.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
    return detections
