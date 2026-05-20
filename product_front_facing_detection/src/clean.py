from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def clean_mask(binary_mask: np.ndarray, config: Config) -> np.ndarray:
    # If segmentation marked almost the whole frame as foreground,
    # flip polarity so products become foreground components.
    white_ratio = float(np.count_nonzero(binary_mask) / max(binary_mask.size, 1))
    work_mask = cv2.bitwise_not(binary_mask) if white_ratio > 0.72 else binary_mask

    opened = cv2.morphologyEx(work_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    morphed = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morphed, connectivity=8)
    cleaned = np.zeros_like(work_mask)
    min_component_area = min(config.min_component_area, config.min_area)
    max_component_area = int(work_mask.shape[0] * work_mask.shape[1] * 0.35)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_component_area <= area <= max_component_area:
            cleaned[labels == i] = 255

    return cleaned
