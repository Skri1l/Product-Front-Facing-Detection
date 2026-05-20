from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def _cleanup_components(mask: np.ndarray, config: Config) -> np.ndarray:
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    morphed = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morphed, connectivity=8)
    cleaned = np.zeros_like(mask)
    min_component_area = min(config.min_component_area, config.min_area)
    max_component_area = int(mask.shape[0] * mask.shape[1] * 0.35)

    for i in range(1, n_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        touches_border = x <= 1 or y <= 1 or (x + w) >= (mask.shape[1] - 1) or (y + h) >= (mask.shape[0] - 1)
        if min_component_area <= area <= max_component_area and not touches_border:
            cleaned[labels == i] = 255
    return cleaned


def clean_mask(binary_mask: np.ndarray, config: Config) -> np.ndarray:
    direct = _cleanup_components(binary_mask, config)
    inverse = _cleanup_components(cv2.bitwise_not(binary_mask), config)

    # Prefer polarity that keeps more object-like components.
    direct_count = int(cv2.countNonZero(direct))
    inverse_count = int(cv2.countNonZero(inverse))
    if inverse_count > direct_count * 1.2:
        return inverse
    return direct
