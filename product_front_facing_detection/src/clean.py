from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def clean_mask(binary_mask, config: Config):
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    morphed = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morphed, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= config.min_area:
            cleaned[labels == i] = 255

    return cleaned
