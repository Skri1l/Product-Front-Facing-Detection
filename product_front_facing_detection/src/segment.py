from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def segment_products(enhanced_image: np.ndarray, config: Config) -> np.ndarray:
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    edges = cv2.Canny(gray, 50, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    sat_mask = cv2.inRange(hsv[:, :, 1], 35, 255)

    if config.segmentation_mode == "adaptive":
        mask = adaptive
    elif config.segmentation_mode == "canny":
        mask = edges
    elif config.segmentation_mode == "hsv":
        mask = sat_mask
    else:
        mask = cv2.bitwise_or(otsu, adaptive)
        mask = cv2.bitwise_or(mask, edges)
        mask = cv2.bitwise_and(mask, cv2.bitwise_or(sat_mask, otsu))

    return cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
