from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def segment_products(enhanced_image: np.ndarray, config: Config) -> np.ndarray:
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    grad_x = cv2.Sobel(gray_blur, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_blur, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0))
    edges = cv2.Canny(gray_blur, 40, 140)
    edges = cv2.bitwise_or(edges, cv2.threshold(grad, 40, 255, cv2.THRESH_BINARY)[1])
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    sat_mask = cv2.inRange(hsv[:, :, 1], 28, 255)

    if config.segmentation_mode == "adaptive":
        mask = adaptive
    elif config.segmentation_mode == "canny":
        mask = edges
    elif config.segmentation_mode == "hsv":
        mask = sat_mask
    else:
        base = cv2.bitwise_or(otsu, adaptive)
        base = cv2.bitwise_or(base, edges)
        mask = cv2.bitwise_or(base, sat_mask)

    return cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
