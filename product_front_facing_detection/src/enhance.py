from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def _resize_keep_ratio(image: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= target_width:
        return image
    scale = target_width / float(w)
    return cv2.resize(image, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0 or abs(gamma - 1.0) < 1e-6:
        return image
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def enhance_image(image: np.ndarray, config: Config) -> np.ndarray:
    resized = _resize_keep_ratio(image, config.resize_width)
    denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=50, sigmaSpace=50)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    enhanced = _gamma_correction(enhanced, config.gamma)
    return enhanced
