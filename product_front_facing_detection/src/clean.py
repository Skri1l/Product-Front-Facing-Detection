import cv2
import numpy as np
from pathlib import Path


def cleanProducts(
    imagePath,
    open_kernel=(3, 3),
    close_kernel=(5, 5),
    dilate_kernel=(3, 3),
    min_area=500
):
    segmentedImg = cv2.imread(str(imagePath), cv2.IMREAD_GRAYSCALE)

    if segmentedImg is None:
        raise ValueError(f"Cannot load image: {imagePath}")

    _, binary = cv2.threshold(segmentedImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        open_kernel
    )

    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel_open
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        close_kernel
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        dilate_kernel
    )

    dilated = cv2.dilate(
        closed,
        kernel_dilate,
        iterations=1
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated,
        connectivity=8
    )

    cleaned = np.zeros_like(dilated, dtype=np.uint8)

    for i in range(1, num_labels):  # 0 = background
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == i] = 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    return np.ascontiguousarray(cleaned).astype(np.uint8)
