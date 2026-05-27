import numpy as np
import cv2

def remove_small_components(mask, min_area=80):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned

def clean_mask(mask):
    """
    Clean segmentation mask using morphology.
    """
    mask = (mask > 0).astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    cleaned = remove_small_components(cleaned, min_area=100)

    return cleaned
