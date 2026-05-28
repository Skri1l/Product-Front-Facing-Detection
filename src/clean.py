MODULE_NAME = "Clean"


import numpy as np
import cv2


def clean_mask(mask):
    print(f"{MODULE_NAME}: Cleaning mask...")

    if len(mask.shape) == 2:
        gray = mask
    else:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    blur = cv2.normalize(
        blur, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    edges = cv2.Canny(blur, threshold1=50, threshold2=150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel, iterations=1
    )

    edges = cv2.dilate(edges, kernel, iterations=1)

    print(f"{MODULE_NAME}: Mask cleaned")

    return edges
