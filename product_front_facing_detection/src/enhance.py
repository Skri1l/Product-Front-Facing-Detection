import cv2
import numpy as np


def enhanceImage(image):
    """
    Enhance image for product separation:
    - boosts contrast
    - normalizes lighting
    - enhances edges
    """

    # 1. LAB -> работаем с яркостью отдельно
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. CLAHE (локальный контраст)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # 3. возвращаем LAB
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. усиливаем локальные границы (sharpen)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # 5. лёгкое усиление теней/контраста
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=-10)

    return enhanced
