import cv2
import numpy as np


def cleanProducts(imagePath):
    img = cv2.imread(imagePath)
    if img is None:
        raise ValueError("Cannot load image")

    # перевод в LAB для устойчивости к освещению
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # сглаживание яркости (убираем текст/шумы)
    blur = cv2.GaussianBlur(L, (15, 15), 0)

    # Canny требует 8-bit grayscale
    blur = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- CANNY EDGE DETECTION ---
    edges = cv2.Canny(blur, threshold1=50, threshold2=150)

    # усиливаем контуры (замыкание разрывов)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # можно дополнительно утолщить линии
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges
