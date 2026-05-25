import cv2
import numpy as np


def cleanProducts(imagePath):

    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Cannot load image")

    # 1. лёгкий шумодав (НЕ разрушает контуры)
    img = cv2.bilateralFilter(img, 7, 50, 50)

    # 2. извлекаем реальные границы
    edges = cv2.Canny(img, 40, 120)

    # 3. слегка соединяем разрывы (очень аккуратно)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges.astype(np.uint8)
