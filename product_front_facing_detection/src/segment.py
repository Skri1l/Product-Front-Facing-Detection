import cv2
import numpy as np

def segment_products(img):

    if img is None:
        raise ValueError("Image is None")

    # --- 1. BGR safety ---
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # --- 2. убираем текст/этикетки (очень важно) ---
    # bilateral сохраняет границы бутылок, но убирает шум текста
    smooth = cv2.bilateralFilter(img, 9, 75, 75)

    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)

    # --- 3. выделяем крупные структуры (НЕ края!) ---
    # adaptive threshold лучше чем Canny для полок
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        5
    )

    # --- 4. убираем точки (noise) ---
    kernel_small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # --- 5. закрываем разрывы в объектах ---
    kernel_med = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_med, iterations=2)

    # --- 6. убираем мелкие компоненты ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # важно: не завышать, иначе убьёшь бутылки
        if area > 1000:
            clean[labels == i] = 255

    return clean
