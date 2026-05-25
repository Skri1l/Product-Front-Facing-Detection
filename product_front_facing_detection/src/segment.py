import cv2
import numpy as np

def segment_products(img):
    """
    Сегментация продуктов + очистка + вертикальное объединение масок
    """

    if img is None:
        raise ValueError("Image is None")

    # --- 1. гарантируем BGR ---
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # --- 2. мягкое подавление шума (важно для этикеток) ---
    smooth = cv2.bilateralFilter(img, 9, 75, 75)

    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)

    # --- 3. бинаризация (лучше чем Canny для твоего кейса) ---
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        5
    )

    # --- 4. убрать мелкий шум (точки) ---
    kernel_small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # --- 5. закрыть разрывы внутри объектов ---
    kernel_med = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_med, iterations=2)

    # --- 6. фильтрация маленьких объектов (этикетки/шум) ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > 1000:   # регулируется под сцену
            clean[labels == i] = 255

    # --- 7. ВЕРТИКАЛЬНОЕ ОБЪЕДИНЕНИЕ ПРОДУКТОВ ---
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)

    # --- 8. финальная нормализация ---
    clean = cv2.threshold(clean, 127, 255, cv2.THRESH_BINARY)[1]

    return clean
