import cv2
import numpy as np

def segment_products(enhanced_img):
    """
    Выполняет сегментацию изображения.
    Возвращает бинарную маску (0/255).
    """

    # --- 1. Перевод в HSV ---
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)

    # --- 2. Цветовые диапазоны для продуктов ---
    # Красные банки (Coca-Cola)
    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red_mask = red1 | red2

    # Синие банки (Pepsi)
    blue_mask = cv2.inRange(hsv, (90, 80, 80), (130, 255, 255))

    # Жёлтые/зелёные (Lipton, Sprite)
    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))

    # Белые/серые (молочные продукты)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))

    # --- 3. Объединение всех масок ---
    color_mask = red_mask | blue_mask | yellow_mask | green_mask | white_mask

    # --- 4. Fallback: Otsu thresholding ---
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 5. Итоговая маска ---
    final_mask = cv2.bitwise_or(color_mask, otsu_mask)

    return final_mask
