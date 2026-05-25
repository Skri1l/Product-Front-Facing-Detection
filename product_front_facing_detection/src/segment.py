import cv2
import numpy as np

def segment_products(enhanced_img):
    """
    Выполняет сегментацию изображения.
    Возвращает бинарную маску (0/255).
    """

    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)

    # --- 1. Цветовые маски продуктов ---
    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red_mask = red1 | red2

    blue_mask = cv2.inRange(hsv, (90, 80, 80), (130, 255, 255))

    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))

    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))

    color_mask = red_mask | blue_mask | yellow_mask | green_mask | white_mask

    # --- 2. Маска "фона" (черные / серые / тени) ---
    # ВАЖНО: это ключевая часть
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 70, 80))
    #            H    S   V      H   S   V
    # low V  -> темные области (полки, тени)
    # low S  -> серые поверхности

    # --- 3. Убираем фон из цветовой маски ---
    color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(dark_mask))

    # --- 4. Otsu fallback (тоже чистим от темного) ---
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_mask = cv2.bitwise_and(otsu_mask, cv2.bitwise_not(dark_mask))

    # --- 5. Итог ---
    final_mask = cv2.bitwise_or(color_mask, otsu_mask)

    return final_mask
