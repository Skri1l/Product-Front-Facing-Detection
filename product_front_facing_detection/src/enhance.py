import cv2
import numpy as np

def enhanceImage(image):
    """
    Strong separation: black vs everything else
    """

    # 1. LAB (работаем с яркостью)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. CLAHE (локальный контраст)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # 3. Нормализация в [0..1]
    l = l.astype(np.float32) / 255.0

    # 4. Усиление чёрного (black crush + gamma)
    gamma = 1.6  # >1 делает тени темнее
    l = np.power(l, gamma)

    # 5. Доп. S-кривая для разделения тонов
    l = 1 / (1 + np.exp(-12 * (l - 0.45)))

    l = np.clip(l, 0, 1)
    l = (l * 255).astype(np.uint8)

    # 6. Возвращаем LAB
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 7. Усиление насыщенности (чтобы "не чёрное" выделялось сильнее)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s = cv2.multiply(s, 1.5)
    v = cv2.multiply(v, 1.15)

    hsv = cv2.merge([
        h,
        np.clip(s, 0, 255).astype(np.uint8),
        np.clip(v, 0, 255).astype(np.uint8)
    ])

    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 8. Лёгкая резкость
    kernel = np.array([
        [0, -1, 0],
        [-1, 6, -1],
        [0, -1, 0]
    ])

    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # 9. Финальный “контрастный удар”
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=-25)

    return enhanced
