import cv2
import numpy as np

def enhanceImage(image):
    # 1. Слегка размываем (Denoising), чтобы убить мелкий шум камеры, 
    # сохраняя при этом резкими края самих объектов.
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Переводим в LAB для выравнивания света
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 3. Мягкий CLAHE
    # Снижаем clipLimit до 3.0, чтобы не было "кислотности", 
    # но размер сетки делаем больше (16x16), чтобы выровнять глобальные перепады света.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    l_eq = clahe.apply(l)

    # 4. Собираем обратно
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 5. Легкое углубление теней (Gamma Correction)
    # Гамма 1.2-1.3 отлично затемнит щели между продуктами, не сжигая блики.
    gamma = 1.3
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)

    # 6. Опционально: легкий blur, чтобы еще больше сгладить этикетки 
    # перед этапом сегментации (помогает собрать продукт в одно "пятно")
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return enhanced
