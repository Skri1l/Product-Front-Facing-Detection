import cv2
import numpy as np

def enhanceImage(image):

    # =========================
    # 1. Denoise (мягче)
    # =========================
    denoised = cv2.bilateralFilter(image, 7, 60, 60)

    # =========================
    # 2. LAB contrast enhancement
    # =========================
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # мягкая gamma коррекция вместо грубого shift
    l = np.power(l / 255.0, 0.9) * 255.0
    l = l.astype(np.uint8)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # =========================
    # 3. Illumination normalization (вместо shadow painting)
    # =========================
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    illumination = cv2.GaussianBlur(gray, (0, 0), 25)

    # avoid division artifacts
    illumination = illumination.astype(np.float32) + 1e-6
    gray_f = gray.astype(np.float32)

    normalized = gray_f / illumination
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)

    # blend (не заменяем полностью!)
    normalized_bgr = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(enhanced, 0.7, normalized_bgr, 0.3, 0)

    # =========================
    # 4. Safe sharpening
    # =========================
    blur = cv2.GaussianBlur(result, (0, 0), 1.0)
    result = cv2.addWeighted(result, 1.15, blur, -0.15, 0)

    # =========================
    # 5. Clamp (важно для моделей)
    # =========================
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result
