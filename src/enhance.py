MODULE_NAME = "Enhance"


import numpy as np
import cv2

def enhance_image(image):
    print(f"{MODULE_NAME}: Image enhancment...")
    denoised = cv2.bilateralFilter(image, 7, 60, 60)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    gamma = 0.9
    l = np.power(l / 255.0, gamma) * 255.0
    l = np.clip(l, 0, 255).astype(np.uint8)

    enhanced_lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    print(f"{MODULE_NAME}: Image enhanced")
    return enhanced
