MODULE_NAME = "Segment"

import cv2


def segment_products(enhanced):
    print(f"{MODULE_NAME}: Image segmentation...")
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red_mask = red1 | red2

    blue_mask = cv2.inRange(hsv, (90, 80, 80), (130, 255, 255))

    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))

    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))

    color_mask = red_mask | blue_mask | yellow_mask | green_mask | white_mask

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    final_mask = cv2.bitwise_or(color_mask, otsu_mask)
    print(f"{MODULE_NAME}: Image segmented")
    return final_mask
