import cv2
import numpy as np


def detectProducts(image, mask=None, min_area=500):
    """
    Lightweight detection using contours.
    Works great with segmentation masks.
    """

    if mask is None:
        raise ValueError("Mask is required for lightweight detection")

    # Убедимся что бинарная маска
    mask = (mask > 0).astype(np.uint8) * 255

    # Находим контуры
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        detections.append({
            "bbox": (x, y, x + w, y + h),
            "area": area,
            "confidence": 1.0  # нет модели → уверенность фиксированная
        })

    return detections

def drawDetections(image, detections):
    img = image.copy()

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            img,
            "product",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return img
