import cv2
import numpy as np


def detectProducts(image, mask, min_area=150, max_area_ratio=0.5):
    mask = (mask > 0).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    h_img, w_img = mask.shape[:2]
    img_area = h_img * w_img

    detections = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue

        if area > img_area * max_area_ratio:
            continue

        aspect = w / (h + 1e-6)

        if aspect < 0.1 or aspect > 10:
            continue

        detections.append({
            "bbox": (x, y, x + w, y + h),
            "area": area,
            "confidence": 1.0
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
