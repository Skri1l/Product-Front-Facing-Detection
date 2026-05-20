from __future__ import annotations

import cv2

from .detect import ProductDetection


COLORS = {
    "FRONT_FACING": (0, 200, 0),
    "NOT_FRONT_FACING": (0, 0, 220),
    "UNCERTAIN": (0, 165, 255),
}


def draw_detections(image, detections: list[ProductDetection], final_decision: dict):
    canvas = image.copy()
    for det in detections:
        x, y, w, h = det.bbox
        color = COLORS.get(det.label, (255, 255, 255))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        txt = f"{det.label} {det.confidence * 100:.0f}%"
        cv2.putText(canvas, txt, (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    header = f"Decision: {final_decision['decision']} | Front ratio: {final_decision['front_ratio']:.2f}"
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 760), 30), (20, 20, 20), -1)
    cv2.putText(canvas, header, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas
