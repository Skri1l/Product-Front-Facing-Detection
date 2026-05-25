import cv2
import numpy as np


# -------------------------
# 1. CLEAN MASK (NO MORPH OVERKILL)
# -------------------------
def preprocess(mask):
    mask = (mask > 0).astype(np.uint8)

    # лёгкое сглаживание, НЕ закрываем всё подряд
    mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)

    return mask


# -------------------------
# 2. FIND ROWS VIA Y PROJECTION
# -------------------------
def find_rows(mask, thresh=0.2):
    proj = np.sum(mask, axis=1) / mask.shape[1]

    rows = []
    in_row = False
    start = 0

    for i, v in enumerate(proj):
        if v > thresh and not in_row:
            start = i
            in_row = True
        elif v <= thresh and in_row:
            end = i
            rows.append((start, end))
            in_row = False

    if in_row:
        rows.append((start, len(proj) - 1))

    return rows


# -------------------------
# 3. FIND OBJECT COLUMNS VIA X PROJECTION
# -------------------------
def find_columns(row_mask, thresh=0.2):
    proj = np.sum(row_mask, axis=0) / row_mask.shape[0]

    cols = []
    in_col = False
    start = 0

    for i, v in enumerate(proj):
        if v > thresh and not in_col:
            start = i
            in_col = True
        elif v <= thresh and in_col:
            end = i
            cols.append((start, end))
            in_col = False

    if in_col:
        cols.append((start, len(proj) - 1))

    return cols


# -------------------------
# 4. MAIN DETECTION
# -------------------------
def detectProducts(image, mask):
    mask = preprocess(mask)

    rows = find_rows(mask)

    detections = []

    for y1, y2 in rows:
        row_mask = mask[y1:y2, :]

        cols = find_columns(row_mask)

        for x1, x2 in cols:
            region = mask[y1:y2, x1:x2]

            density = np.mean(region)

            # 🔥 фильтр пустых зон
            if density < 0.15:
                continue

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(density)
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
