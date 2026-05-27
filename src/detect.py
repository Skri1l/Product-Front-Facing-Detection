MODULE_NAME = "Detect"


from ultralytics import YOLO
import numpy as np
import cv2


CONFIDENCE = 0.10
IMGSZ = 1280
TILE_SIZE = 640
PRODUCT_CLASSES = {"bottle", "cup", "wine glass"}
OVERLAP = 0.30
# =========================
# MODEL SETTINGS
# =========================
# yolov8n.pt = faster, weaker
# yolov8s.pt = balanced
# yolov8x.pt = slower, stronger
MODEL_PATH = "yolov8x.pt"


def get_tile_starts(length, tile_size, step):
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def detect_on_tile(model, tile, offset_x, offset_y):
    results = model(
        tile,
        conf=CONFIDENCE,
        imgsz=IMGSZ,
        verbose=False
    )[0]

    detections = []

    if results.boxes is None:
        return detections

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, classes):
        class_name = results.names[int(cls_id)]

        if class_name not in PRODUCT_CLASSES:
            continue

        x1, y1, x2, y2 = box.astype(int)

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "bbox": (
                x1 + offset_x,
                y1 + offset_y,
                x2 + offset_x,
                y2 + offset_y
            ),
            "class_name": class_name,
            "confidence": float(conf)
        })

    return detections


def nms_detections(detections, iou_threshold=0.40):
    if not detections:
        return []

    boxes = []
    scores = []

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        scores.append(float(d["confidence"]))

    indexes = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=CONFIDENCE,
        nms_threshold=iou_threshold
    )

    if indexes is None or len(indexes) == 0:
        return []

    indexes = np.array(indexes).flatten()
    print(f"{MODULE_NAME}: Detected {np.size(indexes)} indexes")
    return [detections[int(i)] for i in indexes]


def detect_products_tiled(enhanced):
    print(f"{MODULE_NAME}: Detection...")
    model = YOLO(MODEL_PATH)

    h, w = enhanced.shape[:2]
    step = int(TILE_SIZE * (1 - OVERLAP))

    x_starts = get_tile_starts(w, TILE_SIZE, step)
    y_starts = get_tile_starts(h, TILE_SIZE, step)

    all_detections = []

    for y in y_starts:
        for x in x_starts:
            x2 = min(x + TILE_SIZE, w)
            y2 = min(y + TILE_SIZE, h)

            tile = enhanced[y:y2, x:x2]

            if tile.shape[0] < 80 or tile.shape[1] < 80:
                continue

            tile_detections = detect_on_tile(model, tile, x, y)
            all_detections.extend(tile_detections)

    return nms_detections(all_detections)
