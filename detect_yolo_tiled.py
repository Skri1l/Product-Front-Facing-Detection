import os
import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(
    BASE_DIR,
    "product_front_facing_detection",
    "input_images"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "detect_yolo_tiled"
)

MODEL_PATH = "yolov8x.pt"   # stronger than yolov8n.pt
CONFIDENCE = 0.10
TILE_SIZE = 640
OVERLAP = 0.30

PRODUCT_CLASSES = {
    "bottle",
    "cup",
    "wine glass"
}


def create_directory(path):
    os.makedirs(path, exist_ok=True)


def nms_boxes(detections, iou_threshold=0.4):
    if not detections:
        return []

    boxes = []
    scores = []

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(d["confidence"])

    indexes = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=CONFIDENCE,
        nms_threshold=iou_threshold
    )

    if len(indexes) == 0:
        return []

    indexes = indexes.flatten()
    return [detections[i] for i in indexes]


def detect_on_tile(model, tile, offset_x, offset_y):
    results = model(
        tile,
        conf=CONFIDENCE,
        imgsz=1280,
        verbose=False
    )[0]

    detections = []

    if results.boxes is None:
        return detections

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, classes):
        class_name = results.names[cls_id]

        if class_name not in PRODUCT_CLASSES:
            continue

        x1, y1, x2, y2 = box.astype(int)

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


def detect_products_tiled(image_path, model):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    h, w = image.shape[:2]

    step = int(TILE_SIZE * (1 - OVERLAP))
    all_detections = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            x2 = min(x + TILE_SIZE, w)
            y2 = min(y + TILE_SIZE, h)

            tile = image[y:y2, x:x2]

            if tile.shape[0] < 100 or tile.shape[1] < 100:
                continue

            detections = detect_on_tile(model, tile, x, y)
            all_detections.extend(detections)

    final_detections = nms_boxes(all_detections)

    return image, final_detections


def draw_detections(image, detections):
    result = image.copy()

    for i, d in enumerate(detections, start=1):
        x1, y1, x2, y2 = d["bbox"]
        label = f'{i}: {d["class_name"]} {d["confidence"]:.2f}'

        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            result,
            label,
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

    return result


def main():
    create_directory(OUTPUT_DIR)

    model = YOLO(MODEL_PATH)

    image_files = [
        file for file in os.listdir(INPUT_DIR)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for image_name in image_files:
        image_path = os.path.join(INPUT_DIR, image_name)

        image, detections = detect_products_tiled(image_path, model)
        output_image = draw_detections(image, detections)

        name_without_ext = os.path.splitext(image_name)[0]
        output_path = os.path.join(
            OUTPUT_DIR,
            f"{name_without_ext}_detected.png"
        )

        cv2.imwrite(output_path, output_image)

        print(f"{image_name}: detected {len(detections)} product-like objects")

    print("Done.")


if __name__ == "__main__":
    main()