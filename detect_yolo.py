# detect_yolo.py

import os
import cv2
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
    "detect_yolo"
)

# Pretrained model. It will detect common objects like bottles/cups.
# For ALL shop products/cans, you need custom training later.
MODEL_PATH = "yolov8n.pt"

CONFIDENCE = 0.25

# COCO classes that are close to products
PRODUCT_LIKE_CLASSES = {
    "bottle",
    "cup",
    "wine glass",
    "bowl"
}


def create_directory(path):
    os.makedirs(path, exist_ok=True)


def detect_products(image_path, model):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    results = model(image, conf=CONFIDENCE)[0]

    detections = []

    if results.boxes is None:
        return image, detections

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, classes):
        class_name = results.names[cls_id]

        # remove this IF if you want to draw every detected class
        if class_name not in PRODUCT_LIKE_CLASSES:
            continue

        x1, y1, x2, y2 = box.astype(int)

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "class_name": class_name,
            "confidence": float(conf)
        })

    return image, detections


def draw_detections(image, detections):
    result = image.copy()

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f'{d["class_name"]} {d["confidence"]:.2f}'

        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            result,
            label,
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return result


def main():
    create_directory(OUTPUT_DIR)

    model = YOLO(MODEL_PATH)

    image_files = [
        file for file in os.listdir(INPUT_DIR)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print("No input images found.")
        return

    for image_name in image_files:
        image_path = os.path.join(INPUT_DIR, image_name)

        image, detections = detect_products(image_path, model)
        output_image = draw_detections(image, detections)

        name_without_ext = os.path.splitext(image_name)[0]
        output_path = os.path.join(
            OUTPUT_DIR,
            f"{name_without_ext}_detected.png"
        )

        cv2.imwrite(output_path, output_image)

        print(f"{image_name}: detected {len(detections)} objects")

    print("Detection completed.")


if __name__ == "__main__":
    main()