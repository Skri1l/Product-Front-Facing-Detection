import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If this file is placed in the project root, this path matches your structure:
# Product-Front-Facing-Detection/product_front_facing_detection/input_images
INPUT_DIR = os.path.join(
    BASE_DIR,
    "input_images"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ORIGINAL_DIR = os.path.join(OUTPUT_DIR, "01_original")
ENHANCED_DIR = os.path.join(OUTPUT_DIR, "02_enhanced")
SEGMENT_DIR = os.path.join(OUTPUT_DIR, "03_segmentation_mask")
CLEAN_DIR = os.path.join(OUTPUT_DIR, "04_cleaned_mask")
DETECT_DIR = os.path.join(OUTPUT_DIR, "05_detection_result")
DECISION_DIR = os.path.join(OUTPUT_DIR, "06_decision")

# =========================
# MODEL SETTINGS
# =========================
# yolov8n.pt = faster, weaker
# yolov8s.pt = balanced
# yolov8x.pt = slower, stronger
MODEL_PATH = "yolov8x.pt"
CONFIDENCE = 0.10
IMGSZ = 1280
TILE_SIZE = 640
OVERLAP = 0.30

# YOLO COCO classes close to products on shelves.
# Pretrained YOLO does not have a universal "product" class.
PRODUCT_CLASSES = {"bottle", "cup", "wine glass"}

# Final store compliance threshold
PASS_THRESHOLD_PERCENT = 70.0


# =========================
# BASIC UTILS
# =========================
def make_dirs():
    for path in [
        OUTPUT_DIR,
        ORIGINAL_DIR,
        ENHANCED_DIR,
        SEGMENT_DIR,
        CLEAN_DIR,
        DETECT_DIR,
        DECISION_DIR,
    ]:
        os.makedirs(path, exist_ok=True)


def read_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return image


def save_image(path, image):
    cv2.imwrite(path, image)


# =========================
# 1. ENHANCE
# =========================
def enhance_image(image):
    """
    Enhance image quality:
    - denoising
    - CLAHE contrast improvement
    - light gamma correction
    """
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

    return enhanced


# =========================
# 2. SEGMENT
# =========================
def segment_products(enhanced):
    """
    Create a binary mask of visually relevant product/label regions.
    It uses HSV color segmentation + edge-based segmentation.
    """
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

    # Colorful labels/products
    red1 = cv2.inRange(hsv, (0, 60, 50), (12, 255, 255))
    red2 = cv2.inRange(hsv, (168, 60, 50), (180, 255, 255))
    blue = cv2.inRange(hsv, (85, 50, 50), (135, 255, 255))
    yellow = cv2.inRange(hsv, (12, 50, 60), (38, 255, 255))
    green = cv2.inRange(hsv, (38, 40, 50), (90, 255, 255))
    orange = cv2.inRange(hsv, (5, 50, 60), (25, 255, 255))

    # Light labels / white bottles, but not too permissive
    white = cv2.inRange(hsv, (0, 0, 160), (180, 55, 255))

    color_mask = red1 | red2 | blue | yellow | green | orange | white

    # Edge mask helps with labels and product contours
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, edge_kernel, iterations=1)

    mask = cv2.bitwise_or(color_mask, edges)
    return mask


# =========================
# 3. CLEAN
# =========================
def remove_small_components(mask, min_area=80):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def clean_mask(mask):
    """
    Clean segmentation mask using morphology.
    """
    mask = (mask > 0).astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    cleaned = remove_small_components(cleaned, min_area=100)

    return cleaned


# =========================
# 4. DETECT
# =========================
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
    return [detections[int(i)] for i in indexes]


def detect_products_tiled(enhanced, model):
    """
    Detect products with tiled YOLO inference.
    Tiling improves detection of small shelf products.
    """
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


# =========================
# 5. DECIDE: FRONT-FACING CHECK
# =========================
def clamp_bbox(bbox, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    return x1, y1, x2, y2


def classify_front_facing(enhanced, cleaned_mask, bbox):
    """
    Heuristic front-facing decision.

    Idea:
    A product is probably front-facing if its central area contains enough
    visible label information: color, contrast, edges, and mask coverage.
    """
    x1, y1, x2, y2 = clamp_bbox(bbox, enhanced.shape)
    crop = enhanced[y1:y2, x1:x2]
    mask_crop = cleaned_mask[y1:y2, x1:x2]

    h, w = crop.shape[:2]

    if h < 20 or w < 10:
        return "uncertain", 0.0

    # central part of object, where label usually appears
    cx1 = int(w * 0.25)
    cx2 = int(w * 0.75)
    cy1 = int(h * 0.20)
    cy2 = int(h * 0.85)

    center = crop[cy1:cy2, cx1:cx2]
    center_mask = mask_crop[cy1:cy2, cx1:cx2]

    if center.size == 0:
        return "uncertain", 0.0

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # colorful / visible label pixels
    color_ratio = np.mean((s > 45) & (v > 60))

    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    center_mask_ratio = np.count_nonzero(center_mask) / center_mask.size
    total_mask_ratio = np.count_nonzero(mask_crop) / max(mask_crop.size, 1)

    # Product visible front usually has reasonable width/height.
    aspect = w / max(h, 1)
    if 0.18 <= aspect <= 1.10:
        aspect_score = 1.0
    elif 0.10 <= aspect <= 1.50:
        aspect_score = 0.6
    else:
        aspect_score = 0.2

    # Normalize edge density because values are usually small
    edge_score = min(edge_density / 0.08, 1.0)

    score = (
        0.35 * color_ratio +
        0.30 * edge_score +
        0.25 * center_mask_ratio +
        0.10 * aspect_score
    )

    # Extra penalty if product area has almost no visual information
    if total_mask_ratio < 0.05:
        score *= 0.65

    if score >= 0.34:
        label = "front-facing"
    else:
        label = "not-front/uncertain"

    return label, float(score)


def make_decision(enhanced, cleaned_mask, detections):
    product_results = []
    front_count = 0

    for d in detections:
        label, score = classify_front_facing(enhanced, cleaned_mask, d["bbox"])

        result = dict(d)
        result["front_label"] = label
        result["front_score"] = score
        product_results.append(result)

        if label == "front-facing":
            front_count += 1

    total = len(product_results)
    not_front_count = total - front_count
    compliance = (front_count / total * 100.0) if total > 0 else 0.0
    final_decision = "PASS" if compliance >= PASS_THRESHOLD_PERCENT else "FAIL"

    summary = {
        "detected_products": total,
        "front_facing": front_count,
        "not_front_or_uncertain": not_front_count,
        "compliance_percent": compliance,
        "decision": final_decision,
        "threshold_percent": PASS_THRESHOLD_PERCENT
    }

    return product_results, summary


# =========================
# VISUALIZATION
# =========================
def draw_detection_result(image, product_results, summary):
    result = image.copy()

    # Header background
    cv2.rectangle(result, (0, 0), (result.shape[1], 85), (0, 0, 0), -1)

    header1 = (
        f"Detected: {summary['detected_products']} | "
        f"Front-facing: {summary['front_facing']} | "
        f"Not-front/uncertain: {summary['not_front_or_uncertain']}"
    )
    header2 = (
        f"Compliance: {summary['compliance_percent']:.1f}% | "
        f"Decision: {summary['decision']} | "
        f"Threshold: {summary['threshold_percent']:.0f}%"
    )

    cv2.putText(result, header1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(result, header2, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    for i, d in enumerate(product_results, start=1):
        x1, y1, x2, y2 = clamp_bbox(d["bbox"], result.shape)

        is_front = d["front_label"] == "front-facing"
        color = (0, 255, 0) if is_front else (0, 0, 255)

        label = f"{i}: {d['front_label']} {d['front_score']:.2f}"

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            result,
            label,
            (x1, max(y1 - 5, 95)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )

    return result


def write_decision_txt(path, image_name, summary):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Image: {image_name}\n")
        f.write(f"Detected products: {summary['detected_products']}\n")
        f.write(f"Front-facing products: {summary['front_facing']}\n")
        f.write(f"Not-front-facing or uncertain: {summary['not_front_or_uncertain']}\n")
        f.write(f"Compliance: {summary['compliance_percent']:.2f}%\n")
        f.write(f"Threshold: {summary['threshold_percent']:.2f}%\n")
        f.write(f"Final decision: {summary['decision']}\n")


# =========================
# MAIN PIPELINE
# =========================
def process_image(image_name, model):
    input_path = os.path.join(INPUT_DIR, image_name)
    name = os.path.splitext(image_name)[0]

    original = read_image(input_path)

    # 0. Save original
    save_image(os.path.join(ORIGINAL_DIR, f"{name}_original.png"), original)

    # 1. Enhance
    enhanced = enhance_image(original)
    save_image(os.path.join(ENHANCED_DIR, f"{name}_enhanced.png"), enhanced)

    # 2. Segment
    segmentation_mask = segment_products(enhanced)
    save_image(os.path.join(SEGMENT_DIR, f"{name}_segmentation_mask.png"), segmentation_mask)

    # 3. Clean
    cleaned_mask = clean_mask(segmentation_mask)
    save_image(os.path.join(CLEAN_DIR, f"{name}_cleaned_mask.png"), cleaned_mask)

    # 4. Detect
    detections = detect_products_tiled(enhanced, model)

    # 5. Decide
    product_results, summary = make_decision(enhanced, cleaned_mask, detections)

    detection_image = draw_detection_result(original, product_results, summary)
    save_image(os.path.join(DETECT_DIR, f"{name}_detection_result.png"), detection_image)

    decision_path = os.path.join(DECISION_DIR, f"{name}_decision.txt")
    write_decision_txt(decision_path, image_name, summary)

    print(
        f"{image_name}: detected={summary['detected_products']}, "
        f"front={summary['front_facing']}, "
        f"compliance={summary['compliance_percent']:.1f}%, "
        f"decision={summary['decision']}"
    )

    return summary


def main():
    make_dirs()

    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(
            f"Input folder not found: {INPUT_DIR}\n"
            "Check INPUT_DIR in the script."
        )

    image_files = [
        file for file in os.listdir(INPUT_DIR)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print(f"No images found in: {INPUT_DIR}")
        return

    model = YOLO(MODEL_PATH)

    csv_path = os.path.join(DECISION_DIR, "summary.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "image",
            "detected_products",
            "front_facing",
            "not_front_or_uncertain",
            "compliance_percent",
            "decision"
        ])

        for image_name in image_files:
            summary = process_image(image_name, model)
            writer.writerow([
                image_name,
                summary["detected_products"],
                summary["front_facing"],
                summary["not_front_or_uncertain"],
                round(summary["compliance_percent"], 2),
                summary["decision"]
            ])

    print("\nDone.")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
