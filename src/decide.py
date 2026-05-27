MODULE_NAME = "Decide"


import numpy as np
import cv2


PASS_THRESHOLD_PERCENT = 70.0


def clamp_bbox(bbox, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    return x1, y1, x2, y2


def classify_front_facing(enhanced, cleaned_mask, bbox):
    print(f"{MODULE_NAME}: Classifying front facing...")
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

    print(f"{MODULE_NAME}: Classified - {label} : {score}")
    return label, float(score)


def make_decision(enhanced, cleaned_mask, detections):
    print(f"{MODULE_NAME}: Making decision...")
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
    
    print(f"{MODULE_NAME}: Decision made {product_results} {summary}")
    return product_results, summary


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
