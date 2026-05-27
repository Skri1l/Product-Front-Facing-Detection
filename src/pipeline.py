from src.utils import (
    INPUT_DIR,
    ORIGINAL_DIR,
    ENHANCED_DIR,
    SEGMENT_DIR,
    CLEAN_DIR,
    DETECT_DIR,
    DECISION_DIR,
    OUTPUT_DIR,

    read_image,
    save_image,
    make_dirs
)
from src.enhance import enhance_image
from src.segment import segment_products
from src.clean import clean_mask
from src.detect import detect_products_tiled
from src.decide import make_decision, draw_detection_result, write_decision_txt

import os
import csv

def run_pipeline():
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
            summary = process_image(image_name)
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


def process_image(image_name):
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
    detections = detect_products_tiled(enhanced)

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
