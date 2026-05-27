MODULE_NAME = "Utils"

import os
import cv2


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ORIGINAL_DIR = os.path.join(OUTPUT_DIR, "01_original")
ENHANCED_DIR = os.path.join(OUTPUT_DIR, "02_enhanced")
SEGMENT_DIR = os.path.join(OUTPUT_DIR, "03_segmentation_mask")
CLEAN_DIR = os.path.join(OUTPUT_DIR, "04_cleaned_mask")
DETECT_DIR = os.path.join(OUTPUT_DIR, "05_detection_result")
DECISION_DIR = os.path.join(OUTPUT_DIR, "06_decision")


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
        print(f"{MODULE_NAME}: Create dir {path}")
        os.makedirs(path, exist_ok=True)


def read_image(path):
    print(f"{MODULE_NAME}: Read images by {path}")
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"{MODULE_NAME}: Cannot open image: {path}")
    return image


def save_image(path, image):
    print(f"{MODULE_NAME}: Save image to {path}")
    cv2.imwrite(path, image)
