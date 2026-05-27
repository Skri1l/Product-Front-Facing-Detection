# Product Front-Facing Detection Pipeline

Automated pipeline for detecting and classifying front-facing product images (bottles, cups, wine glasses). Processes raw photos through 6 sequential stages — enhancement, segmentation, mask cleaning, YOLO detection, and a scoring-based front-facing decision — saving structured results at every step.


## Project Structure

```
project/
├── input_images/                  # Place your input images here (.jpg, .jpeg, .png)
├── outputs/
│   ├── 01_original/               # Copies of original input images
│   ├── 02_enhanced/               # Denoised + CLAHE contrast-enhanced images
│   ├── 03_segmentation_mask/      # HSV color + edge segmentation masks
│   ├── 04_cleaned_mask/           # Morphology-cleaned masks (noise removed)
│   ├── 05_detection_result/       # YOLO detections drawn on original image
│   └── 06_decision/               # Per-image .txt reports + summary.csv
├── src/
│   ├── enhance.py                 # Image enhancement (bilateral filter + CLAHE)
│   ├── segment.py                 # HSV color segmentation + Canny edge mask
│   ├── clean.py                   # Morphological mask cleanup
│   ├── detect.py                  # Tiled YOLOv8 product detection
│   ├── decide.py                  # Front-facing classification + PASS/FAIL logic
│   ├── pipeline.py                # Main pipeline orchestrator
│   └── utils.py                   # Paths, image I/O helpers
├── main.py
└── requirements.txt
```

## Setup

**Python 3.8+ required.**

```bash
pip install -r requirements.txt
```

## Usage

1. Copy your images (`.jpg`, `.jpeg`, `.png`) into `input_images/`
2. Run:

```bash
python main.py
```

3. Results are saved to `outputs/` subfolders automatically.

## Pipeline Stages

| # | Module | Output folder | What it does |
|---|--------|--------------|--------------|
| 0 | `utils.py` | `01_original/` | Saves a copy of the original image |
| 1 | `enhance.py` | `02_enhanced/` | Bilateral denoising → CLAHE on L channel (LAB) → gamma correction |
| 2 | `segment.py` | `03_segmentation_mask/` | HSV color ranges (red, blue, yellow, green, orange, white) combined with Canny edge mask |
| 3 | `clean.py` | `04_cleaned_mask/` | Morphological open/close + removes connected components smaller than 100 px |
| 4 | `detect.py` | `05_detection_result/` | Tiled YOLOv8 inference (640px tiles, 30% overlap) with NMS; detects `bottle`, `cup`, `wine glass` |
| 5 | `decide.py` | `06_decision/` | Scores each detection by color ratio, edge density, mask coverage, aspect ratio → labels `front-facing` or `not-front/uncertain` |

## Decision Logic

Each detected product is scored on a **0–1 scale**:

| Factor | Weight | Description |
|--------|--------|-------------|
| Color ratio | 35% | Colorful/visible pixels in central crop (HSV saturation + value) |
| Edge density | 30% | Canny edge density normalized to 0.08 baseline |
| Mask coverage | 25% | Cleaned mask fill in the central region of the bounding box |
| Aspect ratio | 10% | Favors portrait/square products (0.18–1.10 ratio = full score) |

- Score **≥ 0.34** → `front-facing`
- Score **< 0.34** → `not-front/uncertain`
- If **≥ 70%** of detected products are front-facing → image gets **`PASS`**, otherwise **`FAIL`**

## Output Files

For each input image `photo.jpg`, the pipeline produces:

| File | Location |
|------|----------|
| `photo_original.png` | `01_original/` |
| `photo_enhanced.png` | `02_enhanced/` |
| `photo_segmentation_mask.png` | `03_segmentation_mask/` |
| `photo_cleaned_mask.png` | `04_cleaned_mask/` |
| `photo_detection_result.png` | `05_detection_result/` |
| `photo_decision.txt` | `06_decision/` |
| `summary.csv` | `06_decision/` |

The `summary.csv` aggregates all images with columns: `image`, `detected_products`, `front_facing`, `not_front_or_uncertain`, `compliance_percent`, `decision`.

## Notes

- Supported input formats: `.jpg`, `.jpeg`, `.png`
- Output folders are created automatically on first run
- Each stage saves results independently — partial runs can be inspected at any step
- Detection targets only: **bottle**, **cup**, **wine glass** (configurable via `PRODUCT_CLASSES` in `detect.py`)
