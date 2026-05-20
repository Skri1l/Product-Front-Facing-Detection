# Product Front-Facing Detection

## 0) Quick Fix if “nothing works”
If you see `ModuleNotFoundError: No module named 'cv2'`, install dependencies first from this project folder:

```bash
cd product_front_facing_detection
pip install -r requirements.txt
```

Then run:

```bash
python main.py --input input_images --output output
```

## 1) Problem Description
In retail stores, products should face customers directly so labels are visible and shelf presentation is consistent. Incorrect product orientation can reduce visibility, weaken brand impact, and potentially reduce sales. This project builds a classical Computer Vision pipeline to automatically evaluate front-facing compliance from shelf images.

## 2) Pipeline
`image → enhance → segment → clean → detect → decide`

## 3) Methods Used
- **Enhance:** resize, bilateral denoising, LAB + CLAHE contrast enhancement, optional gamma correction.
- **Segment:** grayscale thresholding (Otsu/adaptive), Canny edges, HSV saturation mask, and mask fusion.
- **Clean:** morphological opening/closing/dilation and connected-component area filtering.
- **Detect:** contour extraction, bounding-box filtering (size/aspect ratio/rectangularity), feature extraction.
- **Decide:** transparent rule-based classification to FRONT_FACING / NOT_FRONT_FACING / UNCERTAIN, then image-level PASS/FAIL.

## 4) Installation
```bash
pip install -r requirements.txt
```

## 5) Run
```bash
python main.py --input input_images --output output
```
Optional:
```bash
python main.py --input input_images --output output --min-area 1500 --pass-threshold 0.7 --debug
```

## 6) Input
Supported image formats in input folder:
- `.jpg`
- `.jpeg`
- `.png`

Unreadable images are skipped with warning messages.

## 7) Output Per Image
For each image, an output subfolder is created:
- `01_original.jpg` – original input
- `02_enhanced.jpg` – enhanced image
- `03_segmentation_mask.jpg` – initial binary product mask
- `04_cleaned_mask.jpg` – cleaned mask after morphology/filtering
- `05_detection_result.jpg` – boxes + labels + confidence + final decision banner
- `06_decision.txt` – detailed per-image and per-product results
- `07_debug_features.txt` – optional debug thresholds/features (`--debug` mode)

## 8) Decision Logic
Each detected candidate is scored by interpretable features:
- rectangularity
- aspect ratio sanity
- edge density
- symmetry score
- front texture score

Per-product labels:
- **FRONT_FACING**: high overall feature agreement.
- **NOT_FRONT_FACING**: low agreement and weak front cues.
- **UNCERTAIN**: mixed cues.

Image-level labels:
- **NO_PRODUCTS_DETECTED** if no valid products.
- **PASS** if `front_facing / total >= pass_threshold`.
- **FAIL** otherwise.

## 9) Limitations
Potential failure cases include:
- very crowded shelves
- strong reflections/glare
- transparent packaging
- rotated camera viewpoint
- poor lighting or blur
- overlapping products
- unusual product shapes

## 10) Suggested Team Roles
- **Lead CV Engineer:** detect + decide + full integration
- **Image Processing Specialist:** enhance + segment
- **Morphology & Report Lead:** clean + visualization + report writing
- **Data & Testing Engineer:** image collection, QA, failure-case analysis
