# Product Front-Facing Detection

## 1. Problem Description
Retail shelves perform best when products are consistently front-facing. In real stores, products can become rotated, tilted, or partially hidden. This project provides an automatic, interpretable Computer Vision pipeline for checking whether products face customers correctly.

## 2. Team Roles and Task Division
| Name | Role | Pipeline Stage | Contribution |
|---|---|---|---|
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

## 3. Pipeline Design
The system follows the required sequence:
`image → enhance → segment → clean → detect → decide`

- **Enhance:** improve visibility and contrast.
- **Segment:** generate binary candidates for product regions.
- **Clean:** suppress noise and connect fragmented product areas.
- **Detect:** extract product contours and feature descriptors.
- **Decide:** classify each product and produce final shelf-level PASS/FAIL.

## 4. Methods Used
### Enhance
- Resize with fixed target width.
- Bilateral denoising.
- LAB color transform + CLAHE on lightness channel.
- Optional gamma correction.

### Segment
- Grayscale thresholding (Otsu + adaptive options).
- Canny edge extraction.
- HSV saturation masking.
- Combined binary mask generation.

### Clean
- Morphological opening and closing.
- Dilation refinement.
- Connected component filtering by area.

### Detect
- Contour extraction from cleaned mask.
- Bounding box and geometric filtering.
- Feature extraction:
  - contour area
  - aspect ratio
  - rectangularity
  - edge density
  - symmetry score
  - front texture score

### Decide
- Rule-based per-product classification:
  - FRONT_FACING
  - NOT_FRONT_FACING
  - UNCERTAIN
- Final image decision:
  - PASS
  - FAIL
  - NO_PRODUCTS_DETECTED

## 5. Results
Add stage-by-stage images for representative samples:
- Sample A: Original / Enhanced / Segmented / Cleaned / Detection
- Sample B: Original / Enhanced / Segmented / Cleaned / Detection
- Sample C: Failure case example

## 6. Failure Cases
Document at least 3 difficult scenarios:
1. [Insert image + explanation]
2. [Insert image + explanation]
3. [Insert image + explanation]

## 7. Conclusion
This project demonstrates a complete classical CV pipeline for product front-facing detection. The method is interpretable, modular, and suitable for university presentation/defense.

## 8. Contribution Statement
We confirm that all team members contributed to this project and report.

- Member Name / Signature / Date:
- Member Name / Signature / Date:
- Member Name / Signature / Date:
- Member Name / Signature / Date:
