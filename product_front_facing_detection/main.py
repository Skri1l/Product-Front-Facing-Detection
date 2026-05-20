from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    if exc.name == "cv2":
        print(
            "[ERROR] Missing dependency: cv2 (OpenCV).\n"
            "Install project dependencies first:\n"
            "  pip install -r requirements.txt\n"
            "Then run again, for example:\n"
            "  python main.py --input input_images --output output",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise

from src.clean import clean_mask
from src.config import Config
from src.decide import classify_product, make_final_decision
from src.detect import detect_products
from src.enhance import enhance_image
from src.segment import segment_products
from src.utils import gather_images, save_image
from src.visualize import draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Product Front-Facing Detection")
    parser.add_argument("--input", required=True, type=Path, help="Input image folder")
    parser.add_argument("--output", required=True, type=Path, help="Output folder")
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--pass-threshold", type=float, default=0.7)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--segmentation-mode", default="combined", choices=["combined", "adaptive", "canny", "hsv"])
    return parser.parse_args()


def process_image(image_path: Path, output_root: Path, config: Config) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] Cannot load image, skipping: {image_path}")
        return {"skipped": True, "image": image_path.name}

    out_dir = output_root / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    save_image(out_dir / "01_original.jpg", image)
    enhanced = enhance_image(image, config)
    save_image(out_dir / "02_enhanced.jpg", enhanced)

    seg_mask = segment_products(enhanced, config)
    save_image(out_dir / "03_segmentation_mask.jpg", seg_mask)

    cleaned = clean_mask(seg_mask, config)
    save_image(out_dir / "04_cleaned_mask.jpg", cleaned)

    detections = detect_products(cleaned, enhanced, config)
    for det in detections:
        det.label, det.confidence = classify_product(det, config)

    final_decision = make_final_decision(detections, config.pass_threshold)
    vis = draw_detections(enhanced, detections, final_decision)
    save_image(out_dir / "05_detection_result.jpg", vis)

    decision_lines = [
        f"image_name: {image_path.name}",
        f"total_detected_products: {final_decision['total']}",
        f"front_facing_count: {final_decision['front_facing']}",
        f"not_front_facing_count: {final_decision['not_front_facing']}",
        f"uncertain_count: {final_decision['uncertain']}",
        f"front_facing_ratio: {final_decision['front_ratio']:.4f}",
        f"final_decision: {final_decision['decision']}",
        "",
        "per_product_features:",
    ]
    for idx, det in enumerate(detections, 1):
        decision_lines.extend(
            [
                f"  product_{idx}:",
                f"    bbox: {det.bbox}",
                f"    contour_area: {det.contour_area:.2f}",
                f"    aspect_ratio: {det.aspect_ratio:.4f}",
                f"    rectangularity: {det.rectangularity:.4f}",
                f"    edge_density: {det.edge_density:.4f}",
                f"    symmetry_score: {det.symmetry_score:.4f}",
                f"    front_texture_score: {det.front_texture_score:.4f}",
                f"    label: {det.label}",
                f"    confidence: {det.confidence:.4f}",
            ]
        )

    (out_dir / "06_decision.txt").write_text("\n".join(decision_lines), encoding="utf-8")

    if config.debug:
        debug_lines = [
            f"thresholds: rect>={config.rectangularity_threshold}, edge>={config.edge_density_threshold}, "
            f"sym>={config.symmetry_threshold}, tex>={config.front_texture_threshold}",
            f"segmentation_mode: {config.segmentation_mode}",
        ]
        for idx, det in enumerate(detections, 1):
            debug_lines.append(
                f"{idx}: rect={det.rectangularity:.3f}, ar={det.aspect_ratio:.3f}, edge={det.edge_density:.3f}, "
                f"sym={det.symmetry_score:.3f}, tex={det.front_texture_score:.3f}, label={det.label}"
            )
        (out_dir / "07_debug_features.txt").write_text("\n".join(debug_lines), encoding="utf-8")

    return {"skipped": False, "image": image_path.name, **final_decision}


def main() -> None:
    args = parse_args()
    cfg = Config(min_area=args.min_area, pass_threshold=args.pass_threshold, debug=args.debug, segmentation_mode=args.segmentation_mode)

    if not args.input.exists() or not args.input.is_dir():
        raise SystemExit(f"Input directory not found: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    image_paths = gather_images(args.input)
    if not image_paths:
        print(f"No valid images found in {args.input}")
        return

    results = [process_image(p, args.output, cfg) for p in image_paths]

    processed = [r for r in results if not r.get("skipped")]
    print("\n=== Processing Summary ===")
    print(f"Total files found: {len(image_paths)}")
    print(f"Processed images: {len(processed)}")
    print(f"Skipped images: {len(results) - len(processed)}")
    for r in processed:
        print(f"- {r['image']}: {r['decision']} (front_ratio={r['front_ratio']:.2f}, total={r['total']})")


if __name__ == "__main__":
    main()
