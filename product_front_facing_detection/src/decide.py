from __future__ import annotations

from collections import Counter

from .config import Config
from .detect import ProductDetection


def classify_product(detection: ProductDetection, config: Config) -> tuple[str, float]:
    score = 0.0

    rect_ok = detection.rectangularity >= config.rectangularity_threshold
    aspect_ok = config.min_aspect_ratio <= detection.aspect_ratio <= config.max_aspect_ratio
    edge_ok = detection.edge_density >= config.edge_density_threshold
    sym_ok = detection.symmetry_score >= config.symmetry_threshold
    texture_ok = detection.front_texture_score >= config.front_texture_threshold

    checks = [rect_ok, aspect_ok, edge_ok, sym_ok, texture_ok]
    score = sum(checks) / len(checks)

    if score >= 0.8:
        return "FRONT_FACING", score
    if score <= 0.35:
        return "NOT_FRONT_FACING", 1.0 - score
    return "UNCERTAIN", 0.5 + abs(score - 0.5)


def make_final_decision(detections: list[ProductDetection], pass_threshold: float) -> dict:
    if not detections:
        return {
            "total": 0,
            "front_facing": 0,
            "not_front_facing": 0,
            "uncertain": 0,
            "front_ratio": 0.0,
            "decision": "NO_PRODUCTS_DETECTED",
        }

    counts = Counter(d.label for d in detections)
    total = len(detections)
    front = counts.get("FRONT_FACING", 0)
    ratio = front / total
    final_label = "PASS" if ratio >= pass_threshold else "FAIL"
    return {
        "total": total,
        "front_facing": front,
        "not_front_facing": counts.get("NOT_FRONT_FACING", 0),
        "uncertain": counts.get("UNCERTAIN", 0),
        "front_ratio": ratio,
        "decision": final_label,
    }
