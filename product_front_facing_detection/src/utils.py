from __future__ import annotations

from pathlib import Path

import cv2


VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def gather_images(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES])


def save_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
