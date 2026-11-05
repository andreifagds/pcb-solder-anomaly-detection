"""Dataset layout and ROI cropping.

data/<board>/
    rois.json          {"roi_size": 512, "rois": {"roi_001": [x, y], ...}}   top-left corners, panel pixels
    train/*.png|jpg    good panel photos (already aligned to the same reference)
    test/good/*.png    optional: good panels for evaluation
    test/defect/*.png  optional: panels with defects (+ test/defect/labels.json: {"file.png": ["roi_007"]})
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT) if folder.exists() else []


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_rois(board_dir: str | Path) -> tuple[int, dict[str, tuple[int, int]]]:
    cfg = json.loads((Path(board_dir) / "rois.json").read_text())
    return int(cfg.get("roi_size", 512)), {k: tuple(v[:2]) for k, v in cfg["rois"].items()}


def crop(panel: np.ndarray, xy: tuple[int, int], size: int) -> np.ndarray:
    x, y = xy
    c = panel[y : y + size, x : x + size]
    if c.shape[:2] != (size, size):
        raise ValueError(f"ROI at {xy} with size {size} falls outside the {panel.shape[1]}x{panel.shape[0]} image")
    return c
