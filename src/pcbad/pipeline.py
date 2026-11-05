"""Train and inspect a whole board: the glue used by scripts/ and app/demo.py.

Two modes, chosen by the presence of `rois.json` in the board folder:
  * ROI mode    : one model per ROI listed in rois.json (the production setup)
  * whole image : no rois.json -> each image is one region (handy for public
                  datasets such as VisA, where every image is a single board)

In ROI mode, training also builds the solder map (see solder.py) and saves it next to the
models as solder_map.png; inspection then ignores anomalies that are away from any solder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from .backbone import BACKBONES
from .data import crop, list_images, load_rois, read_rgb
from .patchcore import RoiPatchCore, preprocess
from .solder import build_map, load_map, near_solder, save_map

SOLDER_MAP = "solder_map.png"
from .visualize import heatmap, outline

WHOLE = "whole_image"


def _regions(board_dir: Path):
    if (board_dir / "rois.json").exists():
        size, rois = load_rois(board_dir)
        return size, rois
    return None, {WHOLE: None}


def _cut(img: np.ndarray, xy, size):
    return img if xy is None else crop(img, xy, size)


def train_board(
    board_dir: str | Path,
    out_dir: str | Path,
    backbone: str = "dinov2_vitb14",
    quantile: float = 0.90,
    seed: int = 42,
    device: str | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    board_dir, out_dir = Path(board_dir), Path(out_dir)
    train_dir = board_dir / "train" if (board_dir / "train").exists() else board_dir
    panels = list_images(train_dir)
    if len(panels) < 5:
        raise ValueError(f"need at least 5 good images in {train_dir}, found {len(panels)}")
    size, rois = _regions(board_dir)
    log(f"{len(panels)} good images · {len(rois)} region(s) · backbone {BACKBONES[backbone].label}")

    crops: dict[str, list] = {r: [] for r in rois}
    for p in panels:
        img = read_rgb(p)
        for r, xy in rois.items():
            crops[r].append(_cut(img, xy, size))

    out_dir.mkdir(parents=True, exist_ok=True)
    s = BACKBONES[backbone].input_size
    summary = {"backbone": backbone, "roi_size": size, "n_images": len(panels), "rois": {}}
    t0 = time.perf_counter()
    for r, imgs in crops.items():
        t = time.perf_counter()
        m = RoiPatchCore(backbone).to(device).fit(preprocess(imgs, s).to(device), seed=seed, calibrate_quantile=quantile)
        m.meta.update(roi=r, roi_xy=None if rois[r] is None else list(rois[r]), roi_size=size)
        m.save(out_dir / f"{r}.pt")
        summary["rois"][r] = {
            "threshold": round(m.threshold, 4),
            "coreset_ratio": round(m.meta["coreset_ratio"], 4),
            "bank_rows": m.meta["bank_shape"][0],
            "seconds": round(time.perf_counter() - t, 1),
        }
        log(f"  {r}: bank {m.meta['bank_shape']}, threshold {m.threshold:.3f}")
    if rois[next(iter(rois))] is not None:  # ROI mode: map of where the solder is, from the same photos
        log("building the solder map")
        save_map(build_map(read_rgb(p) for p in panels), out_dir / SOLDER_MAP)
    summary["total_seconds"] = round(time.perf_counter() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_models(models_dir: str | Path, device: str | None = None) -> dict[str, RoiPatchCore]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = {p.stem: RoiPatchCore.load(p, device) for p in sorted(Path(models_dir).glob("*.pt"))}
    if not models:
        raise FileNotFoundError(f"no models in {models_dir}")
    _ = next(iter(models.values())).backbone  # load the shared backbone now
    return models


def load_solder_map(models_dir: str | Path) -> np.ndarray | None:
    """The solder map saved by train_board, or None (then every flagged ROI is a defect)."""
    return load_map(Path(models_dir) / SOLDER_MAP)


@torch.inference_mode()
def inspect(models: dict[str, RoiPatchCore], img: np.ndarray, margin: float = 1.15,
            solder_zone: np.ndarray | None = None):
    """-> (annotated image, report dict, list of (roi, outlined crop, heat map crop) for flagged ROIs).

    With a solder map, a ROI above the limit whose anomaly does not touch any solder is reported
    with "away_from_solder": True and does not count as a defect (gray frame)."""
    any_m = next(iter(models.values()))
    s = BACKBONES[any_m.backbone_key].input_size
    canvas = img.copy()
    report, flagged = {}, []
    thick = max(2, img.shape[0] // 400)
    for r, m in models.items():
        xy, size = m.meta.get("roi_xy"), m.meta.get("roi_size")
        c = _cut(img, None if xy is None else tuple(xy), size)
        pred = m.predict(preprocess(c, s).to(m.device), margin=margin)[0]
        x, y = (0, 0) if xy is None else xy
        h, w = c.shape[:2]
        defect, away = pred.is_defect, False
        if defect:
            amap = cv2.resize(pred.anomaly_map, (w, h))
            limit = m.threshold * margin
            if solder_zone is not None and xy is not None:
                zone = solder_zone[y : y + h, x : x + w]
                if zone.shape == amap.shape and not near_solder(zone, amap >= limit):
                    defect, away = False, True
            if defect:
                marked = outline(c, amap, min_level=limit)
                canvas[y : y + h, x : x + w] = marked
                flagged.append((r, marked, heatmap(c, amap, vmax=float(amap.max()))))
        report[r] = {"score": round(pred.score, 3), "defect": defect, "away_from_solder": away}
        if xy is not None:
            color = (220, 40, 40) if defect else (150, 150, 150) if away else (40, 180, 40)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thick)
    return canvas, report, flagged
