"""Inspect panel photos with the per-ROI models (one shared backbone in VRAM).

    python scripts/infer.py --models models/board_a --images data/board_a/test --out outputs/board_a

For each image: <name>_result.jpg (every ROI framed green/red, defects
outlined) and <name>_result.json (score and verdict per ROI; score >= 1 = defect).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcbad.data import list_images, read_rgb  # noqa: E402
from pcbad.pipeline import inspect, load_models  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--images", required=True, help="an image or a folder of images (searched recursively one level)")
ap.add_argument("--out", required=True)
ap.add_argument("--margin", type=float, default=1.15, help="limit = calibrated threshold x margin")
ap.add_argument("--device", default=None)
a = ap.parse_args()

models = load_models(a.models, a.device)
vram = f", VRAM in use: {torch.cuda.memory_allocated() / 2**20:.0f} MB" if torch.cuda.is_available() else ""
print(f"{len(models)} ROI models loaded{vram}")

src = Path(a.images)
images = [src] if src.is_file() else list_images(src) + [p for d in sorted(src.iterdir()) if d.is_dir() for p in list_images(d)]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for p in images:
    t = time.perf_counter()
    canvas, report, _ = inspect(models, read_rgb(p), a.margin)
    n_bad = sum(v["defect"] for v in report.values())
    verdict = "DEFECT" if n_bad else "OK"
    ms = (time.perf_counter() - t) * 1000
    print(f"{p.name}: {verdict}  ({n_bad} ROIs flagged, {ms:.0f} ms)")
    cv2.imwrite(str(out / f"{p.stem}_result.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / f"{p.stem}_result.json").write_text(json.dumps({"verdict": verdict, "ms": round(ms), "rois": report}, indent=2))
