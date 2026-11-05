"""Builds every real-photo figure of the README from the production photos.

Put the photos in the usual layout (data/README.md), train the ROI models, and run:

    python scripts/make_photo_figures.py --board data/board_a --models models/board_a \
        --panel data/board_a/test/defect/panel_017.png --roi roi_028

Optional (each one unlocks one more figure):
    --labels data/board_a/test/defect/labels.json   {"panel_017.png": ["roi_028"], ...}  -> roi_mosaic, qualitative_grid
    --intentional panel_003.png:roi_007             a ROI where a bridge is there by design  -> attempt2_intentional_short
    --accidental  panel_017.png:roi_012             a ROI where the same kind of bridge is a defect

Writes into assets/photos/. Check every image for logos, product names and serial
labels before committing: this script cannot see them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcbad import BACKBONES, BackbonePool, RoiPatchCore, preprocess  # noqa: E402
from pcbad.data import crop, list_images, load_rois, read_rgb  # noqa: E402
from pcbad.pipeline import inspect, load_models  # noqa: E402
from pcbad.visualize import heatmap, outline  # noqa: E402

OUT = ROOT / "assets" / "photos"
GREEN, RED, WHITE, BLACK = (40, 180, 40), (220, 40, 40), (255, 255, 255), (0, 0, 0)


def save(name: str, rgb: np.ndarray, quality: int = 88):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix == ".jpg" else []
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)
    print("wrote", path)


def fit_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Caption bar under an image."""
    bar = np.full((44, img.shape[1], 3), 252, np.uint8)
    cv2.putText(bar, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    return np.vstack([img, bar])


def hstack(images: list[np.ndarray], height: int = 420, gap: int = 12) -> np.ndarray:
    tiles = [cv2.resize(im, (round(im.shape[1] * height / im.shape[0]), height)) for im in images]
    sep = np.full((height, gap, 3), 252, np.uint8)
    out = [tiles[0]]
    for t in tiles[1:]:
        out += [sep, t]
    return np.hstack(out)


def vstack(rows: list[np.ndarray], gap: int = 12) -> np.ndarray:
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=252) for r in rows]
    sep = np.full((gap, width, 3), 252, np.uint8)
    out = [rows[0]]
    for r in rows[1:]:
        out += [sep, r]
    return np.vstack(out)


def predict_map(model: RoiPatchCore, crop_rgb: np.ndarray, margin: float = 1.15):
    s = BACKBONES[model.backbone_key].input_size
    p = model.predict(preprocess(crop_rgb, s).to(model.device), margin=margin)[0]
    return p, cv2.resize(p.anomaly_map, (crop_rgb.shape[1], crop_rgb.shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--panel", required=True, help="a panel photo that has a defect")
    ap.add_argument("--roi", required=True, help="the ROI of that panel where the defect is")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--intentional", default=None, help="file:roi with a bridge that is there by design")
    ap.add_argument("--accidental", default=None, help="file:roi with the same kind of bridge as a defect")
    ap.add_argument("--margin", type=float, default=1.15)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    globals()["OUT"] = Path(a.out)

    board = Path(a.board)
    size, rois = load_rois(board)
    models = load_models(a.models)
    panel = read_rgb(a.panel)
    good_panels = list_images(board / "train")
    good = read_rgb(good_panels[0])
    xy = rois[a.roi]
    m = models[a.roi]

    # 1. panel_rois.jpg: every ROI framed and numbered
    canvas = panel.copy()
    t = max(4, panel.shape[0] // 700)
    for r, (x, y) in rois.items():
        cv2.rectangle(canvas, (x, y), (x + size, y + size), GREEN, t)
        num = r.split("_")[-1].lstrip("0") or "0"
        cv2.putText(canvas, num, (x + 12, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 2.4, BLACK, 14, cv2.LINE_AA)
        cv2.putText(canvas, num, (x + 12, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 2.4, WHITE, 5, cv2.LINE_AA)
    save("panel_rois.jpg", fit_width(canvas, 2000))

    # 2. panel_overview.jpg: whole panel with the chosen ROI in red, and the ROI itself
    over = panel.copy()
    cv2.rectangle(over, xy, (xy[0] + size, xy[1] + size), RED, t * 2)
    save("panel_overview.jpg", hstack([label(fit_width(over, 1400), "full panel"),
                                      label(cv2.resize(crop(panel, xy, size), (1400, 1400)), f"{a.roi}, {size} px")],
                                     height=900))

    # 3. hero.png: good joint, same joint with the defect, heat map, outline
    c_good, c_bad = crop(good, xy, size), crop(panel, xy, size)
    pred, amap = predict_map(m, c_bad, a.margin)
    vmax = m.threshold * a.margin * 1.3
    save("hero.png", hstack([label(c_good, "good"), label(c_bad, "defect"),
                             label(heatmap(c_bad, amap, vmax=vmax, vmin=m.threshold * 0.5), "heat map"),
                             label(outline(c_bad, amap, min_level=m.threshold * a.margin),
                                   f"system output, score {pred.score:.2f}")]))

    # 4. infer_output.jpg: what scripts/infer.py writes for this panel
    result, report, _ = inspect(models, panel, a.margin)
    save("infer_output.jpg", fit_width(result, 2000))

    # 5. attempt1_whole_panel.png: one model for the whole panel vs the ROI model
    s = BACKBONES[m.backbone_key].input_size
    whole = RoiPatchCore(m.backbone_key).to(m.device)
    whole.fit(torch.cat([preprocess(read_rgb(p), s) for p in good_panels[:40]]).to(m.device))
    wp = whole.predict(preprocess(panel, s).to(m.device), margin=a.margin)[0]
    wmap = cv2.resize(wp.anomaly_map, (panel.shape[1], panel.shape[0]))
    whole_view = heatmap(panel, wmap, vmax=whole.threshold * a.margin * 1.3, vmin=whole.threshold * 0.5)
    cv2.rectangle(whole_view, xy, (xy[0] + size, xy[1] + size), WHITE, t * 2)
    save("attempt1_whole_panel.png", hstack([
        label(fit_width(whole_view, 1400), f"one model for the panel, score {wp.score:.2f}"),
        label(cv2.resize(heatmap(c_bad, amap, vmax=vmax, vmin=m.threshold * 0.5), (1400, 1400)),
              f"model of {a.roi}, score {pred.score:.2f}")], height=900))
    del whole

    # 6. backbone_heatmaps.png: same defective ROI, WideResNet-50 vs the backbone of the models
    crops_train = [crop(read_rgb(p), xy, size) for p in good_panels]
    panels_bb = []
    for key in ["wide_resnet50", m.backbone_key]:
        mk = RoiPatchCore(key).to(m.device).fit(preprocess(crops_train, BACKBONES[key].input_size).to(m.device))
        pk, ak = predict_map(mk, c_bad, a.margin)
        panels_bb.append(label(heatmap(c_bad, ak, vmax=mk.threshold * a.margin * 1.3, vmin=mk.threshold * 0.5),
                               f"{BACKBONES[key].label}: {pk.score:.2f}"))
        BackbonePool.clear()
    save("backbone_heatmaps.png", hstack([label(c_bad, "defect")] + panels_bb))
    models = load_models(a.models)  # the pool was cleared above

    # 7. roi_mosaic.jpg and 8. qualitative_grid.png need the defect labels
    if a.labels:
        labels = json.loads(Path(a.labels).read_text())
        folder = Path(a.labels).parent
        bad_crops, results = [], {"hit": [], "miss": [], "false alarm": []}
        for fname, bad_rois in labels.items():
            img = read_rgb(folder / fname)
            _, rep, _ = inspect(models, img, a.margin)
            for r, (x, y) in rois.items():
                c = crop(img, (x, y), size)
                is_bad, flagged = r in bad_rois, rep[r]["defect"]
                key = "hit" if is_bad and flagged else "miss" if is_bad else "false alarm" if flagged else None
                if is_bad:
                    bad_crops.append(c)
                if key and len(results[key]) < 4:
                    _, am = predict_map(models[r], c, a.margin)
                    results[key].append(outline(c, am, min_level=models[r].threshold * a.margin))
        good_crops = [crop(read_rgb(p), xy2, size) for p, xy2 in zip(good_panels[:4], list(rois.values())[:4])]
        save("roi_mosaic.jpg", vstack([hstack(good_crops[:4], 300), hstack(bad_crops[:4], 300)]))
        rows = [label(hstack(v, 300), k) for k, v in results.items() if v]
        if rows:
            save("qualitative_grid.png", vstack(rows))

    # 9. attempt2_intentional_short.png: the same appearance, normal in one ROI and a defect in another
    if a.intentional and a.accidental:
        pairs = []
        for spec, text in [(a.intentional, "bridge by design (good)"), (a.accidental, "same bridge, defect")]:
            fname, r = spec.split(":")
            candidates = [Path(fname), board / "test" / "defect" / fname, board / "test" / "good" / fname,
                          board / "train" / fname]
            src = next(p for p in candidates if p.exists())
            pairs.append(label(crop(read_rgb(src), rois[r], size), f"{r}: {text}"))
        save("attempt2_intentional_short.png", hstack(pairs))

    print("\nCheck every image for logos, product names and serial labels before committing.")


if __name__ == "__main__":
    main()
