"""Public, reproducible evidence on the four PCB categories of VisA (CC BY 4.0).

The production photos cannot be released, so every claim of the README that
can be tested on public data is re-tested here:

  main        detection per backbone (WideResNet-50, DINOv2 ViT-S, ViT-B)
  layers      which ViT-B block to read features from
  neighbors   num_neighbors in the image-score re-weighting
  coreset     memory-bank ratio vs. detection
  pooled      Attempt 2: ONE model for pcb1..pcb4 together vs one model each
  mosaic      Attempt 1: the board is one tile of a k x k "panel"
  threshold   leave-one-out threshold x 1.15 on unseen good/defective boards

Setup: VisA "1cls" split; a fixed random subset of good training images per
category (default 100, close to the ~80 per ROI of the production line).

    python scripts/eval_visa.py --visa data/visa/VisA --split data/visa/1cls.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcbad import BackbonePool, RoiPatchCore, preprocess  # noqa: E402
from pcbad.backbone import get_preset  # noqa: E402

CATS = ["pcb1", "pcb2", "pcb3", "pcb4"]
PIX = 256  # resolution at which pixel AUROC is computed


# ------------------------------------------------------------------ data
def read_rgb(p: Path, size: int | None = None) -> np.ndarray:
    im = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) if size else im


def read_mask(p: Path | None, shape) -> np.ndarray:
    if p is None:
        return np.zeros(shape, np.uint8)
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (cv2.resize(m, shape[::-1], interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)


def load_split(visa: Path, split_csv: Path, n_train: int, seed: int):
    rows = defaultdict(lambda: {"train": [], "test": []})
    with open(split_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["object"] not in CATS:
                continue
            label = 0 if r["label"].strip().lower() == "normal" else 1
            mask = visa / r["mask"] if r.get("mask") else None
            rows[r["object"]][r["split"]].append((visa / r["image"], label, mask))
    rng = random.Random(seed)
    for c in CATS:
        rng.shuffle(rows[c]["train"])
        rows[c]["train"] = rows[c]["train"][:n_train]
    return rows


# ------------------------------------------------------------- features
@torch.inference_mode()
def embed(key: str, images: list[np.ndarray], bs: int = 8) -> torch.Tensor:
    """-> (N,C,h,w) on CPU."""
    bb = BackbonePool.get(key, "cuda")
    s = get_preset(key).input_size
    out = []
    for i in range(0, len(images), bs):
        out.append(bb.embed(preprocess(images[i : i + bs], s).cuda()).cpu())
    return torch.cat(out)


def fit(key: str, emb_train: torch.Tensor, **kw) -> RoiPatchCore:
    m = RoiPatchCore(key.split("@")[0]).to("cuda")
    return m.fit_embeddings(emb_train.cuda(), get_preset(key).input_size, **kw)


def evaluate(m: RoiPatchCore, emb_test: torch.Tensor, labels, masks, margin=1.15, pixel=True):
    scores, maps = m.score_embeddings(emb_test.cuda(), PIX)
    scores = scores.cpu().numpy()
    y = np.array(labels)
    res = {"image_auroc": float(roc_auc_score(y, scores))}
    if pixel:
        gt = np.stack(masks).reshape(-1)
        res["pixel_auroc"] = float(roc_auc_score(gt, maps.cpu().numpy().reshape(-1)))
    limit = m.threshold * margin
    pred = scores >= limit
    res.update(
        recall=float(pred[y == 1].mean()),
        false_alarm_rate=float(pred[y == 0].mean()),
        threshold=m.threshold,
        n_test_good=int((y == 0).sum()),
        n_test_defect=int((y == 1).sum()),
    )
    return res, scores, maps


# ---------------------------------------------------------------- mosaic
def mosaic(target: np.ndarray, fillers: list[np.ndarray], k: int, pos: int, out: int = 518) -> np.ndarray:
    """k x k grid of boards, `target` at cell `pos`, the rest good boards."""
    t = out // k
    canvas = np.zeros((t * k, t * k, 3), np.uint8)
    cells = fillers[: k * k]
    cells[pos] = target
    for i, im in enumerate(cells):
        r, c = divmod(i, k)
        canvas[r * t : (r + 1) * t, c * t : (c + 1) * t] = cv2.resize(im, (t, t), interpolation=cv2.INTER_AREA)
    return canvas


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visa", required=True, help="folder with pcb1..pcb4 (raw VisA layout)")
    ap.add_argument("--split", required=True, help="split_csv/1cls.csv from amazon-science/spot-diff")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-size", type=int, default=518, help="images are stored in RAM at this size")
    ap.add_argument("--only", nargs="*", default=None, help="subset of experiments")
    ap.add_argument("--out", default="results/visa_results.json")
    args = ap.parse_args()
    run = lambda name: args.only is None or name in args.only  # noqa: E731

    visa, t0 = Path(args.visa), time.time()
    split = load_split(visa, Path(args.split), args.n_train, args.seed)
    S = args.load_size
    imgs = {c: {k: [read_rgb(p, S) for p, _, _ in split[c][k]] for k in ("train", "test")} for c in CATS}
    labels = {c: [l for _, l, _ in split[c]["test"]] for c in CATS}
    masks = {c: [read_mask(mp if l else None, (PIX, PIX)) for _, l, mp in split[c]["test"]] for c in CATS}
    print(f"loaded in {time.time() - t0:.0f}s:", {c: (len(imgs[c]["train"]), len(imgs[c]["test"])) for c in CATS})

    out_path = Path(args.out)
    R = json.loads(out_path.read_text()) if out_path.exists() else {}
    R["setup"] = {
        "dataset": "VisA (pcb1-pcb4), 1cls split",
        "n_train_per_category": args.n_train,
        "seed": args.seed,
        "margin": 1.15,
        "threshold": "leave-one-out, quantile 0.90",
        "pixel_auroc_resolution": PIX,
        "gpu": torch.cuda.get_device_name(0),
    }

    def save():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(R, indent=2))

    examples = {}
    # ---------------------------------------------------- main + per-backbone
    if run("main"):
        R["main"] = {}
        for key in ["wide_resnet50", "dinov2_vits14", "dinov2_vitb14"]:
            R["main"][key] = {}
            for c in CATS:
                t = time.time()
                m = fit(key, embed(key, imgs[c]["train"]), seed=args.seed)
                res, scores, maps = evaluate(m, embed(key, imgs[c]["test"]), labels[c], masks[c])
                res["seconds"] = round(time.time() - t, 1)
                R["main"][key][c] = res
                print(f"[main] {key:15s} {c}: img {res['image_auroc']:.3f} pix {res['pixel_auroc']:.3f} "
                      f"recall {res['recall']:.2f} FA {res['false_alarm_rate']:.2f}")
                if key == "dinov2_vitb14":
                    examples[c] = (scores, maps.cpu().numpy(), m.threshold)
                del m
            BackbonePool.clear()
            save()
        np.savez_compressed(
            out_path.with_name("visa_vitb_examples.npz"),
            **{f"{c}_scores": examples[c][0] for c in examples},
            **{f"{c}_thr": np.array(examples[c][2]) for c in examples},
            **{f"{c}_maps": examples[c][1].astype(np.float16) for c in examples},
        )

    # --------------------------- paper configuration (WideResNet layer2+layer3)
    if run("paper"):
        R.setdefault("main", {})["wide_resnet50_l23"] = {}
        k = "wide_resnet50_l23"
        for c in CATS:
            m = fit(k, embed(k, imgs[c]["train"]), seed=args.seed)
            res, _, _ = evaluate(m, embed(k, imgs[c]["test"]), labels[c], masks[c])
            R["main"][k][c] = res
            print(f"[paper] {k}: {c} img {res['image_auroc']:.3f} pix {res['pixel_auroc']:.3f}")
        BackbonePool.clear()
        save()

    key = "dinov2_vitb14"
    # ------------------------------ cosine distance (L2-normalized features)
    if run("cosine"):
        R["cosine"] = {}
        l2 = lambda e: F.normalize(e, dim=1)  # noqa: E731
        for c in CATS:
            m = fit(key, l2(embed(key, imgs[c]["train"])), seed=args.seed)
            res, _, _ = evaluate(m, l2(embed(key, imgs[c]["test"])), labels[c], masks[c])
            R["cosine"][c] = res
            print(f"[cosine] {c}: img {res['image_auroc']:.3f} pix {res['pixel_auroc']:.3f}")
        BackbonePool.clear()
        save()

    # --------------------------------------------------------------- layers
    if run("layers"):
        R["layers"] = {}
        for layer in ["blocks.2", "blocks.5", "blocks.8", "blocks.11"]:
            k = f"{key}@{layer}"
            R["layers"][layer] = {}
            for c in CATS:
                m = fit(k, embed(k, imgs[c]["train"]), seed=args.seed)
                res, _, _ = evaluate(m, embed(k, imgs[c]["test"]), labels[c], masks[c])
                R["layers"][layer][c] = res
                print(f"[layers] {layer}: {c} img {res['image_auroc']:.3f} pix {res['pixel_auroc']:.3f}")
            BackbonePool.clear()
            save()

    # ------------------------------------------ neighbors + coreset + pooled
    if run("neighbors") or run("coreset") or run("pooled"):
        tr = {c: embed(key, imgs[c]["train"]) for c in CATS}
        R.setdefault("neighbors", {})
        R.setdefault("coreset", {})
        pooled = None
        if run("pooled"):
            pooled = fit(key, torch.cat([tr[c] for c in CATS]), seed=args.seed)
            R["pooled"] = {"bank_rows": pooled.meta["bank_shape"][0], "per_category": {}, "pooled": {}}
        for c in CATS:
            te = embed(key, imgs[c]["test"])
            if run("neighbors"):
                m = fit(key, tr[c], seed=args.seed)
                R["neighbors"][c] = {}
                for nn in [1, 3, 9]:
                    m._scorer.num_neighbors = nn
                    R["neighbors"][c][str(nn)] = evaluate(m, te, labels[c], masks[c], pixel=False)[0]["image_auroc"]
                print(f"[neighbors] {c}: {R['neighbors'][c]}")
            if run("coreset"):
                R["coreset"][c] = {}
                for ratio in [0.01, 0.025, 0.1, 0.25, 1.0]:
                    m = fit(key, tr[c], seed=args.seed, ratio=ratio)
                    res, _, _ = evaluate(m, te, labels[c], masks[c])
                    res["bank_rows"] = m.meta["bank_shape"][0]
                    R["coreset"][c][str(ratio)] = res
                    print(f"[coreset] {c} r={ratio}: rows {res['bank_rows']} img {res['image_auroc']:.3f}")
                    del m
                    torch.cuda.empty_cache()
            if pooled is not None:
                own = fit(key, tr[c], seed=args.seed)
                R["pooled"]["per_category"][c] = evaluate(own, te, labels[c], masks[c])[0]
                R["pooled"]["pooled"][c] = evaluate(pooled, te, labels[c], masks[c])[0]
                print(f"[pooled] {c}: own {R['pooled']['per_category'][c]['image_auroc']:.3f} "
                      f"pooled {R['pooled']['pooled'][c]['image_auroc']:.3f}")
            save()
        del tr, pooled
        torch.cuda.empty_cache()

    # --------------------------------------------------------------- mosaic
    if run("mosaic"):
        R["mosaic"] = {}
        rng = random.Random(args.seed)
        for k in [1, 2, 3, 4]:
            R["mosaic"][str(k)] = {}
            for c in CATS:
                good = imgs[c]["train"]
                train_m = [mosaic(good[i], rng.sample(good, k * k), k, rng.randrange(k * k)) for i in range(len(good))]
                test_m = [mosaic(im, rng.sample(good, k * k), k, rng.randrange(k * k)) for im in imgs[c]["test"]]
                m = fit(key, embed(key, train_m), seed=args.seed)
                res, _, _ = evaluate(m, embed(key, test_m), labels[c], masks[c], pixel=False)
                R["mosaic"][str(k)][c] = res
                print(f"[mosaic] {k}x{k}: {c} img {res['image_auroc']:.3f}")
            save()

    R["total_minutes"] = round((time.time() - t0) / 60, 1)
    save()
    print("saved", out_path)


if __name__ == "__main__":
    main()
