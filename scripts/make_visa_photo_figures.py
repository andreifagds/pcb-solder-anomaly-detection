"""Image figures built from VisA (public), used while the production photos are not in:

  visa_attempt1.png      one defective board alone vs. hidden in a 4x4 mosaic of good boards
  visa_backbones_maps.png the same defective board seen by WideResNet-50, ViT-B block 8 and ViT-B block 2
  visa_qualitative.png   hits, misses and false alarms with ViT-B block 2 at threshold x 1.15

    python scripts/make_visa_photo_figures.py --visa data/visa/VisA --split data/visa/1cls.csv
    FIG_LANG=pt python scripts/make_visa_photo_figures.py ...   # Portuguese versions, *_pt.png
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import make_figures as mf  # noqa: E402
from eval_visa import CATS, embed, fit, load_split, mosaic, read_mask, read_rgb  # noqa: E402
from pcbad import BackbonePool  # noqa: E402
from pcbad.visualize import heatmap, outline  # noqa: E402

plt = mf.plt
FIG = mf.FIG
T, N, out = mf.T, mf.N, mf.out
B2 = "dinov2_vitb14@blocks.2"


def scored(key, train_imgs, test_imgs):
    m = fit(key, embed(key, train_imgs), seed=42)
    s, maps = m.score_embeddings(embed(key, test_imgs).cuda(), 518)
    return m, s.cpu().numpy(), maps.cpu().numpy()


def scale(m, maps_good):
    """Color scale shared by all maps of one model: floor at the median of good boards,
    top at the 99th percentile of their worst tile."""
    return float(np.median(maps_good)), float(np.percentile(maps_good.max(axis=(1, 2)), 99))


def show(ax, img, title=None, color=None):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(color is not None)
        if color:
            sp.set_color(color)
            sp.set_linewidth(3)
    if title:
        ax.set_title(title, fontsize=9.5, color=mf.INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visa", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--category", default="pcb1")
    a = ap.parse_args()
    split = load_split(Path(a.visa), Path(a.split), 100, 42)
    c = a.category
    train = [read_rgb(p, 518) for p, _, _ in split[c]["train"]]
    items = split[c]["test"]
    test = [read_rgb(p, 518) for p, _, _ in items]
    y = np.array([l for _, l, _ in items])

    # ---- block 2 on the board alone: pick a typical correct detection
    m2, s2, maps2 = scored(B2, train, test)
    lo2, hi2 = scale(m2, maps2[y == 0])
    lim2 = m2.threshold * 1.15
    cand = np.where((y == 1) & (s2 >= lim2))[0]
    i = int(cand[np.argsort(s2[cand])[len(cand) // 2]])
    board, gt = test[i], read_mask(items[i][2], (518, 518))

    # ---- Figure: attempt 1 (board alone vs inside a 4x4 mosaic), same block-2 model family
    rng = random.Random(42)
    k = 4
    train_mos = [mosaic(train[j], rng.sample(train, k * k), k, rng.randrange(k * k)) for j in range(len(train))]
    pos = 5
    fillers = rng.sample(train, k * k)
    test_mos = [mosaic(board, list(fillers), k, pos)] + [mosaic(test[j], rng.sample(train, k * k), k, pos)
                                                        for j in np.where(y == 0)[0][:40]]
    mm, sm, mapsm = scored(B2, train_mos, test_mos)
    lom, him = scale(mm, mapsm[1:])
    mos_img = test_mos[0]
    mos_view = heatmap(mos_img, mapsm[0], vmax=him, vmin=lom)
    t = 518 // k
    r, col = divmod(pos, k)
    cv2.rectangle(mos_view, (col * t, r * t), (col * t + t, r * t + t), (255, 255, 255), 2)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0))
    show(axes[0], mos_img, T("16 boards in one image (defective one boxed)", "16 placas numa imagem (a com defeito marcada)"))
    show(axes[1], mos_view, T(f"heat map, score {sm[0] / (mm.threshold * 1.15):.2f}× the limit",
           f"mapa de calor, nota {N(sm[0] / (mm.threshold * 1.15), '.2f')}× o limite"))
    show(axes[2], heatmap(board, maps2[i], vmax=hi2, vmin=lo2),
         T(f"the same board alone, score {s2[i] / lim2:.2f}× the limit",
           f"a mesma placa sozinha, nota {N(s2[i] / lim2, '.2f')}× o limite"))
    fig.suptitle(T("Attempt 1 on a public board: the defect disappears when the board is a small part of the image",
                   "Tentativa 1 numa placa pública: o defeito some quando a placa é uma parte pequena da imagem"),
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=mf.INK)
    fig.tight_layout()
    fig.savefig(out("visa_attempt1.png"), dpi=130)
    plt.close(fig)
    BackbonePool.clear()

    # ---- Figure: backbones on the same defective board
    panels = [(board, T("input", "entrada")),
              (np.where(gt[..., None] > 0, 255, board).astype(np.uint8), T("ground truth (white)", "gabarito (branco)"))]
    for key, name in [("wide_resnet50", "WideResNet-50"), ("dinov2_vitb14", T("ViT-B block 8 (production)", "ViT-B bloco 8 (produção)")),
                      (B2, T("ViT-B block 2", "ViT-B bloco 2"))]:
        mk, sk, mapsk = scored(key, train, test)
        lok, hik = scale(mk, mapsk[y == 0])
        panels.append((heatmap(board, mapsk[i], vmax=hik, vmin=lok), f"{name}\n{N(sk[i] / (mk.threshold * 1.15), '.2f')}× " + T("the limit", "o limite")))
        BackbonePool.clear()
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.3))
    for ax, (img, title) in zip(axes, panels):
        show(ax, img, title)
    fig.tight_layout()
    fig.savefig(out("visa_backbones_maps.png"), dpi=130)
    plt.close(fig)

    # ---- Figure: qualitative grid over the four categories with block 2
    rows = {"caught": [], "missed": [], "false alarm": []}
    for cat in CATS:
        tr = [read_rgb(p, 518) for p, _, _ in split[cat]["train"]]
        it = split[cat]["test"]
        te = [read_rgb(p, 518) for p, _, _ in it]
        yy = np.array([l for _, l, _ in it])
        mc, sc, mapsc = scored(B2, tr, te)
        lim = mc.threshold * 1.15
        lo, hi = scale(mc, mapsc[yy == 0])
        groups = {"caught": np.where((yy == 1) & (sc >= lim))[0], "missed": np.where((yy == 1) & (sc < lim))[0],
                  "false alarm": np.where((yy == 0) & (sc >= lim))[0]}
        for name, idx in groups.items():
            if len(idx):
                j = int(idx[np.argsort(sc[idx])[len(idx) // 2]])
                img = outline(te[j], mapsc[j], min_level=hi) if name != "missed" else heatmap(te[j], mapsc[j], vmax=hi, vmin=lo)
                rows[name].append((img, f"{cat}, {N(sc[j] / lim, '.2f')}×"))
        BackbonePool.clear()
    used = {k: v for k, v in rows.items() if v}
    fig, axes = plt.subplots(len(used), 4, figsize=(11, 2.9 * len(used)), squeeze=False)
    ROWNAME = {"caught": T("caught", "detectou"), "missed": T("missed", "deixou passar"),
               "false alarm": T("false alarm", "alarme falso")}
    color = {"caught": mf.GOOD, "missed": mf.CRIT, "false alarm": mf.WARN}
    for r, (name, items_) in enumerate(used.items()):
        for cidx in range(4):
            ax = axes[r, cidx]
            if cidx < len(items_):
                show(ax, items_[cidx][0], items_[cidx][1], color=color[name])
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(ROWNAME[name], fontsize=11, color=mf.INK)
        axes[r, 0].yaxis.set_visible(True)
    fig.tight_layout()
    fig.savefig(out("visa_qualitative.png"), dpi=120)
    plt.close(fig)
    print("figures in", FIG, "| language:", mf.LANG)


if __name__ == "__main__":
    main()
