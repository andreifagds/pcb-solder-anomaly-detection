"""Figures for the public VisA PCB evidence (reads results/visa_results.json).

    python scripts/make_visa_figures.py --visa data/visa/VisA --split data/visa/1cls.csv
    FIG_LANG=pt python scripts/make_visa_figures.py ...     # Portuguese versions, *_pt.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import make_figures as mf  # noqa: E402  (palette + rcParams + language helpers)
from eval_visa import CATS, load_split, read_mask, read_rgb  # noqa: E402
from pcbad.visualize import heatmap, outline  # noqa: E402

plt = mf.plt
RES, FIG = mf.RES, mf.FIG
T, N, out, ticks = mf.T, mf.N, mf.out, mf.decimal_ticks
LABEL = {"wide_resnet50": "WideResNet-50 (layer2)",
         "wide_resnet50_l23": T("WideResNet-50 (layer2+3, paper)", "WideResNet-50 (layer2+3, artigo)"),
         "dinov2_vits14": T("DINOv2 ViT-S (block 8)", "DINOv2 ViT-S (bloco 8)"),
         "dinov2_vitb14": T("DINOv2 ViT-B (block 8, production)", "DINOv2 ViT-B (bloco 8, produção)"),
         "dinov2_vitb14_b2": T("DINOv2 ViT-B (block 2)", "DINOv2 ViT-B (bloco 2)")}
COLOR = {"wide_resnet50": mf.AQUA, "wide_resnet50_l23": "#eda100", "dinov2_vits14": mf.ORANGE,
         "dinov2_vitb14": mf.BLUE, "dinov2_vitb14_b2": "#6250d6"}
MEAN = T("mean", "média")


def bars_backbones(R):
    main = dict(R["main"])
    if "layers" in R and "blocks.2" in R["layers"]:  # best block of the layer test, shown next to the others
        main["dinov2_vitb14_b2"] = R["layers"]["blocks.2"]
    keys = [k for k in LABEL if k in main]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    names = [T("image AUROC (is the board defective?)", "AUROC de imagem (a placa tem defeito?)"),
             T("pixel AUROC (where is the defect?)", "AUROC de pixel (onde está o defeito?)")]
    for ax, metric, name in zip(axes, ["image_auroc", "pixel_auroc"], names):
        x = np.arange(len(CATS) + 1)
        w = 0.8 / len(keys)
        for i, k in enumerate(keys):
            vals = [main[k][c][metric] for c in CATS]
            vals.append(np.mean(vals))
            off = (i - (len(keys) - 1) / 2) * w
            ax.bar(x + off, vals, w - 0.02, color=COLOR[k], label=LABEL[k])
            ax.text(x[-1] + off, vals[-1] + 0.004, N(vals[-1], ".3f"), ha="center", va="bottom",
                    fontsize=7.5, color=mf.INK2, rotation=90)
        ax.set_xticks(x, CATS + [MEAN])
        lo = min(main[k][c][metric] for k in keys for c in CATS)
        ax.set_ylim(max(0.5, np.floor(lo * 20) / 20 - 0.05), 1.03)
        ticks(ax, "y")
        ax.set_title(name, loc="left", fontsize=10.5, color=mf.INK)
        ax.grid(axis="x", visible=False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    n = R["setup"]["n_train_per_category"]
    mf.title(fig, T("Public PCBs (VisA): detection per backbone", "PCBs públicas (VisA): detecção por backbone"),
             T(f"{n} good training images per category · same pipeline as the production line",
               f"{n} imagens boas de treino por categoria · o mesmo pipeline da linha de produção"))
    fig.tight_layout(rect=(0, 0.12, 1, 0.86))
    fig.savefig(out("visa_backbones.png"), dpi=160)
    plt.close(fig)


def story(R):
    """Attempt 1 (mosaic) and Attempt 2 (pooled) reproduced on public data."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    # mosaic
    ax = axes[0]
    ks = sorted(R["mosaic"], key=int)
    for c, col in zip(CATS, [mf.BLUE, mf.ORANGE, mf.AQUA, mf.INK2]):
        ax.plot([int(k) ** 2 for k in ks], [R["mosaic"][k][c]["image_auroc"] for k in ks], marker="o", ms=4,
                color=col, lw=1.6, label=c)
    mean = [np.mean([R["mosaic"][k][c]["image_auroc"] for c in CATS]) for k in ks]
    ax.plot([int(k) ** 2 for k in ks], mean, color=mf.INK, lw=2.6, label=MEAN)
    ax.set_xscale("log", base=2)
    ax.set_xticks([int(k) ** 2 for k in ks], [f"{int(k)**2}" for k in ks])
    ax.set_xlabel(T("boards in the same image (k × k mosaic)", "placas na mesma imagem (mosaico k × k)"))
    ax.set_ylabel(T("image AUROC", "AUROC de imagem"))
    ax.set_ylim(0.45, 1.02)
    ticks(ax, "y")
    ax.axhline(0.5, color=mf.MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(1.05, 0.505, T("chance", "chute"), color=mf.INK2, fontsize=8, va="bottom")
    ax.set_title(T("Attempt 1 · more boards per image, less signal",
                   "Tentativa 1 · mais placas por imagem, menos sinal"), loc="left", fontsize=10.5)
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    # pooled
    ax = axes[1]
    x = np.arange(len(CATS))
    own = [R["pooled"]["per_category"][c]["image_auroc"] for c in CATS]
    pool = [R["pooled"]["pooled"][c]["image_auroc"] for c in CATS]
    ax.bar(x - 0.2, own, 0.37, color=mf.BLUE, label=T("one model per board type", "um modelo por tipo de placa"))
    ax.bar(x + 0.2, pool, 0.37, color=mf.ORANGE, label=T("one model for all four", "um modelo para as quatro"))
    for xi, a, b in zip(x, own, pool):
        ax.text(xi + 0.2, b + 0.004, N(b - a, "+.3f"), ha="center", va="bottom", fontsize=8.5, color=mf.INK2)
    ax.set_xticks(x, CATS)
    ax.set_ylim(max(0.5, min(own + pool) - 0.08), 1.03)
    ticks(ax, "y")
    ax.set_ylabel(T("image AUROC", "AUROC de imagem"))
    ax.set_title(T("Attempt 2 · one model for four different boards",
                   "Tentativa 2 · um modelo para quatro placas diferentes"), loc="left", fontsize=10.5)
    ax.legend(fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="x", visible=False)
    mf.title(fig, T("The first two attempts, tried on public PCBs", "As duas primeiras tentativas, refeitas em PCBs públicas"),
             "VisA pcb1–pcb4 · DINOv2 ViT-B")
    fig.set_size_inches(11, 4.4)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out("visa_story.png"), dpi=160)
    plt.close(fig)


def ablations(R):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    # layers
    ax = axes[0]
    layers = list(R["layers"])
    xi = np.arange(len(layers))
    for metric, col, lab in [("image_auroc", mf.BLUE, T("image AUROC", "AUROC de imagem")),
                             ("pixel_auroc", mf.ORANGE, T("pixel AUROC", "AUROC de pixel"))]:
        ax.plot(xi, [np.mean([R["layers"][l][c][metric] for c in CATS]) for l in layers], marker="o", color=col,
                label=lab)
    sizes_path = RES / "vitb_layer_sizes.json"
    sizes = json.loads(sizes_path.read_text()) if sizes_path.exists() else {}
    labels = []
    for l in layers:
        s = sizes.get(l)
        extra = T(f"\n{s['backbone_mb']:.0f} MB\nsaves ≤{s['limit_saving']:.0%}",
                  f"\n{s['backbone_mb']:.0f} MB\neconomiza ≤{s['limit_saving']:.0%}") if s else ""
        labels.append(l.replace("blocks.", T("block ", "bloco ")) + extra)
    ax.set_xticks(xi, labels, fontsize=8)
    ax.set_title(T("Which ViT-B block (of 12)", "Qual bloco do ViT-B (de 12)"), loc="left", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower left")
    # neighbors
    ax = axes[1]
    ns = ["1", "3", "9"]
    for c, col in zip(CATS, [mf.BLUE, mf.ORANGE, mf.AQUA, mf.INK2]):
        ax.plot(ns, [R["neighbors"][c][n] for n in ns], marker="o", ms=4, color=col, lw=1.6, label=c)
    ax.set_xlabel(T("num_neighbors", "vizinhos (num_neighbors)"))
    ax.set_title(T("Score re-weighting neighbors", "Vizinhos na reponderação da nota"), loc="left", fontsize=10.5)
    ax.legend(fontsize=8, ncol=2)
    # coreset
    ax = axes[2]
    ratios = list(R["coreset"]["pcb1"])
    rows = [np.mean([R["coreset"][c][r]["bank_rows"] for c in CATS]) for r in ratios]
    auc = [np.mean([R["coreset"][c][r]["image_auroc"] for c in CATS]) for r in ratios]
    ax.plot(rows, auc, marker="o", color=mf.BLUE)
    for r, x_, y_ in zip(ratios, rows, auc):
        ax.annotate(f"{float(r):g}".replace(".", "," if mf.LANG == "pt" else "."), (x_, y_), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8, color=mf.INK2)
    ax.set_xscale("log")
    ax.set_ylim(0.70, 0.85)  # same span as the neighbors panel, so small differences look small
    ax.set_xlabel(T("vectors kept in the memory bank (log)", "vetores guardados no banco (log)"))
    ax.set_title(T("Coreset ratio (labels) vs. mean image AUROC", "Razão do coreset (rótulos) × AUROC médio"),
                 loc="left", fontsize=10.5)
    for a in axes:
        a.set_ylabel("AUROC")
        ticks(a, "y")
    mf.title(fig, T("Every parameter, tested on public PCBs", "Cada parâmetro, testado em PCBs públicas"),
             T("VisA pcb1–pcb4 · DINOv2 ViT-B · means over the 4 categories unless noted",
               "VisA pcb1–pcb4 · DINOv2 ViT-B · média das 4 categorias, salvo indicação"))
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(out("visa_ablations.png"), dpi=160)
    plt.close(fig)


def threshold_hist(R, ex):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=False)
    for ax, c in zip(axes, CATS):
        s = ex[f"{c}_scores"]
        thr = float(ex[f"{c}_thr"])
        y = np.array(R["_labels"][c])
        bins = np.linspace(min(s.min(), thr), max(s.max(), thr * 1.15) * 1.02, 30)
        ax.hist(s[y == 0], bins=bins, color=mf.BLUE, alpha=0.85, label=T("good boards", "placas boas"))
        ax.hist(s[y == 1], bins=bins, color=mf.ORANGE, alpha=0.75, label=T("defective boards", "placas com defeito"))
        ax.axvline(thr * 1.15, color=mf.INK, lw=1.4)
        res = R["main"]["dinov2_vitb14"][c]
        ax.set_title(T(f"{c} · recall {res['recall']:.0%} · false alarms {res['false_alarm_rate']:.0%}",
                       f"{c} · recall {res['recall']:.0%} · alarme falso {res['false_alarm_rate']:.0%}"),
                     loc="left", fontsize=9.5)
        ax.set_xlabel(T("PatchCore score", "nota do PatchCore"))
        ticks(ax)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8)
    axes[0].text(0.98, 0.95, T("black line =\nthreshold × 1.15", "linha preta =\nlimiar × 1,15"),
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=8, color=mf.INK2)
    mf.title(fig, T("A threshold calibrated without a single defective image",
                    "Um limiar calibrado sem nenhuma imagem com defeito"),
             T("leave-one-out on the 100 good training images, quantile 0.90, × 1.15 · scored on unseen test boards",
               "leave-one-out nas 100 imagens boas de treino, quantil 0,90, × 1,15 · avaliado em placas de teste nunca vistas"))
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(out("visa_threshold.png"), dpi=160)
    plt.close(fig)


def examples(R, ex8, ex2, split):
    """One defective board per category: input · ground truth · heat map at block 8 (production)
    · heat map at block 2 · outline drawn from block 2."""
    cols = [T("input", "entrada"), T("ground truth (white)", "gabarito (branco)"),
            T("heat map, block 8\n(production setting)", "mapa de calor, bloco 8\n(config. de produção)"),
            T("heat map, block 2", "mapa de calor, bloco 2"), T("outline, block 2", "contorno, bloco 2")]
    fig, axes = plt.subplots(len(CATS), 5, figsize=(12, 2.55 * len(CATS)))
    for r, c in enumerate(CATS):
        items = split[c]["test"]
        y = np.array([l for _, l, _ in items])
        s2, thr2 = ex2[f"{c}_scores"], float(ex2[f"{c}_thr"]) * 1.15
        maps2, maps8 = ex2[f"{c}_maps"].astype(np.float32), ex8[f"{c}_maps"].astype(np.float32)
        # a typical correct detection at block 2: defective, above the limit, median score among those
        cand = np.where((y == 1) & (s2 >= thr2))[0]
        i = cand[np.argsort(s2[cand])[len(cand) // 2]] if len(cand) else int(np.argmax(s2))
        img = read_rgb(items[i][0], 518)
        gt = read_mask(items[i][2], (518, 518))
        a8 = cv2.resize(maps8[i], (518, 518))
        a2 = cv2.resize(maps2[i], (518, 518))
        # same scale rule for both maps: floor at the median value of good boards, top at the
        # 99th percentile of the worst tile of good boards
        v8 = float(np.percentile(maps8[y == 0].max(axis=(1, 2)), 99))
        v2 = float(np.percentile(maps2[y == 0].max(axis=(1, 2)), 99))
        f8, f2 = float(np.median(maps8[y == 0])), float(np.median(maps2[y == 0]))
        gt_img = img.copy()
        cnts, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_img, cnts, -1, (255, 255, 255), 3)
        panels = [img, gt_img, heatmap(img, a8, vmax=v8, vmin=f8), heatmap(img, a2, vmax=v2, vmin=f2),
                  outline(img, a2, min_level=v2)]
        for j, p in enumerate(panels):
            ax = axes[r, j]
            ax.imshow(p)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(cols[j], fontsize=9.5, color=mf.INK)
        axes[r, 0].set_ylabel(c, fontsize=10, color=mf.INK2)
    fig.tight_layout()
    fig.savefig(out("visa_examples.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visa", required=True)
    ap.add_argument("--split", required=True)
    args = ap.parse_args()
    R = json.loads((RES / "visa_results.json").read_text())
    split = load_split(Path(args.visa), Path(args.split), R["setup"]["n_train_per_category"], R["setup"]["seed"])
    R["_labels"] = {c: [l for _, l, _ in split[c]["test"]] for c in CATS}
    if "main" in R:
        bars_backbones(R)
    if "mosaic" in R and "pooled" in R:
        story(R)
    if all(k in R for k in ("layers", "neighbors", "coreset")):
        ablations(R)
    npz = RES / "visa_vitb_examples.npz"
    if npz.exists():
        ex = np.load(npz)
        threshold_hist(R, ex)
        npz2 = RES / "visa_vitb_blocks2_examples.npz"
        if npz2.exists():
            examples(R, ex, np.load(npz2), split)
    print("figures in", FIG, "| language:", mf.LANG)


if __name__ == "__main__":
    main()
