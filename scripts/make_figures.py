"""Builds every chart used in the README from the JSON files in results/.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcbad.coreset import coreset_ratio  # noqa: E402

FIG = ROOT / "assets" / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)

# Reference palette (validated, light surface)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL = "#d03b3b"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
    }
)


def title(fig, main, sub=None):
    fig.text(0.012, 0.975, main, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    if sub:
        fig.text(0.012, 0.915, sub, ha="left", va="top", fontsize=9.5, color=INK2)


# --------------------------------------------------------------------------- VRAM
def fig_vram(bench):
    keys = [k for k in ["wide_resnet50", "dinov2_vits14", "dinov2_vitb14"] if k in bench["backbones"]]
    gpu_gb = bench["gpu_total_mb"] / 1024
    fig, axes = plt.subplots(1, len(keys), figsize=(4.1 * len(keys), 3.9))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        b = bench["backbones"][k]
        runs = b["runs"]
        n = np.array([r["n"] for r in runs])
        sh = np.array([r["shared_resident_mb"] for r in runs]) / 1024
        de = np.array([r["dedicated_resident_mb"] for r in runs]) / 1024
        meas = np.array([r["dedicated_measured"] for r in runs])

        ax.axhline(gpu_gb, color=CRITICAL, lw=1, ls=(0, (4, 3)))
        ax.text(n[0], gpu_gb, f" GPU memory ({gpu_gb:.0f} GB)", color=INK2, fontsize=8, va="bottom")
        last_meas = np.where(meas)[0].max()
        ax.plot(n[: last_meas + 1], de[: last_meas + 1], color=ORANGE, marker="o", ms=4, label="dedicated (measured)")
        if (~meas).any():
            idx = np.r_[last_meas, np.where(~meas)[0]]
            ax.plot(n[idx], de[idx], color=ORANGE, ls=(0, (2, 2)), lw=1.6, label="dedicated (projected)")
        ax.plot(n, sh, color=BLUE, marker="o", ms=4, label="shared backbone (measured)")

        ax.set_title(b["label"], loc="left", fontsize=10.5, color=INK, pad=6)
        ax.set_xlabel("ROI models loaded")
        ax.set_ylabel("resident VRAM (GB)")
        ax.set_xlim(0, n.max() * 1.18)
        top = max(de.max(), gpu_gb) * 1.12
        ax.set_ylim(0, top)
        # direct labels at the right end
        ax.annotate(f"{de[-1]:.1f} GB", (n[-1], de[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
        ax.annotate(f"{sh[-1]:.2f} GB", (n[-1], sh[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
        ax.text(0.03, 0.80, f"−{runs[-1]['saving']:.0%} at {n[-1]} ROIs", transform=ax.transAxes, ha="left",
                va="center", fontsize=11, fontweight="bold", color=INK)
    seen = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    title(fig, "VRAM grows with the backbone copies, not with the ROIs",
          f"{bench['crops_per_roi']} good crops per ROI · measured on {bench['gpu']}")
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    fig.savefig(FIG / "vram_vs_rois.png", dpi=160)
    plt.close(fig)


def fig_saving_curve(bench):
    keys = [k for k in ["wide_resnet50", "dinov2_vits14", "dinov2_vitb14"] if k in bench["backbones"]]
    colors = {"wide_resnet50": AQUA, "dinov2_vits14": ORANGE, "dinov2_vitb14": BLUE}
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    nn = np.arange(1, 141)
    for k in keys:
        b = bench["backbones"][k]
        B, M = b["backbone_mb"], b["bank_mb"]
        theory = 1 - (B + nn * M) / (nn * (B + M))
        ax.plot(nn, theory * 100, color=colors[k], lw=1.6, alpha=0.9)
        n = [r["n"] for r in b["runs"] if r["dedicated_measured"]]
        s = [r["saving"] * 100 for r in b["runs"] if r["dedicated_measured"]]
        ax.plot(n, s, ls="none", marker="o", ms=6, color=colors[k], mec=SURFACE, mew=1.5)
        ax.axhline(b["asymptotic_saving"] * 100, color=colors[k], lw=0.8, ls=(0, (3, 3)))
        dy = {"dinov2_vitb14": 7, "dinov2_vits14": -7}.get(k, 0)
        ax.annotate(f"{b['label']}  → {b['asymptotic_saving']:.0%}   (backbone {B:.0f} MB, bank {M:.0f} MB)",
                    (140, b["asymptotic_saving"] * 100), xytext=(6, dy), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 100)
    ax.set_xlabel("ROI models")
    ax.set_ylabel("VRAM saved vs. dedicated (%)")
    ax.plot([], [], color=MUTED, lw=1.6, label="expected from the sizes")
    ax.plot([], [], color=MUTED, ls="none", marker="o", ms=6, label="measured (while the full models still fit)")
    ax.legend(loc="center right", fontsize=9, bbox_to_anchor=(1.0, 0.62))
    title(fig, "The bigger the backbone, the more sharing it saves",
          "VRAM saved by loading one backbone instead of one per ROI model")
    fig.tight_layout(rect=(0, 0, 0.70, 0.88))
    fig.savefig(FIG / "saving_curve.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------ coreset
def fig_coreset():
    crops = np.arange(10, 401)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ref = 50
    series = [(0.0, ORANGE, "keep always 10% (bank grows with the data)"), (0.5, BLUE, "what I used"),
              (1.0, AQUA, "fixed bank size (more data adds nothing)")]
    for e, c, lab in series:
        kept = np.array([n * coreset_ratio(n, exponent=e) for n in crops])
        rel = kept / (ref * 0.10)
        ax.plot(crops, rel, color=c, lw=2.4 if e == 0.5 else 1.6, label=lab)
        ax.annotate(f"{rel[-1]:.1f}×", (crops[-1], rel[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.axvline(ref, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(ref, ax.get_ylim()[1] * 0.97, "  reference = 50 crops", color=INK2, fontsize=8.5, va="top")
    ax.set_xlabel("good crops used to train one ROI model")
    ax.set_ylabel("memory bank size (× size at 50 crops)")
    ax.set_xlim(0, 430)
    ax.set_ylim(0)
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0.0, 0.92))
    title(fig, "Twice the photos, +41% of memory instead of +100%",
          "the share of vectors kept starts at 10% and goes down slowly as more photos come in")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / "coreset_growth.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------ journey
GOOD, CRIT, WARN = "#0ca30c", "#d03b3b", "#fab219"
BOARD, JOINT, COPPER = "#2f6b3a", "#c9c7c0", "#7a5a2e"


def _card(ax, x, w, num, title, status, color, lines):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, 0.02), w, 0.96, boxstyle="round,pad=0,rounding_size=0.025",
                                fc="#ffffff", ec=GRID, lw=1.2))
    ax.text(x + 0.018, 0.93, f"{num}", fontsize=20, fontweight="bold", color=MUTED, va="top")
    ax.text(x + 0.055, 0.925, title, fontsize=11.5, fontweight="bold", color=INK, va="top")
    ax.add_patch(FancyBboxPatch((x + 0.018, 0.80), w - 0.036, 0.055, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=color, ec="none"))
    ax.text(x + w / 2, 0.8275, status, fontsize=9.5, fontweight="bold", color="#ffffff" if color != WARN else INK,
            ha="center", va="center")
    for i, ln in enumerate(lines):
        ax.text(x + 0.018, 0.255 - i * 0.062, ln, fontsize=9, color=INK2, va="top")


ASPECT = 13 / 4.4  # figure width / height: keeps circles round in a [0,1]x[0,1] axes


def _joint(ax, cx, cy, r, bridge_to=None):
    from matplotlib.patches import Ellipse, Rectangle

    if bridge_to is not None:  # draw the solder bridge first, under both joints
        ax.add_patch(Rectangle((cx, cy - r * 0.5 * ASPECT), bridge_to - cx, r * ASPECT, fc=JOINT, ec="none"))
    ax.add_patch(Ellipse((cx, cy), 2 * r, 2 * r * ASPECT, fc=JOINT, ec="none"))
    ax.add_patch(Ellipse((cx, cy), 0.7 * r, 0.7 * r * ASPECT, fc=COPPER, ec="none"))


def fig_journey():
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig = plt.figure(figsize=(13, 4.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_aspect("auto")
    w, gap, x0 = 0.235, 0.012, 0.012
    xs = [x0 + i * (w + gap) for i in range(4)]

    # 1 — whole panel
    _card(ax, xs[0], w, 1, "Whole panel, one model", "FAILED · defect too small", CRIT,
          ["5120 px panel → 518 px input", "1 feature ≈ 6 mm of board", "a 1 mm short is 1/6 of it"])
    x, y, s = xs[0] + 0.03, 0.33, 0.042
    for r in range(4):
        for c in range(4):
            ax.add_patch(Rectangle((x + c * (s + 0.006), y + r * (s * 2.2 + 0.012)), s, s * 2.2, fc=BOARD, ec="none"))
    from matplotlib.patches import Ellipse

    dx, dy = x + 2 * (s + 0.006) + s * 0.5, y + 1 * (s * 2.2 + 0.012) + s * 1.1
    ax.add_patch(Ellipse((dx, dy), 0.004, 0.004 * ASPECT, color=CRIT))
    ax.annotate("1 mm short", (dx, dy), xytext=(dx + 0.012, dy + 0.13), fontsize=8.5, color=CRIT,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="none"),
                arrowprops=dict(arrowstyle="-", color="#ffffff", lw=1.2))

    # 2 — one model for all ROIs
    _card(ax, xs[1], w, 2, "ROIs, one shared model", "FAILED · position is lost", CRIT,
          ["some shorts are intentional", "\"normal somewhere\" becomes", "\"normal everywhere\""])
    for j, (lab, col) in enumerate([("ROI 7 · by design", GOOD), ("ROI 12 · defect", CRIT)]):
        bx = xs[1] + 0.02 + j * 0.107
        ax.add_patch(FancyBboxPatch((bx, 0.36), 0.095, 0.36, boxstyle="round,pad=0,rounding_size=0.01", fc=BOARD, ec="none"))
        _joint(ax, bx + 0.025, 0.56, 0.014, bridge_to=bx + 0.07)
        _joint(ax, bx + 0.07, 0.56, 0.014)
        _joint(ax, bx + 0.025, 0.45, 0.014)
        _joint(ax, bx + 0.07, 0.45, 0.014)
        ax.text(bx + 0.0475, 0.665, lab, fontsize=7.5, color="#ffffff", ha="center", va="center")
        ax.text(bx + 0.0475, 0.385, "OK" if j == 0 else "→ approved ✗", fontsize=8, color=col if j else "#ffffff",
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="none") if j else None)

    # 3 — one model per ROI
    _card(ax, xs[2], w, 3, "One full model per ROI", "DETECTS · but 35 GB of VRAM", WARN,
          ["each model = backbone + bank", "130 identical backbone copies", "do not fit in a 6 GB GPU"])
    for i in range(6):
        bx, by = xs[2] + 0.022 + (i % 3) * 0.068, 0.55 - (i // 3) * 0.19
        ax.add_patch(Rectangle((bx, by), 0.058, 0.13, fc=ORANGE, ec="none"))
        ax.add_patch(Rectangle((bx, by), 0.058, 0.025, fc=BLUE, ec="none"))
    ax.text(xs[2] + w / 2, 0.315, "orange = backbone (250 MB)   blue = bank (25 MB)", fontsize=7.5, color=INK2,
            ha="center")

    # 4 — shared backbone
    _card(ax, xs[3], w, 4, "Shared backbone", "WORKS · −90% VRAM, same scores", GOOD,
          ["backbone loaded once, frozen", "each ROI keeps only its bank", "130 ROIs in 3.5 GB"])
    ax.add_patch(Rectangle((xs[3] + 0.022, 0.55), w - 0.044, 0.15, fc=ORANGE, ec="none"))
    ax.text(xs[3] + w / 2, 0.625, "one DINOv2 backbone", fontsize=9, color="#ffffff", ha="center", va="center",
            fontweight="bold")
    for i in range(10):
        ax.add_patch(Rectangle((xs[3] + 0.022 + i * 0.0194, 0.40), 0.016, 0.11, fc=BLUE, ec="none"))
    ax.text(xs[3] + w / 2, 0.355, "130 small banks", fontsize=8, color=INK2, ha="center")

    fig.savefig(FIG / "journey.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------- VRAM animation
def gif_vram(bench, key="dinov2_vitb14"):
    """Two 6 GB 'memory gauges' filling as ROI models are loaded (dedicated vs shared)."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import FancyBboxPatch, Rectangle

    b = bench["backbones"][key]
    cap = bench["gpu_total_mb"]
    # straight lines fitted to the MEASURED points (dedicated: only those that fit in the GPU)
    sh = np.polyfit([r["n"] for r in b["runs"]], [r["shared_resident_mb"] for r in b["runs"]], 1)
    meas = [r for r in b["runs"] if r["dedicated_measured"]]
    de = np.polyfit([r["n"] for r in meas], [r["dedicated_resident_mb"] for r in meas], 1)
    ns = list(range(1, 131, 2)) + [130] * 18  # hold the last frame
    fig = plt.figure(figsize=(8, 3.6))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rows = [("One full model per ROI", 0.60, lambda n: float(np.polyval(de, n)), ORANGE),
            ("Shared backbone (this work)", 0.22, lambda n: float(np.polyval(sh, n)), BLUE)]
    x0, w, h = 0.05, 0.72, 0.16
    artists = []
    for label, y, _, color in rows:
        ax.text(x0, y + h + 0.035, label, fontsize=12, fontweight="bold", color=INK)
        ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0,rounding_size=0.02", fc="#f0efec", ec=AXIS))
        bar = Rectangle((x0, y), 0, h, fc=color, ec="none")
        ax.add_patch(bar)
        txt = ax.text(x0 + w + 0.02, y + h / 2, "", fontsize=12, va="center", color=INK, fontweight="bold")
        artists.append((bar, txt))
    counter = ax.text(0.97, 0.95, "", ha="right", va="top", fontsize=13, color=INK2)
    ax.text(x0, 0.06, f"GPU: {bench['gpu']} · {cap / 1024:.0f} GB · DINOv2 ViT-B · measured per-model costs",
            fontsize=8.5, color=MUTED)

    def update(i):
        n = ns[i]
        for (bar, txt), (_, y, f, color) in zip(artists, rows):
            used = f(n)
            frac = min(used / cap, 1.0)
            bar.set_width(w * frac)
            if used > cap:
                bar.set_facecolor(CRITICAL)
                txt.set_text(f"out of memory\n({used / 1024:.1f} GB needed)")
                txt.set_color(CRITICAL)
            else:
                bar.set_facecolor(color)
                txt.set_text(f"{used / 1024:.2f} GB")
                txt.set_color(INK)
        counter.set_text(f"{n} ROI models loaded")
        return []

    anim = FuncAnimation(fig, update, frames=len(ns), interval=70)
    anim.save(FIG / "vram_animation.gif", writer=PillowWriter(fps=14), dpi=90)
    plt.close(fig)


def social_preview(bench):
    """1280x640 card GitHub shows when the repository link is shared."""
    fig = plt.figure(figsize=(12.8, 6.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#0f1720")
    ax.text(0.06, 0.82, "Solder-defect detection on PCBs", fontsize=34, fontweight="bold", color="#ffffff")
    ax.text(0.06, 0.73, "PatchCore trained only on good boards · one shared DINOv2 backbone for every region",
            fontsize=16, color="#c3c2b7")
    b = bench["backbones"]["dinov2_vitb14"]
    last = b["runs"][-1]
    stats = [(f"−{last['saving']:.0%}", "VRAM"), (f"{last['n']}", "region models on a 6 GB GPU"),
             ("10⁻⁵", "score difference vs. reference"), (f"{b['lean_ckpt_mb']:.0f} MB", "per model (was "
                                                                                 f"{b['full_ckpt_mb']:.0f} MB)")]
    for i, (big, small) in enumerate(stats):
        x = 0.06 + i * 0.225
        ax.text(x, 0.45, big, fontsize=40, fontweight="bold", color="#6da7ec")
        ax.text(x, 0.37, small, fontsize=13, color="#c3c2b7")
    ax.text(0.06, 0.10, "Final project · Computer Vision Master (VC MASTER) · ICA / PUC-Rio", fontsize=14,
            color="#898781")
    fig.savefig(FIG / "social_preview.png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    fig_journey()
    fig_coreset()
    bench_path = RES / "vram_benchmark.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        fig_vram(bench)
        fig_saving_curve(bench)
        gif_vram(bench)
        social_preview(bench)
    print("figures in", FIG)
