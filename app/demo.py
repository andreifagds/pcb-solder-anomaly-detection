"""Small web interface: train, configure and inspect (Gradio).

    pip install gradio==5.9.1
    python app/demo.py            # then open http://127.0.0.1:7860

Optional: --models <dir> --image <file> opens straight on the Inspect tab with
that models folder loaded and that image inspected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcbad import BACKBONES, BackbonePool  # noqa: E402
from pcbad.pipeline import inspect, load_models, train_board  # noqa: E402

STATE: dict = {"models": None, "dir": None}

HOW_IT_WORKS = """
1. **Train.** Every good image is cut into the ROIs and a frozen DINOv2 turns each ROI into 37×37 feature vectors.
   A coreset keeps the most spread-out ones as that ROI's memory bank. The threshold is the 0.90 quantile of the
   leave-one-out scores of the good images.
2. **Inspect.** Each ROI of a new image is compared with its own memory bank, and the distance of the most unusual
   tile is the score. A score of 1 or more (raw score at or above threshold × margin) is a defect.
3. **Why it fits in a small GPU.** The backbone is never trained, so all ROI models share one copy of it. Only the
   banks (about 25 MB each) are per ROI: 130 ROIs need about 3.5 GB instead of 35 GB.

See the project README for the full story, measurements and references.
"""


def vram() -> str:
    if not torch.cuda.is_available():
        return "running on CPU"
    used = torch.cuda.memory_allocated() / 2**20
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    return f"{used:,.0f} MB of {total:,.0f} MB VRAM in use · {torch.cuda.get_device_name(0)}"


# ----------------------------------------------------------------- train
def do_train(board, out, backbone, quantile, progress=gr.Progress()):
    lines = []

    def log(msg):
        lines.append(msg)
        progress(0, desc=msg)

    try:
        s = train_board(board, out, backbone, quantile, log=log)
    except Exception as e:  # noqa: BLE001 - show any problem to the operator
        return "\n".join(lines + [f"ERROR: {e}"]), None
    rows = [[r, v["bank_rows"], v["coreset_ratio"], v["threshold"], v["seconds"]] for r, v in s["rois"].items()]
    lines.append(f"done in {s['total_seconds']} s · models saved in {out}")
    return "\n".join(lines), rows


# --------------------------------------------------------------- inspect
def do_load(models_dir):
    try:
        BackbonePool.clear()
        STATE["models"] = load_models(models_dir)
        STATE["dir"] = models_dir
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    m = next(iter(STATE["models"].values()))
    return f"{len(STATE['models'])} ROI model(s) · backbone {BACKBONES[m.backbone_key].label} (shared) · {vram()}"


def do_inspect(image, margin):
    if STATE["models"] is None:
        return None, "Load a models folder first.", None, []
    if image is None:
        return None, "Upload an image.", None, []
    canvas, report, flagged = inspect(STATE["models"], np.asarray(image), margin)
    bad = [r for r, v in report.items() if v["defect"]]
    verdict = (f"## ❌ DEFECT — {len(bad)} of {len(report)} ROI(s) flagged" if bad
               else f"## ✅ OK — {len(report)} ROI(s) inspected")
    table = sorted(([r, v["score"], "defect" if v["defect"] else "ok"] for r, v in report.items()),
                   key=lambda x: -x[1])
    gallery = [(o, f"{r} · outline") for r, o, _ in flagged] + [(h, f"{r} · heat map") for r, _, h in flagged]
    return canvas, verdict + f"\n\n<sub>{vram()}</sub>", table, gallery


# -------------------------------------------------------------------- UI
cli = argparse.ArgumentParser()
cli.add_argument("--models", default="")
cli.add_argument("--image", default="")
cli.add_argument("--port", type=int, default=7860)
ARGS, _ = cli.parse_known_args()

with gr.Blocks(title="PCB solder anomaly detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# PCB solder anomaly detection\n"
                "PatchCore trained **only on good boards** · one memory bank per ROI · "
                "**one shared DINOv2 backbone** in VRAM.")
    with gr.Tabs(selected="inspect" if ARGS.models else "train"):
        with gr.Tab("1 · Train", id="train"):
            gr.Markdown("Point to a board folder with `train/` (good images) and, optionally, `rois.json`. "
                        "Without `rois.json`, each image is one region.")
            with gr.Row():
                board = gr.Textbox(label="Board folder", placeholder="data/board_a")
                out = gr.Textbox(label="Save models to", placeholder="models/board_a")
            with gr.Row():
                backbone = gr.Dropdown(list(BACKBONES), value="dinov2_vitb14", label="Backbone")
                quantile = gr.Slider(0.5, 1.0, value=0.90, step=0.01, label="Threshold quantile (leave-one-out)")
            btn_train = gr.Button("Train", variant="primary")
            log = gr.Textbox(label="Log", lines=8)
            summary = gr.Dataframe(headers=["ROI", "bank vectors", "coreset ratio", "threshold", "seconds"],
                                   label="Trained models")
            btn_train.click(do_train, [board, out, backbone, quantile], [log, summary])

        with gr.Tab("2 · Inspect", id="inspect"):
            with gr.Row():
                models_dir = gr.Textbox(label="Models folder", placeholder="models/board_a", value=ARGS.models,
                                        scale=3)
                btn_load = gr.Button("Load models", scale=1)
            status = gr.Markdown()
            btn_load.click(do_load, models_dir, status)
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(label="Panel / board image", type="numpy", value=ARGS.image or None)
                    margin = gr.Slider(0.8, 2.0, value=1.15, step=0.05,
                                       label="Sensitivity margin (limit = threshold × margin)")
                    btn_inspect = gr.Button("Inspect", variant="primary")
                with gr.Column(scale=2):
                    verdict = gr.Markdown()
                    result = gr.Image(label="Result")
            with gr.Row():
                scores = gr.Dataframe(headers=["ROI", "score (≥ 1 = defect)", "verdict"], label="Scores")
                gallery = gr.Gallery(label="Flagged regions", columns=4, height=260)
            btn_inspect.click(do_inspect, [image, margin], [result, verdict, scores, gallery])

        with gr.Tab("3 · How it works", id="how"):
            gr.Markdown(HOW_IT_WORKS)

    if ARGS.models:  # preload for a one-shot view (used for the README screenshot)
        demo.load(do_load, models_dir, status).then(do_inspect, [image, margin], [result, verdict, scores, gallery])

if __name__ == "__main__":
    import os

    # corporate proxies configured in Windows can make Gradio think localhost is unreachable
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    demo.launch(server_name="127.0.0.1", server_port=ARGS.port)
