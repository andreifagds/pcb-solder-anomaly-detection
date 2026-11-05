"""Small web interface: train, configure and inspect (Gradio).

    pip install gradio==5.9.1
    python app/demo.py            # then open http://127.0.0.1:7860
    python app/demo.py --lang pt  # interface em português

Optional: --models <dir> --image <file> opens straight on the Inspect tab with
that models folder loaded and that image inspected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcbad import BACKBONES, BackbonePool  # noqa: E402
from pcbad.pipeline import inspect, load_models, load_solder_map, train_board  # noqa: E402

cli = argparse.ArgumentParser()
cli.add_argument("--models", default="")
cli.add_argument("--image", default="")
cli.add_argument("--port", type=int, default=7860)
cli.add_argument("--lang", choices=["en", "pt"], default="en")
cli.add_argument("--auto", action="store_true", help="with --models and --image: inspect right away")
ARGS, _ = cli.parse_known_args()


def T(en: str, pt: str) -> str:
    return pt if ARGS.lang == "pt" else en


STATE: dict = {"models": None, "dir": None, "zone": None}

HOW_IT_WORKS = T("""
1. **Train.** Every good image is cut into the ROIs and a frozen DINOv2 turns each ROI into 37×37 feature vectors.
   A coreset keeps the most spread-out ones as that ROI's memory bank. The threshold is the 0.90 quantile of the
   leave-one-out scores of the good images.
2. **Inspect.** Each ROI of a new image is compared with its own memory bank, and the distance of the most unusual
   tile is the score. A score of 1 or more (raw score at or above threshold × margin) is a defect.
3. **Why it fits in a small GPU.** The backbone is never trained, so all ROI models share one copy of it. Only the
   banks (about 25 MB each) are per ROI: 130 ROIs need about 3.5 GB instead of 35 GB.

See the project README for the full story, measurements and references.
""", """
1. **Treino.** Cada foto boa é cortada nas ROIs e um DINOv2 congelado transforma cada ROI em 37×37 vetores.
   O coreset guarda os mais espalhados como o banco de memória daquela ROI. O limiar é o quantil 0,90 das notas
   leave-one-out das fotos boas.
2. **Inspeção.** Cada ROI de uma foto nova é comparada com o seu próprio banco, e a distância do pedacinho mais
   diferente é a nota. Nota 1 ou mais (nota bruta acima de limiar × margem) é defeito.
3. **Por que cabe numa GPU pequena.** O backbone nunca é treinado, então todos os modelos de ROI dividem uma cópia
   só. Só os bancos (uns 25 MB cada) são por ROI: 130 ROIs precisam de uns 3,5 GB em vez de 35 GB.

A história completa, as medições e as referências estão no README do projeto.
""")


def vram() -> str:
    if not torch.cuda.is_available():
        return T("running on CPU", "rodando na CPU")
    used = torch.cuda.memory_allocated() / 2**20
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    return T(f"{used:,.0f} MB of {total:,.0f} MB VRAM in use", f"{used:,.0f} MB de {total:,.0f} MB de VRAM em uso"
             ).replace(",", "." if ARGS.lang == "pt" else ",") + f" · {torch.cuda.get_device_name(0)}"


# ----------------------------------------------------------------- train
def do_train(board, out, backbone, quantile, progress=gr.Progress()):
    lines = []

    def log(msg):
        lines.append(msg)
        progress(0, desc=msg)

    try:
        s = train_board(board, out, backbone, quantile, log=log)
    except Exception as e:  # noqa: BLE001 - show any problem to the operator
        return "\n".join(lines + [f"{T('ERROR', 'ERRO')}: {e}"]), None
    rows = [[r, v["bank_rows"], v["coreset_ratio"], v["threshold"], v["seconds"]] for r, v in s["rois"].items()]
    lines.append(T(f"done in {s['total_seconds']} s · models saved in {out}",
                   f"pronto em {s['total_seconds']} s · modelos salvos em {out}"))
    return "\n".join(lines), rows


# --------------------------------------------------------------- inspect
def do_load(models_dir):
    try:
        BackbonePool.clear()
        STATE["models"] = load_models(models_dir)
        STATE["dir"] = models_dir
        STATE["zone"] = load_solder_map(models_dir)
    except Exception as e:  # noqa: BLE001
        return f"{T('ERROR', 'ERRO')}: {e}"
    m = next(iter(STATE["models"].values()))
    zone = T(" · solder map on", " · mapa de solda ligado") if STATE["zone"] is not None else ""
    return T(f"{len(STATE['models'])} ROI model(s) · backbone {BACKBONES[m.backbone_key].label} (shared){zone} · {vram()}",
             f"{len(STATE['models'])} modelo(s) de ROI · backbone {BACKBONES[m.backbone_key].label} (compartilhado){zone} · {vram()}")


def do_inspect(image, margin):
    if STATE["models"] is None:
        return None, T("Load a models folder first.", "Carregue uma pasta de modelos primeiro."), None, []
    if image is None:
        return None, T("Upload an image.", "Envie uma imagem."), None, []
    canvas, report, flagged = inspect(STATE["models"], np.asarray(image), margin, STATE["zone"])
    if canvas.shape[1] > 2048:  # a 5120 px panel is too heavy to send to the browser as is
        k = 2048 / canvas.shape[1]
        canvas = cv2.resize(canvas, (2048, round(canvas.shape[0] * k)), interpolation=cv2.INTER_AREA)
    bad = [r for r, v in report.items() if v["defect"]]
    verdict = (T(f"## ❌ DEFECT — {len(bad)} of {len(report)} ROI(s) flagged",
                 f"## ❌ REPROVADO — {len(bad)} de {len(report)} ROI(s) com anomalia") if bad
               else T(f"## ✅ OK — {len(report)} ROI(s) inspected", f"## ✅ APROVADO — {len(report)} ROI(s) inspecionadas"))
    away = sum(v.get("away_from_solder", False) for v in report.values())
    if away:
        verdict += T(f"\n\n{away} anomaly(ies) away from solder (ignored, gray frame)",
                     f"\n\n{away} anomalia(s) fora da solda (ignoradas, quadro cinza)")

    def label(v):
        if v["defect"]:
            return T("defect", "defeito")
        return T("away from solder", "fora da solda") if v.get("away_from_solder") else "ok"

    table = sorted(([r, v["score"], label(v)] for r, v in report.items()), key=lambda x: -x[1])
    gallery = ([(o, f"{r} · {T('outline', 'contorno')}") for r, o, _ in flagged]
               + [(h, f"{r} · {T('heat map', 'mapa de calor')}") for r, _, h in flagged])
    return canvas, verdict + f"\n\n<sub>{vram()}</sub>", table, gallery


# -------------------------------------------------------------------- UI
with gr.Blocks(title=T("PCB solder anomaly detection", "Detecção de anomalias em solda de PCB"),
               theme=gr.themes.Soft()) as demo:
    gr.Markdown(T("# PCB solder anomaly detection\n"
                  "PatchCore trained **only on good boards** · one memory bank per ROI · "
                  "**one shared DINOv2 backbone** in VRAM.",
                  "# Detecção de anomalias em solda de PCB\n"
                  "PatchCore treinado **só com placas boas** · um banco de memória por ROI · "
                  "**um único DINOv2 compartilhado** na VRAM."))
    with gr.Tabs(selected="inspect" if ARGS.models else "train"):
        with gr.Tab(T("1 · Train", "1 · Treinar"), id="train"):
            gr.Markdown(T("Point to a board folder with `train/` (good images) and, optionally, `rois.json`. "
                          "Without `rois.json`, each image is one region.",
                          "Aponte para a pasta de uma placa com `train/` (fotos boas) e, se tiver, o `rois.json`. "
                          "Sem `rois.json`, cada imagem vira uma região só."))
            with gr.Row():
                board = gr.Textbox(label=T("Board folder", "Pasta da placa"), placeholder="data/board_a")
                out = gr.Textbox(label=T("Save models to", "Salvar modelos em"), placeholder="models/board_a")
            with gr.Row():
                backbone = gr.Dropdown(list(BACKBONES), value="dinov2_vitb14", label="Backbone")
                quantile = gr.Slider(0.5, 1.0, value=0.90, step=0.01,
                                     label=T("Threshold quantile (leave-one-out)", "Quantil do limiar (leave-one-out)"))
            btn_train = gr.Button(T("Train", "Treinar"), variant="primary")
            log = gr.Textbox(label="Log", lines=8)
            summary = gr.Dataframe(headers=["ROI", T("bank vectors", "vetores no banco"), T("coreset ratio", "razão do coreset"),
                                            T("threshold", "limiar"), T("seconds", "segundos")],
                                   label=T("Trained models", "Modelos treinados"))
            btn_train.click(do_train, [board, out, backbone, quantile], [log, summary])

        with gr.Tab(T("2 · Inspect", "2 · Inspecionar"), id="inspect"):
            with gr.Row():
                models_dir = gr.Textbox(label=T("Models folder", "Pasta dos modelos"), placeholder="models/board_a",
                                        value=ARGS.models, scale=3)
                btn_load = gr.Button(T("Load models", "Carregar modelos"), scale=1)
            status = gr.Markdown()
            btn_load.click(do_load, models_dir, status)
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(label=T("Panel / board image", "Foto do painel / placa"), type="numpy",
                                     value=ARGS.image or None)
                    margin = gr.Slider(0.8, 2.0, value=1.15, step=0.05,
                                       label=T("Sensitivity margin (limit = threshold × margin)",
                                               "Margem de sensibilidade (limite = limiar × margem)"))
                    btn_inspect = gr.Button(T("Inspect", "Inspecionar"), variant="primary")
                with gr.Column(scale=2):
                    verdict = gr.Markdown()
                    result = gr.Image(label=T("Result", "Resultado"))
            with gr.Row():
                scores = gr.Dataframe(headers=["ROI", T("score (≥ 1 = defect)", "nota (≥ 1 = defeito)"),
                                               T("verdict", "resultado")], label=T("Scores", "Notas"))
                gallery = gr.Gallery(label=T("Flagged regions", "Regiões reprovadas"), columns=4, height=260)
            btn_inspect.click(do_inspect, [image, margin], [result, verdict, scores, gallery])

        with gr.Tab(T("3 · How it works", "3 · Como funciona"), id="how"):
            gr.Markdown(HOW_IT_WORKS)

    if ARGS.models:  # preload the models (and, with --auto, inspect the image right away)
        ev = demo.load(do_load, models_dir, status)
        if ARGS.auto:
            ev.then(do_inspect, [image, margin], [result, verdict, scores, gallery])

if __name__ == "__main__":
    import os

    # corporate proxies configured in Windows can make Gradio think localhost is unreachable
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    demo.launch(server_name="127.0.0.1", server_port=ARGS.port)
