"""VRAM / load time / latency: N independent PatchCore models vs. one shared backbone.

  dedicated : N x anomalib PatchcoreModel, each with its own backbone copy + bank
  shared    : 1 backbone (BackbonePool) + N memory banks (RoiPatchCore)

Memory banks have the realistic size of an 80-image ROI dataset after the
dynamic coreset (values are random: VRAM and kNN cost depend only on shape).
Configurations whose projected footprint exceeds the GPU budget are NOT run
(on Windows the driver would silently spill into system RAM); their value is
projected from the measured per-model cost and flagged as such.

    python scripts/benchmark_vram.py --out results/vram_benchmark.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # noqa: E402

from pcbad import BACKBONES, BackbonePool, RoiPatchCore, bank_rows, coreset_ratio  # noqa: E402

MB = 2**20


def sync():
    torch.cuda.synchronize()


def reset():
    gc.collect()
    BackbonePool.clear()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sync()


def latency(fn, x, models, repeats=3):
    """Mean ms per ROI over a sweep of all models (one image each)."""
    for m in models[: min(3, len(models))]:
        fn(m, x)  # warm-up
    sync()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        for m in models:
            fn(m, x)
        sync()
        times.append((time.perf_counter() - t) * 1000 / len(models))
    return statistics.median(times)


def run_shared(key, n, rows, dim, x):
    reset()
    base = torch.cuda.memory_allocated()
    BackbonePool.get(key, "cuda")
    models = []
    for _ in range(n):
        m = RoiPatchCore(key).to("cuda")
        m._scorer.memory_bank = torch.randn(rows, dim, device="cuda")
        m.threshold = 1.0
        models.append(m)
    sync()
    resident = (torch.cuda.memory_allocated() - base) / MB
    ms = latency(lambda m, x: m.predict(x), x, models)
    peak = (torch.cuda.max_memory_allocated() - base) / MB
    del models
    return resident, peak, ms


def run_dedicated(key, n, rows, dim, x):
    reset()
    p = BACKBONES[key]
    base = torch.cuda.memory_allocated()
    models = []
    for _ in range(n):
        m = PatchcoreModel(layers=list(p.layers), backbone=p.timm_name, pre_trained=False, num_neighbors=9)
        m = m.cuda().eval()
        m.memory_bank = torch.randn(rows, dim, device="cuda")
        models.append(m)
    sync()
    resident = (torch.cuda.memory_allocated() - base) / MB

    def f(m, x):
        with torch.inference_mode():
            m(x)

    ms = latency(f, x, models)
    peak = (torch.cuda.max_memory_allocated() - base) / MB
    del models
    return resident, peak, ms


def load_times(key, rows, dim, tmp: Path, k=5):
    """Cold load of one ROI model from disk: full anomalib checkpoint vs lean bank-only file."""
    p = BACKBONES[key]
    full = PatchcoreModel(layers=list(p.layers), backbone=p.timm_name, pre_trained=False)
    full.memory_bank = torch.randn(rows, dim)
    torch.save(full.state_dict(), tmp / "full.pt")
    lean = RoiPatchCore(key)
    lean._scorer.memory_bank = torch.randn(rows, dim)
    lean.threshold = 1.0
    lean.save(tmp / "lean.pt")
    sizes = {"full_ckpt_mb": (tmp / "full.pt").stat().st_size / MB, "lean_ckpt_mb": (tmp / "lean.pt").stat().st_size / MB}
    del full, lean

    BackbonePool.get(key, "cuda")  # the shared backbone is already resident in production
    t_full, t_lean = [], []
    for _ in range(k):
        sync()
        t = time.perf_counter()
        m = PatchcoreModel(layers=list(p.layers), backbone=p.timm_name, pre_trained=False)
        m.load_state_dict(torch.load(tmp / "full.pt", map_location="cpu"), strict=False)
        m.cuda()
        sync()
        t_full.append(time.perf_counter() - t)
        del m
        t = time.perf_counter()
        m = RoiPatchCore.load(tmp / "lean.pt", "cuda")
        sync()
        t_lean.append(time.perf_counter() - t)
        del m
    return {**sizes, "load_full_s": statistics.median(t_full), "load_lean_s": statistics.median(t_lean)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["wide_resnet50", "dinov2_vits14", "dinov2_vitb14"])
    ap.add_argument("--n", nargs="+", type=int, default=[1, 5, 10, 20, 40, 80, 130])
    ap.add_argument("--crops", type=int, default=80, help="training images per ROI")
    ap.add_argument("--budget", type=float, default=0.85, help="fraction of GPU memory we allow ourselves")
    ap.add_argument("--out", default="results/vram_benchmark.json")
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    budget_mb = props.total_memory / MB * args.budget
    out = {
        "gpu": props.name,
        "gpu_total_mb": props.total_memory / MB,
        "budget_mb": budget_mb,
        "torch": torch.__version__,
        "crops_per_roi": args.crops,
        "coreset_ratio": coreset_ratio(args.crops),
        "backbones": {},
    }
    tmp = Path(tempfile.mkdtemp())

    for key in args.backbones:
        p = BACKBONES[key]
        reset()
        base = torch.cuda.memory_allocated()
        bb = BackbonePool.get(key, "cuda")
        backbone_mb = (torch.cuda.memory_allocated() - base) / MB
        dim = bb.feature_dim
        del bb  # must not outlive this iteration, or it inflates the next baseline
        grid = p.input_size // p.stride
        rows = bank_rows(args.crops, grid)
        bank_mb = rows * dim * 4 / MB
        x = torch.randn(1, 3, p.input_size, p.input_size, device="cuda")
        info = {
            "label": p.label,
            "input_size": p.input_size,
            "grid": grid,
            "feature_dim": dim,
            "bank_rows": rows,
            "bank_mb": bank_mb,
            "backbone_mb": backbone_mb,
            "asymptotic_saving": backbone_mb / (backbone_mb + bank_mb),
            "runs": [],
        }
        print(f"\n== {p.label}: backbone {backbone_mb:.1f} MB, bank {rows}x{dim} = {bank_mb:.1f} MB")
        info.update(load_times(key, rows, dim, tmp))

        per_model_dedicated = None
        for n in args.n:
            row = {"n": n}
            r, pk, ms = run_shared(key, n, rows, dim, x)
            row.update(shared_resident_mb=r, shared_peak_mb=pk, shared_ms=ms)

            projected = (per_model_dedicated or (backbone_mb + bank_mb)) * n
            if projected < budget_mb:
                r, pk, ms = run_dedicated(key, n, rows, dim, x)
                per_model_dedicated = r / n
                row.update(dedicated_resident_mb=r, dedicated_peak_mb=pk, dedicated_ms=ms, dedicated_measured=True)
            else:
                row.update(dedicated_resident_mb=projected, dedicated_measured=False)
            row["saving"] = 1 - row["shared_resident_mb"] / row["dedicated_resident_mb"]
            print(
                f"  N={n:4d}  shared {row['shared_resident_mb']:8.1f} MB ({row['shared_ms']:.1f} ms/ROI)   "
                f"dedicated {row['dedicated_resident_mb']:8.1f} MB {'' if row['dedicated_measured'] else '(projected)'}  "
                f"saving {row['saving']:.1%}"
            )
            info["runs"].append(row)
        out["backbones"][key] = info

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
