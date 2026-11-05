"""Scores and heat maps of DINOv2 ViT-B read at block 2 on the VisA test boards,
to show next to the block-8 maps (production setting) in the examples figure.

    python scripts/visa_layer_examples.py --visa data/visa/VisA --split data/visa/1cls.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from eval_visa import CATS, embed, fit, load_split, read_rgb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--visa", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--layer", default="blocks.2")
ap.add_argument("--n-train", type=int, default=100)
a = ap.parse_args()

key = f"dinov2_vitb14@{a.layer}"
split = load_split(Path(a.visa), Path(a.split), a.n_train, 42)
out = {}
for c in CATS:
    tr = [read_rgb(p, 518) for p, _, _ in split[c]["train"]]
    te = [read_rgb(p, 518) for p, _, _ in split[c]["test"]]
    m = fit(key, embed(key, tr), seed=42)
    s, maps = m.score_embeddings(embed(key, te).cuda(), 128)
    out[f"{c}_scores"] = s.cpu().numpy()
    out[f"{c}_thr"] = np.array(m.threshold)
    out[f"{c}_maps"] = maps.cpu().numpy().astype(np.float16)
    print(c, "done", flush=True)
dst = HERE.parent / "results" / f"visa_vitb_{a.layer.replace('.', '')}_examples.npz"
np.savez_compressed(dst, **out)
print("saved", dst)
