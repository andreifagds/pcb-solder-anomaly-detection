"""95% bootstrap intervals for image AUROC (block 8 vs block 2 of DINOv2 ViT-B on VisA),
resampling the test boards. Uses the scores saved by eval_visa.py and visa_layer_examples.py.

    python scripts/bootstrap_ci.py --visa data/visa/VisA --split data/visa/1cls.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from eval_visa import CATS, load_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--visa", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()

res = HERE.parent / "results"
b8, b2 = np.load(res / "visa_vitb_examples.npz"), np.load(res / "visa_vitb_blocks2_examples.npz")
split = load_split(Path(a.visa), Path(a.split), 100, 42)
rng = np.random.default_rng(0)
out = {}
for c in CATS:
    y = np.array([l for _, l, _ in split[c]["test"]])
    s8, s2 = b8[f"{c}_scores"], b2[f"{c}_scores"]
    d, a8, a2 = [], [], []
    for _ in range(a.n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        x8, x2 = roc_auc_score(y[i], s8[i]), roc_auc_score(y[i], s2[i])
        a8.append(x8)
        a2.append(x2)
        d.append(x2 - x8)
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]  # noqa: E731
    out[c] = {"block8": [round(roc_auc_score(y, s8), 3), ci(a8)], "block2": [round(roc_auc_score(y, s2), 3), ci(a2)],
              "diff_block2_minus_block8": [round(roc_auc_score(y, s2) - roc_auc_score(y, s8), 3), ci(d)]}
    print(c, out[c])
(res / "visa_bootstrap_ci.json").write_text(json.dumps(out, indent=2))
