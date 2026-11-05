"""Builds a small ready-to-train board folder from a VisA PCB category.

    python scripts/prepare_visa_demo.py --visa data/visa/VisA --split data/visa/1cls.csv --category pcb1

-> data/visa_demo/pcb1/train/        50 good boards
   data/visa_demo/pcb1/test/good/    10 unseen good boards
   data/visa_demo/pcb1/test/defect/  10 defective boards
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--visa", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--category", default="pcb1", choices=["pcb1", "pcb2", "pcb3", "pcb4"])
ap.add_argument("--n-train", type=int, default=50)
ap.add_argument("--n-test", type=int, default=10)
ap.add_argument("--out", default="data/visa_demo")
a = ap.parse_args()

rows = {"train": [], "good": [], "defect": []}
with open(a.split, newline="") as f:
    for r in csv.DictReader(f):
        if r["object"] != a.category:
            continue
        if r["split"] == "train":
            rows["train"].append(r["image"])
        else:
            rows["good" if r["label"] == "normal" else "defect"].append(r["image"])

rng = random.Random(42)
out = Path(a.out) / a.category
plan = [("train", "train", a.n_train), ("good", "test/good", a.n_test), ("defect", "test/defect", a.n_test)]
for key, sub, n in plan:
    d = out / sub
    d.mkdir(parents=True, exist_ok=True)
    for rel in rng.sample(rows[key], n):
        shutil.copy(Path(a.visa) / rel, d / Path(rel).name)
    print(f"{d}: {n} images")
