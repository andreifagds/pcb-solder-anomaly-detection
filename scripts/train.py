"""Train one PatchCore model per ROI, all sharing the same backbone.

    python scripts/train.py --board data/board_a --out models/board_a

Writes models/<board>/<roi>.pt (memory bank + threshold only) and summary.json.
Without a rois.json in the board folder, every image is treated as one region.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcbad import BACKBONES  # noqa: E402
from pcbad.pipeline import train_board  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--board", required=True, help="folder with train/ (and optionally rois.json)")
ap.add_argument("--out", required=True)
ap.add_argument("--backbone", default="dinov2_vitb14", choices=list(BACKBONES))
ap.add_argument("--quantile", type=float, default=0.90)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--device", default=None)
a = ap.parse_args()

summary = train_board(a.board, a.out, a.backbone, a.quantile, a.seed, a.device)
print(f"done in {summary['total_seconds']} s -> {a.out}")
