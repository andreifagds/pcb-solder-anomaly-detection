"""Sanity check: the shared-backbone ROI model gives the SAME scores as a vanilla
anomalib PatchcoreModel holding its own copy of the backbone and the same bank.

    python scripts/check_equivalence.py --backbone dinov2_vitb14
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anomalib.models.image.patchcore.torch_model import PatchcoreModel  # noqa: E402

from pcbad import BACKBONES, BackbonePool, RoiPatchCore  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", default="dinov2_vitb14", choices=list(BACKBONES))
ap.add_argument("--out", default=None)
args = ap.parse_args()

torch.manual_seed(0)
preset = BACKBONES[args.backbone]
s = preset.input_size
train = torch.randn(12, 3, s, s) * 0.5
test = torch.randn(4, 3, s, s) * 0.5

ours = RoiPatchCore(args.backbone).to("cuda").fit(train.cuda())
pred = ours.predict(test.cuda(), margin=1.0)

ref = PatchcoreModel(layers=list(preset.layers), backbone=preset.timm_name, pre_trained=True, num_neighbors=9)
ref = ref.cuda().eval()
ref.memory_bank = ours.memory_bank.clone()
with torch.inference_mode():
    out = ref(test.cuda())

ours_s = torch.tensor([p.raw_score for p in pred])
ref_s = out.pred_score.float().cpu()
ours_m = torch.stack([torch.from_numpy(p.anomaly_map) for p in pred])
ref_m = out.anomaly_map[:, 0].float().cpu()

# backbone weights: shared copy vs the vanilla model's own copy
shared = BackbonePool.get(args.backbone, "cuda").extractor.state_dict()
own = ref.feature_extractor.state_dict()
w_diff = max((shared[k].float() - own[k].float()).abs().max().item() for k in own)

res = {
    "backbone": preset.label,
    "max_abs_score_diff": float((ours_s - ref_s).abs().max()),
    "max_rel_score_diff": float(((ours_s - ref_s).abs() / ref_s.abs()).max()),
    "max_abs_map_diff": float((ours_m - ref_m).abs().max()),
    "map_range": float(ref_m.max()),
    "backbone_weight_max_diff": w_diff,
    "n_backbone_tensors": len(own),
}
print(json.dumps(res, indent=2))
if args.out:
    Path(args.out).write_text(json.dumps(res, indent=2))
