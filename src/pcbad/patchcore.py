"""One PatchCore model per ROI, without its own backbone.

A per-ROI model is just:
  * memory_bank  (K x C)  feature vectors of good solder at that ROI (coreset)
  * threshold    (float)  calibrated on the good images themselves (leave-one-out)

The backbone is borrowed from `BackbonePool` at call time. Scoring reuses the
exact code of anomalib's PatchcoreModel (nearest neighbour + re-weighting +
anomaly map), so the numbers match the reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from anomalib.models.components import KCenterGreedy
from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from .backbone import BackbonePool, SharedBackbone
from .coreset import coreset_ratio

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess(images_rgb: list[np.ndarray] | np.ndarray, size: int) -> torch.Tensor:
    """uint8 RGB HxWx3 image(s) -> normalized float tensor (B,3,size,size)."""
    if isinstance(images_rgb, np.ndarray) and images_rgb.ndim == 3:
        images_rgb = [images_rgb]
    x = torch.stack([torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1) for im in images_rgb]).float() / 255
    x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class SeparableBlur(torch.nn.Module):
    """anomalib smooths the anomaly map with a 33x33 Gaussian in one 2D pass.
    A Gaussian is separable, so 33x1 followed by 1x33 is the same filter at a
    fraction of the cost (measured 68 ms -> 0.55 ms per ROI on the target GPU)."""

    def __init__(self, original):
        super().__init__()
        k = original.kernel[0, 0].detach().float()
        p = k.sum(dim=1)
        p = p / p.sum()
        self.pad = list(original.padding_shape)
        self.mode = original.border_type
        self.register_buffer("v", p.view(1, 1, -1, 1))
        self.register_buffer("h", p.view(1, 1, 1, -1))

    def forward(self, x):
        x = F.pad(x, self.pad, mode=self.mode)
        return F.conv2d(F.conv2d(x, self.v.to(x.dtype)), self.h.to(x.dtype))


class _Scorer(PatchcoreModel):
    """anomalib's PatchcoreModel minus the line that builds a feature extractor."""

    def __init__(self, num_neighbors: int = 9):
        super(PatchcoreModel, self).__init__()  # skip PatchcoreModel.__init__ on purpose
        self.tiler = None
        self.num_neighbors = num_neighbors
        self.feature_extractor = None  # <- the whole point: no per-ROI backbone
        self.anomaly_map_generator = AnomalyMapGenerator()
        self.anomaly_map_generator.blur = SeparableBlur(self.anomaly_map_generator.blur)
        self.register_buffer("memory_bank", torch.empty(0))
        self.embedding_store = []


@dataclass
class Prediction:
    raw_score: float  # PatchCore image score
    score: float  # raw / (threshold * margin); >= 1.0 means defect
    is_defect: bool
    anomaly_map: np.ndarray  # (H, W) float32, raw patch distances, input resolution


@dataclass
class RoiPatchCore:
    backbone_key: str
    num_neighbors: int = 9
    threshold: float | None = None
    calibration_scores: list[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self._scorer = _Scorer(self.num_neighbors)

    # ---------------------------------------------------------------- helpers
    @property
    def backbone(self) -> SharedBackbone:
        return BackbonePool.get(self.backbone_key, self.device)

    @property
    def device(self) -> str:
        return self.meta.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def memory_bank(self) -> torch.Tensor:
        return self._scorer.memory_bank

    def to(self, device: str) -> "RoiPatchCore":
        self.meta["device"] = str(device)
        self._scorer.to(device)
        return self

    def _embed(self, images: torch.Tensor, batch_size: int = 8) -> torch.Tensor:
        return torch.cat([self.backbone.embed(images[i : i + batch_size]) for i in range(0, len(images), batch_size)])

    def _score(self, emb: torch.Tensor, bank: torch.Tensor, out_size: int):
        """emb (B,C,h,w) -> (image scores (B,), anomaly maps (B,1,out,out))."""
        b, c, h, w = emb.shape
        flat = emb.permute(0, 2, 3, 1).reshape(-1, c)
        self._scorer.memory_bank = bank
        patch, loc = self._scorer.nearest_neighbors(flat, n_neighbors=1)
        patch, loc = patch.reshape(b, -1), loc.reshape(b, -1)
        img_scores = self._scorer.compute_anomaly_score(patch, loc, flat)
        maps = self._scorer.anomaly_map_generator(patch.reshape(b, 1, h, w), (out_size, out_size))
        return img_scores, maps

    # ------------------------------------------------------------------- train
    @torch.inference_mode()
    def fit(self, images: torch.Tensor, seed: int = 42, calibrate_quantile: float = 0.90, **coreset_kw) -> "RoiPatchCore":
        """images: preprocessed good crops of ONE ROI (N,3,S,S)."""
        return self.fit_embeddings(self._embed(images), images.shape[-1], seed, calibrate_quantile, **coreset_kw)

    @torch.inference_mode()
    def fit_embeddings(
        self,
        emb: torch.Tensor,
        out_size: int,
        seed: int = 42,
        calibrate_quantile: float = 0.90,
        ratio: float | None = None,
        **coreset_kw,
    ) -> "RoiPatchCore":
        """Same as fit(), from backbone embeddings (N,C,h,w) already computed."""
        n, c, h, w = emb.shape
        flat = emb.permute(0, 2, 3, 1).reshape(-1, c)
        ratio = ratio if ratio is not None else coreset_ratio(n, **coreset_kw)
        torch.manual_seed(seed)
        if ratio >= 1.0:  # full memory bank, no coreset
            idx = torch.arange(len(flat), device=flat.device)
        else:
            idx = torch.tensor(KCenterGreedy(flat, sampling_ratio=ratio).select_coreset_idxs(), device=flat.device)
        bank = flat[idx].contiguous()
        source = idx // (h * w)  # which training image each bank vector came from
        self._scorer.memory_bank = bank

        # Leave-one-out calibration: score each good image against the bank
        # WITHOUT its own vectors, i.e. as if it were a new good part.
        scores = []
        for i in range(n):
            s, _ = self._score(emb[i : i + 1], bank[source != i], out_size)
            scores.append(float(s))
        self._scorer.memory_bank = bank
        self.calibration_scores = scores
        self.threshold = float(np.quantile(scores, calibrate_quantile))
        self.meta.update(n_images=n, coreset_ratio=ratio, bank_shape=list(bank.shape), grid=[h, w])
        return self

    # --------------------------------------------------------------- inference
    @torch.inference_mode()
    def score_embeddings(self, emb: torch.Tensor, out_size: int, batch_size: int | None = None):
        """Raw image scores (B,) and anomaly maps (B,out,out) from embeddings."""
        if batch_size is None:  # keep the (patches x bank) distance matrix under ~1 GB
            per_image = emb.shape[-1] * emb.shape[-2] * len(self.memory_bank)
            batch_size = max(1, min(8, int(2.5e8 // per_image)))
        s, m = [], []
        for i in range(0, len(emb), batch_size):
            a, b = self._score(emb[i : i + batch_size], self.memory_bank, out_size)
            s.append(a.float())
            m.append(b[:, 0].float())
        return torch.cat(s), torch.cat(m)

    @torch.inference_mode()
    def predict(self, images: torch.Tensor, margin: float = 1.15) -> list[Prediction]:
        emb = self._embed(images)
        scores, maps = self._score(emb, self.memory_bank, images.shape[-1])
        limit = self.threshold * margin
        return [
            Prediction(float(s), float(s) / limit, bool(s >= limit), m[0].float().cpu().numpy())
            for s, m in zip(scores, maps)
        ]

    # ---------------------------------------------------------------- storage
    def save(self, path: str | Path) -> None:
        """Lean checkpoint: memory bank + threshold only (no backbone weights)."""
        torch.save(
            {
                "backbone_key": self.backbone_key,
                "num_neighbors": self.num_neighbors,
                "threshold": self.threshold,
                "calibration_scores": self.calibration_scores,
                "meta": self.meta,
                "memory_bank": self.memory_bank.detach().cpu(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> "RoiPatchCore":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(ck["backbone_key"], ck["num_neighbors"], ck["threshold"], ck["calibration_scores"], ck["meta"])
        m._scorer.memory_bank = ck["memory_bank"]
        return m.to(device)
