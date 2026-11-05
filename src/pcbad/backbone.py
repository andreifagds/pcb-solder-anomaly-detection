"""Shared, frozen feature extractor.

PatchCore never trains its backbone: there is no optimizer, the extractor runs
in eval() under no_grad(), so every per-ROI model carries a bit-for-bit
identical copy of it. This module keeps exactly ONE copy per
(backbone, layers, device) and lends it to every ROI model.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from anomalib.models.components import TimmFeatureExtractor


@dataclass(frozen=True)
class BackbonePreset:
    key: str
    timm_name: str
    layers: tuple[str, ...]
    input_size: int  # square input fed to the network (pixels)
    stride: int  # pixels of the resized input covered by one feature vector
    label: str


BACKBONES: dict[str, BackbonePreset] = {
    # Final choice of this work: wider gap between real defects and false alarms.
    "dinov2_vitb14": BackbonePreset(
        "dinov2_vitb14", "vit_base_patch14_reg4_dinov2.lvd142m", ("blocks.8",), 518, 14, "DINOv2 ViT-B/14"
    ),
    "dinov2_vits14": BackbonePreset(
        "dinov2_vits14", "vit_small_patch14_reg4_dinov2.lvd142m", ("blocks.8",), 518, 14, "DINOv2 ViT-S/14"
    ),
    # WideResNet-50 truncated at layer2 (the first production setup).
    "wide_resnet50": BackbonePreset("wide_resnet50", "wide_resnet50_2", ("layer2",), 320, 8, "WideResNet-50 (layer2)"),
    # Exact layers of the original PatchCore paper (layer2 + layer3, 1536 channels).
    "wide_resnet50_l23": BackbonePreset(
        "wide_resnet50_l23", "wide_resnet50_2", ("layer2", "layer3"), 320, 8, "WideResNet-50 (layer2+3)"
    ),
}


def get_preset(key: str) -> BackbonePreset:
    """'dinov2_vitb14' or, for ablations, 'dinov2_vitb14@blocks.5' (another layer)."""
    if "@" not in key:
        return BACKBONES[key]
    base, layer = key.split("@", 1)
    p = BACKBONES[base]
    return BackbonePreset(key, p.timm_name, (layer,), p.input_size, p.stride, f"{p.label} [{layer}]")


def _enable_fused_attention(module: torch.nn.Module) -> int:
    """anomalib builds timm models with exportable=True, which turns fused attention
    off. Same math, ~30% slower on ViTs, so we turn it back on."""
    n = 0
    for m in module.modules():
        if getattr(m, "fused_attn", None) is False:
            m.fused_attn = True
            n += 1
    return n


class SharedBackbone(torch.nn.Module):
    """Frozen extractor + the 3x3 average pooling PatchCore applies to its features."""

    def __init__(self, preset: BackbonePreset | str, device: str | torch.device = "cuda"):
        super().__init__()
        self.preset = get_preset(preset) if isinstance(preset, str) else preset
        self.device = torch.device(device)
        self.extractor = TimmFeatureExtractor(
            backbone=self.preset.timm_name, layers=list(self.preset.layers), pre_trained=True
        ).eval()
        _enable_fused_attention(self.extractor)
        self.pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.to(self.device)
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        with torch.inference_mode():
            x = torch.zeros(1, 3, self.preset.input_size, self.preset.input_size, device=self.device)
            return int(self.embed(x).shape[1])

    @torch.inference_mode()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """images (B,3,H,W), already normalized -> embedding (B, C, h, w)."""
        feats = self.extractor(images.to(self.device))
        feats = [self.pooler(feats[layer]) for layer in self.preset.layers]
        emb = feats[0]
        for f in feats[1:]:  # multi-layer case: upsample to the first layer's grid and concat
            f = torch.nn.functional.interpolate(f, size=emb.shape[-2:], mode="bilinear")
            emb = torch.cat([emb, f], dim=1)
        return emb


class BackbonePool:
    """Process-wide cache: one SharedBackbone per (preset, device)."""

    _items: dict[tuple[str, str], SharedBackbone] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, preset: str, device: str = "cuda") -> SharedBackbone:
        key = (preset, str(torch.device(device)))
        with cls._lock:
            if key not in cls._items:
                cls._items[key] = SharedBackbone(preset, device)
            return cls._items[key]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._items.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
