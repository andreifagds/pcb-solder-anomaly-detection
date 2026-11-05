"""How much of the training data each ROI model keeps (coreset ratio).

A fixed ratio makes the memory bank grow linearly with the dataset: double the
photos, double the VRAM. We let the ratio shrink as the dataset grows, but
less than proportionally:

    ratio = base * (reference / n_crops) ** exponent      (clamped to [floor, base])

exponent 0   -> fixed ratio (bank doubles when data doubles)
exponent 0.5 -> bank grows ~41% when data doubles   <- used here
exponent 1   -> bank never grows (more data adds nothing)

The floor (0.01) is the headline setting of the PatchCore paper: we never keep
less than what the method was validated with.
"""

from __future__ import annotations

DEFAULTS = {"base": 0.10, "reference": 50, "exponent": 0.5, "floor": 0.01}


def coreset_ratio(n_crops: int, base=0.10, reference=50, exponent=0.5, floor=0.01) -> float:
    if n_crops <= reference:
        return base
    return max(floor, base * (reference / n_crops) ** exponent)


def bank_rows(n_crops: int, grid_side: int, **kw) -> int:
    """Number of feature vectors kept: crops x grid positions x ratio."""
    return max(1, int(n_crops * grid_side * grid_side * coreset_ratio(n_crops, **kw)))


def bank_megabytes(n_crops: int, grid_side: int, dim: int, **kw) -> float:
    return bank_rows(n_crops, grid_side, **kw) * dim * 4 / 2**20
