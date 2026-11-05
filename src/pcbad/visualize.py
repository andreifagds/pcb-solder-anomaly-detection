"""Draw the anomaly on the crop: an outline at a fraction of the heat-map peak.

An outline (instead of painting the region) keeps the solder visible under the
mark. Yellow and magenta were picked because they stay distinct from each other
and from green solder mask under the common types of color blindness.
"""

from __future__ import annotations

import cv2
import numpy as np

YELLOW = (255, 221, 0)
MAGENTA = (255, 0, 200)


def defect_mask(anomaly_map: np.ndarray, fraction_of_peak: float = 0.5, min_level: float | None = None) -> np.ndarray:
    """Pixels above `fraction_of_peak` x max of the map and, if given, above `min_level`
    (the model's own decision limit, in the same distance units) -> uint8 0/255."""
    peak = float(anomaly_map.max())
    level = peak * fraction_of_peak
    if min_level is not None and min_level < peak:  # the blurred map can peak below the image score
        level = max(level, min_level)
    return (anomaly_map >= level).astype(np.uint8) * 255


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) of the largest connected region."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, :4]
    return int(x), int(y), int(w), int(h)


def outline(crop_rgb: np.ndarray, anomaly_map: np.ndarray, fraction_of_peak: float = 0.5,
            min_level: float | None = None) -> np.ndarray:
    """crop (H,W,3 uint8 RGB) + map (h,w) -> copy of the crop with the defect outlined."""
    h, w = crop_rgb.shape[:2]
    amap = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = defect_mask(amap, fraction_of_peak, min_level)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = crop_rgb.copy()
    t = max(2, round(max(h, w) / 260))  # line width that reads the same on a 512 px ROI or a full image
    cv2.drawContours(out, contours, -1, (0, 0, 0), 2 * t, cv2.LINE_AA)  # dark halo for contrast
    cv2.drawContours(out, contours, -1, YELLOW, t, cv2.LINE_AA)
    for c in contours:  # magenta dashes on top of the yellow line
        pts = c[:, 0, :]
        for i in range(0, len(pts) - 1, 8):
            cv2.line(out, tuple(map(int, pts[i])), tuple(map(int, pts[min(i + 3, len(pts) - 1)])), MAGENTA, t, cv2.LINE_AA)
    return out


def heatmap(crop_rgb: np.ndarray, anomaly_map: np.ndarray, vmax: float | None = None, alpha: float = 0.5,
            vmin: float = 0.0) -> np.ndarray:
    """Color overlay of the raw map between vmin and vmax (for figures)."""
    h, w = crop_rgb.shape[:2]
    amap = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
    vmax = vmax or float(amap.max()) or 1.0
    norm = np.clip((amap - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    return (crop_rgb * (1 - alpha) + color * alpha).astype(np.uint8)
