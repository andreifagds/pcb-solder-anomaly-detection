"""Solder map: where the solder is on a board, learned from the good photos.

PatchCore flags what is new, not what matters: a scratch on the solder mask or a speck of
dust scores as high as a short. On the line, most of those false alarms land far from any
solder joint. The solder map keeps a flagged ROI as a defect only when the part of it above
the limit touches a solder joint (with a margin around it); otherwise the anomaly is reported
as "away from solder" and ignored.

How a pixel counts as solder, on each good photo. The jig has three LED strips: red up high next
to the camera, green in the middle, blue low next to the board. Shiny solder mirrors one strip per
slope: flat parts look red, gentle slopes green, steep slopes blue (scripts/make_lighting_figure.py).
  A) red/orange reflection that stands out from its surroundings: the flat, shiny part of a joint.
     Silkscreen, traces and labels do not have that metal shine.
  B) a filled, bright, saturated and roundish blob: the sloped sides of a joint, in blue/green.
     An opening erases thin strokes (letters, silkscreen, traces).
A pixel enters the map when it was solder in at least 10% of the good photos, and every joint
gets a 25 px margin. A missing joint is still caught: the map comes from the good photos,
so an empty pad is inside it.

Simplified from the filter that runs on the production line (which also checks for metal
shine that appeared or disappeared, to catch solder balls far from the joints).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULTS = {
    "margin_px": 25,          # margin around each joint
    "min_frequency": 0.10,    # solder in >= 10% of the good photos
    "region_margin_px": 10,   # the flagged region is grown by this before testing contact
    "red_s": 90, "red_v": 70, "red_contrast": 25,
    "blob_v": 150, "blob_s": 120, "blob_open_px": 7,
    "blob_area_min": 40, "blob_area_max": 8000, "blob_circularity": 0.50,
}


def _local_contrast(rgb: np.ndarray) -> np.ndarray:
    """How much the strongest channel of each pixel is above the median of its ~40 px surroundings."""
    mx = rgb.max(axis=2)
    small = cv2.resize(mx, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, 11), (mx.shape[1], mx.shape[0]), interpolation=cv2.INTER_LINEAR)
    return mx.astype(np.int16) - bg.astype(np.int16)


def solder_pixels(rgb: np.ndarray, p: dict = DEFAULTS) -> np.ndarray:
    """0/1 mask of the solder pixels of one photo (no margin). `rgb` is H x W x 3, RGB."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h < 14) | (h > 165)) & (s >= p["red_s"]) & (v >= p["red_v"])
           & (_local_contrast(rgb) > p["red_contrast"])).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    bright = ((v >= p["blob_v"]) & (s >= p["blob_s"])).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["blob_open_px"], p["blob_open_px"]))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k)
    blobs = np.zeros_like(bright)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if not p["blob_area_min"] <= area <= p["blob_area_max"]:
            continue
        perim = cv2.arcLength(c, True)
        if perim and 4 * np.pi * area / (perim * perim) >= p["blob_circularity"]:
            cv2.drawContours(blobs, [c], -1, 1, -1)
    return red | blobs


def _grow(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, k)


def build_map(photos, p: dict = DEFAULTS) -> np.ndarray:
    """Solder zone (0/1, with the margin) from the good photos (any iterable of RGB arrays, all
    aligned to the same reference; a generator keeps only one photo in memory at a time)."""
    count, n = None, 0
    for rgb in photos:
        m = solder_pixels(rgb, p).astype(np.uint16)
        count = m if count is None else count + m
        n += 1
    solder = (count >= max(1, round(p["min_frequency"] * n))).astype(np.uint8)
    return _grow(solder, p["margin_px"])


def save_map(zone: np.ndarray, path: str | Path) -> None:
    cv2.imwrite(str(path), zone * 255)


def load_map(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    return (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)


def near_solder(zone_crop: np.ndarray, above_limit: np.ndarray, p: dict = DEFAULTS) -> bool:
    """True when the part of the ROI above the limit (0/1, same size as the crop) touches the solder zone."""
    region = _grow(above_limit.astype(np.uint8), p["region_margin_px"])
    return bool((region & zone_crop).any())
