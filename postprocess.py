"""Post-processing for generated illustrations.

whiten_background: SDXL ignores "white background" and paints a flat grey or blue
one. Pixels close to the border color that are connected to the image edge are the
background; they are replaced with pure white. The sticker outline stops the fill
at the subject's edge.

looks_like_pattern: some seeds produce a wallpaper of many small copies instead of
one subject. After whitening, such images have little background and many separate
objects, which is cheap to measure.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

BORDER = 8          # px of frame used to estimate the background color
TOLERANCE = 38.0    # RGB distance still counted as background
MIN_OBJECT = 0.01   # objects smaller than this share of the image are ignored


HOLE_TOLERANCE = 14.0   # stricter match for background seen through holes in the subject
MIN_HOLE = 0.002


def _background_mask(rgb: np.ndarray) -> np.ndarray:
    frame = np.concatenate([
        rgb[:BORDER].reshape(-1, 3), rgb[-BORDER:].reshape(-1, 3),
        rgb[:, :BORDER].reshape(-1, 3), rgb[:, -BORDER:].reshape(-1, 3),
    ])
    color = np.median(frame, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - color, axis=2)
    labels, _ = ndimage.label(dist < TOLERANCE)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    bg = np.isin(labels, edge[edge > 0])
    if color.min() > 235:  # already white: holes are invisible, and white subject parts must stay
        return bg
    # Enclosed patches (e.g. inside a rainbow's arch) only count when they match the
    # background color closely, so a grey subject on a grey background is kept.
    holes, _ = ndimage.label((dist < HOLE_TOLERANCE) & ~bg)
    sizes = np.bincount(holes.ravel())[1:] / bg.size
    return bg | np.isin(holes, np.flatnonzero(sizes >= MIN_HOLE) + 1)


def whiten_background(image: Image.Image) -> tuple[Image.Image, dict[str, float]]:
    rgb = np.asarray(image.convert("RGB")).copy()
    bg = _background_mask(rgb)
    labels, _ = ndimage.label(~bg)
    sizes = np.bincount(labels.ravel())[1:] / bg.size
    # Specks floating in the background (stray sparkles, dust) go white too.
    specks = np.isin(labels, np.flatnonzero(sizes < MIN_OBJECT) + 1)
    rgb[bg | specks] = 255
    stats = {
        "background_fraction": round(float(bg.mean()), 3),
        "objects": int((sizes >= MIN_OBJECT).sum()),
    }
    return Image.fromarray(rgb), stats


def looks_like_pattern(stats: dict[str, float]) -> bool:
    return stats["background_fraction"] < 0.3 or stats["objects"] > 3
