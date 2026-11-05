"""Post-processing for generated illustrations.

whiten_background: SDXL ignores "white background" and paints a flat grey or blue
one. Pixels close to the border color that are connected to the image edge are the
background; they are replaced with pure white. The sticker outline stops the fill
at the subject's edge.

needs_retry: some seeds produce a wallpaper of many small copies instead of one
subject, or an extra object cut off by the image edge. After whitening, such images
have little background, many separate objects, or an object touching the frame,
which is cheap to measure.
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
HAZE_MIN = 200          # a soft drop shadow left around the sticker: every channel at least this,
HAZE_CHROMA = 40        # and nearly grey (max - min channel at most this)


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
    # A light grey haze touching the background (a soft shadow around the sticker's
    # white border) goes white as well. The sticker's dark outline keeps the fill
    # out of the subject's own light parts.
    light = (rgb.min(axis=2) >= HAZE_MIN) & (np.ptp(rgb, axis=2) <= HAZE_CHROMA)
    near, _ = ndimage.label(light | bg)
    haze = np.isin(near, np.unique(near[bg])) & ~bg
    white = bg | specks | haze
    rgb[white] = 255
    # Objects touching the frame are cut off: a dictionary picture shows its subject whole.
    parts, _ = ndimage.label(~white)
    part_sizes = np.bincount(parts.ravel()) / white.size
    edge = np.unique(np.concatenate([parts[0], parts[-1], parts[:, 0], parts[:, -1]]))
    stats = {
        "background_fraction": round(float(bg.mean()), 3),
        "objects": int((sizes >= MIN_OBJECT).sum()),
        "edge_objects": int(sum(1 for p in edge if p > 0 and part_sizes[p] >= MIN_OBJECT)),
    }
    return Image.fromarray(rgb), stats


def needs_retry(stats: dict[str, float]) -> bool:
    """A repeating pattern, nothing left after cleanup, or something cut off by the edge: try another seed."""
    return (stats["background_fraction"] < 0.3 or not 1 <= stats["objects"] <= 3
            or stats["edge_objects"] > 0)
