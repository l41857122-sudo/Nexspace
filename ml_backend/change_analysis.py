"""
change_analysis.py
-------------------
Registered before/after image-pair comparison: pixel-level difference
heatmap plus a lightweight textual summary of change magnitude/location.

This is a classical (non-learned) diff pipeline, which is appropriate for
already co-registered image pairs. If your imagery isn't co-registered,
add an alignment step (e.g. ORB feature matching + homography) before
calling `analyze()`.

Requires:
    pip install pillow numpy
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from PIL import Image


@dataclass
class ChangeResult:
    heatmap: Image.Image          # RGBA heatmap image, same size as inputs
    overlay: Image.Image          # heatmap alpha-blended over image B
    changed_fraction: float       # 0..1, fraction of pixels flagged as changed
    mean_intensity_delta: float   # average absolute intensity difference
    summary: str                  # human-readable textual comparison


def _to_gray_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def _colorize_heatmap(diff_norm: np.ndarray) -> Image.Image:
    """Map a 0..1 normalized diff array to a red-intensity heatmap (RGBA)."""
    h, w = diff_norm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255                              # R
    rgba[..., 1] = (255 * (1 - diff_norm)).astype(np.uint8)  # G fades out with change
    rgba[..., 2] = (255 * (1 - diff_norm)).astype(np.uint8)  # B fades out with change
    rgba[..., 3] = (diff_norm * 255).astype(np.uint8)        # alpha = change strength
    return Image.fromarray(rgba, mode="RGBA")


def analyze(
    image_a: Image.Image,
    image_b: Image.Image,
    change_threshold: float = 0.15,
    resize_to: Tuple[int, int] = None,
) -> ChangeResult:
    """
    Compare two co-registered images (A = before, B = after).

    change_threshold: fraction of dynamic range (0..1) above which a pixel
                       is counted as "changed" for the `changed_fraction`
                       summary statistic.
    resize_to: optionally force both images to the same size before
               diffing (required if A and B differ in resolution).
    """
    if resize_to:
        image_a = image_a.resize(resize_to)
        image_b = image_b.resize(resize_to)
    elif image_a.size != image_b.size:
        # Auto resize image_b to image_a size if different
        image_b = image_b.resize(image_a.size)

    a = _to_gray_array(image_a)
    b = _to_gray_array(image_b)

    diff = np.abs(b - a)
    max_val = diff.max() if diff.max() > 0 else 1.0
    diff_norm = diff / max_val  # normalize to 0..1 for visualization

    changed_mask = diff_norm >= change_threshold
    changed_fraction = float(changed_mask.mean())
    mean_intensity_delta = float(diff.mean())

    heatmap = _colorize_heatmap(diff_norm)
    overlay = image_b.convert("RGBA")
    overlay = Image.alpha_composite(overlay, heatmap)

    if changed_fraction < 0.02:
        magnitude_desc = "negligible change"
    elif changed_fraction < 0.10:
        magnitude_desc = "minor localized change"
    elif changed_fraction < 0.30:
        magnitude_desc = "moderate change across a notable portion of the scene"
    else:
        magnitude_desc = "extensive change across most of the scene"

    summary = (
        f"Change analysis detected {magnitude_desc} between Image A (before) and "
        f"Image B (after): {changed_fraction * 100:.1f}% of pixels exceeded the "
        f"{change_threshold:.0%} change threshold, with a mean intensity delta of "
        f"{mean_intensity_delta:.1f} (0-255 scale)."
    )

    return ChangeResult(
        heatmap=heatmap,
        overlay=overlay,
        changed_fraction=changed_fraction,
        mean_intensity_delta=mean_intensity_delta,
        summary=summary,
    )
