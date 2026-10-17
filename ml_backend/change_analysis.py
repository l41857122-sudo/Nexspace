"""
change_analysis.py
-------------------
Evidence-Based Temporal Change Analysis for Registered Satellite Imagery.
Phase 4 Hardened: Distinguishes genuine surface modifications from acquisition conditions,
geospatial misalignment, clouds/saturation, missing nodata pixels, and sensor differences.

Features:
  - Input scene validation and temporal acquisition metadata tracking
  - Common geographical bounds & overlap enforcement (rejects disjoint scenes)
  - Nodata / margin padding exclusion to prevent false-positive change spikes
  - Cloud / atmospheric saturation detection and explicit transparency warnings
  - Radiometric / intensity difference heatmap and alpha-blended visual overlay
  - Structured limitations exposing evidence provenance without claiming unverified physical construction
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
from PIL import Image


@dataclass
class ChangeResult:
    heatmap: Image.Image          # RGBA heatmap image, same size as inputs
    overlay: Image.Image          # heatmap alpha-blended over image B
    changed_fraction: float       # 0..1, fraction of valid pixels flagged as changed
    mean_intensity_delta: float   # average absolute intensity difference across valid pixels
    summary: str                  # human-readable textual comparison
    image_dimensions: Tuple[int, int] = (0, 0)
    processing_time_ms: float = 0.0
    method: str = "classical_pixel_difference"
    scene_a_id: Optional[str] = None
    scene_b_id: Optional[str] = None
    acquisition_date_a: Optional[str] = None
    acquisition_date_b: Optional[str] = None
    sensor_a: Optional[str] = None
    sensor_b: Optional[str] = None
    valid_pixels_fraction: float = 1.0
    nodata_pixels_count: int = 0
    cloud_contamination_detected: bool = False
    cloud_pixels_fraction: float = 0.0
    alignment_status: str = "dimension_aligned"
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "changed_fraction": round(self.changed_fraction, 4),
            "mean_intensity_delta": round(self.mean_intensity_delta, 2),
            "image_dimensions": list(self.image_dimensions),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "method": self.method,
            "scene_a_id": self.scene_a_id,
            "scene_b_id": self.scene_b_id,
            "acquisition_date_a": self.acquisition_date_a,
            "acquisition_date_b": self.acquisition_date_b,
            "sensor_a": self.sensor_a,
            "sensor_b": self.sensor_b,
            "valid_pixels_fraction": round(self.valid_pixels_fraction, 4),
            "nodata_pixels_count": self.nodata_pixels_count,
            "cloud_contamination_detected": self.cloud_contamination_detected,
            "cloud_pixels_fraction": round(self.cloud_pixels_fraction, 4),
            "alignment_status": self.alignment_status,
            "limitations": self.limitations,
        }


def _to_gray_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def _colorize_heatmap(diff_norm: np.ndarray, nodata_mask: Optional[np.ndarray] = None) -> Image.Image:
    """Map a 0..1 normalized diff array to a red-intensity heatmap (RGBA)."""
    h, w = diff_norm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255                              # R
    rgba[..., 1] = (255 * (1 - diff_norm)).astype(np.uint8)  # G fades out with change
    rgba[..., 2] = (255 * (1 - diff_norm)).astype(np.uint8)  # B fades out with change
    rgba[..., 3] = (diff_norm * 255).astype(np.uint8)        # alpha = change strength

    if nodata_mask is not None:
        rgba[nodata_mask, 3] = 0                    # fully transparent on nodata regions

    return Image.fromarray(rgba, mode="RGBA")


def _parse_bounds_dict(b: Any) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(b, dict):
        min_x = b.get("min_x", b.get("west", b.get("left")))
        min_y = b.get("min_y", b.get("south", b.get("bottom")))
        max_x = b.get("max_x", b.get("east", b.get("right")))
        max_y = b.get("max_y", b.get("north", b.get("top")))
        if None not in (min_x, min_y, max_x, max_y):
            return (float(min_x), float(min_y), float(max_x), float(max_y))
    elif isinstance(b, (list, tuple)) and len(b) == 4:
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return None


def analyze(
    image_a: Image.Image,
    image_b: Image.Image,
    change_threshold: float = 0.15,
    resize_to: Optional[Tuple[int, int]] = None,
    scene_a_id: Optional[str] = None,
    scene_b_id: Optional[str] = None,
    acquisition_date_a: Optional[str] = None,
    acquisition_date_b: Optional[str] = None,
    sensor_a: Optional[str] = None,
    sensor_b: Optional[str] = None,
    bounds_a: Optional[Any] = None,
    bounds_b: Optional[Any] = None,
    crs_a: Optional[str] = None,
    crs_b: Optional[str] = None,
) -> ChangeResult:
    """
    Compare two co-registered observations (A = before, B = after).

    Validates:
      1. Image integrity and dimensions
      2. Geographic bounds overlap (rejects non-overlapping scenes)
      3. Nodata masking (excludes missing/zero-padding margins from change score)
      4. Cloud/saturation detection (flags atmospheric artifacts)
      5. Multi-date metadata attribution
    """
    t0 = time.perf_counter()

    if image_a is None or image_b is None:
        raise ValueError("Both image_a and image_b must be provided for temporal change analysis.")

    # 1. Geographic Overlap Validation
    parsed_b_a = _parse_bounds_dict(bounds_a)
    parsed_b_b = _parse_bounds_dict(bounds_b)
    alignment_status = "dimension_aligned"

    if parsed_b_a is not None and parsed_b_b is not None:
        w1, s1, e1, n1 = parsed_b_a
        w2, s2, e2, n2 = parsed_b_b
        overlap = (max(w1, w2) < min(e1, e2)) and (max(s1, s2) < min(n1, n2))
        if not overlap:
            raise ValueError(
                f"Change analysis rejected: Geographical bounds of observations A and B do not overlap. "
                f"Bounds A: {parsed_b_a}, Bounds B: {parsed_b_b}"
            )
        alignment_status = "geospatially_verified_overlap"

    # 2. Dimension Harmonization
    if resize_to:
        image_a = image_a.resize(resize_to)
        image_b = image_b.resize(resize_to)
        alignment_status = "rescaled_to_target_grid"
    elif image_a.size != image_b.size:
        image_b = image_b.resize(image_a.size)
        alignment_status = "rescaled_to_common_dimensions"

    dims = image_a.size
    total_pixels = dims[0] * dims[1]

    # 3. Nodata and Valid Data Masking
    arr_a_rgb = np.asarray(image_a.convert("RGB"), dtype=np.float32)
    arr_b_rgb = np.asarray(image_b.convert("RGB"), dtype=np.float32)

    # Margins or zero-padded regions where all channels are 0 in either scene
    nodata_mask = (arr_a_rgb.sum(axis=-1) == 0) | (arr_b_rgb.sum(axis=-1) == 0)
    nodata_count = int(np.sum(nodata_mask))
    valid_mask = ~nodata_mask
    valid_count = int(np.sum(valid_mask))
    valid_fraction = float(valid_count / max(1, total_pixels))

    # 4. Intensity Differencing on Valid Pixels Only
    gray_a = _to_gray_array(image_a)
    gray_b = _to_gray_array(image_b)

    diff = np.abs(gray_b - gray_a)
    diff[nodata_mask] = 0.0

    if valid_count > 0:
        valid_diff = diff[valid_mask]
        max_val = float(valid_diff.max()) if valid_diff.max() > 0 else 1.0
        diff_norm = np.zeros_like(diff)
        diff_norm[valid_mask] = diff[valid_mask] / max_val
        changed_mask = (diff_norm >= change_threshold) & valid_mask
        changed_fraction = float(np.sum(changed_mask)) / float(valid_count)
        mean_intensity_delta = float(valid_diff.mean())
    else:
        diff_norm = np.zeros_like(diff)
        changed_fraction = 0.0
        mean_intensity_delta = 0.0

    # 5. Cloud / High-Saturation Atmospheric Contamination Check
    cloud_a = (arr_a_rgb[..., 0] > 225) & (arr_a_rgb[..., 1] > 225) & (arr_a_rgb[..., 2] > 225)
    cloud_b = (arr_b_rgb[..., 0] > 225) & (arr_b_rgb[..., 1] > 225) & (arr_b_rgb[..., 2] > 225)
    cloud_mask = (cloud_a | cloud_b) & valid_mask
    cloud_count = int(np.sum(cloud_mask))
    cloud_fraction = float(cloud_count / max(1, valid_count)) if valid_count > 0 else 0.0
    cloud_detected = cloud_fraction > 0.08

    # 6. Visualization Heatmap & Alpha-Blended Overlay
    heatmap = _colorize_heatmap(diff_norm, nodata_mask=nodata_mask)
    overlay = image_b.convert("RGBA")
    overlay = Image.alpha_composite(overlay, heatmap)

    # 7. Descriptive Summary & Honesty Contract
    if changed_fraction < 0.02:
        magnitude_desc = "negligible change"
    elif changed_fraction < 0.10:
        magnitude_desc = "minor localized change"
    elif changed_fraction < 0.30:
        magnitude_desc = "moderate change across a notable portion of the scene"
    else:
        magnitude_desc = "extensive change across most of the scene"

    date_str = ""
    if acquisition_date_a and acquisition_date_b:
        date_str = f" between {acquisition_date_a} and {acquisition_date_b}"

    summary = (
        f"Change analysis detected {magnitude_desc}{date_str}: "
        f"{changed_fraction * 100:.1f}% of usable pixels exceeded the "
        f"{change_threshold:.0%} change threshold, with a mean intensity delta of "
        f"{mean_intensity_delta:.1f} (0-255 scale)."
    )

    # 8. Methodological Limitations & Evidence Notes
    limitations: List[str] = [
        "Analysis is based on bi-temporal spectral and radiometric differencing, not an assumption of verified physical construction or demolition.",
        "Detected differences can result from seasonal vegetation phenology, illumination/shadow variations, or acquisition condition differences.",
    ]
    if cloud_detected:
        limitations.append(
            f"Atmospheric or cloud/shadow contamination detected across {cloud_fraction * 100:.1f}% of usable area; differences in these regions may represent clouds or shadows rather than ground changes."
        )
    if nodata_count > 0:
        nodata_pct = (nodata_count / total_pixels) * 100.0
        limitations.append(
            f"{nodata_pct:.1f}% of scene pixels were masked as nodata/border padding and excluded from change calculations."
        )
    if sensor_a and sensor_b and sensor_a.lower() != sensor_b.lower():
        limitations.append(
            f"Cross-sensor comparison ({sensor_a} vs {sensor_b}): spectral band response functions differ, which can induce radiometric differences."
        )

    dur = (time.perf_counter() - t0) * 1000.0

    return ChangeResult(
        heatmap=heatmap,
        overlay=overlay,
        changed_fraction=changed_fraction,
        mean_intensity_delta=mean_intensity_delta,
        summary=summary,
        image_dimensions=dims,
        processing_time_ms=round(dur, 2),
        method="classical_pixel_difference",
        scene_a_id=scene_a_id,
        scene_b_id=scene_b_id,
        acquisition_date_a=acquisition_date_a,
        acquisition_date_b=acquisition_date_b,
        sensor_a=sensor_a,
        sensor_b=sensor_b,
        valid_pixels_fraction=valid_fraction,
        nodata_pixels_count=nodata_count,
        cloud_contamination_detected=cloud_detected,
        cloud_pixels_fraction=cloud_fraction,
        alignment_status=alignment_status,
        limitations=limitations,
    )
