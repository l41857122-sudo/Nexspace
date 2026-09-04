"""
anomaly_engine.py
-----------------
Dynamic Anomaly Extraction & Spatial Evidence Engine for Remote Sensing.

Converts classical differential change analysis, object grounding detections,
and multimodal sensor indicators into structured, traceable, georeferenced spatial evidence.

Features:
- Dynamic thresholding (Otsu's method, Percentile, Adaptive)
- Connected components spatial segmentation
- Noise filtering with minimum area thresholds
- Normalized bounding boxes ([x1, y1, x2, y2] & [x1/W, y1/H, x2/W, y2/H])
- Honest georeferencing checks (no fabricated lat/lon)
- Standardized evidence schemas for Change, Grounding, and Multimodal indicators
- Strict coordinate validation & boundary verification
"""

import time
import math
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import scipy.ndimage

from geospatial import GeospatialEngine, GeoMetadata


class AnomalyEngine:
    """Modular engine for spatial anomaly extraction and evidence normalization."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Dynamic Thresholding Computations
    # ------------------------------------------------------------------
    @staticmethod
    def compute_otsu_threshold(diff_array_0_255: np.ndarray) -> float:
        """
        Computes Otsu's optimal global threshold minimizing intra-class variance.
        Returns threshold value in 0..255 scale.
        """
        flat = diff_array_0_255.flatten()
        if flat.size == 0 or np.all(flat == flat[0]):
            return 38.25  # Default ~15% threshold for flat images

        hist, bin_edges = np.histogram(flat, bins=256, range=(0, 256))
        total = flat.size
        current_max = 0.0
        threshold = 38.25
        sum_total = np.dot(np.arange(256), hist)
        sum_b = 0.0
        w_b = 0.0

        for t in range(256):
            w_b += hist[t]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break

            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * ((m_b - m_f) ** 2)

            if var_between > current_max:
                current_max = var_between
                threshold = float(t)

        # Enforce minimum noise threshold floor (e.g. at least 15/255 intensity delta)
        return max(15.0, threshold)

    @staticmethod
    def compute_percentile_threshold(diff_array_0_255: np.ndarray, percentile: float = 95.0) -> float:
        """Computes threshold at specified percentile of non-zero intensity differences."""
        non_zero = diff_array_0_255[diff_array_0_255 > 5.0]
        if non_zero.size == 0:
            return 38.25
        val = float(np.percentile(non_zero, percentile))
        return max(15.0, val)

    # ------------------------------------------------------------------
    # Spatial Region Extraction
    # ------------------------------------------------------------------
    def extract_spatial_regions(
        self,
        diff_array: np.ndarray,
        threshold_strategy: str = "otsu",
        custom_threshold: Optional[float] = None,
        min_pixel_area: int = 20,
        min_area_fraction: float = 0.00005,
    ) -> Dict[str, Any]:
        """
        Segments a 2D intensity difference map into candidate change regions
        using dynamic thresholding and connected component labeling.
        """
        t0 = time.perf_counter()
        h, w = diff_array.shape
        total_pixels = h * w

        # Ensure diff_array is in 0..255 range
        diff_255 = np.clip(diff_array, 0.0, 255.0)

        # Determine threshold
        if threshold_strategy == "otsu":
            thresh_val = self.compute_otsu_threshold(diff_255)
            thresh_method = "otsu_optimal_variance"
        elif threshold_strategy == "percentile":
            thresh_val = self.compute_percentile_threshold(diff_255, 95.0)
            thresh_method = "percentile_95"
        elif threshold_strategy == "fixed" and custom_threshold is not None:
            thresh_val = float(custom_threshold * 255.0 if custom_threshold <= 1.0 else custom_threshold)
            thresh_method = "fixed_threshold"
        else:
            thresh_val = 38.25  # 15% of 255
            thresh_method = "default_15_percent"

        binary_mask = diff_255 >= thresh_val

        # 8-connectivity structure
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_features = scipy.ndimage.label(binary_mask, structure=structure)
        slices = scipy.ndimage.find_objects(labeled_array)

        regions_before_filter = num_features
        filtered_regions: List[Dict[str, Any]] = []

        for idx, s in enumerate(slices, start=1):
            if s is None:
                continue

            # Compute actual pixel area belonging to this connected component
            region_mask = labeled_array[s] == idx
            area_pixels = int(np.sum(region_mask))
            area_fraction = float(area_pixels / total_pixels)

            # Noise filtering
            if area_pixels < min_pixel_area or area_fraction < min_area_fraction:
                continue

            y_slice, x_slice = s
            x1 = float(x_slice.start)
            y1 = float(y_slice.start)
            x2 = float(x_slice.stop)
            y2 = float(y_slice.stop)

            # Validation
            if not self.validate_coordinates(x1, y1, x2, y2, w, h):
                continue

            # Compute regional statistics
            region_diff_vals = diff_255[s][region_mask]
            mean_delta = float(np.mean(region_diff_vals)) if region_diff_vals.size > 0 else 0.0
            max_delta = float(np.max(region_diff_vals)) if region_diff_vals.size > 0 else 0.0

            change_score = round(float(np.clip(mean_delta / 255.0, 0.0, 1.0)), 4)
            # Heuristic severity combines intensity delta and spatial extent
            severity = round(float(np.clip(change_score * 0.6 + min(1.0, area_fraction * 20.0) * 0.4, 0.0, 1.0)), 4)

            region_dict = {
                "id": f"change_region_{idx:03d}",
                "evidence_type": "change_region",
                "bbox_pixel": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "bbox_normalized": [
                    round(x1 / w, 4),
                    round(y1 / h, 4),
                    round(x2 / w, 4),
                    round(y2 / h, 4),
                ],
                "area_pixels": area_pixels,
                "area_fraction": round(area_fraction, 6),
                "mean_intensity_delta": round(mean_delta, 2),
                "max_intensity_delta": round(max_delta, 2),
                "change_score": change_score,
                "severity_score": severity,
                "severity_score_type": "heuristic",
                "source": "classical_change_analysis",
            }
            filtered_regions.append(region_dict)

        # Sort regions by severity descending
        filtered_regions.sort(key=lambda r: r["severity_score"], reverse=True)

        dur = (time.perf_counter() - t0) * 1000.0
        total_changed_pixels = int(np.sum([r["area_pixels"] for r in filtered_regions]))
        changed_frac = round(float(total_changed_pixels / total_pixels), 6)

        return {
            "regions": filtered_regions,
            "total_regions": len(filtered_regions),
            "regions_before_filter": regions_before_filter,
            "regions_after_filter": len(filtered_regions),
            "threshold_method": thresh_method,
            "threshold_value_255": round(thresh_val, 2),
            "threshold_value_normalized": round(thresh_val / 255.0, 4),
            "total_changed_pixels": total_changed_pixels,
            "changed_fraction": changed_frac,
            "image_dimensions": [w, h],
            "processing_time_ms": round(dur, 2),
        }

    # ------------------------------------------------------------------
    # End-to-End Image Pair Extraction
    # ------------------------------------------------------------------
    def extract_change_anomalies(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
        threshold_strategy: str = "otsu",
        custom_threshold: Optional[float] = None,
        min_pixel_area: int = 20,
    ) -> Dict[str, Any]:
        """Runs full change detection and anomaly extraction on two images."""
        if not isinstance(image_a, Image.Image) or not isinstance(image_b, Image.Image):
            raise ValueError("Both image_a and image_b must be valid PIL Image objects.")

        w, h = image_a.size
        if image_b.size != (w, h):
            image_b = image_b.resize((w, h), Image.Resampling.BILINEAR)

        if image_a.mode != "L" and image_b.mode != "L":
            arr_a = np.asarray(image_a.convert("RGB"), dtype=np.float32)
            arr_b = np.asarray(image_b.convert("RGB"), dtype=np.float32)
            diff = np.max(np.abs(arr_b - arr_a), axis=-1)
        else:
            arr_a = np.asarray(image_a.convert("L"), dtype=np.float32)
            arr_b = np.asarray(image_b.convert("L"), dtype=np.float32)
            diff = np.abs(arr_b - arr_a)

        result = self.extract_spatial_regions(
            diff_array=diff,
            threshold_strategy=threshold_strategy,
            custom_threshold=custom_threshold,
            min_pixel_area=min_pixel_area,
        )

        # Check for geospatial metadata (GeoTIFF tags)
        geo_meta = GeospatialEngine.extract_metadata(image_a)
        if not geo_meta.geospatial_available:
            geo_meta = GeospatialEngine.extract_metadata(image_b)

        result["geospatial_coordinates_available"] = geo_meta.geospatial_available
        if geo_meta.geospatial_available:
            result["geo_metadata"] = geo_meta.to_dict()
            enriched_regions = []
            for r in result.get("regions", []):
                enriched_regions.append(GeospatialEngine.enrich_evidence_item(r, geo_meta))
            result["regions"] = enriched_regions
        else:
            for r in result.get("regions", []):
                r["geospatial_coordinates_available"] = False

        return result

    # ------------------------------------------------------------------
    # Evidence Normalization Framework
    # ------------------------------------------------------------------
    def normalize_evidence(
        self,
        raw_items: List[Dict[str, Any]],
        source_type: str,
        image_dimensions: Tuple[int, int] = (512, 512),
        image_ref: Optional[Image.Image] = None,
    ) -> List[Dict[str, Any]]:
        """
        Normalizes various specialist tool outputs (Change, Grounding, Multimodal)
        into standardized spatial evidence structures with optional geospatial enrichment.
        """
        w, h = image_dimensions
        normalized: List[Dict[str, Any]] = []
        geo_meta = GeospatialEngine.extract_metadata(image_ref) if image_ref is not None else GeoMetadata(geospatial_available=False)

        if source_type == "change_analysis":
            for item in raw_items:
                c_item = {
                    "type": "change_region",
                    "id": item.get("id", "change_region"),
                    "bbox_pixel": item.get("bbox_pixel", [0, 0, 0, 0]),
                    "bbox_normalized": item.get("bbox_normalized", [0.0, 0.0, 0.0, 0.0]),
                    "area_pixels": item.get("area_pixels", 0),
                    "area_fraction": item.get("area_fraction", 0.0),
                    "change_score": item.get("change_score", 0.0),
                    "severity_score": item.get("severity_score", 0.0),
                    "severity_score_type": "heuristic",
                    "threshold_method": item.get("threshold_method", "otsu"),
                    "source": "classical_change_analysis",
                    "geospatial_coordinates_available": False,
                }
                if geo_meta.geospatial_available:
                    c_item = GeospatialEngine.enrich_evidence_item(c_item, geo_meta)
                normalized.append(c_item)

        elif source_type == "grounding":
            for idx, item in enumerate(raw_items, start=1):
                box = item.get("box", [0, 0, 0, 0])
                x1, y1, x2, y2 = box
                norm_box = [
                    round(x1 / w, 4) if w > 0 else 0.0,
                    round(y1 / h, 4) if h > 0 else 0.0,
                    round(x2 / w, 4) if w > 0 else 0.0,
                    round(y2 / h, 4) if h > 0 else 0.0,
                ]
                g_item = {
                    "type": "object_detection",
                    "id": f"detection_{idx:03d}",
                    "label": item.get("label", "object"),
                    "bbox_pixel": [round(c, 2) for c in box],
                    "bbox_normalized": norm_box,
                    "score": round(float(item.get("score", 0.0)), 4),
                    "source": item.get("source", "Grounding_DINO"),
                    "geospatial_coordinates_available": False,
                }
                if geo_meta.geospatial_available:
                    g_item = GeospatialEngine.enrich_evidence_item(g_item, geo_meta)
                normalized.append(g_item)

        elif source_type == "optical_sar":
            for item in raw_items:
                normalized.append({
                    "type": "cross_modal_indicator",
                    "optical_source": item.get("optical_source", ""),
                    "sar_source": item.get("sar_source", ""),
                    "fusion_type": item.get("fusion_type", "feature_fusion_baseline"),
                    "alignment_status": item.get("alignment_status", "dimension_match_only"),
                    "cross_modal_cosine_similarity": item.get("cross_modal_cosine_similarity"),
                    "interpretation": "indicator_only",
                    "geospatial_coordinates_available": False,
                })

        return normalized

    # ------------------------------------------------------------------
    # Validation & Georeferencing Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def validate_coordinates(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> bool:
        """Validates bounding box geometry and limits."""
        if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
            return False
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        if (x2 - x1) * (y2 - y1) <= 0.0:
            return False
        return True

    def validate_anomaly_result(self, region: Dict[str, Any], width: int, height: int) -> bool:
        """Validates region schema and numerical correctness."""
        box = region.get("bbox_pixel")
        if not box or len(box) != 4:
            return False
        x1, y1, x2, y2 = box
        if not self.validate_coordinates(x1, y1, x2, y2, width, height):
            return False

        norm_box = region.get("bbox_normalized")
        if not norm_box or len(norm_box) != 4:
            return False
        for c in norm_box:
            if c < 0.0 or c > 1.0 or math.isnan(c) or math.isinf(c):
                return False

        if region.get("area_pixels", 0) <= 0:
            return False
        if region.get("area_fraction", 0.0) <= 0.0 or region.get("area_fraction", 0.0) > 1.0:
            return False
        return True

    @staticmethod
    def check_geospatial_metadata(img: Union[Image.Image, str, bytes]) -> bool:
        """Inspects if image contains genuine GeoTIFF spatial tags."""
        meta = GeospatialEngine.extract_metadata(img)
        return meta.geospatial_available


anomaly_engine = AnomalyEngine()
