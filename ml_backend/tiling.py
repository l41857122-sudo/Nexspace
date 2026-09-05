"""
tiling.py
----------
High-Resolution Satellite Image Tiling & Multi-Scale Inference Engine.

Features:
  - Memory-safe sliding window grid slicing with configurable tile size & overlap stride
  - Precise tile-to-global image coordinate remapping
  - Vectorized Non-Maximum Suppression (NMS) to eliminate duplicate detections across overlapping tile boundaries
  - Canonical coordinate compliance: [xmin, ymin, xmax, ymax] and [0, 1000] normalized space
  - Strict non-inversion validation (xmin < xmax, ymin < ymax)
  - Seamless CRS and geospatial footprint preservation
"""

from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from model_runtime import GroundingDINORuntime
from geospatial import GeospatialEngine, GeoMetadata


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return float(inter_area / max(1e-6, union_area))


def apply_nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
    """Applies greedy Non-Maximum Suppression (NMS) over candidate detections."""
    if not detections:
        return []

    # Sort descending by score
    sorted_dets = sorted(detections, key=lambda d: d.get("score", 0.0), reverse=True)
    kept: List[Dict[str, Any]] = []

    for cand in sorted_dets:
        cand_box = cand.get("bbox_pixel") or cand.get("box")
        if not cand_box:
            continue

        discard = False
        for k in kept:
            k_box = k.get("bbox_pixel") or k.get("box")
            if compute_iou(cand_box, k_box) > iou_threshold:
                discard = True
                break

        if not discard:
            kept.append(cand)

    return kept


class TiledInferenceEngine:
    """
    Executes tiled / multi-scale vision model inference for large high-resolution satellite imagery.
    """

    def __init__(self, default_tile_size: int = 512, default_overlap: float = 0.20):
        self.default_tile_size = default_tile_size
        self.default_overlap = default_overlap

    def generate_tiles(
        self,
        image: Image.Image,
        tile_size: Optional[int] = None,
        overlap: Optional[float] = None,
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Slices an image into overlapping tiles.
        Returns a list of (tile_image, (x_offset, y_offset, tile_w, tile_h)).
        """
        w, h = image.size
        ts = tile_size or self.default_tile_size
        ov = overlap or self.default_overlap
        stride = int(round(ts * (1.0 - ov)))
        stride = max(32, stride)

        # If image is smaller than or equal to tile size, return single full tile
        if w <= ts and h <= ts:
            return [(image, (0, 0, w, h))]

        tiles = []
        y = 0
        while y < h:
            x = 0
            tile_h = min(ts, h - y)
            while x < w:
                tile_w = min(ts, w - x)
                crop_box = (x, y, x + tile_w, y + tile_h)
                tile_crop = image.crop(crop_box)
                tiles.append((tile_crop, (x, y, tile_w, tile_h)))
                if x + tile_w >= w:
                    break
                x += stride
            if y + tile_h >= h:
                break
            y += stride

        return tiles

    def run_tiled_grounding(
        self,
        image: Image.Image,
        target_phrase: str,
        grounding_runtime: Optional[GroundingDINORuntime] = None,
        tile_size: int = 512,
        overlap: float = 0.20,
        iou_threshold: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Runs tiled open-vocabulary object detection across large high-res satellite frames.
        """
        t0 = time.perf_counter()
        w, h = image.size
        geo_meta = GeospatialEngine.extract_metadata(image)

        # 1. Determine if tiling is necessary
        if max(w, h) <= tile_size:
            # Single-pass execution
            if grounding_runtime and grounding_runtime.is_available():
                res = grounding_runtime.infer(image=image, target_phrase=target_phrase)
                raw_dets = res.get("detections", [])
            else:
                raw_dets = []

            return {
                "tiling_used": False,
                "tiles_processed": 1,
                "target_phrase": target_phrase,
                "detections": raw_dets,
                "count": len(raw_dets),
                "image_width": w,
                "image_height": h,
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        # 2. Slice Tiles
        tiles = self.generate_tiles(image, tile_size=tile_size, overlap=overlap)
        raw_detections: List[Dict[str, Any]] = []

        # 3. Process Tiles and Remap Coordinates
        for tile_img, (off_x, off_y, tile_w, tile_h) in tiles:
            if grounding_runtime and grounding_runtime.is_available():
                t_res = grounding_runtime.infer(image=tile_img, target_phrase=target_phrase)
                t_dets = t_res.get("detections", [])

                for d in t_dets:
                    box = d.get("bbox_pixel") or d.get("box", [0, 0, tile_w, tile_h])
                    lx1, ly1, lx2, ly2 = box

                    # Global coordinate remapping
                    gx1 = max(0.0, min(float(w), float(lx1 + off_x)))
                    gy1 = max(0.0, min(float(h), float(ly1 + off_y)))
                    gx2 = max(0.0, min(float(w), float(lx2 + off_x)))
                    gy2 = max(0.0, min(float(h), float(ly2 + off_y)))

                    # Canonical box check (xmin < xmax and ymin < ymax)
                    if gx2 <= gx1 or gy2 <= gy1 or (gx2 - gx1) * (gy2 - gy1) < 4.0:
                        continue

                    # Normalized [0, 1000] coordinates
                    xmin_1000 = int(round((gx1 / max(1, w)) * 1000))
                    ymin_1000 = int(round((gy1 / max(1, h)) * 1000))
                    xmax_1000 = int(round((gx2 / max(1, w)) * 1000))
                    ymax_1000 = int(round((gy2 / max(1, h)) * 1000))

                    ev_item = {
                        "type": "bounding_box",
                        "label": d.get("label", target_phrase),
                        "box": [round(gx1, 1), round(gy1, 1), round(gx2, 1), round(gy2, 1)],
                        "bbox_pixel": [round(gx1, 1), round(gy1, 1), round(gx2, 1), round(gy2, 1)],
                        "box_2d": [xmin_1000, ymin_1000, xmax_1000, ymax_1000],
                        "bbox_normalized": [xmin_1000, ymin_1000, xmax_1000, ymax_1000],
                        "score": d.get("score", 0.50),
                        "source": "tiled_grounding_dino",
                        "image_dimensions": [w, h],
                    }

                    if geo_meta.geospatial_available:
                        ev_item = GeospatialEngine.enrich_evidence_item(ev_item, geo_meta)

                    raw_detections.append(ev_item)

        # 4. Apply Non-Maximum Suppression (NMS) to eliminate duplicate border detections
        merged_detections = apply_nms(raw_detections, iou_threshold=iou_threshold)
        dur = (time.perf_counter() - t0) * 1000.0

        return {
            "tiling_used": True,
            "tile_size": tile_size,
            "overlap_ratio": overlap,
            "tiles_processed": len(tiles),
            "raw_detections_count": len(raw_detections),
            "detections": merged_detections,
            "count": len(merged_detections),
            "image_width": w,
            "image_height": h,
            "duration_ms": round(dur, 2),
        }


# Global Tiled Inference Engine instance
tiled_inference_engine = TiledInferenceEngine()


def generate_sliding_tiles(
    image: Image.Image,
    tile_size: int = 512,
    overlap: float = 0.20,
) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    """Convenience helper to slice an image into sliding window tiles."""
    return tiled_inference_engine.generate_tiles(image, tile_size=tile_size, overlap=overlap)


def merge_tile_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """Convenience helper to merge candidate detections with NMS."""
    return apply_nms(detections, iou_threshold=iou_threshold)


def normalize_bbox_1000(
    box: List[float],
    img_w: int,
    img_h: int,
) -> List[int]:
    """
    Normalizes pixel bounding box [x1, y1, x2, y2] to standard [0, 1000] integer format.
    Guarantees xmin < xmax and ymin < ymax.
    """
    x1, y1, x2, y2 = box
    xmin = max(0, min(1000, int(round((x1 / max(1, img_w)) * 1000))))
    ymin = max(0, min(1000, int(round((y1 / max(1, img_h)) * 1000))))
    xmax = max(0, min(1000, int(round((x2 / max(1, img_w)) * 1000))))
    ymax = max(0, min(1000, int(round((y2 / max(1, img_h)) * 1000))))

    # Enforce minimum 1-unit extent to avoid point/inverted collapse
    if xmax <= xmin:
        xmax = min(1000, xmin + 1)
        if xmax <= xmin and xmin > 0:
            xmin = xmax - 1
    if ymax <= ymin:
        ymax = min(1000, ymin + 1)
        if ymax <= ymin and ymin > 0:
            ymin = ymax - 1

    return [xmin, ymin, xmax, ymax]

