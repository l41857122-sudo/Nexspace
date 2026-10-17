"""
tiling.py
----------
Generalized High-Resolution Satellite Image Tiling & Multi-Scale Detection Engine.

Features:
  - Memory-safe sliding window grid slicing with configurable tile size & overlap stride
  - Boundary-aligned grid coverage with exact image edge preservation
  - Precise tile-to-global image coordinate remapping
  - Generalized class-aware Non-Maximum Suppression (NMS) and deduplication
  - Semantic-aware geometric box sanity filtering (compact vs macro targets)
  - Full support for arbitrary single and multi-target classes
  - Canonical coordinate compliance: [xmin, ymin, xmax, ymax] in [0, 1000] normalized space
  - Seamless CRS and geospatial metadata enrichment
"""

from __future__ import annotations
import os
import sys
import time
import math
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from model_runtime import GroundingDINORuntime
from geospatial import GeospatialEngine, GeoMetadata
from router import build_generalized_aerial_prompt, get_canonical_target_label, normalize_detection_label, MACRO_EXTENDED_CATEGORIES


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection-over-Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
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


def remap_tile_boxes_to_global(
    boxes: List[Dict[str, Any]],
    tile_x1: int,
    tile_y1: int,
    tile_w: int,
    tile_h: int,
    image_w: int,
    image_h: int,
) -> List[Dict[str, Any]]:
    """
    Remaps detections from tile-relative space into global image space and canonical [0, 1000] coordinates.
    """
    remapped = []
    for d in boxes:
        raw_box = d.get("box") or d.get("bbox_pixel") or d.get("box_2d")
        if not raw_box:
            continue
        lx1, ly1, lx2, ly2 = raw_box
        
        # If coordinates are in [0, 1000] tile-relative space, convert to tile pixels
        if max(lx1, ly1, lx2, ly2) <= 1000 and isinstance(lx1, int) and isinstance(lx2, int):
            tx1 = (lx1 / 1000.0) * tile_w
            ty1 = (ly1 / 1000.0) * tile_h
            tx2 = (lx2 / 1000.0) * tile_w
            ty2 = (ly2 / 1000.0) * tile_h
        else:
            tx1, ty1, tx2, ty2 = float(lx1), float(ly1), float(lx2), float(ly2)
            
        gx1 = max(0.0, min(float(image_w), tx1 + tile_x1))
        gy1 = max(0.0, min(float(image_h), ty1 + tile_y1))
        gx2 = max(0.0, min(float(image_w), tx2 + tile_x1))
        gy2 = max(0.0, min(float(image_h), ty2 + tile_y1))
        
        if gx2 <= gx1 or gy2 <= gy1:
            continue
            
        xmin_1000 = int(round(max(0.0, min(1.0, gx1 / max(1, image_w))) * 1000))
        ymin_1000 = int(round(max(0.0, min(1.0, gy1 / max(1, image_h))) * 1000))
        xmax_1000 = int(round(max(0.0, min(1.0, gx2 / max(1, image_w))) * 1000))
        ymax_1000 = int(round(max(0.0, min(1.0, gy2 / max(1, image_h))) * 1000))
        
        new_d = dict(d)
        new_d["box"] = [xmin_1000, ymin_1000, xmax_1000, ymax_1000]
        new_d["box_2d"] = [xmin_1000, ymin_1000, xmax_1000, ymax_1000]
        new_d["bbox_pixel"] = [round(gx1, 2), round(gy1, 2), round(gx2, 2), round(gy2, 2)]
        new_d["bbox_normalized"] = [xmin_1000, ymin_1000, xmax_1000, ymax_1000]
        remapped.append(new_d)
    return remapped


def apply_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.45,
    class_aware: bool = True,
    cross_class_iou_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Applies confidence-aware Non-Maximum Suppression (NMS) over candidate detections.
    If class_aware=True, NMS is performed per semantic label to prevent suppressing different classes.
    """
    if not detections:
        return []

    # Sort descending by score
    sorted_dets = sorted(detections, key=lambda d: d.get("score", 0.0), reverse=True)

    if not class_aware:
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

    # Class-aware NMS: group by normalized label
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for d in sorted_dets:
        lbl = str(d.get("label", "default")).strip().lower()
        by_class.setdefault(lbl, []).append(d)

    kept: List[Dict[str, Any]] = []
    for cls_name, cls_dets in by_class.items():
        cls_kept: List[Dict[str, Any]] = []
        for cand in cls_dets:
            cand_box = cand.get("bbox_pixel") or cand.get("box")
            if not cand_box:
                continue
            discard = False
            for k in cls_kept:
                k_box = k.get("bbox_pixel") or k.get("box")
                if compute_iou(cand_box, k_box) > iou_threshold:
                    discard = True
                    break
            if not discard:
                cls_kept.append(cand)
        kept.extend(cls_kept)

    # Optional cross-class near-identical duplicate filter if requested
    if cross_class_iou_threshold is not None:
        final_kept: List[Dict[str, Any]] = []
        sorted_final = sorted(kept, key=lambda d: d.get("score", 0.0), reverse=True)
        for cand in sorted_final:
            cand_box = cand.get("bbox_pixel") or cand.get("box")
            if not cand_box:
                continue
            discard = False
            for k in final_kept:
                k_box = k.get("bbox_pixel") or k.get("box")
                if compute_iou(cand_box, k_box) > cross_class_iou_threshold:
                    discard = True
                    break
            if not discard:
                final_kept.append(cand)
        final_kept.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return final_kept

    # Sort final kept list descending by confidence score
    kept.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return kept


# ---------------------------------------------------------------------------
# Target Geometry Classification
# ---------------------------------------------------------------------------
def classify_target_geometry(target_phrase: str) -> str:
    """
    Classifies query target into:
      - 'discrete': localized compact objects (buildings, houses, warehouses, ships, boats, vehicles, aircraft, containers, tanks, towers)
      - 'linear': connected linear infrastructure (roads, highways, bridges, railways, tracks, runways)
      - 'region': extended spatial regions (construction sites, water bodies, vegetation, farmland, forests, industrial zones, ports)
    """
    try:
        from region_segmenter import classify_target_geometry as rs_classify, TargetGeometry
        geom = rs_classify(target_phrase)
        if geom == TargetGeometry.LINEAR_FEATURE:
            return "linear"
        elif geom == TargetGeometry.REGION_FEATURE:
            return "region"
        return "discrete"
    except Exception:
        t = (target_phrase or "").lower().strip()
        import re
        if any(re.search(rf"\b{re.escape(k)}\b", t) for k in ["road", "highway", "street", "avenue", "freeway", "motorway", "interchange", "bridge", "viaduct", "overpass", "railway", "track", "rail", "runway", "corridor", "lane"]):
            return "linear"
        if any(re.search(rf"\b{re.escape(k)}\b", t) for k in ["construction", "water", "river", "ocean", "sea", "lake", "canal", "estuary", "coastline", "shoreline", "vegetation", "forest", "tree", "crop", "farm", "farmland", "field", "agriculture", "zone", "area", "port", "harbor", "wetland"]):
            return "region"
        return "discrete"


def filter_boxes_by_semantics(
    detections: List[Dict[str, Any]],
    target_phrase: str,
    img_w: int,
    img_h: int,
    max_area_ratio: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Applies geometry-aware geometric filtering based on query semantics.
    Discards degenerate collapsed boxes and scene-spanning giant boxes for localized discrete objects (cars, buildings, ships),
    while fully preserving elongated shapes, extreme aspect ratios, and broad networks for linear features (roads, highways,
    bridges, runways, railways) and continuous regions (rivers, coastlines, water bodies, forests).
    """
    total_area = float(max(1, img_w * img_h))
    geometry_type = classify_target_geometry(target_phrase)
    is_linear = (geometry_type == "linear")
    is_region = (geometry_type == "region")

    # Geometry-specific maximum area ratios:
    # Linear features (roads, highways, bridges, runways) and continuous regions (rivers, coastlines, water, forests)
    # can span up to 95% of the frame and exhibit extreme aspect ratios.
    # Discrete objects (airplanes, cars, ships, containers) cannot cover > 45% of an entire satellite scene.
    if is_linear or is_region:
        effective_max_area = 0.95 if max_area_ratio is None else max(max_area_ratio, 0.95)
    else:
        effective_max_area = 0.45 if max_area_ratio is None else min(max_area_ratio, 0.45)

    # Check if smaller candidate detections exist
    has_smaller_boxes = any(
        ((abs(float((d.get("bbox_pixel") or d.get("box") or [0, 0, 0, 0])[2] - (d.get("bbox_pixel") or d.get("box") or [0, 0, 0, 0])[0])) *
          abs(float((d.get("bbox_pixel") or d.get("box") or [0, 0, 0, 0])[3] - (d.get("bbox_pixel") or d.get("box") or [0, 0, 0, 0])[1]))) / total_area) <= 0.35
        for d in detections if (d.get("bbox_pixel") or d.get("box")) and len(d.get("bbox_pixel") or d.get("box") or []) == 4
    )

    filtered: List[Dict[str, Any]] = []
    for d in detections:
        box = d.get("bbox_pixel") or d.get("box")
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = box
        bw = max(0.0, float(x2) - float(x1))
        bh = max(0.0, float(y2) - float(y1))
        box_area = bw * bh

        # 1. Reject degenerate collapsed boxes (< 2px width or height, or < 4px^2 area)
        if bw < 2.0 or bh < 2.0 or box_area < 4.0:
            continue

        area_frac = box_area / total_area

        # 2. Filter out any bounding box whose area exceeds 35% of total image area when smaller detections exist
        if has_smaller_boxes and area_frac > 0.35:
            continue

        # 3. Reject discrete objects exceeding maximum realistic ground footprint
        if not (is_linear or is_region) and area_frac > effective_max_area:
            continue

        # 4. Reject giant low-confidence full-screen boundary artifacts
        if area_frac > 0.90 and bw >= 0.90 * img_w and bh >= 0.90 * img_h:
            continue

        filtered.append(d)

    return filtered


class TiledInferenceEngine:
    """
    High-Resolution Tiled & Multi-Scale Vision Model Inference Engine.
    Enables native-resolution object detection for large remote sensing rasters.
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
        Slices an image into overlapping tiles with guaranteed full-boundary coverage.
        Returns list of (tile_image, (x_offset, y_offset, tile_w, tile_h)).
        """
        w, h = image.size
        ts = tile_size or self.default_tile_size
        ov = overlap or self.default_overlap
        stride = max(32, int(round(ts * (1.0 - ov))))

        if w <= ts and h <= ts:
            return [(image, (0, 0, w, h))]

        # Generate X anchors ensuring right boundary is reached
        x_points: List[int] = []
        x = 0
        while x + ts < w:
            x_points.append(x)
            x += stride
        x_points.append(max(0, w - ts))
        x_points = sorted(list(set(x_points)))

        # Generate Y anchors ensuring bottom boundary is reached
        y_points: List[int] = []
        y = 0
        while y + ts < h:
            y_points.append(y)
            y += stride
        y_points.append(max(0, h - ts))
        y_points = sorted(list(set(y_points)))

        tiles = []
        for y_off in y_points:
            for x_off in x_points:
                tw = min(ts, w - x_off)
                th = min(ts, h - y_off)
                crop_box = (x_off, y_off, x_off + tw, y_off + th)
                tile_crop = image.crop(crop_box)
                tiles.append((tile_crop, (x_off, y_off, tw, th)))

        return tiles

    def run_tiled_grounding(
        self,
        image: Image.Image,
        target_phrase: str,
        grounding_runtime: Optional[Any] = None,
        tile_size: int = 512,
        overlap: float = 0.25,
        iou_threshold: float = 0.45,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        max_area_ratio: Optional[float] = None,
        enable_multi_scale: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes multi-scale hierarchical open-vocabulary detection across remote sensing imagery.
        Combines contextual full-frame inference with native overlapping high-resolution tiles.
        """
        t0 = time.perf_counter()
        w, h = image.size
        geo_meta = GeospatialEngine.extract_metadata(image)
        geometry_type = classify_target_geometry(target_phrase)
        geom_str = str(getattr(geometry_type, "value", geometry_type)).lower()
        tp_lower = (target_phrase or "").lower()

        is_linear = "linear" in geom_str or any(k in tp_lower for k in [
            "road", "roads", "highway", "highways", "street", "streets", "avenue", "avenues",
            "freeway", "freeways", "motorway", "motorways", "bridge", "bridges", "viaduct",
            "overpass", "railway", "railroad", "track", "rail", "runway", "runways", "pipeline",
            "interchange", "intersection", "lane", "pathway", "corridor"
        ])
        is_region = "region" in geom_str or any(k in tp_lower for k in [
            "water", "river", "rivers", "ocean", "sea", "lake", "lakes", "canal", "waterway",
            "estuary", "coastline", "shoreline", "vegetation", "forest", "forests", "tree", "trees",
            "crop", "crops", "farm", "farmland", "field", "fields", "agriculture", "zone", "area",
            "port", "harbor", "wetland", "construction"
        ])
        is_discrete = not (is_linear or is_region)

        # Unpack adapter runtime handle if passed as adapter instance
        actual_runtime = getattr(grounding_runtime, "runtime", grounding_runtime)
        source_id = "RS_Adapted_Vision_Grounder" if (actual_runtime and ("rs" in getattr(actual_runtime, "model_id", "").lower() or "rs" in getattr(actual_runtime, "name", "").lower())) else "Grounding_DINO"

        # 1. Build generalized aerial prompt for detector and clean canonical label for UI
        clean_prompt = build_generalized_aerial_prompt(target_phrase)
        canonical_label = get_canonical_target_label(target_phrase)

        raw_detections: List[Dict[str, Any]] = []
        tiles_processed = 0

        # 2. Pass A: Contextual Full-Frame Pass (Captures medium/macro structures and global relationships)
        if actual_runtime and actual_runtime.is_available():
            try:
                full_res = actual_runtime.infer(
                    image=image,
                    target_phrase=clean_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                tiles_processed += 1
                for d in full_res.get("detections", []):
                    b_norm = d.get("bbox_normalized") or d.get("box_2d") or d.get("box") or [0, 0, 1000, 1000]
                    b_pix = d.get("bbox_pixel") or [
                        round((b_norm[0] / 1000.0) * w, 2),
                        round((b_norm[1] / 1000.0) * h, 2),
                        round((b_norm[2] / 1000.0) * w, 2),
                        round((b_norm[3] / 1000.0) * h, 2),
                    ]
                    raw_lbl = str(d.get("label", "")).strip()
                    lbl = normalize_detection_label(raw_lbl, target_phrase) if " . " not in target_phrase else raw_lbl
                    score = round(float(d.get("score", 0.50)), 4)

                    ev_item = {
                        "type": "bounding_box",
                        "label": lbl,
                        "box": b_norm,
                        "box_2d": b_norm,
                        "bbox_pixel": b_pix,
                        "bbox_normalized": b_norm,
                        "score": score,
                        "source": source_id,
                        "image_dimensions": [w, h],
                        "geospatial_coordinates_available": False,
                    }
                    if geo_meta.geospatial_available:
                        ev_item = GeospatialEngine.enrich_evidence_item(ev_item, geo_meta)
                    raw_detections.append(ev_item)
            except Exception as ex:
                pass

        # 3. Pass B: High-Resolution Native Overlapping Tiles
        # STRICT RULE: NEVER Run Grounding DINO More Than ONCE Per Query on CPU!
        # Since images are already resized to max_dim=1024, DO NOT run multi-scale slicing or multiple forward passes.
        # Run a SINGLE full-image forward pass (outputs = model(**inputs)).
        should_tile = False
        
        if should_tile:
            # For 512x512 images, slice 2x2 grid (384x384 tiles with 25% overlap)
            effective_tile_size = 384 if (max(w, h) <= 768 and min(w, h) >= 384) else tile_size
            tiles = self.generate_tiles(image, tile_size=effective_tile_size, overlap=overlap)

            for tile_img, (off_x, off_y, tile_w, tile_h) in tiles:
                # If tile is full image and we already processed full image, skip
                if tile_w == w and tile_h == h:
                    continue

                if actual_runtime and actual_runtime.is_available():
                    try:
                        t_res = actual_runtime.infer(
                            image=tile_img,
                            target_phrase=clean_prompt,
                            box_threshold=box_threshold,
                            text_threshold=text_threshold,
                        )
                        tiles_processed += 1
                        t_dets = t_res.get("detections", [])

                        for d in t_dets:
                            box = d.get("bbox_pixel") or d.get("box", [0, 0, tile_w, tile_h])
                            lx1, ly1, lx2, ly2 = box

                            # Filter out pathological tile-spanning boxes within the local tile
                            tile_box_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
                            if is_discrete and tile_box_area > (0.65 * tile_w * tile_h):
                                continue

                            # Global coordinate remapping
                            gx1 = max(0.0, min(float(w), float(lx1 + off_x)))
                            gy1 = max(0.0, min(float(h), float(ly1 + off_y)))
                            gx2 = max(0.0, min(float(w), float(lx2 + off_x)))
                            gy2 = max(0.0, min(float(h), float(ly2 + off_y)))

                            # Canonical box check
                            if gx2 <= gx1 or gy2 <= gy1 or (gx2 - gx1) * (gy2 - gy1) < 4.0:
                                continue

                            # Normalized [0, 1000] canonical coordinates
                            xmin_1000 = int(round(max(0.0, min(1.0, gx1 / max(1, w))) * 1000))
                            ymin_1000 = int(round(max(0.0, min(1.0, gy1 / max(1, h))) * 1000))
                            xmax_1000 = int(round(max(0.0, min(1.0, gx2 / max(1, w))) * 1000))
                            ymax_1000 = int(round(max(0.0, min(1.0, gy2 / max(1, h))) * 1000))

                            raw_lbl = str(d.get("label", "")).strip().lower()
                            if canonical_label.lower() in raw_lbl or any(k in raw_lbl for k in tp_lower.split() if len(k) > 2):
                                lbl = canonical_label
                            else:
                                lbl = get_canonical_target_label(raw_lbl) if (raw_lbl and " . " not in raw_lbl) else canonical_label
                            score = round(float(d.get("score", 0.50)), 4)

                            ev_item = {
                                "type": "bounding_box",
                                "label": lbl,
                                "box": [xmin_1000, ymin_1000, xmax_1000, ymax_1000],
                                "box_2d": [xmin_1000, ymin_1000, xmax_1000, ymax_1000],
                                "bbox_pixel": [round(gx1, 2), round(gy1, 2), round(gx2, 2), round(gy2, 2)],
                                "bbox_normalized": [xmin_1000, ymin_1000, xmax_1000, ymax_1000],
                                "score": score,
                                "source": source_id,
                                "image_dimensions": [w, h],
                                "geospatial_coordinates_available": False,
                            }

                            if geo_meta.geospatial_available:
                                ev_item = GeospatialEngine.enrich_evidence_item(ev_item, geo_meta)

                            raw_detections.append(ev_item)
                    except Exception:
                        pass

        # 4. Geometry-Aware Semantic Box Filtering
        filtered_detections = filter_boxes_by_semantics(
            raw_detections,
            target_phrase=target_phrase,
            img_w=w,
            img_h=h,
            max_area_ratio=max_area_ratio,
        )

        # 5. Generalized Class-Aware NMS across tile boundaries
        merged_detections = apply_nms(
            filtered_detections,
            iou_threshold=iou_threshold,
            class_aware=True,
        )

        # 6. Promptable SAM 2 & High-Precision Pixel-Level Polygon Segmentation
        try:
            from sam2_engine import sam2_segmentation_engine
            merged_detections = sam2_segmentation_engine.enrich_detections(image, merged_detections)
        except Exception:
            pass

        dur = (time.perf_counter() - t0) * 1000.0

        return {
            "tiling_used": should_tile,
            "tile_size": tile_size,
            "overlap_ratio": overlap,
            "tiles_processed": tiles_processed,
            "target_phrase": target_phrase,
            "prompt_used": clean_prompt,
            "raw_detections_count": len(raw_detections),
            "detections": merged_detections,
            "count": len(merged_detections),
            "image_width": w,
            "image_height": h,
            "inference_time_ms": round(dur, 2),
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
    class_aware: bool = True,
) -> List[Dict[str, Any]]:
    """Convenience helper to merge candidate detections with class-aware NMS."""
    return apply_nms(detections, iou_threshold=iou_threshold, class_aware=class_aware)


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

