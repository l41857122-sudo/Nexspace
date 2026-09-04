"""
audit_grounding_quality.py
--------------------------
Evaluates Grounding DINO zero-shot detection behavior and diagnostics
on real satellite/aerial imagery.
"""

import os
import sys
import time
from PIL import Image
import numpy as np

# Ensure ml_backend is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_runtime import GroundingDINORuntime


def run_grounding_audit():
    print("=== GROUNDING DINO QUALITY & DIAGNOSTIC AUDIT ===")

    runtime = GroundingDINORuntime()
    print("Loading Grounding DINO weights...")
    t_load_0 = time.perf_counter()
    loaded = runtime.load()
    t_load = (time.perf_counter() - t_load_0) * 1000.0
    print(f"Model loaded: {loaded} in {t_load:.1f} ms on device '{runtime.device}'.")

    if not loaded:
        print(f"Grounding DINO load failed: {runtime.load_error}")
        return

    # Use real satellite image
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_satellite.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data", "BiTemporal_After_20231025.png")

    print(f"Testing on image: {img_path}")
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    total_img_area = float(w * h)
    print(f"Image dimensions: {w}x{h} (Total pixels: {int(total_img_area)})")

    prompts = [
        "buildings",
        "roads",
        "vehicles",
        "water",
        "trees",
        "rooftops",
        "ships",
    ]

    all_diagnostics = []

    for p in prompts:
        t0 = time.perf_counter()
        res = runtime.infer(image=image, target_phrase=p, box_threshold=0.25, text_threshold=0.25)
        dur = (time.perf_counter() - t0) * 1000.0

        detections = res.get("detections", [])
        num_det = len(detections)
        scores = [d["score"] for d in detections]
        
        box_areas = []
        near_full_image_boxes = 0

        for d in detections:
            x1, y1, x2, y2 = d["box"]
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            b_area = box_w * box_h
            area_ratio = b_area / total_img_area
            box_areas.append(area_ratio)
            if area_ratio > 0.90:
                near_full_image_boxes += 1

        mean_score = float(np.mean(scores)) if scores else 0.0
        median_score = float(np.median(scores)) if scores else 0.0
        mean_area_ratio = float(np.mean(box_areas)) if box_areas else 0.0
        near_full_pct = (near_full_image_boxes / num_det * 100.0) if num_det > 0 else 0.0

        diag = {
            "prompt": p,
            "num_detections": num_det,
            "mean_score": round(mean_score, 4),
            "median_score": round(median_score, 4),
            "mean_area_ratio": round(mean_area_ratio, 4),
            "near_full_image_count": near_full_image_boxes,
            "near_full_image_pct": round(near_full_pct, 1),
            "inference_ms": round(dur, 1),
        }
        all_diagnostics.append(diag)

        print(f"\nPrompt: '{p}'")
        print(f"  - Detections: {num_det}")
        print(f"  - Scores: Mean={mean_score:.4f}, Median={median_score:.4f}")
        print(f"  - Mean Box/Image Area Ratio: {mean_area_ratio:.4f}")
        print(f"  - Near-Full-Image Detections (>90% area): {near_full_image_boxes} ({near_full_pct:.1f}%)")
        print(f"  - Inference Latency: {dur:.1f} ms")

    return all_diagnostics


if __name__ == "__main__":
    run_grounding_audit()
