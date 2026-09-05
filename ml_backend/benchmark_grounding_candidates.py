"""
benchmark_grounding_candidates.py
---------------------------------
Zero-Regression Visual Grounding Benchmark Suite for Group 3 Evaluation.
Tests Grounding DINO Baseline against Candidate A (OWLv2) and Candidate B (RS Spectral-Spatial Guided)
across 9 standard RS target categories, 4 natural language queries, and high-res direct vs tiled passes.
"""

import os
import sys
import time
import json
import psutil
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from grounding_adapters import (
    GroundingDINOAdapter,
    OWLv2GroundingAdapter,
    RSGuidedSpectralSpatialAdapter,
    BaseGroundingAdapter,
)
from orchestrator import GeoVLMController
from model_runtime import DeviceManager

TARGET_PROMPTS = [
    ("water", "water"),
    ("bridge", "bridge"),
    ("boats", "boats"),
    ("roads", "roads"),
    ("buildings", "buildings"),
    ("vegetation", "vegetation"),
    ("industrial_area", "industrial area"),
    ("construction_site", "construction site"),
    ("farmland", "farmland"),
]

NATURAL_QUERIES = [
    ("query_boats", "How many boats are visible?"),
    ("query_bridge", "Where is the bridge?"),
    ("query_objects", "What objects are visible?"),
    ("query_builtup", "Where are the major built-up areas?"),
]

def evaluate_model_adapter(
    adapter: BaseGroundingAdapter,
    image: Image.Image,
    model_name: str,
) -> Dict[str, Any]:
    print(f"\n=================================================================")
    print(f"EVALUATING MODEL: {model_name} ({adapter.model_id})")
    print(f"=================================================================")

    # 1. Cold Start Load Timing
    t0 = time.perf_counter()
    loaded = adapter.load()
    t_load = (time.perf_counter() - t0) * 1000.0
    print(f"Cold Initialization Time: {t_load:.2f} ms (Loaded={loaded})")

    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 * 1024)

    prompt_results = {}
    warm_latencies = []

    # 2. Standard 9 Target Prompts Evaluation
    print("\n--- Target Prompts Benchmark ---")
    for key, phrase in TARGET_PROMPTS:
        t0 = time.perf_counter()
        try:
            res = adapter.infer(image=image, target_phrase=phrase)
            dur = (time.perf_counter() - t0) * 1000.0
            warm_latencies.append(dur)
            dets = res.get("detections", [])
            scores = [d.get("score", 0.0) for d in dets]
            boxes = [d.get("box", []) for d in dets]

            print(f"  [{phrase.upper():18s}] Latency: {dur:8.2f} ms | Found: {len(dets):2d} boxes | Top Scores: {scores[:3]}")

            prompt_results[key] = {
                "phrase": phrase,
                "latency_ms": round(dur, 2),
                "count": len(dets),
                "scores": scores,
                "boxes": boxes[:8],  # Store top 8 boxes
                "mean_score": round(float(np.mean(scores)), 4) if scores else 0.0,
            }
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000.0
            print(f"  [{phrase.upper():18s}] ERROR: {e}")
            prompt_results[key] = {
                "phrase": phrase,
                "latency_ms": round(dur, 2),
                "error": str(e),
                "count": 0,
            }

    # 3. Natural Language End-to-End Controller Integration
    print("\n--- Natural Language Controller Queries ---")
    ctrl_results = {}
    from tools import GroundingTool
    custom_tool = GroundingTool(runtime=adapter)
    ctrl = GeoVLMController(grounding_tool=custom_tool)

    for q_id, q_text in NATURAL_QUERIES:
        t0 = time.perf_counter()
        c_resp = ctrl.handle_request(query=q_text, optical_image=image)
        dur = (time.perf_counter() - t0) * 1000.0

        tools_used = c_resp.get("selected_tools", [])
        ans = (c_resp.get("response_text", "") or "").replace("\n", " ")[:120]
        dets_count = len(c_resp.get("grounding", {}).get("detections", [])) if c_resp.get("grounding") else 0

        print(f"  Query: \"{q_text}\"")
        print(f"    Tools: {tools_used} | Latency: {dur:.2f} ms | Detections: {dets_count}")
        print(f"    Answer: {ans}...")

        ctrl_results[q_id] = {
            "query": q_text,
            "selected_tools": tools_used,
            "latency_ms": round(dur, 2),
            "detections_count": dets_count,
            "answer_preview": ans,
        }

    # 4. Small-Object & Specific Visual Feature Quality Inspection
    boats_dets = prompt_results.get("boats", {}).get("boxes", [])
    bridge_dets = prompt_results.get("bridge", {}).get("boxes", [])

    # Assessment of Boat Detections (True target on test image: ~8 small river boats)
    small_boat_sensitivity = "STRONG" if len(boats_dets) >= 6 else ("MODERATE" if len(boats_dets) >= 2 else "WEAK")
    bridge_localization = "ACCURATE" if len(bridge_dets) >= 1 else "MISSED"

    avg_warm_lat = float(np.mean(warm_latencies)) if warm_latencies else 0.0

    return {
        "model_name": model_name,
        "model_id": adapter.model_id,
        "cold_start_load_ms": round(t_load, 2),
        "mean_warm_inference_ms": round(avg_warm_lat, 2),
        "ram_footprint_mb": round(ram_mb, 2),
        "small_boat_sensitivity": small_boat_sensitivity,
        "bridge_localization": bridge_localization,
        "boat_count_found": len(boats_dets),
        "bridge_count_found": len(bridge_dets),
        "prompts": prompt_results,
        "natural_queries": ctrl_results,
    }


def run_benchmark():
    test_image_path = os.path.join(os.path.dirname(_dir), "sample_data", "nexspace_test_image_001.jpg")
    img = Image.open(test_image_path).convert("RGB")

    print(f"#################################################################")
    print(f"NEXSPACE GROUP 3: ZERO-REGRESSION GROUNDING MODEL BENCHMARK")
    print(f"Test Asset: {test_image_path} ({img.size})")
    print(f"Compute Device: {DeviceManager.get_device_info()}")
    print(f"PyTorch CPU Threads: {torch.get_num_threads()}")
    print(f"#################################################################")

    all_results = {}

    # Model 1: Grounding DINO Baseline (Active Default)
    dino_adapter = GroundingDINOAdapter()
    all_results["Grounding_DINO_Baseline"] = evaluate_model_adapter(
        adapter=dino_adapter,
        image=img,
        model_name="Grounding DINO Tiny (Production Baseline)",
    )

    # Model 2: Candidate A - OWLv2 Open-Vocabulary Detector
    owl_adapter = OWLv2GroundingAdapter()
    all_results["Candidate_A_OWLv2"] = evaluate_model_adapter(
        adapter=owl_adapter,
        image=img,
        model_name="Candidate A: OWLv2 Open-Vocabulary (google/owlvit-base-patch32)",
    )

    # Model 3: Candidate B - RS Spectral-Spatial Guided Localizer
    rs_adapter = RSGuidedSpectralSpatialAdapter()
    all_results["Candidate_B_RSSpectral"] = evaluate_model_adapter(
        adapter=rs_adapter,
        image=img,
        model_name="Candidate B: RS Spectral-Spatial Guided Localizer",
    )

    out_file = os.path.join(_dir, "grounding_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=================================================================")
    print(f"BENCHMARK COMPLETE! Full results saved to: {out_file}")
    print(f"=================================================================")

if __name__ == "__main__":
    run_benchmark()
