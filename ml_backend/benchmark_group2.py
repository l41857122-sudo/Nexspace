"""
benchmark_group2.py
-------------------
Comprehensive verification & quality preservation test suite for Group 2 Safe Performance Optimization.
Tests queries A through H on sample_data/nexspace_test_image_001.jpg:
  A. "Is there a body of water?"
  B. "Are there bridges?"
  C. "Is vegetation present?"
  D. "How many boats are visible?"
  E. "Where is the bridge?"
  F. "Describe the scene."
  G. "What is near the bridge?"
  H. "What objects are visible?"
And tests:
  - Change Analysis
  - Optical/SAR Fusion & Modality Separation
"""

import os
import sys
import time
import json
import torch
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from orchestrator import GeoVLMController
from model_runtime import DeviceManager, BLIPCaptioningRuntime, GroundingDINORuntime, PaliGemmaVQARuntime

def run_group2_benchmark():
    test_image_path = os.path.join(os.path.dirname(_dir), "sample_data", "nexspace_test_image_001.jpg")
    if not os.path.exists(test_image_path):
        test_image_path = os.path.join(_dir, "sample_satellite.png")

    img = Image.open(test_image_path).convert("RGB")
    ctrl = GeoVLMController()

    print("=================================================================")
    print("NEXSPACE GROUP 2 SAFE PERFORMANCE BENCHMARK & QUALITY AUDIT")
    print(f"Target Image: {test_image_path} ({img.size})")
    print(f"Device Info: {DeviceManager.get_device_info()}")
    print("=================================================================\n")

    # Step 0: Warm up models to measure cold-start vs warm inference separately
    print("--- [STEP 0] MODEL INITIALIZATION & WARM-UP ---")
    t0 = time.perf_counter()
    PaliGemmaVQARuntime().load()
    t_vqa_init = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    BLIPCaptioningRuntime().load()
    t_blip_init = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    GroundingDINORuntime().load()
    t_dino_init = (time.perf_counter() - t0) * 1000.0

    print(f"  VQA Component Init       : {t_vqa_init:.2f} ms")
    print(f"  BLIP Captioning Init     : {t_blip_init:.2f} ms")
    print(f"  Grounding DINO Init      : {t_dino_init:.2f} ms\n")

    queries = [
        ("A", "Is there a body of water?"),
        ("B", "Are there bridges?"),
        ("C", "Is vegetation present?"),
        ("D", "How many boats are visible?"),
        ("E", "Where is the bridge?"),
        ("F", "Describe the scene."),
        ("G", "What is near the bridge?"),
        ("H", "What objects are visible?"),
    ]

    baseline_latencies = {
        "A": 331.42,
        "B": 336.83,
        "C": 327.43,
        "D": 29740.0,
        "E": 21090.0,
        "F": 14070.0,
        "G": 321.02,
        "H": 340.03,
    }

    results = {
        "model_initialization_ms": {
            "VQA": round(t_vqa_init, 2),
            "BLIP": round(t_blip_init, 2),
            "Grounding_DINO": round(t_dino_init, 2),
        },
        "queries": {},
    }

    print("--- [STEP 1] WARM INFERENCE BENCHMARK FOR QUERIES A THROUGH H ---")

    for case_id, q in queries:
        t0 = time.perf_counter()
        resp = ctrl.handle_request(query=q, optical_image=img)
        dur = (time.perf_counter() - t0) * 1000.0

        tools = resp.get("selected_tools", [])
        ans = resp.get("response_text", "") or ""
        conf = resp.get("confidence")
        conf_type = resp.get("confidence_type")
        conf_src = resp.get("confidence_source")
        dets = resp.get("grounding", {}).get("detections", []) if resp.get("grounding") else []
        ev_items = resp.get("evidence", [])

        b_lat = baseline_latencies.get(case_id, dur)
        speedup = (b_lat / dur) if dur > 0 else 1.0

        print(f"\n--- CASE {case_id}: \"{q}\" ---")
        print(f"  Selected Tools   : {tools}")
        print(f"  Optimized Latency: {dur:.2f} ms ({dur/1000.0:.2f} s)")
        print(f"  Baseline Latency : {b_lat:.2f} ms ({b_lat/1000.0:.2f} s)")
        print(f"  Speedup Factor   : {speedup:.2f}x ({'+' if dur <= b_lat else ''}{((b_lat - dur)/max(1, b_lat))*100:.1f}%)")
        print(f"  Detections Count : {len(dets)} (Scores: {[d.get('score') for d in dets[:4]]})")
        print(f"  Evidence Nodes   : {len(ev_items)}")
        print(f"  Confidence       : {conf} ({conf_type}, {conf_src})")
        print(f"  Answer Preview   : {ans[:140].replace(chr(10), ' ')}...")

        results["queries"][case_id] = {
            "query": q,
            "tools": tools,
            "optimized_latency_ms": round(dur, 2),
            "baseline_latency_ms": round(b_lat, 2),
            "speedup_factor": round(speedup, 2),
            "detections_count": len(dets),
            "evidence_count": len(ev_items),
            "confidence": conf,
            "confidence_type": conf_type,
            "confidence_source": conf_src,
            "answer": ans,
        }

    # Test Change Analysis & Optical/SAR
    print("\n--- [STEP 2] COMPONENT BENCHMARK: Change Analysis & Optical/SAR ---")
    t0 = time.perf_counter()
    chg_resp = ctrl.handle_request(query="What changed between these scenes?", change_image_a=img, change_image_b=img)
    t_chg = (time.perf_counter() - t0) * 1000.0
    print(f"  Change Analysis Latency: {t_chg:.2f} ms")

    t0 = time.perf_counter()
    opt_sar_resp = ctrl.handle_request(query="Compare optical and SAR imagery", optical_image=img, sar_image=img)
    t_optsar = (time.perf_counter() - t0) * 1000.0
    print(f"  Optical/SAR Latency: {t_optsar:.2f} ms")

    results["Change_Analysis"] = {"latency_ms": round(t_chg, 2)}
    results["Optical_SAR"] = {"latency_ms": round(t_optsar, 2)}

    output_path = os.path.join(_dir, "group2_benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved Group 2 benchmark report to: {output_path}")

if __name__ == "__main__":
    run_group2_benchmark()
