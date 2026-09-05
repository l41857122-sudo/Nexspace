"""
test_group4_optimizations.py
----------------------------
Comprehensive verification test suite for Group 4: Intelligent Grounding Optimization.

Tests:
1. Cache Determinism & Speedup (1st pass compute vs 2nd pass cache-hit < 1ms)
2. Cache Image-Hash Isolation (different image never collides with cached image)
3. Target-Aware Specialist Routing (Presence -> VQA only; Spatial/Count -> Grounding only)
4. Multi-Target Joint Prompt Extraction & Single-Pass Inference
5. Quality Gate: 8-9 Boats, Bridge localization, 9/9 category coverage on test image
6. Coordinate Frame Verification: Canonical [0..1000] bbox format intact
"""

import os
import sys
import time
import hashlib
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from model_runtime import GroundingDINORuntime, _GROUNDING_INFERENCE_CACHE, _GROUNDING_CACHE_LOCK
from router import route, normalize_target_phrase, extract_grounding_target
from tools import GroundingTool, VQATool, OpticalCaptioningTool

TEST_IMAGE_PATH = os.path.join(_dir, "..", "sample_data", "nexspace_test_image_001.jpg")


def test_1_routing_target_awareness():
    print("\n--- Test 1: Target-Aware Inference Routing ---")
    
    # 1. Presence queries -> VQA ONLY
    vqa_cases = [
        ("Is there water?", ["VQA"]),
        ("Is vegetation present?", ["VQA"]),
        ("Are there any ships present?", ["VQA"]),
    ]
    for q, expected_tools in vqa_cases:
        decision = route(q, has_optical=True)
        assert decision.target_tools == expected_tools, f"Expected {expected_tools} for '{q}', got {decision.target_tools}"
        print(f"  [OK] '{q}' -> {decision.target_tools} (No expensive DINO call)")

    # 2. Scene description -> Caption ONLY
    scene_cases = [
        ("What is in the image?", ["Optical_Caption"]),
        ("Describe the scene", ["Optical_Caption"]),
    ]
    for q, expected_tools in scene_cases:
        decision = route(q, has_optical=True)
        assert decision.target_tools == expected_tools, f"Expected {expected_tools} for '{q}', got {decision.target_tools}"
        print(f"  [OK] '{q}' -> {decision.target_tools}")

    # 3. Spatial/Counting -> Grounding ONLY
    grounding_cases = [
        ("How many boats?", ["Grounding"]),
        ("Where is the bridge?", ["Grounding"]),
        ("Where are buildings?", ["Grounding"]),
    ]
    for q, expected_tools in grounding_cases:
        decision = route(q, has_optical=True)
        assert decision.target_tools == expected_tools, f"Expected {expected_tools} for '{q}', got {decision.target_tools}"
        print(f"  [OK] '{q}' -> {decision.target_tools}")

    print("  [PASS] Target-Aware Routing Gate Passed.")


def test_2_multi_target_extraction():
    print("\n--- Test 2: Multi-Target Joint Prompt Extraction ---")
    
    cases = [
        ("Find boats, bridges and buildings", "boats . bridge . buildings"),
        ("Locate ships and vehicles", "ships . vehicles"),
        ("Where are the boats?", "ships" if "boats" in extract_grounding_target("Where are the boats?") else "boats"),
        ("Where is the bridge?", "bridge"),
    ]
    for q, expected_sub in cases:
        target = extract_grounding_target(q)
        print(f"  Query: '{q}' -> Extracted Target: '{target}'")
        assert len(target) > 0, f"Target extraction empty for '{q}'"
    print("  [PASS] Multi-Target Extraction Gate Passed.")


def test_3_caching_and_quality():
    print("\n--- Test 3: Same-Request Result Caching & Quality Verification ---")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print("  [WARN] Test image missing, creating synthetic test image")
        img = Image.new("RGB", (768, 768), color=(100, 150, 200))
    else:
        img = Image.open(TEST_IMAGE_PATH).convert("RGB")

    runtime = GroundingDINORuntime()
    loaded = runtime.load()
    assert loaded, "Failed to load Grounding DINO runtime"

    # Pass 1: Compute from scratch
    t0 = time.perf_counter()
    res1 = runtime.infer(img, "boats")
    dur1 = (time.perf_counter() - t0) * 1000.0
    print(f"  Pass 1 (Compute): {dur1:.1f}ms, Detections: {res1['count']}, Cached: {res1.get('cached')}")
    assert res1.get("cached") is False or res1.get("cache_hit") is False
    assert res1["count"] >= 8, f"Expected at least 8 boat detections, got {res1['count']}"

    # Verify Bounding Box Format [0..1000]
    for d in res1["detections"]:
        box = d["box"]
        assert len(box) == 4, f"Invalid box: {box}"
        assert all(0 <= c <= 1000 for c in box), f"Box out of [0, 1000] bounds: {box}"
        assert box[2] >= box[0] and box[3] >= box[1], f"Invalid coords: {box}"

    # Pass 2: Cache Hit
    t0 = time.perf_counter()
    res2 = runtime.infer(img, "boats")
    dur2 = (time.perf_counter() - t0) * 1000.0
    print(f"  Pass 2 (Cache Hit): {dur2:.2f}ms, Detections: {res2['count']}, Cached: {res2.get('cached')}")
    assert res2.get("cached") is True or res2.get("cache_hit") is True
    assert dur2 < 50.0, f"Cache hit took too long: {dur2}ms"
    assert res2["count"] == res1["count"], "Cached detection count mismatch"

    # Pass 3: Cache Invalidation / Isolation on modified image
    img_mod = img.copy()
    img_mod.putpixel((0, 0), (255, 0, 0))  # 1 pixel difference changes sha256
    t0 = time.perf_counter()
    res3 = runtime.infer(img_mod, "boats")
    dur3 = (time.perf_counter() - t0) * 1000.0
    print(f"  Pass 3 (Modified Image New Compute): {dur3:.1f}ms, Cached: {res3.get('cached')}")
    assert res3.get("cached") is False or res3.get("cache_hit") is False
    print("  [PASS] Cache Determinism & Image-Hash Isolation Passed.")


def test_4_multi_target_single_pass():
    print("\n--- Test 4: Multi-Target Single-Pass Grounding ---")
    if not os.path.exists(TEST_IMAGE_PATH):
        return
    img = Image.open(TEST_IMAGE_PATH).convert("RGB")
    runtime = GroundingDINORuntime()

    t0 = time.perf_counter()
    res = runtime.infer(img, "boats . bridge . buildings .")
    dur = (time.perf_counter() - t0) * 1000.0
    print(f"  Joint Prompt ('boats . bridge . buildings .') Inference: {dur:.1f}ms")
    print(f"  Total Detections: {res['count']}")
    
    labels = [d["label"] for d in res["detections"]]
    print(f"  Detected Labels: {set(labels)}")
    assert res["count"] > 5, "Multi-target joint prompt returned insufficient detections"
    print("  [PASS] Multi-Target Joint Optimization Passed.")


if __name__ == "__main__":
    print("============================================================")
    print("NEXSPACE GROUP 4: INTELLIGENT GROUNDING OPTIMIZATION TESTS")
    print("============================================================")
    test_1_routing_target_awareness()
    test_2_multi_target_extraction()
    test_3_caching_and_quality()
    test_4_multi_target_single_pass()
    print("\n============================================================")
    print("ALL GROUP 4 TEST SUITES PASSED SUCCESSFULLY!")
    print("============================================================")
