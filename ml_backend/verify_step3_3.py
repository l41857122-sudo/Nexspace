"""
verify_step3_3.py
-----------------
Verification suite for STEP 3.3:
1. Pure captioning routing vs. VQA routing vs. Multi-task routing
2. Real BLIP inference on aerial/satellite sample image (512x512)
3. End-to-end FastAPI API query execution for "Describe this image" and "Describe this image and locate the buildings"
"""

import sys
import os
import time
import json
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ml_backend is on path
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from router import IntentClassifier, TaskType
from model_runtime import BLIPCaptioningRuntime
from tools import optical_caption_tool, grounding_tool
from server import app, encode_image_b64
from fastapi.testclient import TestClient


def test_routing():
    print("======================================================================")
    print("STEP 3.3 [1]: INTENT CLASSIFICATION & ROUTING TEST")
    print("======================================================================")
    classifier = IntentClassifier()

    cases = [
        ("Describe this image", ["Optical_Caption"], TaskType.CAPTIONING),
        ("What does this image look like?", ["Optical_Caption"], TaskType.CAPTIONING),
        ("Give me a description of this image", ["Optical_Caption"], TaskType.CAPTIONING),
        ("Describe the scene", ["Optical_Caption"], TaskType.CAPTIONING),
        ("What is visible in this image?", ["Optical_Caption"], TaskType.CAPTIONING),
        ("Is there water in this image?", ["VQA"], TaskType.VQA),
        ("Are there buildings?", ["VQA"], TaskType.VQA),
        ("How many buildings are visible?", ["VQA"], TaskType.VQA),
        ("Describe this image and locate the buildings", ["Optical_Caption", "Grounding"], TaskType.MULTI_TASK),
    ]

    all_passed = True
    for query, expected_tools, expected_task in cases:
        res = classifier.classify(query=query, has_optical=True)
        match_tools = res.target_tools == expected_tools
        match_task = res.task_type == expected_task
        passed = match_tools and match_task
        if not passed:
            all_passed = False
        print(f"Query: '{query}'")
        print(f"  -> task_type: {res.task_type.value} (Expected: {expected_task.value}) | Match: {match_task}")
        print(f"  -> selected_tools: {res.target_tools} (Expected: {expected_tools}) | Match: {match_tools}")
        print(f"  -> VQA queries: {res.restructured_vqa_queries}")
        print(f"  -> Status: {'PASS' if passed else 'FAIL'}\n")

    return all_passed


def test_real_blip_satellite_image():
    print("======================================================================")
    print("STEP 3.3 [2]: REAL BLIP INFERENCE ON SATELLITE / AERIAL IMAGE")
    print("======================================================================")
    img_path = os.path.join(_dir, "sample_satellite.png")
    if not os.path.exists(img_path):
        from create_aerial_sample import generate_satellite_sample
        generate_satellite_sample(img_path)

    img = Image.open(img_path)
    runtime = BLIPCaptioningRuntime()

    t0 = time.perf_counter()
    ok = runtime.load()
    t_load = (time.perf_counter() - t0) * 1000.0

    print(f"Model ID: {runtime.model_id}")
    print(f"Device: {runtime.device}")
    print(f"Image dimensions: {img.size}")
    print(f"Load duration: {t_load:.2f}ms")

    t1 = time.perf_counter()
    inf_res = runtime.infer(image=img, modality="optical")
    t_inf = (time.perf_counter() - t1) * 1000.0

    print(f"Inference duration: {t_inf:.2f}ms")
    print(f"Generated Caption: \"{inf_res['caption']}\"")
    print(f"Model Capability: {inf_res['model_capability']}")
    print(f"Confidence Type: {inf_res['confidence_type']}")
    print(f"Confidence Source: {inf_res['confidence_source']}")

    return {
        "real_inference": ok,
        "model": runtime.model_id,
        "device": runtime.device,
        "dimensions": img.size,
        "load_time_ms": round(t_load, 2),
        "inference_time_ms": round(t_inf, 2),
        "caption": inf_res["caption"],
        "fallback": not ok,
    }


def test_api_pipeline():
    print("======================================================================")
    print("STEP 3.3 [3]: COMPLETE API PIPELINE TESTS VIA FASTAPI TESTCLIENT")
    print("======================================================================")
    client = TestClient(app)
    img_path = os.path.join(_dir, "sample_satellite.png")
    img = Image.open(img_path)
    b64_img = encode_image_b64(img)

    # API Health
    h_res = client.get("/api/health")
    print(f"GET /api/health -> HTTP {h_res.status_code}")

    # Query 1: "Describe this image"
    print("\n--- Query 1: 'Describe this image' ---")
    q1_res = client.post("/api/query", json={"query": "Describe this image", "optical_image": b64_img})
    q1 = q1_res.json()
    print(f"HTTP Status: {q1_res.status_code}")
    print(f"task_type: {q1.get('task_type')}")
    print(f"selected_tools: {q1.get('selected_tools')}")
    print(f"tool_statuses: {[(r.get('tool_name'), r.get('status')) for r in q1.get('results', [])]}")
    print(f"confidence: {q1.get('confidence')} ({q1.get('confidence_type')}) [Source: {q1.get('confidence_source')}]")
    print(f"execution_trace: {[e.get('stage') for e in q1.get('execution_trace', [])]}")
    print(f"response_text:\n{q1.get('response_text')}")

    # Query 2: "Describe this image and locate the buildings"
    print("\n--- Query 2: 'Describe this image and locate the buildings' ---")
    q2_res = client.post("/api/query", json={"query": "Describe this image and locate the buildings", "optical_image": b64_img})
    q2 = q2_res.json()
    print(f"HTTP Status: {q2_res.status_code}")
    print(f"task_type: {q2.get('task_type')}")
    print(f"selected_tools: {q2.get('selected_tools')}")
    print(f"tool_statuses: {[(r.get('tool_name'), r.get('status')) for r in q2.get('results', [])]}")
    print(f"confidence: {q2.get('confidence')} ({q2.get('confidence_type')}) [Source: {q2.get('confidence_source')}]")
    print(f"execution_trace: {[e.get('stage') for e in q2.get('execution_trace', [])]}")
    print(f"response_text:\n{q2.get('response_text')}")

    return q1, q2


if __name__ == "__main__":
    test_routing()
    blip_info = test_real_blip_satellite_image()
    test_api_pipeline()
