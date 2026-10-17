import os
import sys
import time
import json
import unittest
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from grounding_adapters import GroundingDINOAdapter, RSAdaptedVisionGroundingAdapter
from tools import GroundingTool, tool_registry
from orchestrator import GeoVLMController
from router import IntentClassifier, TaskType

print("==================================================================", flush=True)
print("NEXSPACE PRODUCTION COMPATIBILITY & REGRESSION SUITE", flush=True)
print("==================================================================", flush=True)

img_path = os.path.join(os.path.dirname(_dir), "sample_data", "nexspace_test_image_001.jpg")
if os.path.exists(img_path):
    test_img = Image.open(img_path)
else:
    test_img = Image.new("RGB", (512, 512), color=(50, 100, 150))

w, h = test_img.size
print(f"Test Image: {w}x{h}, Mode={test_img.mode}\n", flush=True)

controller = GeoVLMController()

# ==================================================================
# PHASE 7: 10 REPRESENTATIVE QUERIES
# ==================================================================
phase7_queries = [
    ("Locate buildings", "RS specialist: buildings"),
    ("Find warehouses", "RS specialist: warehouses"),
    ("Locate ships", "RS specialist: ships"),
    ("Find boats", "RS specialist: boats"),
    ("Locate bridges", "RS specialist: bridges"),
    ("Find airplanes", "RS specialist: airplanes"),
    ("Find vehicles", "RS specialist: vehicles"),
    ("Find roads", "RS specialist: roads"),
    ("Find telecommunication antenna tower", "Novel / arbitrary target"),
    ("Find boats and buildings", "Multi-target query"),
]

print("=" * 70, flush=True)
print("PHASE 7 — LIVE END-TO-END QUERIES EVALUATION", flush=True)
print("=" * 70, flush=True)

phase7_results = []
for q, desc in phase7_queries:
    t0 = time.perf_counter()
    res = controller.handle_request(query=q, optical_image=test_img)
    dur = (time.perf_counter() - t0) * 1000.0

    status = res.get("status")
    task_type = res.get("task_type")
    selected_tools = res.get("selected_tools", [])
    ev_count = len(res.get("evidence", []))
    summary = res.get("investigation_report", {}).get("summary", "")[:120]

    # Find the grounding tool result
    grounding_res = next((r for r in res.get("results", []) if r.get("tool_name") == "Grounding"), None)
    selected_engine = grounding_res.get("confidence_source") if grounding_res else "N/A"
    dets_count = len(grounding_res.get("data", {}).get("detections", [])) if grounding_res else 0

    record = {
        "query": q,
        "description": desc,
        "status": status,
        "task_type": task_type,
        "selected_tools": selected_tools,
        "selected_engine": selected_engine,
        "detections_count": dets_count,
        "evidence_count": ev_count,
        "duration_ms": round(dur, 2),
    }
    phase7_results.append(record)

    print(f"Query: \"{q}\" ({desc})", flush=True)
    print(f"  -> Status: {status} | Task: {task_type} | Engine: {selected_engine}", flush=True)
    print(f"  -> Detections: {dets_count} | Evidence: {ev_count} | Latency: {dur:.1f}ms", flush=True)
    if ev_count > 0:
        sample_ev = res["evidence"][0]
        print(f"  -> Sample Box: {sample_ev.get('box')} | Score: {sample_ev.get('score')} | Src: {sample_ev.get('source')}", flush=True)
    print("-" * 70, flush=True)

# ==================================================================
# PHASE 8: FALLBACK TEST (Force RS Adapter Unavailable)
# ==================================================================
print("\n" + "=" * 70, flush=True)
print("PHASE 8 — FALLBACK TEST (RS Adapter Unavailable)", flush=True)
print("=" * 70, flush=True)

class MockUnavailableRSAdapter(RSAdaptedVisionGroundingAdapter):
    def is_available(self) -> bool:
        return False

# Test with RS adapter disabled
fallback_tool = GroundingTool()
fallback_tool.rs_adapter = MockUnavailableRSAdapter()
fallback_controller = GeoVLMController(grounding_tool=fallback_tool)

t0 = time.perf_counter()
fb_res = fallback_controller.handle_request(query="Where are the buildings?", optical_image=test_img)
fb_dur = (time.perf_counter() - t0) * 1000.0

fb_g_res = next((r for r in fb_res.get("results", []) if r.get("tool_name") == "Grounding"), None)
fb_engine = fb_g_res.get("confidence_source") if fb_g_res else "N/A"

print(f"Fallback Query: \"Where are the buildings?\" (RS adapter forced unavailable)", flush=True)
print(f"  -> Status: {fb_res.get('status')} | Engine: {fb_engine}", flush=True)
print(f"  -> Fallback handled gracefully without crash: {fb_res.get('status') == 'completed'}", flush=True)
print(f"  -> Latency: {fb_dur:.1f}ms", flush=True)

# ==================================================================
# PHASE 9: CORE UNIT TEST SUITES
# ==================================================================
print("\n" + "=" * 70, flush=True)
print("PHASE 9 — RUNNING CORE UNIT TESTS", flush=True)
print("=" * 70, flush=True)

loader = unittest.TestLoader()
suite = unittest.TestSuite()

from test_step4_grounding import TestStep4GroundingPipeline
from test_controller import TestController
from test_matrix_image001 import TestMatrixImage001

suite.addTests(loader.loadTestsFromTestCase(TestStep4GroundingPipeline))
suite.addTests(loader.loadTestsFromTestCase(TestController))
suite.addTests(loader.loadTestsFromTestCase(TestMatrixImage001))

runner = unittest.TextTestRunner(verbosity=1)
test_res = runner.run(suite)

print("\n" + "=" * 70, flush=True)
print("COMPREHENSIVE AUDIT & VALIDATION SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"Phase 7 Live Queries Passed: {all(r['status'] == 'completed' for r in phase7_results)}/10")
print(f"Phase 8 Fallback Successful: {fb_res.get('status') == 'completed'}")
print(f"Phase 9 Unit Tests Passed: {test_res.wasSuccessful()} (Ran {test_res.testsRun} tests, Errors: {len(test_res.errors)}, Failures: {len(test_res.failures)})")
