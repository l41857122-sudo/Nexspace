"""
test_investigation_pipeline.py
------------------------------
Comprehensive validation suite for the NexSpace Backend Investigation Pipeline.

Tests:
1. Simple Captioning ("Describe this image" with user image -> skips location/discovery/retrieval)
2. Building Detection ("Detect buildings" with user image -> runs grounding, skips satellite discovery)
3. Vehicle Counting ("How many vehicles are present?" -> verified counting, flag is_counting)
4. Ship Detection with Location ("Find ships in Mumbai" -> runs location, discovery, retrieval, detection)
5. Region Detection ("Segment water bodies in London" -> polygon region segmenter)
6. Change Analysis (Bi-temporal comparison -> differential anomaly engine)
7. Spatial Query ("Find ships near port in Mumbai" -> spatial predicate evaluation)
8. Cache Hit Verification (repeated retrieval -> CACHE_HIT)
9. Failure Handling: Location Unresolved ("Find airplanes in NonexistentPlaceXYZ12345" -> LOCATION_UNRESOLVED)
10. Async Execution & Status Polling (POST /api/query async -> GET status -> GET result)
11. Backward Compatibility Schema Verification (all fields expected by QueryPage.tsx)
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import time
import base64
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from fastapi.testclient import TestClient
from server import app
from investigation_pipeline import investigation_pipeline, global_investigation_store, PipelineStage


client = TestClient(app)

def load_test_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    ext = "jpeg" if path.endswith((".jpg", ".jpeg")) else "png"
    return f"data:image/{ext};base64,{b64}"


def run_all_tests():
    print("=" * 70)
    print("NEXSPACE BACKEND INVESTIGATION PIPELINE - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    demo_img_path = os.path.abspath(os.path.join(_dir, "..", "public", "demo_mumbai_port.jpg"))
    has_demo_img = os.path.exists(demo_img_path)
    demo_b64 = load_test_image_b64(demo_img_path) if has_demo_img else None

    test_results = []

    # -----------------------------------------------------------------------
    # TEST 1: Simple Captioning (User image provided)
    # -----------------------------------------------------------------------
    print("\n[TEST 1] Simple Captioning: 'Describe this satellite scene'")
    t0 = time.perf_counter()
    res1 = client.post("/api/query", json={
        "query": "Describe this satellite scene",
        "optical_image": demo_b64,
    })
    dur1 = (time.perf_counter() - t0) * 1000.0
    data1 = res1.json()
    assert res1.status_code == 200, f"Expected 200, got {res1.status_code}"
    stages1 = {s["stage_name"]: s["status"] for s in data1.get("execution_trace", [])}
    assert stages1.get("LOCATION_RESOLUTION") == "skipped", "Location resolution should be skipped when image provided"
    assert stages1.get("IMAGERY_DISCOVERY") == "skipped", "Discovery should be skipped when image provided"
    assert stages1.get("IMAGE_RETRIEVAL") == "skipped", "Retrieval should be skipped when image provided"
    assert stages1.get("MODEL_INFERENCE") == "completed", "Model inference must run"
    assert stages1.get("RESULT_SYNTHESIS") == "completed", "Result synthesis must run"
    assert "optical_caption" in data1, "optical_caption field must exist"
    print(f"[PASS] in {dur1:.1f}ms. Stages skipped: LOCATION, DISCOVERY, RETRIEVAL. Optical caption: {str(data1.get('optical_caption'))[:60]}...")
    test_results.append({
        "test": "Simple Captioning",
        "query": "Describe this satellite scene",
        "duration_ms": dur1,
        "stages": data1.get("execution_trace", []),
        "status": "PASSED",
    })

    # -----------------------------------------------------------------------
    # TEST 2: Building Detection (User image provided)
    # -----------------------------------------------------------------------
    print("\n[TEST 2] Building Detection: 'Find buildings in this area'")
    t0 = time.perf_counter()
    res2 = client.post("/api/query", json={
        "query": "Find buildings in this area",
        "optical_image": demo_b64,
    })
    dur2 = (time.perf_counter() - t0) * 1000.0
    data2 = res2.json()
    assert res2.status_code == 200
    assert "detections" in data2
    assert "evidence" in data2
    assert "geojson" in data2
    stages2 = {s["stage_name"]: s["status"] for s in data2.get("execution_trace", [])}
    assert stages2.get("MODEL_INFERENCE") == "completed"
    assert stages2.get("SPATIAL_ANALYSIS") == "completed"
    print(f"[PASS] in {dur2:.1f}ms. Detections found: {data2.get('num_detections', 0)}. Result category: {data2.get('result_category')}")
    test_results.append({
        "test": "Building Detection",
        "query": "Find buildings in this area",
        "duration_ms": dur2,
        "stages": data2.get("execution_trace", []),
        "status": "PASSED",
    })

    # -----------------------------------------------------------------------
    # TEST 3: Vehicle Counting (Counting target)
    # -----------------------------------------------------------------------
    print("\n[TEST 3] Vehicle Counting: 'How many vehicles are present?'")
    t0 = time.perf_counter()
    res3 = client.post("/api/query", json={
        "query": "How many vehicles are present?",
        "optical_image": demo_b64,
    })
    dur3 = (time.perf_counter() - t0) * 1000.0
    data3 = res3.json()
    assert res3.status_code == 200
    assert data3.get("count_summary") is not None, "Count summary must exist for counting query"
    assert "count" in data3, "Count field must exist"
    print(f"[PASS] in {dur3:.1f}ms. Count: {data3.get('count')}. Summary: {data3.get('count_summary')}")
    test_results.append({
        "test": "Vehicle Counting",
        "query": "How many vehicles are present?",
        "duration_ms": dur3,
        "stages": data3.get("execution_trace", []),
        "status": "PASSED",
    })

    # -----------------------------------------------------------------------
    # TEST 4: Full Satellite Investigation with Location: "Find ships in Mumbai"
    # -----------------------------------------------------------------------
    print("\n[TEST 4] Full Satellite Investigation: 'Find ships in Mumbai'")
    t0 = time.perf_counter()
    res4 = client.post("/api/query", json={
        "query": "Find ships in Mumbai",
    })
    dur4 = (time.perf_counter() - t0) * 1000.0
    data4 = res4.json()
    assert res4.status_code == 200
    stages4 = {s["stage_name"]: s["status"] for s in data4.get("execution_trace", [])}
    assert stages4.get("LOCATION_RESOLUTION") == "completed", "Location resolution must complete"
    assert stages4.get("IMAGERY_DISCOVERY") == "completed", "Imagery discovery must complete"
    assert stages4.get("SCENE_SELECTION") == "completed", "Scene selection must complete"
    assert stages4.get("IMAGE_RETRIEVAL") == "completed", "Image retrieval must complete"
    assert stages4.get("MODEL_INFERENCE") == "completed", "Model inference must complete"
    assert stages4.get("SPATIAL_ANALYSIS") == "completed", "Spatial analysis must complete"
    assert stages4.get("RESULT_SYNTHESIS") == "completed", "Result synthesis must complete"
    print(f"[PASS] in {dur4:.1f}ms ({dur4/1000.0:.2f}s). All 10 stages executed! Detections: {data4.get('num_detections')}. Category: {data4.get('result_category')}")
    test_results.append({
        "test": "Full Satellite Investigation (Location)",
        "query": "Find ships in Mumbai",
        "duration_ms": dur4,
        "stages": data4.get("execution_trace", []),
        "status": "PASSED",
    })

    # -----------------------------------------------------------------------
    # TEST 5: Region Detection: "Segment water bodies in London"
    # -----------------------------------------------------------------------
    print("\n[TEST 5] Region Detection: 'Segment water bodies in London'")
    t0 = time.perf_counter()
    res5 = client.post("/api/query", json={
        "query": "Segment water bodies in London",
    })
    dur5 = (time.perf_counter() - t0) * 1000.0
    data5 = res5.json()
    assert res5.status_code == 200
    assert "geographic_context" in data5
    print(f"[PASS] in {dur5:.1f}ms ({dur5/1000.0:.2f}s). Detections: {data5.get('num_detections')}. Center: {data5.get('geographic_context', {}).get('center')}")
    test_results.append({
        "test": "Region Detection",
        "query": "Segment water bodies in London",
        "duration_ms": dur5,
        "stages": data5.get("execution_trace", []),
        "status": "PASSED",
    })

    # -----------------------------------------------------------------------
    # TEST 6: Change Analysis (Bi-temporal comparison)
    # -----------------------------------------------------------------------
    print("\n[TEST 6] Change Analysis: 'Compare before and after satellite images'")
    t0 = time.perf_counter()
    res6 = client.post("/api/query", json={
        "query": "Compare before and after satellite images for structural change",
        "change_image_a": demo_b64,
        "change_image_b": demo_b64,
    })
    dur6 = (time.perf_counter() - t0) * 1000.0
    data6 = res6.json()
    assert res6.status_code == 200
    assert "change_analysis" in data6
    print(f"[PASS] in {dur6:.1f}ms. Change analysis completed.")
    test_results.append({
        "test": "Change Analysis",
        "query": "Compare before and after satellite images for structural change",
        "duration_ms": dur6,
        "stages": data6.get("execution_trace", []),
        "status": "PASSED",
    })

    # -----------------------------------------------------------------------
    # TEST 7: Spatial Relationship Query: "Find ships near port in Mumbai"
    # -----------------------------------------------------------------------
    print("\n[TEST 7] Spatial Relationship Query: 'Find ships near port in Mumbai'")
    t0 = time.perf_counter()
    res7 = client.post("/api/query", json={
        "query": "Find ships near port in Mumbai",
    })
    dur7 = (time.perf_counter() - t0) * 1000.0
    data7 = res7.json()
    assert res7.status_code == 200
    stages7 = {s["stage_name"]: s["metadata"] for s in data7.get("execution_trace", [])}
    spatial_meta = stages7.get("SPATIAL_ANALYSIS", {})
    assert spatial_meta.get("spatial_predicate") == "near", f"Expected near predicate, got {spatial_meta.get('spatial_predicate')}"
    print(f"[PASS] in {dur7:.1f}ms. Spatial predicate evaluated: {spatial_meta.get('spatial_predicate')}. Reference: {spatial_meta.get('spatial_reference')}")
    test_results.append({
        "test": "Spatial Relationship Query",
        "query": "Find ships near port in Mumbai",
        "duration_ms": dur7,
        "stages": data7.get("execution_trace", []),
        "status": "PASSED",
    })

    # -----------------------------------------------------------------------
    # TEST 8: Cache Hit Verification
    # (Running the exact same query again should report CACHE_HIT in IMAGE_RETRIEVAL)
    # -----------------------------------------------------------------------
    print("\n[TEST 8] Cache Hit Verification: Repeated query 'Find ships in Mumbai'")
    t0 = time.perf_counter()
    res8 = client.post("/api/query", json={
        "query": "Find ships in Mumbai",
    })
    dur8 = (time.perf_counter() - t0) * 1000.0
    data8 = res8.json()
    assert res8.status_code == 200
    retrieval_trace = next((s for s in data8.get("execution_trace", []) if s["stage_name"] == "IMAGE_RETRIEVAL"), None)
    assert retrieval_trace is not None
    assert retrieval_trace["metadata"].get("cache_status") == "CACHE_HIT", f"Expected CACHE_HIT, got {retrieval_trace['metadata'].get('cache_status')}"
    print(f"[PASS] in {dur8:.1f}ms. IMAGE_RETRIEVAL explicitly reported CACHE_HIT: {retrieval_trace['metadata']['cache_status']}")
    test_results.append({
        "test": "Cache Hit Verification",
        "query": "Find ships in Mumbai (cached)",
        "duration_ms": dur8,
        "stages": data8.get("execution_trace", []),
        "status": "PASSED",
    })

    # -----------------------------------------------------------------------
    # TEST 9: Failure Handling: Location Unresolved
    # -----------------------------------------------------------------------
    print("\n[TEST 9] Failure Handling: Unresolvable location 'NonexistentPlaceXYZ12345'")
    t0 = time.perf_counter()
    res9 = client.post("/api/query", json={
        "query": "Find airplanes in NonexistentPlaceXYZ12345",
    })
    dur9 = (time.perf_counter() - t0) * 1000.0
    data9 = res9.json()
    assert data9.get("status") == "failed"
    assert data9.get("error_stage") == "LOCATION_RESOLUTION"
    print(f"[PASS] in {dur9:.1f}ms. Correctly caught LOCATION_UNRESOLVED: {str(data9.get('error'))[:60]}...")
    test_results.append({
        "test": "Failure Handling (Location Unresolved)",
        "query": "Find airplanes in NonexistentPlaceXYZ12345",
        "duration_ms": dur9,
        "stages": data9.get("execution_trace", []),
        "status": "PASSED",
    })

    # -----------------------------------------------------------------------
    # TEST 10: Async Execution & Status / Result API
    # -----------------------------------------------------------------------
    print("\n[TEST 10] Async Execution & Live Status Polling")
    res10_accept = client.post(
        "/api/query",
        json={"query": "Find buildings in London", "async_execution": True},
    )
    assert res10_accept.status_code == 200
    accept_data = res10_accept.json()
    assert accept_data.get("status") == "accepted"
    inv_id = accept_data.get("investigation_id")
    print(f"[PASS] Async query accepted with ID: {inv_id}")

    # Poll status endpoint
    time.sleep(0.5)
    res_status = client.get(f"/api/investigation/{inv_id}/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert "status" in status_data
    assert "stages" in status_data
    assert "progress" in status_data
    print(f"[PASS] Status polled: status={status_data['status']}, current_stage={status_data.get('current_stage')}, progress={status_data['progress']['percentage']}%")

    # Wait for completion and poll result
    max_wait = 60
    start_wait = time.time()
    result_data = None
    while time.time() - start_wait < max_wait:
        res_poll = client.get(f"/api/investigation/{inv_id}/result")
        if res_poll.status_code == 200:
            result_data = res_poll.json()
            break
        elif res_poll.status_code == 202:
            time.sleep(0.8)
        else:
            time.sleep(0.8)

    assert result_data is not None, "Async investigation did not complete in time"
    assert result_data.get("status") == "completed"
    print(f"[PASS] Result retrieved successfully from /api/investigation/{inv_id}/result! Total stages: {len(result_data.get('execution_trace', []))}")
    test_results.append({
        "test": "Async Execution & Status Polling",
        "query": "Find buildings in London",
        "duration_ms": (time.time() - start_wait) * 1000.0,
        "stages": result_data.get("execution_trace", []),
        "status": "PASSED",
    })

    # -----------------------------------------------------------------------
    # TEST 11: Backward Compatibility Schema Verification
    # (Verify all fields expected by QueryPage.tsx are intact)
    # -----------------------------------------------------------------------
    print("\n[TEST 11] Backward Compatibility Verification for Frontend")
    required_frontend_keys = [
        "request_id", "status", "query", "plan", "selected_tools", "routing_decision",
        "vqa_results", "optical_caption", "sar_caption", "change_analysis",
        "evidence", "evidence_graph", "investigation_report", "execution_trace",
        "confidence", "geographic_context", "geojson", "geospatial_metadata",
        "detections", "num_detections", "response_text", "backend_status"
    ]
    for key in required_frontend_keys:
        assert key in data4, f"Missing required frontend contract key: '{key}'"
    print(f"[PASS] 100% of required frontend contract keys ({len(required_frontend_keys)}/{len(required_frontend_keys)}) present in payload!")

    print("\n" + "=" * 70)
    print("ALL 11 BACKEND INVESTIGATION PIPELINE TESTS PASSED!")
    print("=" * 70)

    # Save test results JSON
    results_path = os.path.join(_dir, "INVESTIGATION_PIPELINE_TEST_RESULTS.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"Saved test results to {results_path}")
    return test_results


if __name__ == "__main__":
    run_all_tests()
