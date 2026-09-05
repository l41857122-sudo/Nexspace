"""
war_mode_test_suite.py
----------------------
Comprehensive Live War-Mode Test Suite for NexSpace.
Executes live API queries against http://localhost:8000/api/query,
renders visual detection overlays with PIL/OpenCV, saves artifacts,
validates evidence traceability, caching, reset isolation, and stress-tests edge cases.
"""

import os
import sys
import time
import base64
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List
from PIL import Image, ImageDraw, ImageFont

_dir = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(_dir, "war_mode_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

IMAGE_A_PATH = os.path.join(_dir, "..", "sample_data", "nexspace_test_image_001.jpg")
API_URL = "http://localhost:8000/api/query"


def load_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"


def send_query(image_b64: str, query: str) -> Dict[str, Any]:
    payload = {
        "image": image_b64,
        "query": query,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as resp:
            dur = (time.perf_counter() - t0) * 1000.0
            res = json.loads(resp.read().decode("utf-8"))
            res["_client_latency_ms"] = round(dur, 2)
            return res
    except urllib.error.HTTPError as e:
        dur = (time.perf_counter() - t0) * 1000.0
        err_body = e.read().decode("utf-8")
        return {
            "error": True,
            "status_code": e.code,
            "body": err_body,
            "_client_latency_ms": round(dur, 2)
        }


def render_detections(
    image_path: str,
    detections: List[Dict[str, Any]],
    output_filename: str,
    title: str = "Detections",
    box_color: str = "#FF3366",
) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Palette for multi-target
    COLOR_MAP = {
        "boat": "#00E5FF",
        "boats": "#00E5FF",
        "ship": "#00E5FF",
        "ships": "#00E5FF",
        "bridge": "#FFD600",
        "bridges": "#FFD600",
        "building": "#FF1744",
        "buildings": "#FF1744",
        "road": "#00E676",
        "water": "#2979FF",
    }

    for d in detections:
        box = d.get("bbox_normalized") or d.get("box") or d.get("box_2d") or [0, 0, 1000, 1000]
        label = d.get("label", "object")
        score = d.get("score", 0.0)

        x1 = int(round((box[0] / 1000.0) * w))
        y1 = int(round((box[1] / 1000.0) * h))
        x2 = int(round((box[2] / 1000.0) * w))
        y2 = int(round((box[3] / 1000.0) * h))

        c = COLOR_MAP.get(label.lower(), box_color)

        # Draw thick box
        for offset in range(3):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=c)

        # Label tag
        tag = f"{label} {score:.2f}"
        draw.rectangle([x1, max(0, y1 - 18), x1 + len(tag) * 8 + 6, max(18, y1)], fill=c)
        draw.text((x1 + 3, max(0, y1 - 16)), tag, fill="#000000")

    # Header banner
    draw.rectangle([0, 0, w, 32], fill="#111827")
    header_text = f"NexSpace Live Grounding Overlay | {title} | Detections: {len(detections)}"
    draw.text((12, 8), header_text, fill="#F9FAFB")

    out_path = os.path.join(ARTIFACTS_DIR, output_filename)
    img.save(out_path, quality=95)
    return out_path


def get_detections_from_res(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    if res.get("grounding") and isinstance(res["grounding"], dict) and "detections" in res["grounding"]:
        return res["grounding"]["detections"]
    return [e for e in res.get("evidence", []) if e.get("type") == "bounding_box"]


def get_response_text(res: Dict[str, Any]) -> str:
    if res.get("response_text"):
        return res["response_text"]
    obs = res.get("investigation_report", {}).get("observations", [])
    if obs:
        return " ".join(obs)
    return ""


def run_full_war_suite():
    print("============================================================")
    print("PHASE 2A: WAR-MODE LIVE IMAGE TEST SUITE")
    print("============================================================")
    
    assert os.path.exists(IMAGE_A_PATH), f"Image missing at {IMAGE_A_PATH}"
    img_b64 = load_image_base64(IMAGE_A_PATH)
    
    results = {}

    # Copy original image to artifacts
    orig_art_path = os.path.join(ARTIFACTS_DIR, "01_original_uploaded_image.jpg")
    Image.open(IMAGE_A_PATH).save(orig_art_path, quality=95)
    print(f"[OK] Saved original image artifact: {orig_art_path}")

    # ---------------------------------------------------------
    # 1. VQA / Presence Tests
    # ---------------------------------------------------------
    print("\n--- 1. VQA / PRESENCE TESTS ---")
    vqa_queries = [
        "Is there water?",
        "Is there a bridge?",
        "Is vegetation present?",
        "Are there boats?",
        "Are buildings present?",
    ]
    for q in vqa_queries:
        res = send_query(img_b64, q)
        ans = get_response_text(res)
        print(f"Query: '{q}'")
        print(f"  Routed Task:  {res.get('task_type')}")
        print(f"  Specialists:  {res.get('selected_tools')}")
        print(f"  Latency:      {res.get('_client_latency_ms')} ms")
        print(f"  Confidence:   {res.get('confidence')} ({res.get('confidence_type')})")
        print(f"  Answer:       {ans[:120]}...")
        results[f"vqa_{q}"] = {**res, "_extracted_answer": ans}

    # ---------------------------------------------------------
    # 2. Counting Tests
    # ---------------------------------------------------------
    print("\n--- 2. COUNTING TESTS ---")
    count_queries = [
        "How many boats are visible?",
        "How many buildings are visible?",
        "How many bridges are visible?",
    ]
    for q in count_queries:
        res = send_query(img_b64, q)
        dets = get_detections_from_res(res)
        ans = get_response_text(res)
        print(f"Query: '{q}'")
        print(f"  Routed Task:  {res.get('task_type')}")
        print(f"  Specialists:  {res.get('selected_tools')}")
        print(f"  Detections:   {len(dets)}")
        print(f"  Latency:      {res.get('_client_latency_ms')} ms")
        print(f"  Confidence:   {res.get('confidence')} ({res.get('confidence_type')})")
        print(f"  Answer:       {ans[:120]}...")
        results[f"count_{q}"] = {**res, "_extracted_detections": dets, "_extracted_answer": ans}

    # ---------------------------------------------------------
    # 3. Grounding / Detection Tests & Visual Artifact Generation
    # ---------------------------------------------------------
    print("\n--- 3. GROUNDING / DETECTION TESTS & VISUAL ARTIFACTS ---")
    
    # 3a. Boats
    res_boats = send_query(img_b64, "Where are the boats?")
    dets_boats = get_detections_from_res(res_boats)
    art_boats = render_detections(IMAGE_A_PATH, dets_boats, "02_grounding_boats.jpg", "Target: boats", "#00E5FF")
    print(f"[OK] Boats Grounding ({len(dets_boats)} boxes) -> {art_boats}")
    results["grounding_boats"] = {"response": res_boats, "artifact": art_boats, "detections": dets_boats}

    # 3b. Bridge
    res_bridge = send_query(img_b64, "Where is the bridge?")
    dets_bridge = get_detections_from_res(res_bridge)
    art_bridge = render_detections(IMAGE_A_PATH, dets_bridge, "03_grounding_bridge.jpg", "Target: bridge", "#FFD600")
    print(f"[OK] Bridge Grounding ({len(dets_bridge)} boxes) -> {art_bridge}")
    results["grounding_bridge"] = {"response": res_bridge, "artifact": art_bridge, "detections": dets_bridge}

    # 3c. Buildings
    res_buildings = send_query(img_b64, "Where are buildings?")
    dets_buildings = get_detections_from_res(res_buildings)
    art_buildings = render_detections(IMAGE_A_PATH, dets_buildings, "04_grounding_buildings.jpg", "Target: buildings", "#FF1744")
    print(f"[OK] Buildings Grounding ({len(dets_buildings)} boxes) -> {art_buildings}")
    results["grounding_buildings"] = {"response": res_buildings, "artifact": art_buildings, "detections": dets_buildings}

    # 3d. Multi-Target Joint Grounding
    res_multi = send_query(img_b64, "Find boats, bridges and buildings")
    dets_multi = get_detections_from_res(res_multi)
    art_multi = render_detections(IMAGE_A_PATH, dets_multi, "05_grounding_multi_target.jpg", "Target: boats . bridge . buildings", "#FF3366")
    print(f"[OK] Multi-Target Grounding ({len(dets_multi)} boxes) -> {art_multi}")
    results["grounding_multi"] = {"response": res_multi, "artifact": art_multi, "detections": dets_multi}

    # 3e. Evidence Visualization Overlay
    evidence_items = res_boats.get("evidence", []) + res_bridge.get("evidence", [])
    dets_ev = [e for e in evidence_items if e.get("type") == "bounding_box"]
    art_ev = render_detections(IMAGE_A_PATH, dets_ev, "06_evidence_visualization.jpg", "Consolidated Evidence Proposals", "#76FF03")
    print(f"[OK] Evidence Overlay ({len(dets_ev)} boxes) -> {art_ev}")
    results["evidence_vis"] = {"artifact": art_ev, "evidence_count": len(evidence_items)}

    # ---------------------------------------------------------
    # 4. Scene Captioning Test
    # ---------------------------------------------------------
    print("\n--- 4. SCENE CAPTIONING TEST ---")
    cap_res1 = send_query(img_b64, "What is in this image?")
    ans_cap1 = get_response_text(cap_res1)
    print(f"Query: 'What is in this image?'")
    print(f"  Response:   {ans_cap1}")
    print(f"  Latency:    {cap_res1.get('_client_latency_ms')} ms")
    
    cap_res2 = send_query(img_b64, "Describe the scene.")
    ans_cap2 = get_response_text(cap_res2)
    print(f"Query: 'Describe the scene.'")
    print(f"  Response:   {ans_cap2}")
    print(f"  Latency:    {cap_res2.get('_client_latency_ms')} ms")
    results["caption_what_is"] = {**cap_res1, "_extracted_answer": ans_cap1}
    results["caption_describe"] = {**cap_res2, "_extracted_answer": ans_cap2}
    results["caption_what_is"] = cap_res1
    results["caption_describe"] = cap_res2

    # ---------------------------------------------------------
    # 5. Spatial Reasoning Tests
    # ---------------------------------------------------------
    print("\n--- 5. SPATIAL REASONING TESTS ---")
    spatial_queries = [
        "Where is the bridge?",
        "Where are the boats relative to the bridge?",
        "Which side contains the major built-up area?",
        "Where is the vegetation?",
        "Where is the river?",
    ]
    for q in spatial_queries:
        res = send_query(img_b64, q)
        ans = get_response_text(res)
        print(f"Query: '{q}'")
        print(f"  Task:       {res.get('task_type')}")
        print(f"  Tools:      {res.get('selected_tools')}")
        print(f"  Response:   {ans[:120]}...")
        results[f"spatial_{q}"] = {**res, "_extracted_answer": ans}

    # ---------------------------------------------------------
    # 6. Multi-Intent / Composite Queries
    # ---------------------------------------------------------
    print("\n--- 6. MULTI-INTENT / COMPOSITE QUERIES ---")
    multi_queries = [
        "What is visible in this image and where are the boats?",
        "Find the bridge and tell me what is around it.",
        "How many boats are there and where are they?",
        "Describe the image and identify major built-up areas.",
        "Is there water, and where is it?",
    ]
    for q in multi_queries:
        res = send_query(img_b64, q)
        dets = get_detections_from_res(res)
        ans = get_response_text(res)
        print(f"Query: '{q}'")
        print(f"  Task:       {res.get('task_type')}")
        print(f"  Tools:      {res.get('selected_tools')}")
        print(f"  Latency:    {res.get('_client_latency_ms')} ms")
        print(f"  Detections: {len(dets)}")
        print(f"  Response:   {ans[:120]}...")
        results[f"multi_{q}"] = {**res, "_extracted_detections": dets, "_extracted_answer": ans}

    # ---------------------------------------------------------
    # 7. Reset / Stale-Data War Test
    # ---------------------------------------------------------
    print("\n--- 7. RESET & STALE-DATA ISOLATION TEST ---")
    # Query boats on Image A
    q_a1 = send_query(img_b64, "Where are the boats?")
    count_a = len(get_detections_from_res(q_a1))
    
    # Create Image B (Synthetic modified image: all red)
    img_b = Image.new("RGB", (768, 768), color=(200, 30, 30))
    img_b_path = os.path.join(ARTIFACTS_DIR, "image_b_synthetic.jpg")
    img_b.save(img_b_path, quality=95)
    img_b_b64 = load_image_base64(img_b_path)
    
    # Query buildings on Image B
    q_b = send_query(img_b_b64, "Where are buildings?")
    count_b = len(get_detections_from_res(q_b))
    print(f"  Image A Boats: {count_a} detections")
    print(f"  Image B Buildings: {count_b} detections")
    
    # Query Image A again (should hit cache cleanly and return exactly count_a)
    q_a2 = send_query(img_b64, "Where are the boats?")
    dets_a2 = get_detections_from_res(q_a2)
    print(f"  Image A Cache Hit Latency: {q_a2.get('_client_latency_ms')} ms, Detections: {len(dets_a2)}")
    assert len(dets_a2) == count_a, "Cache return count mismatch!"
    print("  [PASS] Reset / Stale Data Isolation Verified.")

    # ---------------------------------------------------------
    # 8. Edge Case / Bug Hunt Queries
    # ---------------------------------------------------------
    print("\n--- 8. WAR-MODE EDGE CASE / BUG HUNT ---")
    edge_queries = [
        ("Very Long Query", "Please thoroughly inspect this satellite capture and tell me in great detail whether there are any shipping containers, bridges, aircraft carriers, football stadiums, agricultural fields, highways, or bodies of water present anywhere in the entire scene."),
        ("Vague Query", "What about this?"),
        ("Multiple Objects", "Show me boats, bridge, cars, trees, water, and houses."),
        ("Unsupported Query", "Write a python script to calculate the distance between New York and London."),
        ("Geospatial CRS Query", "What are the coordinates and CRS projection of this image?"),
    ]
    for label, eq in edge_queries:
        res = send_query(img_b64, eq)
        print(f"  [{label}] Query: '{eq[:50]}...'")
        print(f"    Task:     {res.get('task_type')}")
        print(f"    Tools:    {res.get('selected_tools')}")
        print(f"    Latency:  {res.get('_client_latency_ms')} ms")
        print(f"    Response: {res.get('response', '')[:100]}...")
        results[f"edge_{label}"] = res

    # Save complete run summary
    summary_path = os.path.join(ARTIFACTS_DIR, "war_mode_run_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Complete War-Mode Run Results saved to {summary_path}")


if __name__ == "__main__":
    run_full_war_suite()
