import os
import sys
import time
import json
import urllib.request
import urllib.error
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from grounding_adapters import GroundingDINOAdapter, RSAdaptedVisionGroundingAdapter
from tools import GroundingTool
from orchestrator import GeoVLMController

print("=" * 80, flush=True)
print("FINAL LIVE END-TO-END INVESTIGATION VERIFICATION", flush=True)
print("=" * 80, flush=True)

# 1. Load canonical test image and prepare base64
img_path = os.path.join(os.path.dirname(_dir), "sample_data", "nexspace_test_image_001.jpg")
with open(img_path, "rb") as f:
    import base64
    b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

queries = [
    ("Locate buildings", "RS specialist: buildings"),
    ("Find warehouses", "RS specialist: warehouses"),
    ("Locate ships", "RS specialist: ships"),
    ("Find boats", "RS specialist: boats"),
    ("Locate bridges", "RS specialist: bridges"),
    ("Find airplanes", "RS specialist: airplanes"),
    ("Find vehicles", "RS specialist: vehicles"),
    ("Find roads", "RS specialist: roads"),
    ("Find telecommunication antenna tower", "Novel arbitrary target -> Grounding DINO"),
    ("Locate buildings and roads", "Multi-target query"),
]

print("\n=== TESTING LIVE NEXT.JS & FASTAPI ENDPOINTS ===", flush=True)
results = []

for q, desc in queries:
    payload = {
        "query": q,
        "optical_image": data_url
    }

    t0 = time.perf_counter()
    req = urllib.request.Request(
        "http://localhost:3000/api/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            http_status = resp.getcode()
            data = json.loads(resp.read().decode("utf-8"))
            dur = (time.perf_counter() - t0) * 1000.0

            task_type = data.get("task_type")
            status = data.get("status")
            backend_status = data.get("backend_status", "online")
            tools = data.get("selected_tools", [])
            evidence = data.get("evidence", [])
            conf = data.get("confidence")

            # Check grounding evidence source
            g_item = next((e for e in evidence if e.get("type") == "bounding_box"), None)
            engine = g_item.get("source") if g_item else ("VQA/BLIP" if "VQA" in tools or "Optical_Caption" in tools else "N/A")

            record = {
                "query": q,
                "description": desc,
                "http_status": http_status,
                "backend_status": backend_status,
                "task_type": task_type,
                "status": status,
                "tools": tools,
                "engine": engine,
                "evidence_count": len(evidence),
                "confidence": conf,
                "duration_ms": round(dur, 2)
            }
            results.append(record)

            print(f"Query: \"{q}\"", flush=True)
            print(f"  -> HTTP: {http_status} | Backend: {backend_status} | Status: {status} | Task: {task_type}", flush=True)
            print(f"  -> Tools: {tools} | Engine: {engine} | Ev Count: {len(evidence)} | Latency: {dur:.1f}ms", flush=True)
            if g_item:
                print(f"  -> BBox: {g_item.get('box')} | Score: {g_item.get('score')} | Src: {g_item.get('source')}", flush=True)
            print("-" * 80, flush=True)
    except Exception as ex:
        print(f"Query \"{q}\" FAILED with exception: {ex}", flush=True)
        results.append({"query": q, "status": "failed", "error": str(ex)})

print("\n=== FALLBACK VERIFICATION (RS ADAPTER DISABLED) ===", flush=True)
class MockUnavailableRS(RSAdaptedVisionGroundingAdapter):
    def is_available(self): return False

fb_tool = GroundingTool()
fb_tool.rs_adapter = MockUnavailableRS()
fb_controller = GeoVLMController(grounding_tool=fb_tool)
test_pil = Image.open(img_path)

t0 = time.perf_counter()
fb_res = fb_controller.handle_request(query="Where are the buildings?", optical_image=test_pil)
fb_dur = (time.perf_counter() - t0) * 1000.0

fb_g = next((r for r in fb_res.get("results", []) if r.get("tool_name") == "Grounding"), None)
fb_src = fb_g.get("confidence_source") if fb_g else "N/A"
print(f"Fallback Query Result: status={fb_res.get('status')}, engine={fb_src}, time={fb_dur:.1f}ms", flush=True)
print(f"Fallback Handled Gracefully: {fb_res.get('status') == 'completed'}", flush=True)

print("\n=== FINAL RECONCILIATION SUMMARY ===", flush=True)
online_count = sum(1 for r in results if r.get("backend_status") == "online")
success_count = sum(1 for r in results if r.get("status") == "completed" and r.get("http_status") == 200)
print(f"Total Queries Tested: {len(queries)}")
print(f"Successful Live API Queries: {success_count}/{len(queries)}")
print(f"Live Neural Online Status: {online_count}/{len(queries)}")
print(f"Fallback Operational: {fb_res.get('status') == 'completed'}")
