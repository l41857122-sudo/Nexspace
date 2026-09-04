import sys
import os
import json
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from server import app, encode_image_b64
from fastapi.testclient import TestClient

client = TestClient(app)
img_path = os.path.join(_dir, "sample_satellite.png")
img = Image.open(img_path)
b64_img = encode_image_b64(img)

# Health
h = client.get("/api/health")
print("=== GET /api/health ===")
print(h.json())

# Query 1: "Describe this image"
print("\n=== Query 1: 'Describe this image' ===")
q1 = client.post("/api/query", json={"query": "Describe this image", "optical_image": b64_img}).json()
print("task_type:", q1.get("task_type"))
print("selected_tools:", q1.get("selected_tools"))
print("statuses:", [(r.get("tool_name"), r.get("status")) for r in q1.get("results", [])])
print("confidence:", q1.get("confidence"), f"({q1.get('confidence_type')})", "source:", q1.get("confidence_source"))
print("evidence count:", len(q1.get("evidence", [])))
print("trace stages:", [e.get("stage") for e in q1.get("execution_trace", [])])
print("response_text:\n", q1.get("response_text"))

# Query 2: "Locate the buildings"
print("\n=== Query 2: 'Locate the buildings' ===")
q2 = client.post("/api/query", json={"query": "Locate the buildings", "optical_image": b64_img}).json()
print("task_type:", q2.get("task_type"))
print("selected_tools:", q2.get("selected_tools"))
print("statuses:", [(r.get("tool_name"), r.get("status")) for r in q2.get("results", [])])
print("confidence:", q2.get("confidence"), f"({q2.get('confidence_type')})", "source:", q2.get("confidence_source"))
print("evidence count:", len(q2.get("evidence", [])))
print("evidence preview:", q2.get("evidence", [])[:2])
print("trace stages:", [e.get("stage") for e in q2.get("execution_trace", [])])
print("response_text:\n", q2.get("response_text"))

# Query 3: "Describe this image and locate the buildings"
print("\n=== Query 3: 'Describe this image and locate the buildings' ===")
q3 = client.post("/api/query", json={"query": "Describe this image and locate the buildings", "optical_image": b64_img}).json()
print("task_type:", q3.get("task_type"))
print("selected_tools:", q3.get("selected_tools"))
print("statuses:", [(r.get("tool_name"), r.get("status")) for r in q3.get("results", [])])
print("confidence:", q3.get("confidence"), f"({q3.get('confidence_type')})", "source:", q3.get("confidence_source"))
print("evidence count:", len(q3.get("evidence", [])))
print("evidence preview:", q3.get("evidence", [])[:2])
print("trace stages:", [e.get("stage") for e in q3.get("execution_trace", [])])
print("response_text:\n", q3.get("response_text"))
