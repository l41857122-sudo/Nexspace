"""
perf_smoke_test.py
------------------
Measures latency across API endpoints, model loading, and multi-specialist pipelines.
"""

import sys
import os
import time
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from server import app, encode_image_b64
from fastapi.testclient import TestClient

client = TestClient(app)

img = Image.new("RGB", (100, 100), (30, 100, 200))
draw = ImageDraw.Draw(img)
draw.rectangle([20, 20, 60, 60], fill=(255, 255, 255))
b64 = encode_image_b64(img)

img_b = img.copy()
draw_b = ImageDraw.Draw(img_b)
draw_b.rectangle([40, 40, 80, 80], fill=(255, 0, 0))
b64_b = encode_image_b64(img_b)

print("=== LATENCY PERFORMANCE SMOKE TEST ===")

# 1. Health check
t0 = time.perf_counter()
res_h = client.get("/api/health")
t_health = (time.perf_counter() - t0) * 1000.0
print(f"1. Health Check (/api/health): {t_health:.2f} ms (Status: {res_h.status_code})")

# 2. Change Analysis
t0 = time.perf_counter()
res_c = client.post("/api/change-analysis", json={"image_a": b64, "image_b": b64_b})
t_chg = (time.perf_counter() - t0) * 1000.0
print(f"2. Change Analysis (/api/change-analysis): {t_chg:.2f} ms (Status: {res_c.status_code})")

# 3. Simple Caption Query
t0 = time.perf_counter()
res_cap = client.post("/api/query", json={"query": "Describe this image", "optical_image": b64})
t_cap = (time.perf_counter() - t0) * 1000.0
print(f"3. Simple Caption Query (/api/query): {t_cap:.2f} ms (Status: {res_cap.status_code})")

# 4. Grounding Query
t0 = time.perf_counter()
res_grd = client.post("/api/query", json={"query": "Locate buildings", "optical_image": b64})
t_grd = (time.perf_counter() - t0) * 1000.0
print(f"4. Grounding Query (/api/query): {t_grd:.2f} ms (Status: {res_grd.status_code})")

# 5. GeoJSON Export
t0 = time.perf_counter()
res_geo = client.post("/api/geojson", json={"evidence": res_c.json().get("evidence", [])})
t_geo = (time.perf_counter() - t0) * 1000.0
print(f"5. GeoJSON Export (/api/geojson): {t_geo:.2f} ms (Status: {res_geo.status_code})")

print("=======================================")
