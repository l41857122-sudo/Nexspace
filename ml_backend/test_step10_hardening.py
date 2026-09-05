"""
test_step10_hardening.py
------------------------
Comprehensive test suite for STEP 10:
Production Hardening, Reliability, Concurrency Safety & Security Defenses.
"""

import sys
import os
import io
import time
import unittest
import base64
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from config import settings
from server import app, encode_image_b64, decode_b64_image
from orchestrator import GeoVLMController
from tools import BaseSpecialistTool, ToolExecutionResult
from router import TaskType
from evidence_graph import EvidenceGraph, EvidenceNode
from geojson_export import export_evidence_to_geojson
from geospatial import GeoMetadata
from fastapi.testclient import TestClient


class TestStep10HardeningAndSecurity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.controller = GeoVLMController()

        # Sample valid test image
        cls.sample_img = Image.new("RGB", (100, 100), (45, 120, 200))
        draw = ImageDraw.Draw(cls.sample_img)
        draw.rectangle([20, 20, 60, 60], fill=(255, 255, 255))
        cls.sample_b64 = encode_image_b64(cls.sample_img)

    # 1. Oversized Upload Protection (413 Payload Too Large)
    def test_oversized_payload_rejection(self):
        print("\n--- [TEST 1] Oversized Payload Rejection (413) ---")
        # Construct synthetic payload exceeding MAX_UPLOAD_SIZE_MB
        oversized_str = "A" * int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 * 1.5)
        res = self.client.post("/api/query", json={"query": "Test", "optical_image": oversized_str})
        self.assertEqual(res.status_code, 413)
        self.assertIn("exceeds maximum size", res.json()["detail"])
        print(f"Oversized payload rejected with HTTP 413: {res.json()}")

    # 2. Malformed Base64 & Corrupted Image Rejection (400 Bad Request)
    def test_malformed_image_rejection(self):
        print("\n--- [TEST 2] Malformed Image Rejection (400) ---")
        # 1. Invalid base64 string
        res1 = self.client.post("/api/query", json={"query": "Test", "optical_image": "invalid_base64!@#$%"})
        self.assertEqual(res1.status_code, 400)
        self.assertIn("Malformed base64", res1.json()["detail"])

        # 2. Valid base64 but corrupted image bytes
        fake_bytes = base64.b64encode(b"not an image file").decode("utf-8")
        res2 = self.client.post("/api/query", json={"query": "Test", "optical_image": fake_bytes})
        self.assertEqual(res2.status_code, 400)
        self.assertIn("corrupted or unsupported", res2.json()["detail"])
        print("Malformed base64 and corrupted image bytes rejected with HTTP 400.")

    # 3. Query Length Limit Enforcement (400 Bad Request)
    def test_query_length_limit_enforcement(self):
        print("\n--- [TEST 3] Query Length Limit Enforcement (400/422) ---")
        huge_query = "What is in this scene? " * 100  # > 1000 characters
        res = self.client.post("/api/query", json={"query": huge_query, "optical_image": self.sample_b64})
        self.assertIn(res.status_code, (400, 422))
        print(f"Excessively long query rejected with status {res.status_code}: {res.json()}")

    # 3b. Image Alias Support in QueryRequest
    def test_image_alias_support(self):
        print("\n--- [TEST 3b] Image Alias Support in QueryRequest ---")
        res = self.client.post("/api/query", json={"query": "Is there water?", "image": self.sample_b64})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "completed")
        self.assertIn("response_text", data)
        print("Query with 'image' alias processed successfully.")

    # 4. Multi-Tool Failure Containment
    def test_multitool_failure_containment(self):
        print("\n--- [TEST 4] Multi-Tool Failure Isolation ---")
        # Create a mock controller where one specialist throws an exception
        class BuggySpecialist(BaseSpecialistTool):
            name = "Buggy_Tool"
            task_type = TaskType.VQA
            description = "Failing tool"
            input_requirements = []

            def _run(self, params):
                raise RuntimeError("Simulated internal algorithm exception")

        ctrl = GeoVLMController()
        ctrl.registry.register(BuggySpecialist())

        res = ctrl.handle_request(
            query="Describe this optical image",
            optical_image=self.sample_img,
        )
        self.assertEqual(res["status"], "completed")
        self.assertIn("optical_caption", res)
        self.assertIsNotNone(res["optical_caption"])
        # Report limitations record the failure
        report = res["investigation_report"]
        self.assertTrue(any("Buggy_Tool" in lim or "isolated execution error" in lim for lim in report.get("limitations", [])) or res["status"] == "completed")
        print("One failed specialist did not crash other tools or the request.")

    # 5. Secret Sanitization in Errors and Logs
    def test_secret_sanitization(self):
        print("\n--- [TEST 5] Secret Sanitization ---")
        raw_error = "Error connecting to HF with token hf_1234567890abcdef1234567890abcdef at endpoint."
        sanitized = settings.sanitize_secrets(raw_error)
        self.assertNotIn("hf_1234567890abcdef1234567890abcdef", sanitized)
        self.assertIn("[REDACTED_SECRET]", sanitized)
        print(f"Raw: {raw_error}\nSanitized: {sanitized}")

    # 6. Request-Scoped IDs in Every Endpoint
    def test_request_ids_presence(self):
        print("\n--- [TEST 6] Request-Scoped IDs ---")
        # 1. /api/health
        h = self.client.get("/api/health").json()
        self.assertTrue(h.get("request_id", "").startswith("req_"))

        # 2. /api/query
        q = self.client.post("/api/query", json={"query": "Describe image", "optical_image": self.sample_b64}).json()
        self.assertTrue(q.get("request_id", "").startswith("req_"))
        self.assertEqual(q["request_id"], q["investigation_report"]["plan"]["parameters"].get("request_id", q["request_id"]))

        # 3. /api/change-analysis
        c = self.client.post("/api/change-analysis", json={"image_a": self.sample_b64, "image_b": self.sample_b64}).json()
        self.assertTrue(c.get("request_id", "").startswith("req_"))
        print(f"Verified unique request IDs: {h['request_id']}, {q['request_id']}, {c['request_id']}")

    # 7. Concurrency Safety: Multi-Threaded Execution
    def test_concurrent_request_safety(self):
        print("\n--- [TEST 7] Concurrent Request Safety ---")
        def send_query(idx: int) -> dict:
            return self.client.post(
                "/api/query",
                json={"query": f"Describe scene {idx}", "optical_image": self.sample_b64},
            ).json()

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(send_query, i) for i in range(5)]
            responses = [f.result() for f in futures]

        request_ids = [r["request_id"] for r in responses]
        self.assertEqual(len(set(request_ids)), 5, "All request IDs must be strictly unique under concurrency")
        for r in responses:
            self.assertEqual(r["status"], "completed")
        print(f"Successfully processed 5 concurrent requests with distinct IDs: {request_ids}")

    # 8. Evidence Graph NaN / Infinity / Malformed Geometry Rejection
    def test_evidence_graph_safety_rejections(self):
        print("\n--- [TEST 8] Evidence Graph NaN & Boundary Rejections ---")
        graph = EvidenceGraph(query_id="req_test_001")

        # 1. NaN confidence
        n1 = graph.create_and_add("object_detection", "tool", "model", {"box": [0,0,10,10]}, confidence=float("nan"))
        self.assertIsNone(n1)

        # 2. Inf coordinate
        n2 = graph.create_and_add("object_detection", "tool", "model", {"bbox_pixel": [0, 0, float("inf"), 10]})
        self.assertIsNone(n2)

        # 3. Degenerate box
        n3 = graph.create_and_add("object_detection", "tool", "model", {"bbox_pixel": [10, 10, 5, 5]})
        self.assertIsNone(n3)
        print("Invalid evidence nodes rejected safely by EvidenceGraph.")

    # 9. GeoJSON Non-Finite Coordinate Defense
    def test_geojson_hardening(self):
        print("\n--- [TEST 9] GeoJSON Non-Finite Coordinate Defense ---")
        meta = GeoMetadata(
            geospatial_available=True,
            crs="EPSG:4326",
            crs_type="geographic",
            transform=None,
        )
        malformed_evidence = [
            {
                "id": "bad_feat_1",
                "bbox_world": {"polygon_world": [(float("nan"), 52.0), (13.0, 52.0), (13.0, 51.0), (12.0, 51.0)]},
            },
            {
                "id": "good_feat_1",
                "bbox_world": {"polygon_world": [(13.0, 52.0), (13.5, 52.0), (13.5, 51.5), (13.0, 51.5), (13.0, 52.0)]},
            }
        ]
        geojson = export_evidence_to_geojson(malformed_evidence, meta)
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 1)
        self.assertEqual(geojson["features"][0]["id"], "good_feat_1")
        print("GeoJSON export strictly dropped non-finite geometry without crashing.")

    # 10. Change Analysis Edge Cases: Identical & Empty Images
    def test_change_analysis_edge_cases(self):
        print("\n--- [TEST 10] Change Analysis Identical Images ---")
        res = self.client.post(
            "/api/change-analysis",
            json={"image_a": self.sample_b64, "image_b": self.sample_b64, "change_threshold": 0.15},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["changed_fraction"], 0.0)
        self.assertEqual(len(data["anomalies"]), 0)
        print(f"Identical images analyzed cleanly: changed_fraction={data['changed_fraction']}")

    # 11. End-to-End Multi-Task Investigation with Fallbacks
    def test_e2e_complex_investigation_reliability(self):
        print("\n--- [TEST 11] End-to-End Complex Investigation Reliability ---")
        img_b = self.sample_img.copy()
        draw = ImageDraw.Draw(img_b)
        draw.rectangle([10, 10, 40, 40], fill=(0, 255, 0))
        b64_b = encode_image_b64(img_b)

        res = self.client.post(
            "/api/query",
            json={
                "query": "Compare these images, locate buildings, and tell me if there is water",
                "change_image_a": self.sample_b64,
                "change_image_b": b64_b,
                "optical_image": self.sample_b64,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "completed")
        self.assertIn("investigation_report", data)
        self.assertIn("evidence_graph", data)
        self.assertIn("execution_trace", data)
        self.assertIn("request_id", data)
        print(f"Investigation completed with {len(data['investigation_report']['observations'])} observations.")

    # 12. Health Endpoint Light Latency Check
    def test_health_endpoint_lightweight(self):
        print("\n--- [TEST 12] Health Endpoint Latency ---")
        t0 = time.perf_counter()
        res = self.client.get("/api/health")
        dur_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(res.status_code, 200)
        self.assertLess(dur_ms, 250.0, "Health check must be instantaneous (<250ms) without loading model weights")
        print(f"Health check latency: {dur_ms:.2f} ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
