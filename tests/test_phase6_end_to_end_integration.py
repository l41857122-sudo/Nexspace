"""
tests/test_phase6_end_to_end_integration.py
--------------------------------------------
Phase 6 Verification Suite:
End-to-End Investigation Integration Across All Five Capabilities:
  1. Execution Trace and Structured Logs
  2. OpenStreetMap / Overpass Integration
  3. Genuine Optical-SAR Processing and Fusion
  4. Evidence-Based Temporal Change Analysis
  5. Reliable Visual Question Answering (VQA)

Verifies the unified conceptual flow:
  USER QUERY -> QUERY UNDERSTANDING -> INPUT VALIDATION ->
  AGENT CONTROLLER -> APPROPRIATE SPECIALISTS ->
  GEOSPATIAL ALIGNMENT -> ANALYSIS & OPTIONAL FUSION ->
  EVIDENCE & PROVENANCE -> EXPLAINABLE RESULT
"""

from __future__ import annotations
import os
import sys
import unittest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_backend"))

from orchestrator import GeoVLMController
from investigation_pipeline import investigation_pipeline, global_investigation_store
from tools import tool_registry
from router import TaskType
from fastapi.testclient import TestClient
from server import app, encode_image_b64


class TestPhase6EndToEndIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.controller = GeoVLMController()
        cls.pipeline = investigation_pipeline
        cls.client = TestClient(app)

        # Authentic test rasters
        cls.opt_img = Image.new("RGB", (256, 256), (150, 150, 150))
        d_opt = ImageDraw.Draw(cls.opt_img)
        d_opt.rectangle([20, 20, 100, 100], fill=(200, 40, 40))   # Built structure
        d_opt.rectangle([120, 20, 240, 120], fill=(30, 180, 40))  # Green vegetation
        d_opt.rectangle([20, 140, 240, 240], fill=(20, 40, 160))  # Water body

        cls.sar_img = Image.new("RGB", (256, 256), (60, 60, 60))
        d_sar = ImageDraw.Draw(cls.sar_img)
        d_sar.rectangle([20, 20, 100, 100], fill=(255, 255, 255)) # Strong corner reflection
        d_sar.rectangle([120, 20, 240, 120], fill=(90, 90, 90))   # Volume scattering

        cls.changed_img = cls.opt_img.copy()
        d_ch = ImageDraw.Draw(cls.changed_img)
        d_ch.rectangle([20, 20, 100, 100], fill=(150, 150, 150)) # Cleared structure

    def test_01_execution_trace_and_logs_integration(self):
        """Investigation executes, generates structured trace and logs, retrievable via API."""
        b64_opt = encode_image_b64(self.opt_img)
        res = self.client.post("/api/query", json={
            "query": "Describe this satellite scene",
            "optical_image": b64_opt,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        req_id = data["request_id"]
        self.assertTrue(req_id.startswith("req_"))

        # Verify structured trace endpoint
        trace_res = self.client.get(f"/api/investigation/{req_id}/trace")
        self.assertEqual(trace_res.status_code, 200)
        trace_data = trace_res.json()
        self.assertIn("tool_executions", trace_data)
        self.assertIn("structured_trace", trace_data)
        self.assertGreater(len(trace_data["stages"]), 0)

        # Verify structured logs endpoint
        logs_res = self.client.get(f"/api/investigation/{req_id}/logs")
        self.assertEqual(logs_res.status_code, 200)
        logs_data = logs_res.json()
        self.assertIn("logs", logs_data)
        self.assertGreater(logs_data["count"], 0)

    def test_02_osm_overpass_integration_flow(self):
        """OSM Overpass endpoint returns GeoJSON and attribution without fabricating data."""
        res = self.client.post("/api/osm-features", json={
            "bbox": [80.9, 26.8, 80.95, 26.85],
            "feature_types": ["buildings", "highways"],
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("geojson", data)
        self.assertEqual(data["geojson"]["type"], "FeatureCollection")
        self.assertIn("OpenStreetMap contributors", data["attribution"])
        self.assertIn("distinction_note", data)

    def test_03_optical_sar_fusion_end_to_end(self):
        """Optical + SAR analysis executes through pipeline and populates breakdown."""
        b64_opt = encode_image_b64(self.opt_img)
        b64_sar = encode_image_b64(self.sar_img)
        res = self.client.post("/api/query", json={
            "query": "Compare optical and SAR imagery",
            "optical_image": b64_opt,
            "sar_image": b64_sar,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("Optical_SAR_Analysis", data["selected_tools"])
        self.assertIn("optical_sar_analysis", data)
        osa = data["optical_sar_analysis"]
        self.assertIsNotNone(osa)
        self.assertEqual(osa.get("fused_feature_dim"), 1536)
        self.assertIn("optical_only_metrics", osa)
        self.assertIn("sar_only_metrics", osa)
        self.assertIn("limitations", osa)

    def test_04_temporal_change_analysis_end_to_end(self):
        """Temporal change analysis executes through /api/change-analysis with evidence and limitations."""
        b64_a = encode_image_b64(self.opt_img)
        b64_b = encode_image_b64(self.changed_img)
        res = self.client.post("/api/change-analysis", json={
            "image_a": b64_a,
            "image_b": b64_b,
            "date_a": "2024-01-01",
            "date_b": "2024-06-01",
            "change_threshold": 0.15,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(data["changed_fraction"], 0.0)
        self.assertIn("overlay_image", data)
        self.assertIn("heatmap_image", data)
        self.assertIn("limitations", data)
        self.assertEqual(data["acquisition_date_a"], "2024-01-01")
        self.assertEqual(data["acquisition_date_b"], "2024-06-01")

    def test_05_reliable_vqa_query_flow(self):
        """VQA query routes to VQA specialist with transparent provenance."""
        b64_opt = encode_image_b64(self.opt_img)
        res = self.client.post("/api/query", json={
            "query": "What type of area is shown?",
            "optical_image": b64_opt,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("VQA", data["selected_tools"])
        self.assertIn("evidence", data)
        vqa_ev = next((e for e in data["evidence"] if e.get("type") == "vqa_answer"), None)
        self.assertIsNotNone(vqa_ev)
        self.assertIn(vqa_ev.get("provenance"), ("MODEL_DERIVED", "HEURISTIC"))


if __name__ == "__main__":
    unittest.main()
