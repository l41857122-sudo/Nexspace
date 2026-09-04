"""
test_step7_anomaly.py
---------------------
Comprehensive test suite for STEP 7:
Dynamic Anomaly Extraction + Spatial Evidence Engine & Agent Controller Integration.
"""

import sys
import os
import unittest
import time
import numpy as np
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from anomaly_engine import AnomalyEngine, anomaly_engine
from tools import ChangeAnalysisTool, AnomalyExtractionTool, GroundingTool, OpticalSARAnalysisTool
from orchestrator import GeoVLMController
from router import IntentClassifier, TaskType
from server import app, encode_image_b64
from fastapi.testclient import TestClient


class TestStep7DynamicAnomalyEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = AnomalyEngine()
        cls.controller = GeoVLMController()
        cls.classifier = IntentClassifier()
        cls.client = TestClient(app)

        # Base 512x512 image (green landscape)
        cls.img_base = Image.new("RGB", (512, 512), (34, 139, 34))

        # Controlled change image 1: Single distinct 100x100 white building/structure
        cls.img_single_change = cls.img_base.copy()
        draw1 = ImageDraw.Draw(cls.img_single_change)
        draw1.rectangle([100, 150, 200, 250], fill=(240, 240, 240))

        # Controlled change image 2: Multiple distinct regions (3 separate patches)
        cls.img_multi_change = cls.img_base.copy()
        draw2 = ImageDraw.Draw(cls.img_multi_change)
        draw2.rectangle([50, 50, 120, 120], fill=(220, 50, 50))     # Region 1 (70x70)
        draw2.rectangle([300, 100, 400, 200], fill=(50, 50, 220))   # Region 2 (100x100)
        draw2.rectangle([200, 350, 280, 420], fill=(240, 240, 50))  # Region 3 (80x70)

        # Real satellite image sample
        cls.img_sat = Image.open(os.path.join(_dir, "sample_satellite.png"))
        cls.img_sar = Image.open(os.path.join(_dir, "sample_sar.png"))

    # 1. Identical Images -> Near-zero Change Regions
    def test_identical_images(self):
        print("\n--- [TEST 1] Identical Images (Zero Change) ---")
        res = self.engine.extract_change_anomalies(self.img_base, self.img_base)
        self.assertEqual(res["total_regions"], 0)
        self.assertEqual(res["total_changed_pixels"], 0)
        self.assertEqual(res["changed_fraction"], 0.0)
        print(f"Zero Change Result: regions={res['total_regions']}, changed_pixels={res['total_changed_pixels']}")

    # 2. Controlled Changed Region -> Region Detected
    def test_controlled_changed_region(self):
        print("\n--- [TEST 2] Controlled Changed Region ---")
        res = self.engine.extract_change_anomalies(self.img_base, self.img_single_change)
        self.assertGreaterEqual(res["total_regions"], 1)
        r0 = res["regions"][0]
        # Target rectangle was [100, 150, 200, 250] (area 100x100 = 10000 pixels)
        self.assertAlmostEqual(r0["bbox_pixel"][0], 100, delta=5)
        self.assertAlmostEqual(r0["bbox_pixel"][1], 150, delta=5)
        self.assertAlmostEqual(r0["bbox_pixel"][2], 200, delta=5)
        self.assertAlmostEqual(r0["bbox_pixel"][3], 250, delta=5)
        print(f"Detected Region BBox: {r0['bbox_pixel']}, Area: {r0['area_pixels']} px")

    # 3. Bounding Box Validity
    def test_bounding_box_validity(self):
        print("\n--- [TEST 3] Bounding Box Geometric Validity ---")
        res = self.engine.extract_change_anomalies(self.img_base, self.img_single_change)
        for r in res["regions"]:
            x1, y1, x2, y2 = r["bbox_pixel"]
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 512)
            self.assertLessEqual(y2, 512)
            self.assertTrue(self.engine.validate_coordinates(x1, y1, x2, y2, 512, 512))
        print("Bounding box geometric constraints fully validated.")

    # 4. Normalized Coordinates [0..1]
    def test_normalized_coordinates(self):
        print("\n--- [TEST 4] Normalized Coordinates Range [0..1] ---")
        res = self.engine.extract_change_anomalies(self.img_base, self.img_single_change)
        for r in res["regions"]:
            nx1, ny1, nx2, ny2 = r["bbox_normalized"]
            self.assertTrue(0.0 <= nx1 <= 1.0)
            self.assertTrue(0.0 <= ny1 <= 1.0)
            self.assertTrue(0.0 <= nx2 <= 1.0)
            self.assertTrue(0.0 <= ny2 <= 1.0)
            self.assertLess(nx1, nx2)
            self.assertLess(ny1, ny2)
        print(f"Normalized BBox: {res['regions'][0]['bbox_normalized']}")

    # 5. Dynamic Thresholding Strategies
    def test_dynamic_thresholding_strategies(self):
        print("\n--- [TEST 5] Dynamic Thresholding Strategies ---")
        otsu_res = self.engine.extract_change_anomalies(self.img_base, self.img_single_change, threshold_strategy="otsu")
        pct_res = self.engine.extract_change_anomalies(self.img_base, self.img_single_change, threshold_strategy="percentile")
        fixed_res = self.engine.extract_change_anomalies(self.img_base, self.img_single_change, threshold_strategy="fixed", custom_threshold=0.2)

        self.assertIn("otsu", otsu_res["threshold_method"])
        self.assertIn("percentile", pct_res["threshold_method"])
        self.assertIn("fixed", fixed_res["threshold_method"])
        print(f"Otsu Thresh: {otsu_res['threshold_value_255']}, Percentile Thresh: {pct_res['threshold_value_255']}, Fixed Thresh: {fixed_res['threshold_value_255']}")

    # 6. Noise Filtering
    def test_noise_filtering(self):
        print("\n--- [TEST 6] Noise Filtering (Speckle Suppression) ---")
        img_noise = self.img_base.copy()
        # Add tiny single-pixel noise dots
        draw_noise = ImageDraw.Draw(img_noise)
        for i in range(10):
            draw_noise.point((i * 30, i * 30), fill=(255, 255, 255))

        # Run extraction with min_pixel_area = 20
        res = self.engine.extract_change_anomalies(self.img_base, img_noise, min_pixel_area=20)
        # All tiny 1-pixel noise dots must be filtered out
        self.assertEqual(res["total_regions"], 0)
        print(f"Noise filtering: before_filter={res['regions_before_filter']}, after_filter={res['regions_after_filter']}")

    # 7. Multiple Connected Regions
    def test_multiple_connected_regions(self):
        print("\n--- [TEST 7] Multiple Connected Change Regions ---")
        res = self.engine.extract_change_anomalies(self.img_base, self.img_multi_change, min_pixel_area=20)
        # 3 distinct painted rectangles
        self.assertEqual(res["total_regions"], 3)
        print(f"Detected {res['total_regions']} distinct non-overlapping regions as expected.")

    # 8. Evidence Schema
    def test_evidence_schema(self):
        print("\n--- [TEST 8] Standardized Evidence Schema ---")
        res = self.engine.extract_change_anomalies(self.img_base, self.img_single_change)
        r0 = res["regions"][0]
        required_keys = [
            "id", "evidence_type", "bbox_pixel", "bbox_normalized",
            "area_pixels", "area_fraction", "mean_intensity_delta",
            "change_score", "severity_score", "severity_score_type", "source"
        ]
        for k in required_keys:
            self.assertIn(k, r0)
        self.assertEqual(r0["evidence_type"], "change_region")
        self.assertEqual(r0["severity_score_type"], "heuristic")
        print("Standardized evidence schema keys verified.")

    # 9. Invalid BBox Rejection
    def test_invalid_bbox_rejection(self):
        print("\n--- [TEST 9] Invalid BBox Rejection ---")
        # Inverted box [200, 200, 100, 100]
        self.assertFalse(self.engine.validate_coordinates(200, 200, 100, 100, 512, 512))
        # NaN box
        self.assertFalse(self.engine.validate_coordinates(float("nan"), 0, 100, 100, 512, 512))
        # Out of bounds
        self.assertFalse(self.engine.validate_coordinates(-10, 0, 600, 512, 512, 512))
        print("Invalid bounding boxes rejected correctly.")

    # 10. Geospatial Metadata Unavailable Handling
    def test_geospatial_metadata_handling(self):
        print("\n--- [TEST 10] Geospatial Metadata Handling ---")
        res = self.engine.extract_change_anomalies(self.img_base, self.img_single_change)
        self.assertFalse(res["geospatial_coordinates_available"])
        print(f"geospatial_coordinates_available: {res['geospatial_coordinates_available']} (No fake lat/lon)")

    # 11. Grounding Evidence Normalization
    def test_grounding_evidence_normalization(self):
        print("\n--- [TEST 11] Grounding Evidence Normalization ---")
        raw_grounding = [
            {"label": "building", "box": [50.0, 60.0, 150.0, 180.0], "score": 0.88, "source": "Grounding_DINO"}
        ]
        norm = self.engine.normalize_evidence(raw_grounding, source_type="grounding", image_dimensions=(500, 500))
        self.assertEqual(len(norm), 1)
        self.assertEqual(norm[0]["type"], "object_detection")
        self.assertEqual(norm[0]["bbox_normalized"], [0.1, 0.12, 0.3, 0.36])
        self.assertEqual(norm[0]["score"], 0.88)
        print(f"Normalized Grounding Evidence: {norm[0]}")

    # 12. Optical + SAR Indicator Handling
    def test_optical_sar_indicator_handling(self):
        print("\n--- [TEST 12] Optical + SAR Indicator Handling ---")
        raw_fusion = [
            {
                "optical_source": "ViT-B",
                "sar_source": "generic_vision_encoder_baseline",
                "fusion_type": "feature_fusion_baseline",
                "alignment_status": "dimension_match_only",
                "cross_modal_cosine_similarity": 0.74,
            }
        ]
        norm = self.engine.normalize_evidence(raw_fusion, source_type="optical_sar")
        self.assertEqual(norm[0]["type"], "cross_modal_indicator")
        self.assertEqual(norm[0]["interpretation"], "indicator_only")
        print(f"Normalized Multimodal Indicator: {norm[0]}")

    # 13. Anomaly Severity Provenance
    def test_anomaly_severity_provenance(self):
        print("\n--- [TEST 13] Anomaly Severity Provenance ---")
        tool = AnomalyExtractionTool()
        res = tool.execute({"image_a": self.img_base, "image_b": self.img_single_change})
        self.assertIsNone(res.confidence)
        self.assertEqual(res.confidence_type, "heuristic")
        self.assertEqual(res.confidence_source, "dynamic_anomaly_engine")
        for r in res.data["anomalies"]:
            self.assertEqual(r["severity_score_type"], "heuristic")
        print("Severity correctly tracked as heuristic rather than model confidence.")

    # 14. Controller Routing for Change / Anomaly Queries
    def test_controller_routing_anomaly(self):
        print("\n--- [TEST 14] Controller Routing: Change & Anomaly Queries ---")
        queries = [
            "Find changes between these images",
            "Show me the changed regions",
            "Where has the scene changed?",
            "What areas changed?",
        ]
        for q in queries:
            res = self.classifier.classify(query=q, has_change_pair=True)
            self.assertIn("Change_Analysis", res.target_tools)
            self.assertIn("Anomaly_Extraction", res.target_tools)
            print(f"Query '{q}' -> Tools: {res.target_tools}")

    # 15. Multi-Tool Execution (Composite Queries)
    def test_multitool_execution(self):
        print("\n--- [TEST 15] Multi-Tool Execution ---")
        # 1. Change + Anomaly + Grounding
        q1 = "Find changed regions and locate buildings"
        r1 = self.classifier.classify(query=q1, has_change_pair=True)
        self.assertEqual(r1.task_type, TaskType.MULTI_TASK)
        self.assertEqual(r1.target_tools, ["Change_Analysis", "Anomaly_Extraction", "Grounding"])

        # 2. Optical-SAR + Anomaly
        q2 = "Analyze optical and SAR imagery and identify unusual regions"
        r2 = self.classifier.classify(query=q2, has_optical=True, has_sar=True)
        self.assertEqual(r2.task_type, TaskType.MULTI_TASK)
        self.assertIn("Optical_SAR_Analysis", r2.target_tools)
        self.assertIn("Anomaly_Extraction", r2.target_tools)

        # 3. Change + Anomaly + Grounding + VQA
        q3 = "Compare these images, locate buildings, and tell me if there is water"
        r3 = self.classifier.classify(query=q3, has_change_pair=True)
        self.assertEqual(r3.task_type, TaskType.MULTI_TASK)
        self.assertEqual(r3.target_tools, ["Change_Analysis", "Anomaly_Extraction", "Grounding", "VQA"])

        print(f"q1 -> {r1.target_tools}")
        print(f"q2 -> {r2.target_tools}")
        print(f"q3 -> {r3.target_tools}")

    # 16. Execution Trace Stages
    def test_execution_trace(self):
        print("\n--- [TEST 16] Execution Trace Stages ---")
        res = self.controller.handle_request(
            query="Show me the changed regions",
            change_image_a=self.img_base,
            change_image_b=self.img_single_change,
        )
        self.assertEqual(res["status"], "completed")
        stages = [e["stage"] for e in res["execution_trace"]]
        self.assertIn("query_received", stages)
        self.assertIn("classification", stages)
        self.assertIn("tool_selection", stages)
        self.assertIn("tool_execution", stages)
        self.assertIn("synthesis", stages)
        print(f"Recorded stages: {stages}")

    # 17. API Compatibility & Endpoints
    def test_api_compatibility(self):
        print("\n--- [TEST 17] API Compatibility: /api/change-analysis & /api/query ---")
        b64_a = encode_image_b64(self.img_base)
        b64_b = encode_image_b64(self.img_single_change)

        # Test POST /api/change-analysis
        res_chg = self.client.post("/api/change-analysis", json={
            "image_a": b64_a,
            "image_b": b64_b,
            "change_threshold": 0.15,
        })
        self.assertEqual(res_chg.status_code, 200)
        data_chg = res_chg.json()
        self.assertIn("summary", data_chg)
        self.assertIn("changed_fraction", data_chg)
        self.assertIn("overlay_image", data_chg)
        self.assertIn("heatmap_image", data_chg)
        self.assertIn("anomalies", data_chg)
        self.assertIn("anomaly_summary", data_chg)
        self.assertGreaterEqual(len(data_chg["anomalies"]), 1)

        # Test POST /api/query
        res_q = self.client.post("/api/query", json={
            "query": "Show me the changed regions",
            "change_image_a": b64_a,
            "change_image_b": b64_b,
        })
        self.assertEqual(res_q.status_code, 200)
        data_q = res_q.json()
        self.assertIn("evidence", data_q)
        self.assertIn("execution_trace", data_q)
        print(f"API /api/change-analysis returned {len(data_chg['anomalies'])} anomalies; /api/query returned HTTP 200 OK.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
