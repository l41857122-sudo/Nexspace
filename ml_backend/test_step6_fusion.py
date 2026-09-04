"""
test_step6_fusion.py
--------------------
Comprehensive test suite for STEP 6:
Real Optical + SAR Multimodal Feature Fusion Baseline & Agent Controller Integration.
"""

import sys
import os
import unittest
import time
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from model_runtime import OpticalSARFusionRuntime, BaseModelRuntime
from tools import OpticalSARAnalysisTool, ToolRegistry
from orchestrator import GeoVLMController
from router import IntentClassifier, TaskType
from server import app, encode_image_b64
from fastapi.testclient import TestClient


class TestStep6OpticalSARFusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.img_optical = Image.open(os.path.join(_dir, "sample_satellite.png"))
        cls.img_sar = Image.open(os.path.join(_dir, "sample_sar.png"))
        cls.controller = GeoVLMController()
        cls.classifier = IntentClassifier()
        cls.client = TestClient(app)

    # 1. Optical + SAR Input Validation
    def test_input_validation(self):
        print("\n--- [TEST 1] Optical + SAR Input Validation ---")
        runtime = OpticalSARFusionRuntime()
        # Invalid types must raise ValueError
        with self.assertRaises(ValueError):
            runtime.infer("not_an_image", self.img_sar) # type: ignore
        with self.assertRaises(ValueError):
            runtime.infer(self.img_optical, None) # type: ignore
        print("Input validation passed: rejects non-PIL or missing images.")

    # 2. Dimension Mismatch Handling & 3. Alignment Status Reporting
    def test_dimension_mismatch_and_alignment(self):
        print("\n--- [TEST 2 & 3] Dimension Mismatch & Alignment Status ---")
        runtime = OpticalSARFusionRuntime()
        # Create a mismatched 256x256 SAR image
        sar_mismatched = self.img_sar.resize((256, 256))
        inf = runtime.infer(self.img_optical, sar_mismatched)

        self.assertEqual(inf["fusion"]["alignment_status"], "dimension_mismatch_rescaled")
        self.assertTrue(inf["fusion"]["alignment_warning"])

        # Test matching dimensions
        inf_matched = runtime.infer(self.img_optical, self.img_sar)
        self.assertEqual(inf_matched["fusion"]["alignment_status"], "dimension_match_only")
        self.assertFalse(inf_matched["fusion"]["alignment_warning"])
        print(f"Mismatched Alignment: {inf['fusion']['alignment_status']} (warning={inf['fusion']['alignment_warning']})")
        print(f"Matched Alignment: {inf_matched['fusion']['alignment_status']} (warning={inf_matched['fusion']['alignment_warning']})")

    # 4. Optical Feature Extraction & 5. SAR Feature Extraction
    def test_feature_extraction(self):
        print("\n--- [TEST 4 & 5] Optical & SAR Feature Extraction ---")
        runtime = OpticalSARFusionRuntime()
        inf = runtime.infer(self.img_optical, self.img_sar)
        fusion = inf["fusion"]

        self.assertEqual(fusion["optical_feature_dim"], 768)
        self.assertEqual(fusion["sar_feature_dim"], 768)
        self.assertIn("vision_model", fusion["optical_encoder"])
        self.assertIn("generic_vision_encoder_baseline", fusion["sar_encoder"])
        print(f"Optical Feature Dim: {fusion['optical_feature_dim']}, SAR Feature Dim: {fusion['sar_feature_dim']}")

    # 6. Actual Numerical Fusion & 7. Fused Output Dimension Validation
    def test_numerical_fusion_and_dimensions(self):
        print("\n--- [TEST 6 & 7] Numerical Fusion & Output Dimensions ---")
        runtime = OpticalSARFusionRuntime()
        inf = runtime.infer(self.img_optical, self.img_sar)
        fusion = inf["fusion"]
        analysis = inf["analysis"]

        # Fused vector dimension must equal optical_dim + sar_dim = 1536
        self.assertEqual(fusion["fused_feature_dim"], 1536)
        self.assertIsInstance(fusion["cross_modal_cosine_similarity"], float)
        self.assertIsInstance(fusion["cross_modal_discrepancy_norm"], float)
        self.assertGreaterEqual(fusion["cross_modal_discrepancy_norm"], 0.0)

        # Cross-modal metrics
        metrics = analysis["cross_modal_metrics"]
        self.assertTrue(metrics["joint_representation_available"])
        self.assertIsInstance(metrics["spatial_pearson_correlation"], float)
        print(f"Fused Dim: {fusion['fused_feature_dim']}, Cosine Sim: {fusion['cross_modal_cosine_similarity']}, Spatial Corr: {metrics['spatial_pearson_correlation']}")

    # 8. No String Concatenation Masquerading as Fusion
    def test_no_string_concatenation_masquerade(self):
        print("\n--- [TEST 8] No String Concatenation Masquerade ---")
        runtime = OpticalSARFusionRuntime()
        inf = runtime.infer(self.img_optical, self.img_sar)
        fusion = inf["fusion"]

        self.assertEqual(fusion["fusion_type"], "feature_fusion_baseline")
        self.assertFalse(fusion["is_trained_fusion_model"])
        self.assertIn("optical_feature_dim", fusion)
        self.assertIn("sar_feature_dim", fusion)
        print(f"Fusion Type explicitly labeled as '{fusion['fusion_type']}', is_trained={fusion['is_trained_fusion_model']}")

    # 9. Fusion Failure Handling
    def test_fusion_failure_handling(self):
        print("\n--- [TEST 9] Fusion Failure Handling ---")
        class MockUnavailableFusionRuntime(BaseModelRuntime):
            def __init__(self):
                super().__init__(model_id="mock/unavailable-fusion", task_name="Optical_SAR_Fusion")
                self.load_error = "Mock backbone unavailable"
                self.is_loaded = False
            def _do_load(self):
                return False
            def is_available(self):
                return False

        tool = OpticalSARAnalysisTool(fusion_runtime=MockUnavailableFusionRuntime())
        res = tool.execute({"optical_image": self.img_optical, "sar_image": self.img_sar})

        self.assertEqual(res.status, "unavailable")
        self.assertEqual(res.evidence, [])
        self.assertEqual(res.data["fusion"]["status"], "unavailable")
        print(f"Unavailable Status: {res.status}, Fusion Status: {res.data['fusion']['status']}")

    # 10. Controller Routing for Optical + SAR Analysis
    def test_controller_routing_optical_sar(self):
        print("\n--- [TEST 10] Controller Routing: Optical + SAR ---")
        queries = [
            "Analyze the optical and SAR images together",
            "Compare optical and SAR imagery",
            "Use both optical and SAR data",
            "Analyze this scene using optical and SAR",
        ]
        for q in queries:
            res = self.classifier.classify(query=q, has_optical=True, has_sar=True)
            self.assertEqual(res.task_type, TaskType.OPTICAL_SAR_ANALYSIS)
            self.assertEqual(res.target_tools, ["Optical_SAR_Analysis"])
            print(f"Query '{q}' -> Task: {res.task_type}, Tools: {res.target_tools}")

    # 11. Multi-Tool Compatibility: Optical_SAR_Analysis + Grounding + VQA
    def test_multitool_compatibility(self):
        print("\n--- [TEST 11] Multi-Tool Planning: Fusion + Grounding + VQA ---")
        # 1. Fusion + Grounding
        q1 = "Analyze optical and SAR imagery and locate buildings"
        r1 = self.classifier.classify(query=q1, has_optical=True, has_sar=True)
        self.assertEqual(r1.task_type, TaskType.MULTI_TASK)
        self.assertEqual(r1.target_tools, ["Optical_SAR_Analysis", "Grounding"])

        # 2. Fusion + Grounding + VQA
        q2 = "Analyze optical and SAR imagery, locate buildings, and tell me if there is water"
        r2 = self.classifier.classify(query=q2, has_optical=True, has_sar=True)
        self.assertEqual(r2.task_type, TaskType.MULTI_TASK)
        self.assertEqual(r2.target_tools, ["Optical_SAR_Analysis", "Grounding", "VQA"])

        print(f"q1 -> {r1.target_tools}")
        print(f"q2 -> {r2.target_tools}")

    # 12. Confidence Provenance
    def test_confidence_provenance(self):
        print("\n--- [TEST 12] Confidence Provenance ---")
        tool = OpticalSARAnalysisTool()
        res = tool.execute({"optical_image": self.img_optical, "sar_image": self.img_sar})
        self.assertIsNone(res.confidence)
        self.assertEqual(res.confidence_type, "unavailable")
        self.assertEqual(res.confidence_source, "optical_sar_feature_fusion_baseline")
        print(f"Confidence: {res.confidence}, Type: {res.confidence_type}, Source: {res.confidence_source}")

    # 13. Execution Telemetry & Traces
    def test_execution_telemetry_trace(self):
        print("\n--- [TEST 13] Execution Telemetry Trace ---")
        res = self.controller.handle_request(
            query="Analyze optical and SAR imagery",
            optical_image=self.img_optical,
            sar_image=self.img_sar,
        )
        self.assertEqual(res["status"], "completed")
        stages = [e["stage"] for e in res["execution_trace"]]
        self.assertIn("query_received", stages)
        self.assertIn("classification", stages)
        self.assertIn("tool_selection", stages)
        self.assertIn("tool_execution", stages)
        self.assertIn("synthesis", stages)
        self.assertIn("completed", stages)
        print(f"Execution stages: {stages}")

    # 14. API Backward Compatibility
    def test_api_compatibility(self):
        print("\n--- [TEST 14] API Backward Compatibility ---")
        b64_opt = encode_image_b64(self.img_optical)
        b64_sar = encode_image_b64(self.img_sar)

        res = self.client.post("/api/query", json={
            "query": "Analyze the optical and SAR images together",
            "optical_image": b64_opt,
            "sar_image": b64_sar,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["task_type"], "OPTICAL_SAR_ANALYSIS")
        self.assertEqual(data["selected_tools"], ["Optical_SAR_Analysis"])
        self.assertIn("evidence", data)
        self.assertIn("execution_trace", data)
        self.assertIn("Multimodal Fusion", data["response_text"])
        print(f"API HTTP 200: Task={data['task_type']}, Tools={data['selected_tools']}, Evidence items={len(data['evidence'])}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
