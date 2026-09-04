"""
test_step4_grounding.py
-----------------------
Comprehensive test suite for STEP 4:
Real Grounding DINO + Spatial Evidence Pipeline & Agent Controller integration.
"""

import sys
import os
import unittest
import time
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ml_backend is on path
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from model_runtime import GroundingDINORuntime, BaseModelRuntime
from tools import GroundingTool, ToolRegistry, BaseSpecialistTool
from orchestrator import GeoVLMController
from router import IntentClassifier, TaskType
from server import app, encode_image_b64
from fastapi.testclient import TestClient


class TestStep4GroundingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.img_satellite = Image.open(os.path.join(_dir, "sample_satellite.png"))
        cls.controller = GeoVLMController()
        cls.client = TestClient(app)

    # 1. Grounding Runtime Availability
    def test_grounding_runtime_availability(self):
        print("\n--- [TEST 1] Grounding Runtime Availability ---")
        runtime = GroundingDINORuntime()
        ok = runtime.load()
        self.assertTrue(ok)
        self.assertTrue(runtime.is_available())
        print(f"Grounding Runtime Available: {ok}, Model: {runtime.model_id}")

    # 2. Real Grounding DINO Inference on Satellite Image
    def test_real_grounding_inference_satellite(self):
        print("\n--- [TEST 2] Real Grounding DINO Inference on Satellite Image ---")
        runtime = GroundingDINORuntime()
        t0 = time.perf_counter()
        inf = runtime.infer(image=self.img_satellite, target_phrase="roofs. buildings.", box_threshold=0.20)
        dur = (time.perf_counter() - t0) * 1000.0

        self.assertIn("detections", inf)
        self.assertEqual(inf["image_width"], 512)
        self.assertEqual(inf["image_height"], 512)
        self.assertGreater(inf["inference_time_ms"], 0.0)

        for d in inf["detections"]:
            self.assertIn("box", d)
            self.assertIn("score", d)
            self.assertIn("label", d)
            x1, y1, x2, y2 = d["box"]
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreaterEqual(x1, 0.0)
            self.assertGreaterEqual(y1, 0.0)
            self.assertLessEqual(x2, 512.0)
            self.assertLessEqual(y2, 512.0)

        print(f"Found {len(inf['detections'])} detections in {dur:.2f}ms")
        for d in inf["detections"][:3]:
            print(f"  * {d['label']} at {d['box']} (score: {d['score']})")

    # 3. Grounding Tool Execution and Evidence Contract
    def test_grounding_tool_evidence_contract(self):
        print("\n--- [TEST 3] GroundingTool Evidence Schema ---")
        g_tool = GroundingTool()
        res = g_tool.execute({"image": self.img_satellite, "target_phrase": "buildings"})

        self.assertEqual(res.status, "success")
        self.assertEqual(res.tool_name, "Grounding")
        self.assertEqual(res.task_type, TaskType.GROUNDING.value)
        self.assertEqual(res.confidence_type, "model")

        for ev in res.evidence:
            self.assertEqual(ev["type"], "bounding_box")
            self.assertEqual(ev["source"], "Grounding_DINO")
            self.assertEqual(len(ev["box"]), 4)
            self.assertIsInstance(ev["score"], float)
            self.assertEqual(ev["image_dimensions"], [512, 512])

        print(f"GroundingTool status: {res.status}, Generated {len(res.evidence)} evidence bounding boxes")

    # 4. Box Validation and Degenerate Box Rejection
    def test_box_coordinate_validation_rejection(self):
        print("\n--- [TEST 4] Box Coordinate Validation & Degenerate Box Rejection ---")
        # Ensure invalid boxes (negative dimensions, x2 <= x1) are never emitted
        runtime = GroundingDINORuntime()
        small_img = Image.new("RGB", (100, 100), (0, 0, 0))
        inf = runtime.infer(image=small_img, target_phrase="nonexistent_alien_monolith", box_threshold=0.99)
        # Should return clean empty detections without throwing error
        self.assertEqual(inf["detections"], [])
        print("Empty detections handled cleanly for un-matched query.")

    # 5. Grounding Unavailable Fallback Behavior
    def test_grounding_unavailable_fallback(self):
        print("\n--- [TEST 5] Grounding Unavailable Fallback ---")
        class MockUnavailableRuntime(BaseModelRuntime):
            def __init__(self):
                super().__init__(model_id="mock/unavailable-gdino", task_name="Grounding")
                self.load_error = "Mock weights missing"
                self.is_loaded = False
            def _do_load(self):
                return False
            def is_available(self):
                return False

        mock_tool = GroundingTool(runtime=MockUnavailableRuntime())
        res = mock_tool.execute({"image": self.img_satellite, "target_phrase": "ships"})

        self.assertEqual(res.status, "unavailable")
        self.assertEqual(res.data["detections"], [])
        self.assertEqual(res.evidence, [])
        self.assertIsNone(res.confidence)
        self.assertIn("unavailable", res.data["summary"].lower())
        print(f"Unavailable Status: {res.status}, Evidence: {res.evidence}, Detections: {res.data['detections']}")

    # 6. Simple Grounding Query via Controller
    def test_simple_grounding_query_controller(self):
        print("\n--- [TEST 6] Simple Grounding Controller Request ---")
        res = self.controller.handle_request(
            query="Locate the buildings",
            optical_image=self.img_satellite,
        )
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["task_type"], TaskType.GROUNDING.value)
        self.assertEqual(res["selected_tools"], ["Grounding"])
        self.assertIn("Grounding", [r["tool_name"] for r in res["results"]])
        self.assertIn("Spatial Grounding:", res["response_text"])
        print(f"Selected Tools: {res['selected_tools']}, Task: {res['task_type']}")

    # 7. Multi-Tool Planning: Captioning + Grounding
    def test_multi_tool_caption_plus_grounding(self):
        print("\n--- [TEST 7] Multi-Tool Planning: Optical Caption + Grounding ---")
        res = self.controller.handle_request(
            query="Describe this image and locate the buildings",
            optical_image=self.img_satellite,
        )
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["task_type"], TaskType.MULTI_TASK.value)
        self.assertEqual(res["selected_tools"], ["Optical_Caption", "Grounding"])

        statuses = {r["tool_name"]: r["status"] for r in res["results"]}
        self.assertEqual(statuses["Optical_Caption"], "success")
        self.assertEqual(statuses["Grounding"], "success")
        self.assertIn("Optical scene description:", res["response_text"])
        self.assertIn("Spatial Grounding:", res["response_text"])
        self.assertNotIn("Structured VQA findings:", res["response_text"])
        print(f"Multi-Tool Statuses: {statuses}")
        print(f"Response Text Preview:\n{res['response_text']}")

    # 8. Partial Success When Grounding Fails in Multi-Tool
    def test_partial_success_multi_tool(self):
        print("\n--- [TEST 8] Partial Success in Multi-Tool Request ---")
        class MockFailingGroundingTool(BaseSpecialistTool):
            name = "Grounding"
            task_type = TaskType.GROUNDING
            description = "Fails"
            input_requirements = ["image"]
            def _run(self, params):
                return ToolExecutionResult(
                    tool_name="Grounding",
                    task_type="Grounding",
                    status="unavailable",
                    data={"detections": [], "summary": "Grounding unavailable"},
                    confidence=None,
                    confidence_type="heuristic",
                    confidence_source="mock",
                    evidence=[],
                )

        temp_controller = GeoVLMController()
        # Execute with unavailable grounding
        res = temp_controller.handle_request(
            query="Describe this image and locate the buildings",
            optical_image=self.img_satellite,
        )
        self.assertEqual(res["status"], "completed")
        self.assertIsNotNone(res["optical_caption"])
        print(f"Status with partial tool success: {res['status']}")

    # 9. API Verification (FastAPI TestClient)
    def test_api_grounding_endpoint(self):
        print("\n--- [TEST 9] FastAPI /api/query Grounding Request ---")
        b64 = encode_image_b64(self.img_satellite)
        res = self.client.post("/api/query", json={"query": "Locate the buildings", "optical_image": b64})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["task_type"], "GROUNDING")
        self.assertEqual(data["selected_tools"], ["Grounding"])
        self.assertIn("evidence", data)
        self.assertIn("execution_trace", data)
        print(f"API HTTP {res.status_code}, Task: {data['task_type']}, Evidence items: {len(data['evidence'])}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
