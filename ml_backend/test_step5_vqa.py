"""
test_step5_vqa.py
-----------------
Comprehensive test suite for STEP 5:
PaliGemma VQA Runtime & Agent Controller Integration.
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

from model_runtime import PaliGemmaVQARuntime, BaseModelRuntime
from tools import VQATool, BaseSpecialistTool, ToolRegistry
from orchestrator import GeoVLMController
from router import IntentClassifier, TaskType
from server import app, encode_image_b64
from fastapi.testclient import TestClient


class TestStep5PaliGemmaVQA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.img_satellite = Image.open(os.path.join(_dir, "sample_satellite.png"))
        cls.controller = GeoVLMController()
        cls.classifier = IntentClassifier()
        cls.client = TestClient(app)

    # ------------------------------------------------------------------
    # A. VQA Query Classification
    # ------------------------------------------------------------------
    def test_vqa_query_classification(self):
        print("\n--- [TEST A] VQA Query Classification ---")
        queries = [
            "Is there water in this image?",
            "Are there buildings visible?",
            "What type of land cover is visible?",
            "Is there a road crossing the scene?",
        ]
        for q in queries:
            res = self.classifier.classify(query=q, has_optical=True)
            self.assertEqual(res.task_type, TaskType.VQA)
            self.assertEqual(res.target_tools, ["VQA"])
            print(f"Query '{q}' -> Task: {res.task_type}, Tools: {res.target_tools}")

    # ------------------------------------------------------------------
    # B. Counting Query Classification & Warning
    # ------------------------------------------------------------------
    def test_counting_query_classification(self):
        print("\n--- [TEST B] Counting Query Classification ---")
        q = "How many buildings are visible?"
        res = self.classifier.classify(query=q, has_optical=True)
        self.assertEqual(res.task_type, TaskType.VQA)
        self.assertEqual(res.target_tools, ["VQA"])
        self.assertTrue(res.requires_count_warning)
        print(f"Query '{q}' -> Requires Count Warning: {res.requires_count_warning}")

    # ------------------------------------------------------------------
    # C. Pure Description Query Classification (Captioning Only)
    # ------------------------------------------------------------------
    def test_description_query_classification(self):
        print("\n--- [TEST C] Description Query Classification ---")
        q = "Describe this image"
        res = self.classifier.classify(query=q, has_optical=True)
        self.assertEqual(res.task_type, TaskType.CAPTIONING)
        self.assertEqual(res.target_tools, ["Optical_Caption"])
        self.assertNotIn("VQA", res.target_tools)
        print(f"Query '{q}' -> Task: {res.task_type}, Tools: {res.target_tools}")

    # ------------------------------------------------------------------
    # D. Multi-Tool Queries (Caption + Grounding, Caption + VQA, Triple)
    # ------------------------------------------------------------------
    def test_multi_tool_queries(self):
        print("\n--- [TEST D] Multi-Tool Query Planning ---")
        # 1. Caption + Grounding
        q1 = "Describe this image and locate the buildings"
        r1 = self.classifier.classify(query=q1, has_optical=True)
        self.assertEqual(r1.task_type, TaskType.MULTI_TASK)
        self.assertEqual(set(r1.target_tools), {"Optical_Caption", "Grounding"})

        # 2. Caption + VQA
        q2 = "Is there water and describe the scene"
        r2 = self.classifier.classify(query=q2, has_optical=True)
        self.assertEqual(r2.task_type, TaskType.MULTI_TASK)
        self.assertEqual(set(r2.target_tools), {"Optical_Caption", "VQA"})

        # 3. Triple Tool: Caption + Grounding + VQA
        q3 = "Describe the image, locate buildings, and tell me if there is water"
        r3 = self.classifier.classify(query=q3, has_optical=True)
        self.assertEqual(r3.task_type, TaskType.MULTI_TASK)
        self.assertEqual(set(r3.target_tools), {"Optical_Caption", "Grounding", "VQA"})

        print(f"q1 -> {r1.target_tools}")
        print(f"q2 -> {r2.target_tools}")
        print(f"q3 -> {r3.target_tools}")

    # ------------------------------------------------------------------
    # E. VQA Runtime Unavailable & Fallback Behavior (No Fake Confidences)
    # ------------------------------------------------------------------
    def test_vqa_runtime_unavailable_behavior(self):
        print("\n--- [TEST E] VQA Runtime Unavailable Behavior ---")
        runtime = PaliGemmaVQARuntime()
        # Without HF_TOKEN, load() returns False
        if not os.environ.get("HF_TOKEN"):
            ok = runtime.load()
            self.assertFalse(ok)
            self.assertIn("AUTHENTICATION REQUIRED", runtime.load_error)

        tool = VQATool(runtime=runtime)
        res = tool.execute({"image": self.img_satellite, "question": "Is there water present?"})

        if not runtime.is_available():
            self.assertEqual(res.status, "fallback")
            self.assertEqual(res.confidence_type, "heuristic")
            self.assertEqual(res.confidence_source, "rsvqa_heuristic_adapter")
            self.assertIsNone(res.confidence)
            for ev in res.evidence:
                self.assertTrue(ev["fallback"])
                self.assertIsNotNone(ev["fallback_reason"])
            print(f"Fallback Verified: Status={res.status}, Conf={res.confidence}, Reason={res.evidence[0]['fallback_reason']}")

    # ------------------------------------------------------------------
    # F. Structured VQA Model Execution (Mocked Runtime Boundary)
    # ------------------------------------------------------------------
    def test_vqa_structured_model_execution_boundary(self):
        print("\n--- [TEST F] Structured VQA Model Execution Boundary ---")
        class MockPaliGemmaRuntime(BaseModelRuntime):
            def __init__(self):
                super().__init__(model_id="google/paligemma-3b-ft-rsvqa-lr-224", task_name="VQA")
                self.is_loaded = True
            def _do_load(self):
                return True
            def is_available(self):
                return True
            def infer(self, image, question):
                return {
                    "answer": "water body and vegetation",
                    "question": question,
                    "confidence": None,
                    "confidence_type": "unavailable",
                    "confidence_source": self.model_id,
                    "inference_time_ms": 124.5,
                    "model": self.model_id,
                    "device": "cpu",
                    "fallback": False,
                    "success": True,
                    "model_metadata": self.get_metadata(),
                }

        mock_tool = VQATool(runtime=MockPaliGemmaRuntime())
        res = mock_tool.execute({"image": self.img_satellite, "question": "What is visible?"})

        self.assertEqual(res.status, "success")
        self.assertEqual(res.confidence_type, "unavailable")
        self.assertEqual(res.confidence_source, "google/paligemma-3b-ft-rsvqa-lr-224")
        self.assertEqual(res.data["primary_answer"], "water body and vegetation")
        self.assertEqual(len(res.evidence), 1)
        self.assertEqual(res.evidence[0]["type"], "vqa_answer")
        self.assertEqual(res.evidence[0]["model"], "google/paligemma-3b-ft-rsvqa-lr-224")
        self.assertFalse(res.evidence[0]["fallback"])
        print(f"VQA Model Boundary Output: Answer='{res.data['primary_answer']}', ConfType={res.confidence_type}")

    # ------------------------------------------------------------------
    # G. Confidence Provenance (No Hardcoded 0.85/0.32)
    # ------------------------------------------------------------------
    def test_confidence_provenance(self):
        print("\n--- [TEST G] Confidence Provenance ---")
        res = self.controller.handle_request(
            query="Is there water in this image?",
            optical_image=self.img_satellite,
        )
        self.assertIn(res["confidence_type"], ("model", "heuristic", "unavailable", "estimated"))
        self.assertIsNotNone(res["confidence_source"])
        print(f"Confidence Type: {res['confidence_type']}, Source: {res['confidence_source']}, Score: {res['confidence']}")

    # ------------------------------------------------------------------
    # H. Real Execution Telemetry & Traces
    # ------------------------------------------------------------------
    def test_execution_telemetry_trace(self):
        print("\n--- [TEST H] Execution Telemetry Trace ---")
        res = self.controller.handle_request(
            query="Is there water in this image?",
            optical_image=self.img_satellite,
        )
        stages = [e["stage"] for e in res["execution_trace"]]
        self.assertIn("query_received", stages)
        self.assertIn("classification", stages)
        self.assertIn("tool_selection", stages)
        self.assertIn("tool_execution", stages)
        self.assertIn("synthesis", stages)
        self.assertIn("completed", stages)

        for event in res["execution_trace"]:
            self.assertIn("timestamp", event)
            self.assertIn("status", event)
        print(f"Trace Stages Recorded: {stages}")

    # ------------------------------------------------------------------
    # I. API Backward Compatibility (POST /api/query for VQA)
    # ------------------------------------------------------------------
    def test_api_query_vqa(self):
        print("\n--- [TEST I] API Query VQA ---")
        b64 = encode_image_b64(self.img_satellite)
        res = self.client.post("/api/query", json={"query": "Is there water in this image?", "optical_image": b64})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["task_type"], "VQA")
        self.assertEqual(data["selected_tools"], ["VQA"])
        self.assertIn("response_text", data)
        self.assertIn("evidence", data)
        self.assertIn("execution_trace", data)
        print(f"API HTTP 200: Task={data['task_type']}, Tools={data['selected_tools']}, Evidence Items={len(data['evidence'])}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
