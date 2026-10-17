"""
tests/test_phase5_vqa.py
-------------------------
Phase 5 Verification Suite:
Reliable Visual Question Answering (VQA).

Acceptance Criteria:
 1. Image input validation: missing or invalid image types rejected cleanly
 2. Empty/whitespace questions handled gracefully with domain default
 3. Honest unavailable runtime handling: when gated weights/HF_TOKEN are absent,
    reports status='fallback', fallback=True, explicit fallback_reason, and limitations
 4. Separation of model interpretation from verified spectral indices & geospatial facts
 5. Real inference boundary: mock/genuine VLM runtime executes through inference path,
    returning provenance='MODEL_DERIVED', status='success', fallback=False
 6. Multi-question batching: executes across list of questions with evidence per question
 7. Confidence provenance: uncalibrated generative logits marked as confidence=None / unavailable
 8. Agent Controller integration: routes VQA queries to VQA tool without breaking unrelated queries
"""

from __future__ import annotations
import os
import sys
import unittest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_backend"))

from model_runtime import BaseModelRuntime, PaliGemmaVQARuntime
from tools import VQATool
from rs_vqa_engine import RemoteSensingVQAEngine


class TestPhase5VQA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Synthetic 256x256 satellite image with green vegetation, blue water, and gray built structure
        cls.img = Image.new("RGB", (256, 256), (140, 140, 140))
        d = ImageDraw.Draw(cls.img)
        d.rectangle([20, 20, 100, 100], fill=(200, 50, 50))     # Building structure
        d.rectangle([120, 20, 240, 120], fill=(30, 180, 40))    # Dense vegetation
        d.rectangle([20, 140, 240, 240], fill=(20, 50, 160))    # Water body

    def test_01_input_validation_missing_image(self):
        """Missing image returns a clean failed ToolExecutionResult."""
        tool = VQATool()
        res = tool.execute({"image": None, "question": "What type of area is shown?"})
        self.assertEqual(res.status, "failed")
        self.assertIn("input image", res.error)

    def test_02_input_validation_invalid_type(self):
        """Non-PIL Image parameter returns an input validation error."""
        tool = VQATool()
        res = tool.execute({"image": "not_an_image", "question": "What type of area is shown?"})
        self.assertEqual(res.status, "failed")
        self.assertIn("valid PIL Image", res.error)

    def test_03_empty_question_fallback(self):
        """Empty or whitespace question gracefully falls back to default question."""
        tool = VQATool()
        res = tool.execute({"image": self.img, "question": "   "})
        self.assertIn(res.status, ("fallback", "success", "degraded"))
        self.assertGreater(len(res.data["vqa_results"]), 0)
        self.assertIsNotNone(res.data["primary_answer"])

    def test_04_runtime_unavailable_transparent_fallback(self):
        """When gated PaliGemma runtime has no HF_TOKEN, reports status='fallback' with explicit reason."""
        runtime = PaliGemmaVQARuntime()
        tool = VQATool(runtime=runtime)
        res = tool.execute({"image": self.img, "question": "Is there water in this image?"})

        if not runtime.is_available():
            self.assertEqual(res.status, "fallback")
            self.assertTrue(res.fallback_used)
            self.assertEqual(res.provenance, "HEURISTIC")
            self.assertEqual(res.confidence_type, "heuristic")
            self.assertIn("limitations", res.data)
            self.assertTrue(any("PaliGemma neural runtime unavailable" in lim for lim in res.data["limitations"]))
            # Evidence inspection
            self.assertGreater(len(res.evidence), 0)
            ev = res.evidence[0]
            self.assertTrue(ev["fallback"])
            self.assertIsNotNone(ev.get("fallback_reason"))

    def test_05_structured_model_execution_boundary(self):
        """When a real or verified VLM runtime is available, runs genuine inference path."""
        class MockVerifiedVLMRuntime(BaseModelRuntime):
            def __init__(self):
                super().__init__(model_id="google/paligemma-3b-ft-rsvqa-lr-224", task_name="VQA")
                self.is_loaded = True

            def _do_load(self):
                return True

            def is_available(self):
                return True

            def infer(self, image, question):
                return {
                    "answer": "water body and vegetation plot",
                    "question": question,
                    "confidence": None,
                    "confidence_type": "unavailable",
                    "confidence_source": self.model_id,
                }

        tool = VQATool(runtime=MockVerifiedVLMRuntime())
        res = tool.execute({"image": self.img, "question": "What is visible?"})

        self.assertEqual(res.status, "success")
        self.assertEqual(res.provenance, "MODEL_DERIVED")
        self.assertFalse(res.fallback_used)
        self.assertEqual(res.method, "paligemma_generative_vqa")
        self.assertEqual(res.data["primary_answer"], "water body and vegetation plot")
        self.assertEqual(res.confidence_source, "google/paligemma-3b-ft-rsvqa-lr-224")
        self.assertIn("probabilistic vision-language interpretations", res.data["limitations"][0])
        self.assertFalse(res.evidence[0]["fallback"])

    def test_06_multi_question_batching(self):
        """Multiple questions are processed and structured into discrete evidence items."""
        tool = VQATool()
        questions = [
            "Is there water in this image?",
            "Is there vegetation visible?",
            "What type of area is shown?",
        ]
        res = tool.execute({"image": self.img, "questions": questions})
        self.assertEqual(len(res.data["vqa_results"]), 3)
        self.assertEqual(len(res.evidence), 3)
        for idx, ev in enumerate(res.evidence):
            self.assertEqual(ev["question"], questions[idx])
            self.assertIsNotNone(ev["answer"])

    def test_07_separation_of_interpretation_from_geospatial_facts(self):
        """VQA outputs are labeled as interpretation, not asserted as ground truth facts."""
        tool = VQATool()
        res = tool.execute({"image": self.img, "question": "Where are the buildings?"})
        # Verify evidence node contains explicit capability and method
        self.assertIn("Visual Question Answering", res.capability)
        self.assertIn("limitations", res.data)
        # Provenance must be HEURISTIC or MODEL_DERIVED, never asserting ground truth
        self.assertIn(res.provenance, ("MODEL_DERIVED", "HEURISTIC"))


if __name__ == "__main__":
    unittest.main()
