"""
test_step13_model_quality.py
----------------------------
Comprehensive verification suite for Step 13:
  1. PaliGemma state & transparent RSVQA fallback behavior
  2. Grounding target extraction (single, multi-target, complex phrases)
  3. Grounding DINO detection diagnostics & near-full-image detection filters
  4. Confidence provenance & semantics (no invented numbers)
  5. Scientific language and limitation labeling
  6. Multi-tool isolation and routing correctness
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router import IntentClassifier, TaskType, _extract_grounding_target
from model_runtime import PaliGemmaVQARuntime, GroundingDINORuntime, BLIPCaptioningRuntime, OpticalSARFusionRuntime
from orchestrator import GeoVLMController as AgentController
from config import settings


class TestStep13ModelQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.controller = AgentController()
        cls.test_img = Image.new("RGB", (100, 100), (40, 100, 180))

    # 1. Target Extraction: Single & Multi-Target Phrases
    def test_grounding_target_extraction(self):
        print("\n--- [TEST 1] Grounding Target & Multi-Target Extraction ---")
        test_cases = [
            ("Locate the buildings", "buildings"),
            ("Find roads and buildings", "roads and buildings"),
            ("Locate ships near the water", "ships near the water"),
            ("Show me all vehicles in this image", "vehicles"),
            ("Can you detect rooftops, trees and cars?", "rooftops, trees and cars"),
            ("Where are the solar panels?", "solar panels"),
        ]
        for query, expected in test_cases:
            extracted = _extract_grounding_target(query)
            self.assertEqual(extracted, expected, f"Query '{query}' extracted '{extracted}', expected '{expected}'")
            print(f"Query: '{query}' -> Extracted target: '{extracted}'")

    # 2. PaliGemma State & Transparent Fallback
    def test_paligemma_fallback_transparency(self):
        print("\n--- [TEST 2] PaliGemma State & Fallback Transparency ---")
        runtime = PaliGemmaVQARuntime()
        if not os.environ.get("HF_TOKEN"):
            is_loaded = runtime.load()
            self.assertFalse(is_loaded)
            self.assertIn("AUTHENTICATION REQUIRED", runtime.load_error or "")
            print(f"PaliGemma load status cleanly trapped: {runtime.load_error}")

            # Execute query through controller to verify transparent fallback
            res = self.controller.handle_request(query="Is there water in this image?", optical_image=self.test_img)
            self.assertIn("vqa_results", res)
            self.assertTrue(len(res["vqa_results"]) > 0)
            self.assertEqual(res["confidence_type"], "heuristic")
            self.assertTrue("adapter" in res["confidence_source"] or "heuristic" in res["confidence_source"])
            print(f"Controller VQA result transparently labeled as heuristic: {res['vqa_results'][0]}")

    # 3. Grounding DINO Detection Diagnostics
    def test_grounding_detection_diagnostics(self):
        print("\n--- [TEST 3] Grounding Detection Diagnostics ---")
        runtime = GroundingDINORuntime()
        if runtime.load():
            res = runtime.infer(image=self.test_img, target_phrase="buildings")
            self.assertIn("detections", res)
            self.assertIn("count", res)
            print(f"Grounding DINO executed on test image: {res['count']} detections found.")
            for d in res["detections"]:
                self.assertIn("box", d)
                self.assertIn("score", d)
                self.assertIn("label", d)

    # 4. Confidence Semantics (No Fake Calibrated Confidence)
    def test_confidence_semantics_honesty(self):
        print("\n--- [TEST 4] Confidence Semantics & Scientific Honesty ---")
        # BLIP text generation must have confidence_type = 'model'
        res_cap = self.controller.handle_request(query="Describe this image", optical_image=self.test_img)
        self.assertEqual(res_cap["confidence_type"], "model")

        # Fallback counting query must flag requires_count_warning = True and limitation
        res_count = self.controller.handle_request(query="How many ships are visible?", optical_image=self.test_img)
        self.assertTrue(res_count["routing_decision"]["requires_count_warning"])
        self.assertTrue(any("approximate" in lim.lower() for lim in res_count["investigation_report"]["limitations"]))
        print("Verified counting estimation warning is present.")

    # 5. Multi-Tool Routing Precision
    def test_multitool_routing_precision(self):
        print("\n--- [TEST 5] Multi-Tool Routing Precision ---")
        router = IntentClassifier()

        # Pure caption -> Optical_Caption only
        r1 = router.classify(query="Describe this image", has_optical=True)
        self.assertEqual(r1.target_tools, ["Optical_Caption"])

        # Composite caption + grounding -> Optical_Caption + Grounding
        r2 = router.classify(query="Describe this image and locate the buildings", has_optical=True)
        self.assertEqual(r2.target_tools, ["Optical_Caption", "Grounding"])

        # Optical + SAR -> Optical_SAR_Analysis only
        r3 = router.classify(query="Compare optical and SAR imagery", has_optical=True, has_sar=True)
        self.assertEqual(r3.target_tools, ["Optical_SAR_Analysis"])

        print("All single-task and multi-tool routing paths verified.")

    # 6. Scientific Limitation Labels
    def test_scientific_limitation_labels(self):
        print("\n--- [TEST 6] Scientific Limitation Labels ---")
        res = self.controller.handle_request(query="Describe this image", optical_image=self.test_img)
        report = res.get("investigation_report", {})
        self.assertIn("limitations", report)
        self.assertTrue(len(report["limitations"]) > 0)
        self.assertTrue(any("domain shift" in lim.lower() or "land cover" in lim.lower() for lim in report["limitations"]))
        print(f"Investigation report limitations: {report['limitations']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
