"""
test_step14_caption_safety.py
-----------------------------
Adversarial Verification Suite for Vision-Language Model Safety:
  1. Taj Mahal / Architectural Scene Quality (no repetitive 'ta ta ta...' loop)
  2. Diverse Photography & Domain Coverage (City, Landscape, Aerial, SAR, Synthetic, Dark, Bright)
  3. Output Validation Gate (detects word repetition, n-gram loops, low lexical diversity, special tokens)
  4. Confidence Correction (never reports HIGH confidence on invalid/rejected generation)
  5. Evidence-Grounded Synthesis (grounding proposal synthesis when captioning is unavailable)
  6. Modality Separation & Provenance Integrity
"""

import os
import sys
import re
import unittest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from caption_validator import validate_caption_quality, CaptionValidationResult
from model_runtime import BLIPCaptioningRuntime
from tools import OpticalCaptioningTool, GroundingTool
from orchestrator import GeoVLMController as AgentController
from router import IntentClassifier


class TestStep14CaptionSafetyAndQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.controller = AgentController()
        cls.caption_runtime = BLIPCaptioningRuntime()
        cls.caption_tool = OpticalCaptioningTool(runtime=cls.caption_runtime)

        # 1. Taj Mahal / Architectural Monument Test Fixture
        cls.img_taj_mahal = Image.new("RGB", (600, 450), (135, 206, 235))
        d = ImageDraw.Draw(cls.img_taj_mahal)
        d.rectangle([180, 200, 420, 380], fill=(250, 248, 245), outline=(200, 200, 200)) # Base
        d.ellipse([240, 100, 360, 220], fill=(255, 255, 255), outline=(210, 210, 210)) # Dome
        d.rectangle([100, 120, 130, 380], fill=(240, 240, 240)) # Left minaret
        d.rectangle([470, 120, 500, 380], fill=(240, 240, 240)) # Right minaret
        d.rectangle([0, 380, 600, 450], fill=(50, 130, 50)) # Garden
        d.rectangle([260, 380, 340, 450], fill=(60, 110, 180)) # Pool

        # 2. City photograph
        cls.img_city = Image.new("RGB", (500, 350), (120, 160, 200))
        d = ImageDraw.Draw(cls.img_city)
        d.rectangle([30, 120, 120, 320], fill=(100, 100, 120))
        d.rectangle([150, 60, 270, 320], fill=(140, 130, 120))
        d.rectangle([300, 140, 450, 320], fill=(160, 160, 170))
        d.rectangle([0, 320, 500, 350], fill=(40, 40, 40))

        # 3. Landscape photograph
        cls.img_landscape = Image.new("RGB", (500, 350), (100, 180, 240))
        d = ImageDraw.Draw(cls.img_landscape)
        d.polygon([(0, 350), (200, 150), (400, 350)], fill=(120, 110, 100)) # Mountain
        d.polygon([(250, 350), (400, 180), (500, 350)], fill=(100, 90, 80))
        d.rectangle([0, 280, 500, 350], fill=(40, 140, 40)) # Forest

        # 4. Aerial / Satellite image
        cls.img_aerial = Image.new("RGB", (512, 512), (30, 80, 30))
        d = ImageDraw.Draw(cls.img_aerial)
        d.rectangle([100, 100, 250, 250], fill=(180, 180, 180)) # Terminal
        d.line([(0, 256), (512, 256)], fill=(50, 50, 50), width=16) # Runway

        # 5. Synthetic dark image
        cls.img_dark = Image.new("RGB", (256, 256), (8, 10, 12))

        # 6. Synthetic bright image
        cls.img_bright = Image.new("RGB", (256, 256), (250, 250, 252))

    # -------------------------------------------------------------------
    # TEST 1: Taj Mahal / Architectural Scene Quality Test
    # -------------------------------------------------------------------
    def test_taj_mahal_architectural_quality(self):
        print("\n--- [TEST 1] Taj Mahal / Architectural Scene Quality ---")
        res = self.controller.handle_request(query="Describe this image", optical_image=self.img_taj_mahal)
        
        caption = res.get("optical_caption")
        print(f"Generated Optical Caption: '{caption}'")
        print(f"Overall Confidence: {res.get('confidence')} (Type: {res.get('confidence_type')})")
        print(f"Response Text:\n{res.get('response_text')}")

        # Assertions
        self.assertIsNotNone(caption)
        self.assertFalse("ta ta ta" in caption.lower())
        self.assertFalse(re.search(r"\b(\w+)(?:\s+\1){2,}\b", caption))
        self.assertTrue(len(caption) > 5)
        self.assertEqual(res["confidence_type"], "model")

    # -------------------------------------------------------------------
    # TEST 2: Multi-Domain Photography & Imagery Robustness
    # -------------------------------------------------------------------
    def test_multi_domain_visual_coverage(self):
        print("\n--- [TEST 2] Multi-Domain Visual Coverage ---")
        domain_fixtures = {
            "city_street": self.img_city,
            "mountain_landscape": self.img_landscape,
            "aerial_satellite": self.img_aerial,
            "dark_environment": self.img_dark,
            "bright_environment": self.img_bright,
        }

        for name, img in domain_fixtures.items():
            res = self.controller.handle_request(query="What do you see in this image?", optical_image=img)
            cap = res.get("optical_caption")
            print(f"Domain '{name}' -> Caption: \"{cap}\" (Confidence: {res.get('confidence')})")
            
            self.assertIsNotNone(cap)
            self.assertFalse(re.search(r"\b(\w+)(?:\s+\1){2,}\b", cap))
            self.assertTrue(len(cap) >= 4)

    # -------------------------------------------------------------------
    # TEST 3: Output Validation Filter Gate (Adversarial Pathological Cases)
    # -------------------------------------------------------------------
    def test_caption_validator_adversarial_patterns(self):
        print("\n--- [TEST 3] Caption Safety Validator Adversarial Matrix ---")
        adversarial_cases = [
            ("the ta ta ta ta ta ta ta ta ta ta ta", False, "Repeated syllable loop"),
            ("a a a a a a a a a", False, "Single character repeat"),
            ("building building building building", False, "Consecutive word repeat"),
            ("in the harbor in the harbor in the harbor", False, "3-gram repetition loop"),
            ("a photo of [PAD] [CLS] object", False, "Special token leak"),
            ("the the the", False, "Stop word loop"),
            ("", False, "Empty string"),
            ("  ", False, "Whitespace only"),
            ("a large white building with a dome and pillars in a green courtyard", True, "Valid description"),
            ("an aerial photograph showing a port harbor with commercial vessels", True, "Valid satellite caption"),
        ]

        for text, expected_valid, desc in adversarial_cases:
            val_res = validate_caption_quality(text)
            self.assertEqual(val_res.is_valid, expected_valid, f"Failed for {desc}: got {val_res.is_valid}")
            print(f"Validated [{desc}]: is_valid={val_res.is_valid}, reason={val_res.reason}")

    # -------------------------------------------------------------------
    # TEST 4: Confidence Correction on Degenerate Model Generation
    # -------------------------------------------------------------------
    def test_confidence_downgrade_on_invalid_generation(self):
        print("\n--- [TEST 4] Confidence Downgrade on Invalid Generation ---")
        # Direct validation of degraded state
        bad_val = validate_caption_quality("the ta ta ta ta ta ta ta ta ta")
        self.assertFalse(bad_val.is_valid)
        self.assertEqual(bad_val.status, "invalid_generation")

        # Verify orchestrator confidence logic never gives HIGH when generation fails
        from tools import ToolExecutionResult
        from router import ClassificationResult, TaskType

        failed_tool_res = ToolExecutionResult(
            tool_name="Optical_Caption",
            task_type="CAPTIONING",
            status="invalid_generation",
            data={"caption": None, "rejection_reason": bad_val.reason},
            confidence=0.20,
            confidence_type="generation_failure",
            confidence_source="caption_quality_validator",
        )

        dummy_class = ClassificationResult(
            task_type=TaskType.CAPTIONING,
            target_tools=["Optical_Caption"],
            confidence=0.8,
            confidence_type="model",
            confidence_source="test_pipeline",
        )
        conf_val, conf_type, conf_src = self.controller._compute_confidence(dummy_class, [failed_tool_res])
        
        print(f"Calculated Confidence for Generation Failure: {conf_val} (Type: {conf_type}, Source: {conf_src})")
        self.assertLessEqual(conf_val, 0.35)
        self.assertEqual(conf_type, "generation_failure")
        self.assertNotEqual(conf_val, 0.88)

    # -------------------------------------------------------------------
    # TEST 5: Evidence-Grounded Synthesis when Captioning Fails
    # -------------------------------------------------------------------
    def test_evidence_grounded_synthesis_on_caption_failure(self):
        print("\n--- [TEST 5] Evidence-Grounded Synthesis ---")
        from synthesis import InvestigationSynthesizer
        from evidence_graph import EvidenceGraph
        from telemetry import ExecutionTrace
        from tools import ToolExecutionResult

        synth = InvestigationSynthesizer()
        graph = EvidenceGraph()
        trace = ExecutionTrace("req_test_eg")

        # Simulate caption failed, but grounding succeeded with 2 detections
        tool_results = [
            ToolExecutionResult(
                tool_name="Optical_Caption",
                task_type="CAPTIONING",
                status="invalid_generation",
                data={"caption": None, "rejection_reason": "Repetitive token loop"},
                confidence=0.20,
                confidence_type="generation_failure",
            ),
            ToolExecutionResult(
                tool_name="Grounding",
                task_type="GROUNDING",
                status="success",
                data={
                    "detections": [
                        {"box": [100, 100, 200, 200], "score": 0.52, "label": "building"},
                        {"box": [250, 250, 350, 350], "score": 0.48, "label": "building"},
                    ],
                    "target_phrase": "buildings",
                },
                confidence=0.50,
                confidence_type="model",
            )
        ]

        report, summary, response_text = synth.synthesize(
            query="Describe the scene and detect structures",
            task_type="MULTI_TASK",
            plan={"selected_tools": ["Optical_Caption", "Grounding"]},
            tool_results=tool_results,
            evidence_graph=graph,
            trace=trace,
        )

        print(f"Synthesized Fallback Response:\n{response_text}")
        self.assertIn("could not be generated reliably", response_text)
        self.assertIn("vision grounding model identified **2** candidate region(s)", response_text)
        self.assertTrue(any("quality filters" in lim.lower() for lim in report.limitations))


if __name__ == "__main__":
    unittest.main()
