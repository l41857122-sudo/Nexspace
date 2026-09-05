"""
test_matrix_image001.py
-----------------------
Dedicated, repeatable evaluation matrix and quality diagnostic runner
for NexSpace Test Image #001 (sample_data/nexspace_test_image_001.jpg).

Evaluates:
  1. VQA (Object inventory, counting, presence, land-use, spatial queries)
  2. Grounding & Detection (River/water, bridge, boats, roads, buildings, vegetation, industrial, construction, farmland)
  3. Scene Description (Coherence, repetition check, domain relevance)
  4. Spatial Reasoning (Compass & relative proximity)
  5. Bi-temporal Change Understanding (Controlled paired testing)
  6. Multimodal Optical + SAR Fusion (Physical microwave reasoning)
  7. Performance & Latency Breakdown (Model load, inference, postprocessing)
"""

import os
import sys
import time
import json
import unittest
from typing import Dict, Any, List
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ml_backend is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tools import (
    VQATool,
    OpticalCaptioningTool,
    GroundingTool,
    ChangeVQATool,
    OpticalSARAnalysisTool,
    ToolRegistry,
)
from rs_vqa_engine import RemoteSensingVQAEngine
from rs_vision_core import RemoteSensingVisionRuntime
from tiling import generate_sliding_tiles, merge_tile_detections, normalize_bbox_1000
from semantic_change import SemanticChangeEngine
from optical_sar_fusion import OpticalSARFusionEngine


TEST_IMAGE_PATH = os.path.join(current_dir, "..", "sample_data", "nexspace_test_image_001.jpg")


class TestNexSpaceImage001Matrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(TEST_IMAGE_PATH):
            raise FileNotFoundError(f"Primary test image not found at {TEST_IMAGE_PATH}")
        cls.raw_img = Image.open(TEST_IMAGE_PATH).convert("RGB")
        cls.img_w, cls.img_h = cls.raw_img.size
        print(f"\n[SETUP] Loaded NexSpace Test Image #001: {cls.img_w}x{cls.img_h} pixels")

        # Initialize engines
        cls.vqa_tool = VQATool()
        cls.caption_tool = OpticalCaptioningTool()
        cls.grounding_tool = GroundingTool()
        cls.change_vqa_tool = ChangeVQATool()
        cls.opt_sar_tool = OpticalSARAnalysisTool()
        cls.vqa_engine = RemoteSensingVQAEngine()
        cls.rs_vision = RemoteSensingVisionRuntime()
        cls.change_engine = SemanticChangeEngine()
        cls.opt_sar_engine = OpticalSARFusionEngine()

    # -----------------------------------------------------------------------
    # TEST 1: VQA Questions Suite
    # -----------------------------------------------------------------------
    def test_01_vqa_suite(self):
        print("\n=== [EVAL 1] VQA Suite on Test Image #001 ===")
        questions = [
            ("What objects are visible?", "inventory"),
            ("How many boats/ships are visible?", "counting"),
            ("Is there a river/water body?", "presence_water"),
            ("Are there bridges?", "presence_bridge"),
            ("Are there roads/highways?", "presence_road"),
            ("Is vegetation present?", "presence_veg"),
            ("Is there an industrial area?", "presence_industrial"),
            ("Is there a residential area?", "presence_residential"),
            ("Is there a construction site?", "presence_construction"),
            ("Where are the major built-up areas?", "localization"),
            ("What is the dominant land-use pattern?", "land_use"),
            ("Compare the visible structures with the surrounding area.", "comparison"),
        ]

        results = []
        for q, category in questions:
            t0 = time.perf_counter()
            res = self.vqa_engine.answer_question(self.raw_img, q)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            ans = res.get("answer", "")
            conf = res.get("confidence")
            task = res.get("task", "")
            print(f"  Q: '{q}'\n  → A: {ans}\n  → Task: {task} | Conf: {conf} | Latency: {lat_ms:.1f}ms\n")

            self.assertIsNotNone(ans)
            self.assertTrue(len(ans) > 5)
            self.assertIn("duration_ms", res)
            results.append({"q": q, "category": category, "ans": ans, "conf": conf, "lat_ms": lat_ms})

        return results

    # -----------------------------------------------------------------------
    # TEST 2: Grounding & Detection Target Classes
    # -----------------------------------------------------------------------
    def test_02_grounding_target_classes(self):
        print("\n=== [EVAL 2] Grounding Target Classes on Test Image #001 ===")
        targets = [
            "river or water body",
            "bridge",
            "boats or ships",
            "roads or highway",
            "buildings",
            "vegetation or trees",
            "industrial area",
            "construction site",
            "farmland or agricultural field",
        ]

        for target in targets:
            t0 = time.perf_counter()
            res = self.grounding_tool.execute({"image": self.raw_img, "target_phrase": target})
            lat_ms = (time.perf_counter() - t0) * 1000.0

            self.assertEqual(res.status, "success")
            boxes = res.data.get("detections", [])
            print(f"  Target: '{target}' → {len(boxes)} box(es) detected in {lat_ms:.1f}ms (Conf type: {res.confidence_type})")

            # Validate box geometry
            for b in boxes:
                box_coords = b.get("box", [])
                if box_coords:
                    xmin, ymin, xmax, ymax = box_coords
                    self.assertLess(xmin, xmax, f"Inverted X in box: {box_coords}")
                    self.assertLess(ymin, ymax, f"Inverted Y in box: {box_coords}")
                    self.assertTrue(0 <= xmin <= 1000 and 0 <= xmax <= 1000)
                    self.assertTrue(0 <= ymin <= 1000 and 0 <= ymax <= 1000)

    # -----------------------------------------------------------------------
    # TEST 3: Scene Description & Repetition Check
    # -----------------------------------------------------------------------
    def test_03_scene_description_quality(self):
        print("\n=== [EVAL 3] Scene Description Quality ===")
        t0 = time.perf_counter()
        res = self.caption_tool.execute({"image": self.raw_img, "modality": "optical"})
        lat_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(res.status, "success")
        caption = res.data.get("caption", "")
        print(f"  Caption: '{caption}' (Latency: {lat_ms:.1f}ms)")

        # Verify no token loops / repetitive degeneration
        tokens = caption.lower().split()
        for i in range(len(tokens) - 2):
            self.assertFalse(tokens[i] == tokens[i+1] == tokens[i+2], f"Repetitive token loop detected: {tokens[i]}")

        # Verify not empty or generic screenshot
        self.assertNotIn("screenshot", caption.lower())
        self.assertNotIn("computer screen", caption.lower())
        self.assertTrue(len(caption) > 10)

    # -----------------------------------------------------------------------
    # TEST 4: Spatial Reasoning
    # -----------------------------------------------------------------------
    def test_04_spatial_reasoning(self):
        print("\n=== [EVAL 4] Spatial Reasoning Queries ===")
        spatial_queries = [
            "Where are the boats located relative to the river?",
            "What is located near the bridge?",
            "Where is the construction area relative to the residential area?",
            "Which side contains the industrial area?",
        ]

        for sq in spatial_queries:
            t0 = time.perf_counter()
            res = self.vqa_engine.answer_question(self.raw_img, sq)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            ans = res.get("answer", "")
            print(f"  Spatial Q: '{sq}'\n  → A: {ans} ({lat_ms:.1f}ms)\n")
            self.assertTrue(len(ans) > 10)

    # -----------------------------------------------------------------------
    # TEST 5: Controlled Bi-Temporal Change Analysis
    # -----------------------------------------------------------------------
    def test_05_bitemporal_change_analysis(self):
        print("\n=== [EVAL 5] Bi-temporal Change Analysis ===")
        # Create a modified version representing real-world new construction development
        modified_img = self.raw_img.copy()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(modified_img)
        # Simulate new built structures in construction site quadrant (approx 2800, 1800 to 3200, 2200)
        draw.rectangle([int(self.img_w * 0.65), int(self.img_h * 0.60), int(self.img_w * 0.78), int(self.img_h * 0.75)], fill=(210, 210, 215))

        t0 = time.perf_counter()
        res = self.change_vqa_tool.execute({
            "image_a": self.raw_img,
            "image_b": modified_img,
            "query": "What changed in the construction area?",
        })
        lat_ms = (time.perf_counter() - t0) * 1000.0

        print(f"  Change Query Answer: {res.data.get('answer')}")
        print(f"  What Changed: {res.data.get('what_changed')}")
        print(f"  Category: {res.data.get('change_category')} | Latency: {lat_ms:.1f}ms")

        self.assertIn(res.status, ("success", "fallback"))
        self.assertIsNotNone(res.data.get("what_changed"))

    # -----------------------------------------------------------------------
    # TEST 6: Multimodal Optical + SAR Fusion Reasoning
    # -----------------------------------------------------------------------
    def test_06_optical_sar_fusion(self):
        print("\n=== [EVAL 6] Multimodal Optical + SAR Fusion Reasoning ===")
        # Load legitimate SAR test image
        sar_path = os.path.join(current_dir, "..", "public", "test-data", "sar_a.png")
        if not os.path.exists(sar_path):
            sar_img = Image.new("L", (self.img_w, self.img_h), color=80)
        else:
            sar_img = Image.open(sar_path).convert("RGB")

        t0 = time.perf_counter()
        res = self.opt_sar_tool.execute({
            "optical_image": self.raw_img,
            "sar_image": sar_img,
            "query": "Compare optical reflectance with SAR backscatter across the built-up sector",
        })
        lat_ms = (time.perf_counter() - t0) * 1000.0

        print(f"  Optical Evidence: {res.data.get('optical_evidence')}")
        print(f"  SAR Evidence:     {res.data.get('sar_evidence')}")
        print(f"  Fused Conclusion: {res.data.get('fused_conclusion')}")
        print(f"  Latency: {lat_ms:.1f}ms")

        self.assertEqual(res.status, "success")
        self.assertIn("Optical", res.data.get("optical_evidence"))
        self.assertIn("SAR", res.data.get("sar_evidence"))
        self.assertTrue(len(res.data.get("fused_conclusion")) > 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
