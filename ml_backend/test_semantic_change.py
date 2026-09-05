"""
test_semantic_change.py
------------------------
Comprehensive test suite for Phase 3: Bi-temporal Semantic Change Understanding.

Validates:
  1. Pixel Change vs Object Change vs Semantic Change taxonomy categorization
  2. Targeted Change-VQA ("What changed near the buildings?")
  3. Bounding box and anomaly cluster segmentation integrity
  4. Bitemporal interpretation provenance and confidence tracking
  5. Fallback containment on identical vs divergent pairs
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from semantic_change import semantic_change_engine, SemanticChangeResult


class TestSemanticChange(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = semantic_change_engine

        # Baseline Image A (Before: Vegetated terrain with 1 building)
        cls.img_before = Image.new("RGB", (256, 256), (34, 139, 34))
        draw_a = ImageDraw.Draw(cls.img_before)
        draw_a.rectangle([40, 40, 90, 90], fill=(200, 60, 40), outline=(255, 255, 255))

        # Divergent Image B (After: Deforestation + 3 new buildings constructed)
        cls.img_after = Image.new("RGB", (256, 256), (180, 160, 120))  # Cleared ground
        draw_b = ImageDraw.Draw(cls.img_after)
        # Original building
        draw_b.rectangle([40, 40, 90, 90], fill=(200, 60, 40), outline=(255, 255, 255))
        # New buildings added
        draw_b.rectangle([120, 40, 170, 90], fill=(200, 60, 40), outline=(255, 255, 255))
        draw_b.rectangle([40, 140, 90, 190], fill=(200, 60, 40), outline=(255, 255, 255))
        draw_b.rectangle([120, 140, 170, 190], fill=(200, 60, 40), outline=(255, 255, 255))

    # Test 1: Full Semantic Change Pipeline Output Integrity
    def test_semantic_change_pipeline_execution(self):
        res = self.engine.analyze_semantic_change(
            image_a=self.img_before,
            image_b=self.img_after,
            query="What changed between these images?",
        )
        self.assertIsInstance(res, SemanticChangeResult)
        self.assertIn("Image A (Before)", res.before_interpretation)
        self.assertIn("Image B (After)", res.after_interpretation)
        self.assertIn(res.change_category, ["SEMANTIC CHANGE", "OBJECT CHANGE", "PIXEL CHANGE"])
        self.assertTrue(res.changed_fraction > 0.05)
        self.assertTrue(len(res.changed_regions) > 0)
        self.assertTrue(len(res.evidence) > 0)
        self.assertIn("Research baseline", res.model_provenance)

    # Test 2: Targeted Query near specific objects ("near the buildings")
    def test_targeted_change_vqa_query(self):
        res = self.engine.analyze_semantic_change(
            image_a=self.img_before,
            image_b=self.img_after,
            query="What changed near the buildings?",
        )
        self.assertIn("building", res.change_vqa_answer.lower())
        self.assertTrue(len(res.changed_regions) > 0)
        for r in res.changed_regions:
            self.assertEqual(len(r.box), 4)
            self.assertEqual(len(r.box_normalized), 4)
            self.assertTrue(r.area_pixels > 0)

    # Test 3: Identical Images should report PIXEL CHANGE or minimal delta
    def test_identical_images_change_handling(self):
        res = self.engine.analyze_semantic_change(
            image_a=self.img_before,
            image_b=self.img_before,
            query="Did anything change?",
        )
        self.assertLessEqual(res.changed_fraction, 0.01)
        self.assertEqual(res.change_category, "PIXEL CHANGE")
        self.assertIn("Minimal", res.what_changed)

    # Test 4: Serialization Schema Compatibility
    def test_semantic_change_serialization_dict(self):
        res = self.engine.analyze_semantic_change(
            image_a=self.img_before,
            image_b=self.img_after,
        )
        d = res.to_dict()
        self.assertIn("before_interpretation", d)
        self.assertIn("after_interpretation", d)
        self.assertIn("what_changed", d)
        self.assertIn("change_category", d)
        self.assertIn("changed_regions", d)
        self.assertIn("change_vqa_answer", d)
        self.assertIn("model_provenance", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
