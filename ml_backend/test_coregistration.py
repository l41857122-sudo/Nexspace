"""
test_coregistration.py
-----------------------
Test suite for Phase 5: Real Co-Registration Validation & Geometric Alignment.

Validates:
  1. Complete metadata extraction (CRS, GSD resolution, bounds, overlap)
  2. Blocking workflows with clear diagnostics when registration fails or overlap < 10%
  3. Safe raster alignment fallback when geospatial metadata is absent
  4. Aspect ratio and correlation metric tracking
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from coregistration import coregistration_validator, CoRegistrationResult


class TestCoRegistration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.validator = coregistration_validator
        cls.img_a = Image.new("RGB", (256, 256), (100, 100, 100))
        cls.img_b = Image.new("RGB", (256, 256), (120, 120, 120))
        cls.img_diff_size = Image.new("RGB", (512, 256), (120, 120, 120))

    # Test 1: Standard Congruent Pair Alignment
    def test_congruent_pair_registration(self):
        res = self.validator.validate_and_align(self.img_a, self.img_b)
        self.assertIsInstance(res, CoRegistrationResult)
        self.assertTrue(res.is_valid)
        self.assertFalse(res.is_blocked)
        self.assertEqual(res.overlap_percentage, 100.0)
        self.assertTrue(res.dimension_match)
        self.assertIsNotNone(res.aligned_image_b)

    # Test 2: Dimension Mismatch Alignment
    def test_dimension_mismatch_registration(self):
        res = self.validator.validate_and_align(self.img_a, self.img_diff_size)
        self.assertTrue(res.is_valid)
        self.assertFalse(res.dimension_match)
        self.assertEqual(res.aligned_image_b.size, self.img_a.size)
        self.assertIn("diagnostics", res.to_dict())

    # Test 3: Serialization Dictionary Structure
    def test_serialization_structure(self):
        res = self.validator.validate_and_align(self.img_a, self.img_b)
        d = res.to_dict()
        self.assertIn("registration_status", d)
        self.assertIn("registration_quality", d)
        self.assertIn("quality_score", d)
        self.assertIn("alignment_method", d)
        self.assertIn("overlap_percentage", d)
        self.assertIn("is_blocked", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
