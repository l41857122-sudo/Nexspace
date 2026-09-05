"""
test_optical_sar.py
--------------------
Test suite for Phase 4: Multimodal Optical + SAR Joint Reasoning & Multimodal Fusion.

Validates:
  1. Complete separation of Optical Evidence, SAR Evidence, and Fused Conclusion
  2. Physical microwave backscatter vs optical multi-spectral index calculation
  3. Spatial and high-level feature correlation metrics
  4. Explicit provenance labeling ("Research baseline — trained multimodal fusion checkpoint unavailable")
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

from optical_sar_fusion import optical_sar_fusion_engine, MultimodalFusionResult


class TestOpticalSARFusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = optical_sar_fusion_engine

        # Synthetic Optical Image (Urban with Vegetation)
        cls.opt_img = Image.new("RGB", (256, 256), (180, 180, 180))
        draw_o = ImageDraw.Draw(cls.opt_img)
        draw_o.rectangle([30, 30, 90, 90], fill=(210, 50, 50))  # Red roof
        draw_o.rectangle([130, 130, 220, 220], fill=(34, 139, 34))  # Vegetation

        # Synthetic SAR Image (High backscatter double-bounce on building, volume on veg)
        cls.sar_img = Image.new("RGB", (256, 256), (70, 70, 70))
        draw_s = ImageDraw.Draw(cls.sar_img)
        draw_s.rectangle([30, 30, 90, 90], fill=(250, 250, 250))  # Intense corner reflector
        draw_s.rectangle([130, 130, 220, 220], fill=(90, 90, 90))  # Moderate volume scatter

    # Test 1: Separated Evidence & Conclusion Output Structure
    def test_optical_sar_separated_output_structure(self):
        res = self.engine.analyze_pair(
            optical_image=self.opt_img,
            sar_image=self.sar_img,
            query="Compare optical and SAR data",
        )
        self.assertIsInstance(res, MultimodalFusionResult)
        self.assertIn("Optical Multi-Spectral Signal", res.optical_evidence)
        self.assertIn("SAR Microwave Radar Signal", res.sar_evidence)
        self.assertTrue(any(term in res.fused_conclusion.lower() for term in ["cross-validate", "complementary", "structural", "optical"]))

    # Test 2: Provenance & Baseline Transparency
    def test_optical_sar_provenance_transparency(self):
        res = self.engine.analyze_pair(
            optical_image=self.opt_img,
            sar_image=self.sar_img,
        )
        self.assertFalse(res.is_trained_model)
        self.assertIn("Research baseline", res.model_provenance)
        self.assertIn("trained multimodal fusion checkpoint unavailable", res.model_provenance)
        self.assertEqual(res.confidence_type, "unavailable")

    # Test 3: Metric Computation
    def test_optical_sar_metrics_computation(self):
        res = self.engine.analyze_pair(
            optical_image=self.opt_img,
            sar_image=self.sar_img,
        )
        self.assertIn("mean_intensity", res.optical_metrics)
        self.assertIn("edge_density", res.optical_metrics)
        self.assertIn("mean_backscatter", res.sar_metrics)
        self.assertIn("high_backscatter_ratio", res.sar_metrics)
        self.assertIn("dynamic_range_db", res.sar_metrics)
        self.assertIn("spatial_correlation", res.cross_modal_metrics)

    # Test 4: Serialization Schema
    def test_optical_sar_serialization(self):
        res = self.engine.analyze_pair(
            optical_image=self.opt_img,
            sar_image=self.sar_img,
        )
        d = res.to_dict()
        self.assertIn("optical_evidence", d)
        self.assertIn("sar_evidence", d)
        self.assertIn("fused_conclusion", d)
        self.assertIn("model_provenance", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
