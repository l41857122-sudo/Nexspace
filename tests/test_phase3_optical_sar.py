"""
tests/test_phase3_optical_sar.py
---------------------------------
Phase 3 Verification Suite:
Genuine Optical-SAR Processing and Fusion with Sensor Breakdown & Transparency.

Acceptance Criteria:
 1. Genuine dual-source inputs combine into 1536-d fused representation
 2. Sensor-specific breakdown: optical_only_metrics, sar_only_metrics, combined_metrics
 3. Source contributions and preprocessing status accurately documented
 4. Honest limitations included (baseline feature concatenation, no claimed accuracy gain without benchmark)
 5. Optical missing: explicit failure, fusion_executed=False, returns sar_only_metrics
 6. SAR missing: explicit failure, fusion_executed=False, returns optical_only_metrics
 7. Both missing: explicit failure
 8. Identical images: rejected as not independent
 9. Non-overlapping geographic bounds: rejected
10. Perturbation test: both optical and SAR perturbations affect fused representation
11. Modality dropout contract
12. Ablation evaluation
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_backend"))

from optical_sar_fusion import (
    OpticalSARFusionEngine,
    optical_sar_fusion_engine,
    MultimodalFusionResult,
)
from tools import OpticalSARAnalysisTool


class TestPhase3OpticalSAR(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = optical_sar_fusion_engine
        cls.tool = OpticalSARAnalysisTool(engine=cls.engine)

        # 1. Authentic Optical Synthetic Raster (RGB, vegetation, water, urban)
        cls.opt_img = Image.new("RGB", (256, 256), (150, 150, 150))
        d_opt = ImageDraw.Draw(cls.opt_img)
        d_opt.rectangle([20, 20, 100, 100], fill=(200, 50, 50))    # Built structure
        d_opt.rectangle([120, 20, 240, 120], fill=(30, 180, 40))   # Photosynthetic vegetation
        d_opt.rectangle([20, 140, 240, 240], fill=(20, 40, 150))   # Water body

        # 2. Authentic SAR Synthetic Raster (L/grayscale, dihedral bounce, diffuse, specular)
        cls.sar_img = Image.new("RGB", (256, 256), (50, 50, 50))
        d_sar = ImageDraw.Draw(cls.sar_img)
        d_sar.rectangle([20, 20, 100, 100], fill=(255, 255, 255))  # Dihedral double-bounce
        d_sar.rectangle([120, 20, 240, 120], fill=(90, 90, 90))    # Volume scattering
        d_sar.rectangle([20, 140, 240, 240], fill=(10, 10, 10))    # Specular smooth reflection

        # Common overlapping UTM bounds
        cls.bounds_utm = {"min_x": 500000.0, "min_y": 1997440.0, "max_x": 502560.0, "max_y": 2000000.0}
        cls.crs_utm = "EPSG:32644"

    def test_01_dual_source_fusion_success(self):
        """Dual-source inputs successfully fuse into 1536-d concatenated vector."""
        res = self.engine.analyze_pair(self.opt_img, self.sar_img)
        self.assertIsInstance(res, MultimodalFusionResult)
        self.assertEqual(res.optical_feature_dim, 768)
        self.assertEqual(res.sar_feature_dim, 768)
        self.assertEqual(res.fused_feature_dim, 1536)
        self.assertEqual(res.fusion_operation, "CONCATENATION")
        self.assertEqual(res.fusion_type, "DUAL_ENCODER_FEATURE_CONCATENATION")
        self.assertFalse(res.is_trained_model)
        self.assertEqual(res.fusion_learning, "NONE")

    def test_02_sensor_specific_breakdown_metrics(self):
        """Result contains independent optical and SAR metrics."""
        res = self.engine.analyze_pair(self.opt_img, self.sar_img)
        d = res.to_dict()

        # Optical breakdown
        self.assertIn("optical_only_metrics", d)
        opt_m = d["optical_only_metrics"]
        self.assertIn("vegetation_fraction", opt_m)
        self.assertIn("water_fraction", opt_m)
        self.assertIn("edge_density", opt_m)
        self.assertGreater(opt_m["vegetation_fraction"], 0.0)

        # SAR breakdown
        self.assertIn("sar_only_metrics", d)
        sar_m = d["sar_only_metrics"]
        self.assertIn("mean_backscatter", sar_m)
        self.assertIn("dynamic_range_db", sar_m)
        self.assertIn("speckle_cv", sar_m)
        self.assertIn("high_backscatter_ratio", sar_m)
        self.assertGreater(sar_m["dynamic_range_db"], 0.0)

    def test_03_source_contributions_and_limitations(self):
        """Result documents source contributions and explicit honest limitations."""
        res = self.engine.analyze_pair(self.opt_img, self.sar_img)
        d = res.to_dict()

        self.assertIn("source_contributions", d)
        self.assertIn("optical", d["source_contributions"])
        self.assertIn("sar", d["source_contributions"])
        self.assertIn("combined", d["source_contributions"])

        self.assertIn("limitations", d)
        self.assertTrue(len(d["limitations"]) >= 2)
        # Verify explicit transparency statement
        lim_text = " ".join(d["limitations"])
        self.assertIn("not learned cross-attention", lim_text)
        self.assertIn("Accuracy improvement over single-sensor baselines has not been established", lim_text)

    def test_04_sar_missing_partial_breakdown(self):
        """When SAR is missing, tool returns status='failed', fusion_executed=False, but provides optical_only_metrics."""
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": None,
        })
        self.assertEqual(res.status, "failed")
        data = res.to_dict()["data"]
        self.assertFalse(data["fusion_executed"])
        self.assertTrue(data["optical_present"])
        self.assertFalse(data["sar_present"])
        self.assertIn("optical_only_metrics", data)
        self.assertIn("vegetation_fraction", data["optical_only_metrics"])
        self.assertIn("limitations", data)

    def test_05_optical_missing_partial_breakdown(self):
        """When Optical is missing, tool returns status='failed', fusion_executed=False, but provides sar_only_metrics."""
        res = self.tool.execute({
            "optical_image": None,
            "sar_image": self.sar_img,
        })
        self.assertEqual(res.status, "failed")
        data = res.to_dict()["data"]
        self.assertFalse(data["fusion_executed"])
        self.assertFalse(data["optical_present"])
        self.assertTrue(data["sar_present"])
        self.assertIn("sar_only_metrics", data)
        self.assertIn("mean_backscatter", data["sar_only_metrics"])
        self.assertIn("limitations", data)

    def test_06_both_missing_rejection(self):
        """When both are missing, tool returns failed without crash."""
        res = self.tool.execute({"optical_image": None, "sar_image": None})
        self.assertEqual(res.status, "failed")
        self.assertFalse(res.to_dict()["data"]["fusion_executed"])

    def test_07_identical_inputs_rejected(self):
        """Identical inputs are rejected to enforce independence."""
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.opt_img,
        })
        self.assertEqual(res.status, "failed")
        self.assertFalse(res.to_dict()["data"]["fusion_executed"])
        self.assertIn("independent", res.error)

    def test_08_non_overlapping_bounds_rejected(self):
        """Images with non-overlapping geographical bounds are rejected."""
        sar_disjoint_bounds = {"min_x": 600000.0, "min_y": 2997440.0, "max_x": 602560.0, "max_y": 3000000.0}
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.sar_img,
            "optical_bounds": self.bounds_utm,
            "sar_bounds": sar_disjoint_bounds,
            "optical_crs": self.crs_utm,
            "sar_crs": self.crs_utm,
        })
        self.assertEqual(res.status, "failed")
        self.assertFalse(res.to_dict()["data"]["fusion_executed"])
        self.assertIn("do not overlap", res.error)

    def test_09_perturbation_sensitivity(self):
        """Perturbing either modality changes the fused representation."""
        perturbation = self.engine.run_perturbation_test(self.opt_img, self.sar_img)
        self.assertTrue(perturbation["fusion_response"]["both_modalities_influence_result"])
        self.assertTrue(perturbation["optical_sensitivity"]["is_sensitive"])
        self.assertTrue(perturbation["sar_sensitivity"]["is_sensitive"])

    def test_10_ablation_evaluation(self):
        """Ablation compares optical-only, SAR-only, and joint representation."""
        ablation = self.engine.run_ablation(self.opt_img, self.sar_img)
        self.assertEqual(ablation["case_a_optical_only"]["feature_dimension"], 768)
        self.assertEqual(ablation["case_b_sar_only"]["feature_dimension"], 768)
        self.assertEqual(ablation["case_c_optical_sar_fusion"]["fused_feature_dimension"], 1536)
        self.assertIn("downstream_accuracy_claim", ablation["findings"])


if __name__ == "__main__":
    unittest.main()
