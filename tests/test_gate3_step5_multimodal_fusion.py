"""
tests/test_gate3_step5_multimodal_fusion.py
--------------------------------------------
Gate 3 Step 5 Verification Test Matrix:
True Multimodal Fusion Validation & Fusion-Truth Hardening

Covers all 26 mandatory cases:
 1. Independent Optical + SAR fusion
 2. Optical-only rejection
 3. SAR-only rejection
 4. Same-image rejection
 5. Non-overlapping rejection
 6. CRS mismatch handled through Gate 3 Step 4
 7. Different GSD handled through Gate 3 Step 4
 8. Feature dimensions verified (768-d optical, 768-d SAR)
 9. Fused dimension verified (1536-d)
10. Feature normalization verified (L2 norm = 1.0)
11. Cosine similarity verified
12. Euclidean discrepancy verified
13. Pearson correlation verified
14. No arbitrary confidence score (confidence=None, type=unavailable)
15. Provenance contract
16. Fusion type contract (DUAL_ENCODER_FEATURE_CONCATENATION, learning=NONE)
17. Optical perturbation changes result where expected
18. SAR perturbation changes result where expected
19. Unrelated SAR cannot fuse (geographically disjoint rejected)
20. Unrelated Optical cannot fuse
21. Modality dropout
22. Deterministic repeated inference
23. Reproducibility within numerical tolerance
24. Gate 3 Step 2 invariant preserved (input integrity & SHA-256)
25. Gate 3 Step 3 invariant preserved (authentic multispectral ingestion)
26. Gate 3 Step 4 invariant preserved (geospatial co-registration & reprojection)
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_backend"))

from optical_sar_fusion import (
    OpticalSARFusionEngine,
    optical_sar_fusion_engine,
    MultimodalFusionResult,
    FusionCapability,
)
from tools import OpticalSARAnalysisTool, MultispectralAnalysisTool
from co_registration import RasterGeometry, GeospatialCoRegistrationEngine
from multispectral import SpectralBand, MultispectralScene, SpectralEngine


class TestGate3Step5MultimodalFusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = optical_sar_fusion_engine
        cls.tool = OpticalSARAnalysisTool()

        # Synthetic Optical Image: Urban scene with distinct red roofs & green vegetation
        cls.opt_img = Image.new("RGB", (256, 256), (180, 180, 180))
        draw_o = ImageDraw.Draw(cls.opt_img)
        draw_o.rectangle([30, 30, 100, 100], fill=(220, 40, 40))   # Red roof
        draw_o.rectangle([140, 140, 220, 220], fill=(30, 160, 30)) # Green vegetation

        # Synthetic SAR Image: Microwave backscatter with high corner reflection & diffuse terrain
        cls.sar_img = Image.new("RGB", (256, 256), (60, 60, 60))
        draw_s = ImageDraw.Draw(cls.sar_img)
        draw_s.rectangle([30, 30, 100, 100], fill=(255, 255, 255)) # Strong double-bounce
        draw_s.rectangle([140, 140, 220, 220], fill=(95, 95, 95))   # Volume scattering

        cls.crs_utm = "EPSG:32644"
        cls.bounds_utm = {"min_x": 500000.0, "min_y": 1997440.0, "max_x": 502560.0, "max_y": 2000000.0}
        cls.transform_utm = (10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)

    # 1. Independent Optical + SAR fusion executes
    def test_01_independent_optical_sar_fusion(self):
        res = self.engine.analyze_pair(self.opt_img, self.sar_img)
        self.assertIsInstance(res, MultimodalFusionResult)
        self.assertEqual(res.fusion_type, "DUAL_ENCODER_FEATURE_CONCATENATION")
        self.assertEqual(res.fused_feature_dim, 1536)
        self.assertEqual(res.fusion_learning, "NONE")
        self.assertFalse(res.is_trained_model)

    # 2. Optical-only rejection
    def test_02_optical_only_rejection(self):
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": None,
        })
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.provenance, "UNAVAILABLE")
        self.assertFalse(res.to_dict()["data"]["fusion_executed"])

    # 3. SAR-only rejection
    def test_03_sar_only_rejection(self):
        res = self.tool.execute({
            "optical_image": None,
            "sar_image": self.sar_img,
        })
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.provenance, "UNAVAILABLE")
        self.assertFalse(res.to_dict()["data"]["fusion_executed"])

    # 4. Same-image rejection (Gate 3 Step 2 invariant)
    def test_04_same_image_rejection(self):
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.opt_img,
        })
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.provenance, "UNAVAILABLE")
        self.assertEqual(res.to_dict()["data"]["alignment_status"], "identical_inputs_rejected")

    # 5. Non-overlapping rejection (Gate 3 Step 4 invariant)
    def test_05_non_overlapping_rejection(self):
        bounds_disjoint = {"min_x": 600000.0, "min_y": 2097440.0, "max_x": 602560.0, "max_y": 2100000.0}
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.sar_img,
            "optical_crs": self.crs_utm,
            "sar_crs": self.crs_utm,
            "optical_bounds": self.bounds_utm,
            "sar_bounds": bounds_disjoint,
        })
        self.assertEqual(res.status, "failed")
        self.assertFalse(res.to_dict()["data"]["fusion_executed"])
        self.assertEqual(res.to_dict()["data"]["alignment_status"], "bounds_non_overlapping")

    # 6. CRS mismatch handled through Gate 3 Step 4 reprojection
    def test_06_crs_mismatch_handled_through_coregistration(self):
        # Transform bounds to WGS84 for SAR input
        bounds_geo = GeospatialCoRegistrationEngine.transform_bounds(
            self.bounds_utm, self.crs_utm, "EPSG:4326"
        )
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.sar_img,
            "optical_crs": self.crs_utm,
            "sar_crs": "EPSG:4326",
            "optical_bounds": self.bounds_utm,
            "sar_bounds": bounds_geo,
            "optical_transform": self.transform_utm,
            "sar_transform": (0.0001, 0.0, bounds_geo["min_x"], 0.0, -0.0001, bounds_geo["max_y"]),
        })
        self.assertEqual(res.status, "success")
        d = res.to_dict()
        self.assertTrue(d["data"]["fusion_executed"])
        self.assertTrue(d["data"]["geospatial_valid"])

    # 7. Different GSD handled through Gate 3 Step 4
    def test_07_different_gsd_handled(self):
        sar_small = self.sar_img.resize((128, 128))
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": sar_small,
            "optical_crs": self.crs_utm,
            "sar_crs": self.crs_utm,
            "optical_bounds": self.bounds_utm,
            "sar_bounds": self.bounds_utm,
            "optical_transform": self.transform_utm,
            "sar_transform": (20.0, 0.0, 500000.0, 0.0, -20.0, 2000000.0),
        })
        self.assertEqual(res.status, "success")
        self.assertTrue(res.to_dict()["data"]["fusion_executed"])

    # 8. Feature dimensions verified
    def test_08_feature_dimensions_verified(self):
        f_opt = self.engine.extract_modality_features(self.opt_img, modality="optical")
        f_sar = self.engine.extract_modality_features(self.sar_img, modality="sar")
        self.assertEqual(f_opt.shape, (768,))
        self.assertEqual(f_sar.shape, (768,))

    # 9. Fused dimension verified (1536-d)
    def test_09_fused_dimension_verified(self):
        res = self.engine.analyze_pair(self.opt_img, self.sar_img)
        self.assertEqual(res.fused_feature_dim, 1536)
        self.assertEqual(len(res.fused_representation), 1536)

    # 10. Feature normalization verified (L2 norm = 1.0)
    def test_10_feature_normalization_verified(self):
        f_opt = self.engine.extract_modality_features(self.opt_img, modality="optical")
        f_sar = self.engine.extract_modality_features(self.sar_img, modality="sar")
        norm_opt = float(np.linalg.norm(f_opt))
        norm_sar = float(np.linalg.norm(f_sar))
        self.assertAlmostEqual(norm_opt, 1.0, places=4)
        self.assertAlmostEqual(norm_sar, 1.0, places=4)

    # 11. Cosine similarity verified
    def test_11_cosine_similarity_verified(self):
        res = self.engine.analyze_pair(self.opt_img, self.sar_img)
        cos_sim = res.cross_modal_metrics["cross_modal_feature_cosine_similarity"]
        self.assertGreaterEqual(cos_sim, -1.0)
        self.assertLessEqual(cos_sim, 1.0)

    # 12. Euclidean discrepancy verified
    def test_12_euclidean_discrepancy_verified(self):
        res = self.engine.analyze_pair(self.opt_img, self.sar_img)
        disc = res.cross_modal_metrics["cross_modal_feature_euclidean_discrepancy"]
        cos_sim = res.cross_modal_metrics["cross_modal_feature_cosine_similarity"]
        # Mathematical identity: ||f1 - f2||_2 = sqrt(2 - 2*cos_sim) for unit vectors
        expected_disc = float(np.sqrt(max(0.0, 2.0 - 2.0 * cos_sim)))
        self.assertAlmostEqual(disc, expected_disc, places=3)

    # 13. Pearson correlation verified
    def test_13_pearson_correlation_verified(self):
        res = self.engine.analyze_pair(self.opt_img, self.sar_img)
        corr = res.cross_modal_metrics["spatial_intensity_pearson_correlation"]
        self.assertGreaterEqual(corr, -1.0)
        self.assertLessEqual(corr, 1.0)

    # 14. No arbitrary confidence score
    def test_14_no_arbitrary_confidence_score(self):
        res = self.engine.analyze_pair(self.opt_img, self.sar_img)
        self.assertIsNone(res.confidence)
        self.assertEqual(res.confidence_type, "unavailable")
        self.assertEqual(res.confidence_source, "optical_sar_physics_fusion_baseline")

    # 15. Provenance contract
    def test_15_provenance_contract(self):
        res = self.engine.analyze_pair(self.opt_img, self.sar_img)
        self.assertEqual(res.optical_provenance, "MODEL_DERIVED")
        self.assertEqual(res.sar_provenance, "MODEL_DERIVED")
        self.assertEqual(res.alignment_provenance, "ALGORITHMIC")
        self.assertEqual(res.fusion_provenance, "ALGORITHMIC")
        self.assertEqual(res.confidence_provenance, "UNAVAILABLE")

    # 16. Fusion type contract
    def test_16_fusion_type_contract(self):
        res = self.engine.analyze_pair(self.opt_img, self.sar_img)
        self.assertEqual(res.fusion_type, "DUAL_ENCODER_FEATURE_CONCATENATION")
        self.assertEqual(res.fusion_operation, "CONCATENATION")
        self.assertEqual(res.fusion_learning, "NONE")
        self.assertFalse(res.is_trained_model)

    # 17. Optical perturbation changes result
    def test_17_optical_perturbation_changes_result(self):
        pert = self.engine.run_perturbation_test(self.opt_img, self.sar_img)
        self.assertTrue(pert["optical_sensitivity"]["is_sensitive"])
        self.assertGreater(pert["optical_sensitivity"]["perturbation_representation_distance"], 0.01)

    # 18. SAR perturbation changes result
    def test_18_sar_perturbation_changes_result(self):
        pert = self.engine.run_perturbation_test(self.opt_img, self.sar_img)
        self.assertTrue(pert["sar_sensitivity"]["is_sensitive"])
        self.assertGreater(pert["sar_sensitivity"]["perturbation_representation_distance"], 0.01)

    # 19. Unrelated SAR cannot fuse (geographically disjoint rejected)
    def test_19_unrelated_sar_rejected_by_geography(self):
        bounds_unrelated = {"min_x": 800000.0, "min_y": 3000000.0, "max_x": 802560.0, "max_y": 3002560.0}
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.sar_img,
            "optical_crs": self.crs_utm,
            "sar_crs": self.crs_utm,
            "optical_bounds": self.bounds_utm,
            "sar_bounds": bounds_unrelated,
        })
        self.assertEqual(res.status, "failed")
        self.assertFalse(res.to_dict()["data"]["fusion_executed"])

    # 20. Unrelated Optical cannot fuse
    def test_20_unrelated_optical_rejected_by_geography(self):
        bounds_unrelated = {"min_x": 100000.0, "min_y": 1000000.0, "max_x": 102560.0, "max_y": 1002560.0}
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.sar_img,
            "optical_crs": self.crs_utm,
            "sar_crs": self.crs_utm,
            "optical_bounds": bounds_unrelated,
            "sar_bounds": self.bounds_utm,
        })
        self.assertEqual(res.status, "failed")
        self.assertFalse(res.to_dict()["data"]["fusion_executed"])

    # 21. Modality dropout
    def test_21_modality_dropout(self):
        drop_opt = self.engine.run_modality_dropout(self.opt_img, None)
        self.assertFalse(drop_opt["fusion_available"])
        self.assertEqual(drop_opt["status"], "unavailable")

        drop_sar = self.engine.run_modality_dropout(None, self.sar_img)
        self.assertFalse(drop_sar["fusion_available"])
        self.assertEqual(drop_sar["status"], "unavailable")

        drop_both = self.engine.run_modality_dropout(self.opt_img, self.sar_img)
        self.assertTrue(drop_both["fusion_available"])
        self.assertEqual(drop_both["status"], "success")

    # 22. Deterministic repeated inference
    def test_22_deterministic_repeated_inference(self):
        res1 = self.engine.analyze_pair(self.opt_img, self.sar_img)
        res2 = self.engine.analyze_pair(self.opt_img, self.sar_img)
        self.assertEqual(res1.fused_feature_dim, res2.fused_feature_dim)
        self.assertEqual(res1.optical_metrics, res2.optical_metrics)
        self.assertEqual(res1.sar_metrics, res2.sar_metrics)
        self.assertEqual(res1.cross_modal_metrics, res2.cross_modal_metrics)

    # 23. Reproducibility within numerical tolerance
    def test_23_reproducibility_within_numerical_tolerance(self):
        res1 = self.engine.analyze_pair(self.opt_img, self.sar_img)
        res2 = self.engine.analyze_pair(self.opt_img, self.sar_img)
        arr1 = np.array(res1.fused_representation)
        arr2 = np.array(res2.fused_representation)
        np.testing.assert_allclose(arr1, arr2, rtol=1e-5, atol=1e-5)

    # 24. Existing Gate 3 Step 2 invariant preserved
    def test_24_gate3_step2_invariant_preserved(self):
        # Identical image attack must fail before fusion
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.opt_img,
            "optical_crs": self.crs_utm,
            "sar_crs": self.crs_utm,
            "optical_bounds": self.bounds_utm,
            "sar_bounds": self.bounds_utm,
        })
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.provenance, "UNAVAILABLE")
        self.assertFalse(res.to_dict()["data"]["fusion_executed"])

    # 25. Existing Gate 3 Step 3 invariant preserved (Multispectral authenticity)
    def test_25_gate3_step3_invariant_preserved(self):
        # RGB only NDVI must be unavailable
        tool = MultispectralAnalysisTool()
        res = tool.execute({"image": self.opt_img, "indices": ["NDVI"]})
        self.assertEqual(res.status, "unavailable")
        self.assertEqual(res.provenance, "UNAVAILABLE")

    # 26. Existing Gate 3 Step 4 invariant preserved (Geospatial Co-Registration)
    def test_26_gate3_step4_invariant_preserved(self):
        geom_opt = RasterGeometry.from_image_or_metadata(
            self.opt_img, crs=self.crs_utm, bounds=self.bounds_utm, transform=self.transform_utm
        )
        geom_sar = RasterGeometry.from_image_or_metadata(
            self.sar_img, crs=self.crs_utm, bounds=self.bounds_utm, transform=self.transform_utm
        )
        self.assertTrue(geom_opt.is_valid_geospatial())
        self.assertTrue(geom_sar.is_valid_geospatial())
        grid, err = GeospatialCoRegistrationEngine.establish_target_grid(geom_opt, geom_sar)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.crs, self.crs_utm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
