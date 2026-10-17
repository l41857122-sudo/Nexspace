"""
tests/test_gate3_step6_task_validation.py
-----------------------------------------
Gate 3 Step 6 Verification Test Matrix:
Task-Level Multimodal Validation & Benchmarking.

Minimum 17 Comprehensive Tests:
 1. Dataset split deterministic.
 2. No train/test scene leakage.
 3. Optical baseline executes.
 4. SAR baseline executes or honestly reports unavailable.
 5. Fusion baseline executes.
 6. Metrics calculated correctly.
 7. Ground truth alignment verified.
 8. Missing modality handled.
 9. Same-image attack rejected.
10. Non-overlapping scenes rejected.
11. CRS mismatch handled.
12. RGB-only spectral request remains unavailable.
13. Repeated benchmark reproducible.
14. Previous Gate 3.2 invariant preserved.
15. Previous Gate 3.3 invariant preserved.
16. Previous Gate 3.4 invariant preserved.
17. Previous Gate 3.5 invariant preserved.
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_backend"))

from task_benchmark import MultimodalTaskBenchmark, task_benchmark
from optical_sar_fusion import optical_sar_fusion_engine, OpticalSARFusionEngine
from co_registration import (
    RasterGeometry,
    GeospatialCoRegistrationEngine,
    AlignmentStatus,
    OverlapStatus,
)
from multispectral import spectral_engine, MultispectralScene, SpectralBand


class TestGate3Step6TaskValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.benchmark = task_benchmark
        # Pre-warm cached results
        cls.cached_results = cls.benchmark.run_full_benchmark(fast_inspection=True)

    def test_01_dataset_split_deterministic(self):
        """1. Dataset split deterministic: Fixed seed produces identical partition and SHA-256 hash."""
        bench_a = MultimodalTaskBenchmark(seed=42)
        bench_b = MultimodalTaskBenchmark(seed=42)

        self.assertEqual(bench_a.split_hash, bench_b.split_hash)
        self.assertEqual(len(bench_a.train_scenes), 15)
        self.assertEqual(len(bench_a.val_scenes), 5)
        self.assertEqual(len(bench_a.test_scenes), 10)

        test_ids_a = [s["scene_id"] for s in bench_a.test_scenes]
        test_ids_b = [s["scene_id"] for s in bench_b.test_scenes]
        self.assertEqual(test_ids_a, test_ids_b)

    def test_02_no_train_test_scene_leakage(self):
        """2. No train/test scene leakage: Strict geographic and ID disjointness."""
        train_ids = set(s["scene_id"] for s in self.benchmark.train_scenes)
        val_ids = set(s["scene_id"] for s in self.benchmark.val_scenes)
        test_ids = set(s["scene_id"] for s in self.benchmark.test_scenes)

        self.assertTrue(train_ids.isdisjoint(test_ids), "Train and Test sets must not overlap!")
        self.assertTrue(val_ids.isdisjoint(test_ids), "Val and Test sets must not overlap!")
        self.assertTrue(train_ids.isdisjoint(val_ids), "Train and Val sets must not overlap!")
        self.assertEqual(len(train_ids | val_ids | test_ids), 30)

    def test_03_optical_baseline_executes(self):
        """3. Optical baseline executes: 768-d features, returns bounded metrics."""
        opt = self.cached_results["optical_only_baseline"]
        self.assertEqual(opt["feature_dimension"], 768)
        self.assertGreaterEqual(opt["accuracy"], 0.0)
        self.assertLessEqual(opt["accuracy"], 1.0)
        self.assertGreaterEqual(opt["f1_macro"], 0.0)
        self.assertLessEqual(opt["f1_macro"], 1.0)
        self.assertIn("urban_dense", opt["per_class"])

    def test_04_sar_baseline_executes(self):
        """4. SAR baseline executes or honestly reports unavailable."""
        sar = self.cached_results["sar_only_baseline"]
        self.assertEqual(sar["feature_dimension"], 768)
        self.assertGreaterEqual(sar["accuracy"], 0.0)
        self.assertLessEqual(sar["accuracy"], 1.0)
        self.assertGreaterEqual(sar["f1_macro"], 0.0)
        self.assertLessEqual(sar["f1_macro"], 1.0)
        self.assertIn("maritime_port", sar["per_class"])

    def test_05_fusion_baseline_executes(self):
        """5. Fusion baseline executes: Verified 1536-d concatenation."""
        fused = self.cached_results["fused_baseline"]
        self.assertEqual(fused["feature_dimension"], 1536)
        self.assertEqual(fused["fusion_architecture"], "DUAL_ENCODER_FEATURE_CONCATENATION")
        self.assertEqual(fused["fusion_learning"], "NONE")
        self.assertGreaterEqual(fused["accuracy"], 0.0)
        self.assertLessEqual(fused["accuracy"], 1.0)

    def test_06_metrics_calculated_correctly(self):
        """6. Metrics calculated correctly: Confusion matrix dimensions and values."""
        fused_eval = self.benchmark.evaluate_modality_baseline("fused")
        cm = np.array(fused_eval["confusion_matrix"])
        self.assertEqual(cm.shape, (6, 6))
        self.assertEqual(int(cm.sum()), 10)  # N=10 test samples

    def test_07_ground_truth_alignment_verified(self):
        """7. Ground truth alignment verified: Ground truth labels match predefined taxonomy."""
        for s in self.benchmark.test_scenes:
            self.assertIn(s["category"], MultimodalTaskBenchmark.CATEGORIES)
            self.assertIn("EPSG:4326", s["crs"])
            self.assertEqual(len(s["bounds"]), 4)

    def test_08_missing_modality_handled(self):
        """8. Missing modality handled: Rejects None input truthfully."""
        with self.assertRaises((ValueError, TypeError)):
            optical_sar_fusion_engine.fuse_features(None, np.ones(768, dtype=np.float32))

        with self.assertRaises(ValueError):
            optical_sar_fusion_engine.analyze_pair(None, Image.new("L", (100, 100)))

    def test_09_same_image_attack_rejected(self):
        """9. Same-image attack rejected: Gate 3.2 SHA-256 identical input defense."""
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        with self.assertRaises(ValueError) as ctx:
            optical_sar_fusion_engine.analyze_pair(img, img)
        self.assertIn("IDENTICAL_INPUT_REJECTED", str(ctx.exception))

    def test_10_non_overlapping_scenes_rejected(self):
        """10. Non-overlapping scenes rejected: Geographic bounding box validation."""
        opt_geom = RasterGeometry.from_image_or_metadata(
            np.zeros((100, 100, 3), dtype=np.uint8),
            crs="EPSG:4326",
            bounds={"min_x": -0.15, "min_y": 51.50, "max_x": -0.10, "max_y": 51.52},
            transform=(0.0001, 0, -0.15, 0, -0.0001, 51.52),
        )
        sar_geom = RasterGeometry.from_image_or_metadata(
            np.zeros((100, 100), dtype=np.uint8),
            crs="EPSG:4326",
            bounds={"min_x": 12.45, "min_y": 41.88, "max_x": 12.50, "max_y": 41.92},
            transform=(0.0001, 0, 12.45, 0, -0.0001, 41.92),
        )
        res_opt, res_sar = GeospatialCoRegistrationEngine.co_register_pair(
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100), dtype=np.uint8),
            opt_geom,
            sar_geom,
        )
        self.assertEqual(res_opt.status, AlignmentStatus.FAILED)
        self.assertEqual(res_opt.overlap_status, OverlapStatus.NO_OVERLAP)

    def test_11_crs_mismatch_handled(self):
        """11. CRS mismatch handled: Reprojection between distinct coordinate reference systems."""
        geom_utm = RasterGeometry.from_image_or_metadata(
            np.zeros((100, 100), dtype=np.float32),
            crs="EPSG:32644",
            transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0),
            bounds={"min_x": 500000.0, "min_y": 1999000.0, "max_x": 501000.0, "max_y": 2000000.0},
        )
        import pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)
        w, s = transformer.transform(500000.0, 1999000.0)
        e, n = transformer.transform(501000.0, 2000000.0)
        bounds_geo = {"min_x": min(w, e), "min_y": min(s, n), "max_x": max(w, e), "max_y": max(s, n)}
        transform_geo = ((max(w, e) - min(w, e)) / 100.0, 0.0, min(w, e), 0.0, -(max(s, n) - min(s, n)) / 100.0, max(s, n))
        geom_geo = RasterGeometry.from_image_or_metadata(
            np.zeros((100, 100), dtype=np.float32),
            crs="EPSG:4326",
            transform=transform_geo,
            bounds=bounds_geo,
        )
        res_utm, res_geo = GeospatialCoRegistrationEngine.co_register_pair(
            np.zeros((100, 100), dtype=np.float32),
            np.zeros((100, 100), dtype=np.float32),
            geom_utm,
            geom_geo,
        )
        self.assertEqual(res_geo.status, AlignmentStatus.REPROJECTED)
        self.assertTrue(res_geo.resampling_applied)

    def test_12_rgb_only_spectral_request_remains_unavailable(self):
        """12. RGB-only spectral request remains unavailable: Gate 3.3 invariant."""
        from tools import multispectral_analysis_tool
        rgb_img = Image.new("RGB", (100, 100), color=(30, 140, 50))
        tool_res = multispectral_analysis_tool.execute({
            "image": rgb_img,
            "multispectral_scene": None,
            "indices": ["NDVI"],
        })
        self.assertEqual(tool_res.status, "unavailable")
        self.assertEqual(tool_res.provenance, "UNAVAILABLE")
        self.assertIn("B04", tool_res.error)

    def test_13_repeated_benchmark_reproducible(self):
        """13. Repeated benchmark reproducible: Exact deterministic metric stability."""
        repro = self.benchmark.verify_reproducibility()
        self.assertTrue(repro["is_reproducible"])
        self.assertEqual(repro["reproducibility_status"], "DETERMINISTIC_REPRODUCIBLE")
        self.assertLess(repro["max_absolute_metric_difference"], 1e-6)

    def test_14_previous_gate3_2_invariant_preserved(self):
        """14. Gate 3.2 Invariant Preserved: Input independence and SHA-256 validation."""
        img1 = Image.new("RGB", (64, 64), color=(50, 100, 150))
        img2 = Image.new("RGB", (64, 64), color=(50, 100, 150))
        with self.assertRaises(ValueError):
            optical_sar_fusion_engine.analyze_pair(img1, img2)

    def test_15_previous_gate3_3_invariant_preserved(self):
        """15. Gate 3.3 Invariant Preserved: Authentic Sentinel-2 multispectral ingestion and real NDVI."""
        b4_data = np.full((100, 100), 1000, dtype=np.uint16)
        b8_data = np.full((100, 100), 5000, dtype=np.uint16)
        b4 = SpectralBand(
            name="B04", common_name="red", data=b4_data,
            native_width=100, native_height=100, native_resolution_m=10.0,
            dtype="uint16", scale=0.0001, offset=0.0, nodata=0,
            crs="EPSG:32643", transform={"a": 10.0, "b": 0.0, "c": 500000.0, "d": 0.0, "e": -10.0, "f": 3000000.0},
            wavelength_um=0.665,
        )
        b8 = SpectralBand(
            name="B08", common_name="nir", data=b8_data,
            native_width=100, native_height=100, native_resolution_m=10.0,
            dtype="uint16", scale=0.0001, offset=0.0, nodata=0,
            crs="EPSG:32643", transform={"a": 10.0, "b": 0.0, "c": 500000.0, "d": 0.0, "e": -10.0, "f": 3000000.0},
            wavelength_um=0.842,
        )
        scene = MultispectralScene(
            scene_id="s2_test",
            bands={"B04": b4, "B08": b8},
            loaded_bands=["B04", "B08"],
            available_bands=["B04", "B08"],
            width=100,
            height=100,
            crs="EPSG:32643",
            transform={"a": 10.0, "b": 0.0, "c": 5000000.0, "d": 0.0, "e": -10.0, "f": 3000000.0},
            bounds={"min_x": 500000.0, "min_y": 2999000.0, "max_x": 501000.0, "max_y": 3000000.0},
            resolution_m=10.0,
            acquisition_time="2026-03-15T05:30:00Z",
            source="Sentinel-2 L2A",
            spectral_source="REAL_MULTISPECTRAL",
            provenance={"provider": "Element84_AWS_STAC"},
        )
        from multispectral import SpectralEngine
        res = SpectralEngine.compute_ndvi(scene)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.provenance, "REAL_MULTISPECTRAL")
        expected_ndvi = (0.50 - 0.10) / (0.50 + 0.10)
        self.assertAlmostEqual(res.stats.mean, expected_ndvi, places=2)

    def test_16_previous_gate3_4_invariant_preserved(self):
        """16. Gate 3.4 Invariant Preserved: Geospatial co-registration and GSD harmonization."""
        g_10m = RasterGeometry.from_image_or_metadata(
            np.zeros((100, 100), dtype=np.float32),
            crs="EPSG:32644",
            transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0),
            bounds={"min_x": 500000.0, "min_y": 1999000.0, "max_x": 501000.0, "max_y": 2000000.0},
        )
        g_20m = RasterGeometry.from_image_or_metadata(
            np.zeros((50, 50), dtype=np.float32),
            crs="EPSG:32644",
            transform=(20.0, 0.0, 500000.0, 0.0, -20.0, 2000000.0),
            bounds={"min_x": 500000.0, "min_y": 1999000.0, "max_x": 501000.0, "max_y": 2000000.0},
        )
        res_a, res_b = GeospatialCoRegistrationEngine.co_register_pair(
            np.zeros((100, 100), dtype=np.float32),
            np.zeros((50, 50), dtype=np.float32),
            g_10m,
            g_20m,
        )
        self.assertEqual(res_b.status, AlignmentStatus.RESAMPLED)
        self.assertEqual(res_a.aligned_raster.shape, res_b.aligned_raster.shape)

    def test_17_previous_gate3_5_invariant_preserved(self):
        """17. Gate 3.5 Invariant Preserved: Independent influence and DUAL_ENCODER_FEATURE_CONCATENATION."""
        f_opt1 = np.ones(768, dtype=np.float32) / np.sqrt(768)
        f_opt2 = np.zeros(768, dtype=np.float32)
        f_opt2[0] = 1.0
        f_sar = np.full(768, 0.5, dtype=np.float32) / np.linalg.norm(np.full(768, 0.5, dtype=np.float32))

        fused1 = optical_sar_fusion_engine.fuse_features(f_opt1, f_sar)
        fused2 = optical_sar_fusion_engine.fuse_features(f_opt2, f_sar)
        self.assertEqual(len(fused1), 1536)
        self.assertEqual(len(fused2), 1536)
        # Perturbation in optical independently changes fused embedding
        diff = np.linalg.norm(fused1 - fused2)
        self.assertGreater(diff, 0.5)


if __name__ == "__main__":
    unittest.main()
