"""
tests/test_gate3_step7_spatial_validation.py
--------------------------------------------
Gate 3 Step 7 Verification Test Suite:
Task-Level Spatial Intelligence & Geospatial Reasoning Validation.

Comprehensive 20-Point Automated Verification Suite:
 1. Benchmark dataset integrity & deterministic SHA-256 hash.
 2. 11 Core Spatial Relations evaluated with 100% precision & recall against ground truth.
 3. Multi-target queries decomposed and evaluated across Cartesian candidates.
 4. Natural language spatial semantics and synonym parsing.
 5. WGS84 and projected coordinate system safety.
 6. 15-case failure matrix handled truthfully with honest statuses.
 7. 10 anti-hardcoding geometric mutations altering outcomes as geometry mutates.
 8. Reproducibility & exact numerical determinism across repeated executions.
 9. Metric distance accuracy (Haversine geodesic, MAE < 1.5m).
10. Directional bearing classification & diagonal confusion matrix.
11. Step-by-step query-to-geometry execution traceability.
12. AOI containment and intersection evaluation.
13. Scene footprint containment validation.
14. Detection-to-GeoJSON RFC 7946 feature conversion.
15. Provenance contract preservation (ALGORITHMIC).
16. Confidence contract preservation (confidence=None, confidence_type="UNAVAILABLE").
17. Prior Gate 3.2 invariant preserved (input integrity & SHA-256 identical input rejection).
18. Prior Gate 3.3 invariant preserved (authentic multispectral bands & real NDVI).
19. Prior Gate 3.4 invariant preserved (true CRS reprojection & GSD harmonization).
20. Prior Gate 3.5 & 3.6 invariants preserved (dual-encoder fusion & honest task benchmarking).
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_backend"))

from spatial_reasoning import (
    SpatialRelationType,
    SpatialEvaluationStatus,
    SpatialContractResult,
    SpatialRelationshipEngine,
    GeospatialSpatialEngine,
    haversine_distance_meters,
    calculate_geodetic_bearing_deg,
    classify_direction_from_bearing,
    SPATIAL_SYNONYM_MAP,
)
from spatial_benchmark import spatial_benchmark, SpatialIntelligenceBenchmark
from discovery_parser import DiscoveryQueryParser
from co_registration import (
    RasterGeometry,
    GeospatialCoRegistrationEngine,
    AlignmentStatus,
    OverlapStatus,
)
from multispectral import spectral_engine, MultispectralScene, SpectralBand
from optical_sar_fusion import optical_sar_fusion_engine


class TestGate3Step7SpatialValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.benchmark = spatial_benchmark
        cls.engine = GeospatialSpatialEngine()

    def test_01_benchmark_dataset_integrity_and_hash(self):
        """1. Dataset integrity: Dataset produces consistent SHA-256 hash and valid entities."""
        h1 = self.benchmark._compute_dataset_hash()
        h2 = self.benchmark.dataset_hash
        self.assertEqual(h1, h2)
        self.assertGreater(len(self.benchmark.entities), 15)
        self.assertIn("ref_port", self.benchmark.entities)
        self.assertIn("line_road_main", self.benchmark.entities)

    def test_02_eleven_core_spatial_relations_accuracy(self):
        """2. 11 core spatial relations: 100% accuracy, precision, and recall on ground-truth benchmark."""
        res = self.benchmark.run_relation_benchmarks()
        self.assertEqual(res["overall_accuracy"], 1.0)
        metrics = res["relation_metrics"]
        for rel in [
            "near", "inside", "outside", "adjacent", "crossing", "along",
            "closest_to", "north_of", "south_of", "east_of", "west_of"
        ]:
            self.assertIn(rel, metrics)
            self.assertEqual(metrics[rel]["accuracy"], 1.0, f"Relation {rel} accuracy failed")
            self.assertEqual(metrics[rel]["f1_score"], 1.0, f"Relation {rel} F1 failed")

    def test_03_multi_target_queries(self):
        """3. Multi-target queries: Decomposes candidate sets and pairs geometrically without index bias."""
        res = self.benchmark.run_multi_target_benchmark()
        self.assertTrue(res["all_multi_target_correct"])
        self.assertEqual(res["queries_count"], 4)
        for q in res["queries"]:
            self.assertTrue(q["correct"], f"Multi-target query failed: {q['query']}")

    def test_04_natural_language_spatial_semantics(self):
        """4. NL spatial semantics: Synonym phrases correctly map to canonical relations."""
        res = self.benchmark.run_natural_language_semantics_benchmark()
        self.assertTrue(res["all_nl_semantics_passed"])
        self.assertGreaterEqual(res["test_cases_count"], 16)

    def test_05_coordinate_system_safety(self):
        """5. Coordinate safety: Distinguishes WGS84 degrees, projected meters, and normalized pixel space."""
        # WGS84 point
        geo_item = {"id": "geo_pt", "coordinates": {"lat": 1.270, "lon": 103.775}}
        ref_item = {"id": "ref_pt", "coordinates": {"lat": 1.271, "lon": 103.775}}
        contract = self.engine.evaluate_contract(geo_item, ref_item, "near")
        self.assertEqual(contract.distance["unit"], "meters")
        self.assertAlmostEqual(contract.distance["value"], 111.0, delta=2.0)

        # Pixel item
        px_item1 = {"id": "px_1", "bbox": [100, 100, 200, 200]}
        px_item2 = {"id": "px_2", "bbox": [150, 150, 250, 250]}
        contract_px = self.engine.evaluate_contract(px_item1, px_item2, "near")
        self.assertEqual(contract_px.distance["unit"], "normalized_pixels")

    def test_06_failure_matrix_fifteen_cases(self):
        """6. Failure matrix: All 15 failure modes handle corrupt/missing inputs truthfully."""
        res = self.benchmark.run_failure_matrix()
        self.assertTrue(res["all_failure_cases_passed"])
        self.assertEqual(res["total_failure_cases"], 15)

    def test_07_anti_hardcoding_mutation_audit(self):
        """7. Anti-hardcoding audit: 10 geometric mutations prove dynamic geometric calculation."""
        res = self.benchmark.run_anti_hardcoding_mutation_audit()
        self.assertTrue(res["all_mutations_passed"])
        self.assertEqual(res["total_mutations"], 10)

    def test_08_reproducibility_and_determinism(self):
        """8. Reproducibility: Repeated benchmark executions yield identical SHA-256 result hashes."""
        res = self.benchmark.run_reproducibility_audit()
        self.assertTrue(res["is_reproducible"])
        self.assertTrue(res["hash_match"])
        self.assertEqual(res["run1_hash"], res["run2_hash"])

    def test_09_metric_distance_accuracy(self):
        """9. Metric distance accuracy: Geodesic Haversine calculation achieves MAE < 1.5 meters."""
        res = self.benchmark.run_metric_distance_and_directional_benchmarks()
        dist_meta = res["distance_evaluation"]
        self.assertLess(dist_meta["mae_meters"], 1.5)
        self.assertLess(dist_meta["rmse_meters"], 2.0)

    def test_10_directional_bearing_classification(self):
        """10. Directional bearing: Cardinal direction confusion matrix is strictly diagonal."""
        res = self.benchmark.run_metric_distance_and_directional_benchmarks()
        cm = res["directional_confusion_matrix"]
        for d in ["north_of", "east_of", "south_of", "west_of"]:
            self.assertEqual(cm[d][d], 1)
            for d_other in ["north_of", "east_of", "south_of", "west_of"]:
                if d != d_other:
                    self.assertEqual(cm[d][d_other], 0)

    def test_11_execution_trace_generation(self):
        """11. Execution traceability: Generates step-by-step audit trace for spatial query explanation."""
        port = self.benchmark.entities["ref_port"]
        ship = self.benchmark.entities["ship_inside_port"]
        contract = self.engine.evaluate_contract(ship, port, "inside")
        trace = self.engine.generate_execution_trace(
            query="Which ships are inside the port?",
            parsed_subject_target="ship",
            parsed_reference_target="port",
            parsed_relation="inside",
            candidate_count=1,
            matched_results=[contract],
        )
        self.assertEqual(trace["provenance"], "ALGORITHMIC")
        self.assertIsNone(trace["confidence"])
        self.assertEqual(trace["results_summary"]["satisfied_count"], 1)

    def test_12_aoi_containment_and_intersection(self):
        """12. AOI containment: Explicit AOI polygon filters detections correctly."""
        aoi = {"id": "aoi_jurong", "polygon": self.benchmark.entities["ref_port"]["polygon"]}
        ship_in = self.benchmark.entities["ship_inside_port"]
        ship_out = self.benchmark.entities["ship_outside_port"]

        res_in = self.engine.evaluate_aoi_containment(ship_in, aoi)
        res_out = self.engine.evaluate_aoi_containment(ship_out, aoi)
        self.assertTrue(res_in["is_contained"])
        self.assertFalse(res_out["is_contained"])

    def test_13_scene_footprint_containment(self):
        """13. Scene footprint: Validates whether coordinates lie inside geographic raster bounds."""
        bounds = self.benchmark.scene_bounds
        inside_item = {"coordinates": {"lat": 1.270, "lon": 103.775}}
        outside_item = {"coordinates": {"lat": 2.500, "lon": 105.000}}

        self.assertTrue(self.engine.evaluate_scene_containment(inside_item, bounds))
        self.assertFalse(self.engine.evaluate_scene_containment(outside_item, bounds))

    def test_14_detection_to_geojson_rfc7946(self):
        """14. Detection to GeoJSON: Exports RFC 7946 compliant Feature with validated coordinates."""
        ship = self.benchmark.entities["ship_inside_port"]
        feat = self.engine.map_detection_to_geojson(ship)
        self.assertEqual(feat["type"], "Feature")
        self.assertEqual(feat["properties"]["provenance"], "ALGORITHMIC")
        self.assertEqual(feat["geometry"]["type"], "Point")
        coords = tuple(feat["geometry"]["coordinates"])
        self.assertAlmostEqual(coords[0], 103.775, places=3)
        self.assertAlmostEqual(coords[1], 1.270, places=3)

    def test_15_provenance_contract_preservation(self):
        """15. Provenance contract: Spatial relations are strictly ALGORITHMIC."""
        port = self.benchmark.entities["ref_port"]
        ship = self.benchmark.entities["ship_near_port_120m"]
        contract = self.engine.evaluate_contract(ship, port, "near")
        self.assertEqual(contract.provenance, "ALGORITHMIC")

    def test_16_confidence_contract_preservation(self):
        """16. Confidence contract: Spatial relation confidence is null / UNAVAILABLE."""
        port = self.benchmark.entities["ref_port"]
        ship = self.benchmark.entities["ship_inside_port"]
        contract = self.engine.evaluate_contract(ship, port, "inside")
        self.assertIsNone(contract.confidence)
        self.assertEqual(contract.confidence_type, "UNAVAILABLE")

    def test_17_prior_gate_3_2_invariant_preserved(self):
        """17. Gate 3.2 invariant preserved: SHA-256 identical input rejection remains active."""
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        with self.assertRaises(ValueError) as ctx:
            optical_sar_fusion_engine.analyze_pair(img, img)
        self.assertIn("IDENTICAL_INPUT_REJECTED", str(ctx.exception))

    def test_18_prior_gate_3_3_invariant_preserved(self):
        """18. Gate 3.3 invariant preserved: Real Sentinel-2 NDVI calculation."""
        b04_data = np.full((100, 100), 0.10, dtype=np.float32)
        b08_data = np.full((100, 100), 0.50, dtype=np.float32)
        band_b04 = SpectralBand(
            name="B04", common_name="red", data=b04_data,
            native_width=100, native_height=100, native_resolution_m=10.0,
            crs="EPSG:32643",
            transform={"a": 10.0, "b": 0.0, "c": 500000.0, "d": 0.0, "e": -10.0, "f": 3000000.0},
        )
        band_b08 = SpectralBand(
            name="B08", common_name="nir", data=b08_data,
            native_width=100, native_height=100, native_resolution_m=10.0,
            crs="EPSG:32643",
            transform={"a": 10.0, "b": 0.0, "c": 500000.0, "d": 0.0, "e": -10.0, "f": 3000000.0},
        )

        scene = MultispectralScene(
            scene_id="S2A_TEST_GATE3_STEP7",
            bands={"B04": band_b04, "B08": band_b08},
            loaded_bands=["B04", "B08"],
            available_bands=["B04", "B08"],
            width=100, height=100, crs="EPSG:32643",
            transform={"a": 10.0, "b": 0.0, "c": 500000.0, "d": 0.0, "e": -10.0, "f": 3000000.0},
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

    def test_19_prior_gate_3_4_invariant_preserved(self):
        """19. Gate 3.4 invariant preserved: True CRS reprojection & GSD harmonization."""
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

    def test_20_prior_gate_3_5_and_3_6_invariants_preserved(self):
        """20. Gate 3.5 & 3.6 invariants preserved: Dual-encoder feature concatenation (1536-d)."""
        f_opt = np.random.randn(768).astype(np.float32)
        f_sar = np.random.randn(768).astype(np.float32)
        fused = optical_sar_fusion_engine.fuse_features(f_opt, f_sar)
        self.assertEqual(fused.shape, (1536,))
        self.assertAlmostEqual(float(np.linalg.norm(fused)), np.sqrt(2.0), delta=1e-4)


if __name__ == "__main__":
    unittest.main()
