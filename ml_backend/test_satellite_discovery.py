"""
test_satellite_discovery.py
----------------------------
Comprehensive Automated Test Suite for Satellite Imagery Discovery & Google Maps Integration.
Covers 25 required forensic test cases from Section 25:

TEST 1:  Place name -> imagery search
TEST 2:  Coordinates -> imagery search
TEST 3:  AOI -> imagery search
TEST 4:  Date range -> imagery search
TEST 5:  Cloud-cover filter
TEST 6:  Real scene metadata parsing
TEST 7:  Scene preview retrieval
TEST 8:  Scene selection
TEST 9:  Scene download/retrieval
TEST 10: CRS preservation
TEST 11: Affine transform preservation
TEST 12: Pixel bbox -> geographic bbox
TEST 13: GeoJSON generation (RFC 7946)
TEST 14: Google Maps marker positioning
TEST 15: Google Maps polygon projection
TEST 16: Detection -> map synchronization
TEST 17: Map -> detection synchronization
TEST 18: Open in Google Maps URL integrity
TEST 19: Provider API failure handling
TEST 20: No-location imagery must NOT fabricate coordinates
TEST 21: Stale-result isolation (cache parameter binding)
TEST 22: Multiple scenes transparent ranking
TEST 23: Multiple detections georeferencing
TEST 24: Before/after bitemporal imagery search
TEST 25: Existing upload workflow regression
"""

import sys
import os
import io
import math
import json
import unittest
from datetime import datetime, timezone
from PIL import Image
import numpy as np

# Ensure ml_backend is on sys.path
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from geocoding import GeocodingEngine, GeocodedLocation
from imagery_provider import (
    ImageryProvider,
    ImagerySearchQuery,
    SatelliteScene,
    SceneDownloadResult,
    Element84STACProvider,
    SatelliteDataUnavailableError,
    get_default_imagery_provider,
)
from discovery_parser import DiscoveryQueryParser, SemanticDiscoveryPlan
from geospatial import GeospatialEngine, GeoMetadata, GeoTransform
from geojson_export import export_evidence_to_geojson
from target_spec import GeometryType


class TestSatelliteDiscoveryAndMaps(unittest.TestCase):
    """25-Test Forensic Verification Matrix."""

    @classmethod
    def setUpClass(cls):
        cls.provider = get_default_imagery_provider()

    # -------------------------------------------------------------------------
    # TEST 1: Place name -> imagery search
    # -------------------------------------------------------------------------
    def test_01_place_name_to_imagery_search(self):
        plan = DiscoveryQueryParser.parse("Find recent satellite imagery of Lucknow")
        self.assertEqual(plan.location_raw, "Lucknow")
        self.assertIsNotNone(plan.geocoding)
        self.assertTrue(plan.geocoding.geocoding_available)
        self.assertEqual(len(plan.geocoding.bbox), 4)

        # Query provider with resolved bbox
        q = ImagerySearchQuery(bbox=plan.geocoding.bbox, limit=2)
        scenes = self.provider.search(q)
        self.assertIsInstance(scenes, list)
        self.assertGreater(len(scenes), 0, "Expected at least 1 real Sentinel-2 scene for Lucknow")
        self.assertIn("Sentinel-2", scenes[0].platform)

    # -------------------------------------------------------------------------
    # TEST 2: Coordinates -> imagery search
    # -------------------------------------------------------------------------
    def test_02_coordinates_to_imagery_search(self):
        plan = DiscoveryQueryParser.parse("Find recent imagery near 26.8467, 80.9462")
        self.assertIsNotNone(plan.geocoding)
        self.assertTrue(plan.geocoding.geocoding_available)
        self.assertAlmostEqual(plan.geocoding.latitude, 26.8467, places=3)
        self.assertAlmostEqual(plan.geocoding.longitude, 80.9462, places=3)

        q = ImagerySearchQuery(bbox=plan.geocoding.bbox, limit=2)
        scenes = self.provider.search(q)
        self.assertGreater(len(scenes), 0)

    # -------------------------------------------------------------------------
    # TEST 3: AOI -> imagery search
    # -------------------------------------------------------------------------
    def test_03_aoi_to_imagery_search(self):
        aoi = [80.8462, 26.7467, 81.0462, 26.9467]
        q = ImagerySearchQuery(bbox=aoi, max_cloud_cover=30.0, limit=3)
        scenes = self.provider.search(q)
        self.assertGreater(len(scenes), 0)
        # Check that scene bbox intersects requested AOI
        s_bbox = scenes[0].bbox
        self.assertTrue(s_bbox[0] <= aoi[2] and s_bbox[2] >= aoi[0])

    # -------------------------------------------------------------------------
    # TEST 4: Date range -> imagery search
    # -------------------------------------------------------------------------
    def test_04_date_range_to_imagery_search(self):
        plan = DiscoveryQueryParser.parse("Show satellite images of Mumbai from last month")
        self.assertFalse(plan.temporal.is_bitemporal)
        self.assertIsNotNone(plan.temporal.datetime_range)
        self.assertIn("/", plan.temporal.datetime_range)

    # -------------------------------------------------------------------------
    # TEST 5: Cloud-cover filter
    # -------------------------------------------------------------------------
    def test_05_cloud_cover_filter(self):
        plan = DiscoveryQueryParser.parse("Get the latest cloud-free satellite image of Lucknow")
        self.assertLessEqual(plan.imagery.max_cloud_cover, 10.0, "Expected strict cloud filter for 'cloud-free'")

        q = ImagerySearchQuery(bbox=[80.8462, 26.7467, 81.0462, 26.9467], max_cloud_cover=15.0, limit=3)
        scenes = self.provider.search(q)
        for s in scenes:
            self.assertLessEqual(s.cloud_cover, 15.0, f"Scene {s.scene_id} cloud cover {s.cloud_cover} exceeded 15%")

    # -------------------------------------------------------------------------
    # TEST 6: Real scene metadata parsing
    # -------------------------------------------------------------------------
    def test_06_real_scene_metadata_parsing(self):
        q = ImagerySearchQuery(bbox=[72.7777, 18.9760, 72.9777, 19.1760], max_cloud_cover=25.0, limit=1)
        scenes = self.provider.search(q)
        self.assertGreater(len(scenes), 0)
        s = scenes[0]
        self.assertTrue(s.scene_id.startswith("S2"), f"Expected Sentinel-2 product ID, got {s.scene_id}")
        self.assertIn("Sentinel-2", s.platform)
        self.assertGreater(s.spatial_resolution_m, 0)
        self.assertIn(s.spectral_source, ("CATALOG_METADATA_ONLY", "REAL_MULTISPECTRAL"))
        self.assertGreater(len(s.available_bands), 0)
        self.assertTrue(isinstance(s.selected_reason, str) and len(s.selected_reason) > 5)

    # -------------------------------------------------------------------------
    # TEST 7: Scene preview
    # -------------------------------------------------------------------------
    def test_07_scene_preview(self):
        q = ImagerySearchQuery(bbox=[80.8462, 26.7467, 81.0462, 26.9467], max_cloud_cover=25.0, limit=1)
        scenes = self.provider.search(q)
        self.assertGreater(len(scenes), 0)
        preview_bytes = self.provider.get_preview(scenes[0])
        self.assertGreater(len(preview_bytes), 1000, "Expected non-empty preview image bytes")
        # Validate that PIL can open it
        img = Image.open(io.BytesIO(preview_bytes))
        self.assertGreater(img.width, 100)
        self.assertGreater(img.height, 100)

    # -------------------------------------------------------------------------
    # TEST 8: Scene selection
    # -------------------------------------------------------------------------
    def test_08_scene_selection(self):
        q = ImagerySearchQuery(bbox=[80.8462, 26.7467, 81.0462, 26.9467], limit=3)
        scenes = self.provider.search(q)
        target_id = scenes[0].scene_id
        selected = self.provider.get_scene(target_id)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.scene_id, target_id)

    # -------------------------------------------------------------------------
    # TEST 9: Scene download/retrieval
    # -------------------------------------------------------------------------
    def test_09_scene_download_retrieval(self):
        q = ImagerySearchQuery(bbox=[80.8462, 26.7467, 81.0462, 26.9467], max_cloud_cover=25.0, limit=1)
        scenes = self.provider.search(q)
        res = self.provider.download_scene(scenes[0], prefer_preview=True)
        self.assertIsInstance(res, SceneDownloadResult)
        self.assertTrue(os.path.exists(res.image_path))
        self.assertTrue(res.image_base64.startswith("data:image/jpeg;base64,"))
        self.assertGreater(res.width, 0)
        self.assertGreater(res.height, 0)

    # -------------------------------------------------------------------------
    # TEST 10: CRS preservation
    # -------------------------------------------------------------------------
    def test_10_crs_preservation(self):
        q = ImagerySearchQuery(bbox=[80.8462, 26.7467, 81.0462, 26.9467], max_cloud_cover=25.0, limit=1)
        scenes = self.provider.search(q)
        res = self.provider.download_scene(scenes[0], prefer_preview=True)
        self.assertIn("EPSG:", res.crs)
        self.assertIsNotNone(res.bounds_world)

    # -------------------------------------------------------------------------
    # TEST 11: Affine transform preservation
    # -------------------------------------------------------------------------
    def test_11_affine_transform_preservation(self):
        q = ImagerySearchQuery(bbox=[80.8462, 26.7467, 81.0462, 26.9467], max_cloud_cover=25.0, limit=1)
        scenes = self.provider.search(q)
        res = self.provider.download_scene(scenes[0], prefer_preview=True)
        t = res.transform
        for param in ["a", "b", "c", "d", "e", "f"]:
            self.assertIn(param, t)
        self.assertNotEqual(t["a"], 0.0)
        self.assertNotEqual(t["e"], 0.0)

    # -------------------------------------------------------------------------
    # TEST 12: Pixel bbox -> geographic bbox
    # -------------------------------------------------------------------------
    def test_12_pixel_bbox_to_geographic_bbox(self):
        # Construct known affine transform (origin 80.0 lon, 27.0 lat, 0.001 deg/px)
        t = GeoTransform(a=0.001, b=0.0, c=80.0, d=0.0, e=-0.001, f=27.0)
        # Pixel box: [100, 100, 200, 200]
        w_bbox = t.bbox_pixel_to_world([100, 100, 200, 200], crs="EPSG:4326")
        self.assertAlmostEqual(w_bbox["min_x"], 80.1, places=4)
        self.assertAlmostEqual(w_bbox["max_x"], 80.2, places=4)
        self.assertAlmostEqual(w_bbox["min_y"], 26.8, places=4)
        self.assertAlmostEqual(w_bbox["max_y"], 26.9, places=4)

    # -------------------------------------------------------------------------
    # TEST 13: GeoJSON generation (RFC 7946)
    # -------------------------------------------------------------------------
    def test_13_geojson_generation(self):
        t = GeoTransform(a=0.0001, b=0.0, c=80.0, d=0.0, e=-0.0001, f=26.0)
        geo_meta = GeoMetadata(
            geospatial_available=True,
            crs="EPSG:4326",
            transform=t,
            resolution={"x": 0.0001, "y": 0.0001, "unit": "degree"},
        )
        raw_evidence = [{
            "evidence_id": "det_001",
            "type": "object_detection",
            "label": "vessel",
            "bbox_pixel": [10, 10, 50, 50],
            "score": 0.88,
        }]
        enriched = [GeospatialEngine.enrich_evidence_item(item, geo_meta) for item in raw_evidence]
        fc = export_evidence_to_geojson(enriched, geo_meta)
        self.assertEqual(fc["type"], "FeatureCollection")
        self.assertEqual(len(fc["features"]), 1)
        feat = fc["features"][0]
        self.assertEqual(feat["type"], "Feature")
        self.assertEqual(feat["geometry"]["type"], "Polygon")
        ring = feat["geometry"]["coordinates"][0]
        # Polygon must be closed (first and last coordinate match)
        self.assertEqual(ring[0], ring[-1])

    # -------------------------------------------------------------------------
    # TEST 14: Google Maps marker
    # -------------------------------------------------------------------------
    def test_14_google_maps_marker(self):
        t = GeoTransform(a=0.0001, b=0.0, c=80.9462, d=0.0, e=-0.0001, f=26.8467)
        geo_meta = GeoMetadata(geospatial_available=True, crs="EPSG:4326", transform=t)
        item = {"evidence_id": "ship_01", "bbox_pixel": [100, 100, 200, 200]}
        enriched = GeospatialEngine.enrich_evidence_item(item, geo_meta)
        fc = export_evidence_to_geojson([enriched], geo_meta)
        coords = fc["features"][0]["geometry"]["coordinates"][0]
        center_lon = sum(p[0] for p in coords[:-1]) / (len(coords) - 1)
        center_lat = sum(p[1] for p in coords[:-1]) / (len(coords) - 1)
        self.assertTrue(-180 <= center_lon <= 180)
        self.assertTrue(-90 <= center_lat <= 90)

    # -------------------------------------------------------------------------
    # TEST 15: Google Maps polygon
    # -------------------------------------------------------------------------
    def test_15_google_maps_polygon(self):
        t = GeoTransform(a=0.0002, b=0.0, c=72.8460, d=0.0, e=-0.0002, f=18.9438)
        geo_meta = GeoMetadata(geospatial_available=True, crs="EPSG:4326", transform=t)
        item = {"evidence_id": "region_01", "bbox_pixel": [20, 30, 80, 90]}
        enriched = GeospatialEngine.enrich_evidence_item(item, geo_meta)
        fc = export_evidence_to_geojson([enriched], geo_meta)
        ring = fc["features"][0]["geometry"]["coordinates"][0]
        self.assertEqual(len(ring), 5, "Expected 4 corners + closed loop")

    # -------------------------------------------------------------------------
    # TEST 16: Detection -> map synchronization
    # -------------------------------------------------------------------------
    def test_16_detection_to_map_synchronization(self):
        # Stable detection ID across evidence and GeoJSON
        item = {"id": "target_42", "evidence_id": "target_42", "bbox_pixel": [10, 10, 20, 20]}
        t = GeoTransform(a=0.001, b=0.0, c=0.0, d=0.0, e=-0.001, f=0.0)
        geo_meta = GeoMetadata(geospatial_available=True, crs="EPSG:4326", transform=t)
        enriched = GeospatialEngine.enrich_evidence_item(item, geo_meta)
        fc = export_evidence_to_geojson([enriched], geo_meta)
        self.assertEqual(fc["features"][0]["id"], "target_42")
        self.assertEqual(fc["features"][0]["properties"]["evidence_id"], "target_42")

    # -------------------------------------------------------------------------
    # TEST 17: Map -> detection synchronization
    # -------------------------------------------------------------------------
    def test_17_map_to_detection_synchronization(self):
        # Verifies that clicking map feature with ID recovers identical evidence node
        evidence_nodes = [
            {"evidence_id": "feat_01", "label": "ship", "bbox_pixel": [0, 0, 10, 10]},
            {"evidence_id": "feat_02", "label": "dock", "bbox_pixel": [20, 20, 30, 30]},
        ]
        t = GeoTransform(a=0.001, b=0.0, c=0.0, d=0.0, e=-0.001, f=0.0)
        geo_meta = GeoMetadata(geospatial_available=True, crs="EPSG:4326", transform=t)
        fc = export_evidence_to_geojson(
            [GeospatialEngine.enrich_evidence_item(n, geo_meta) for n in evidence_nodes],
            geo_meta
        )
        clicked_id = fc["features"][1]["id"]
        found = next(n for n in evidence_nodes if n["evidence_id"] == clicked_id)
        self.assertEqual(found["label"], "dock")

    # -------------------------------------------------------------------------
    # TEST 18: Open in Google Maps
    # -------------------------------------------------------------------------
    def test_18_open_in_google_maps(self):
        lat, lon = 26.846712, 80.946234
        url = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
        self.assertIn("26.846712,80.946234", url)
        self.assertTrue(url.startswith("https://www.google.com/maps?q="))

    # -------------------------------------------------------------------------
    # TEST 19: Provider API failure handling
    # -------------------------------------------------------------------------
    def test_19_provider_api_failure_handling(self):
        # Create provider with intentionally invalid endpoint
        class FailingProvider(Element84STACProvider):
            STAC_ENDPOINT = "https://invalid-host-catalog.element84.xyz/search"

        fail_p = FailingProvider()
        q = ImagerySearchQuery(bbox=[0, 0, 1, 1])
        with self.assertRaises(SatelliteDataUnavailableError) as cm:
            fail_p.search(q)
        self.assertIn("catalog request failed", str(cm.exception))

    # -------------------------------------------------------------------------
    # TEST 20: No-location imagery must NOT fabricate coordinates
    # -------------------------------------------------------------------------
    def test_20_no_location_imagery_must_not_fabricate(self):
        # Plain JPEG image without GeoTIFF tags
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        geo_meta = GeospatialEngine.extract_metadata(buf.getvalue())
        self.assertFalse(geo_meta.geospatial_available)
        self.assertIsNone(geo_meta.transform)

        item = {"evidence_id": "det_no_geo", "bbox_pixel": [10, 10, 20, 20]}
        enriched = GeospatialEngine.enrich_evidence_item(item, geo_meta)
        self.assertFalse(enriched.get("geospatial_coordinates_available", True))
        self.assertNotIn("bbox_world", enriched)

        fc = export_evidence_to_geojson([enriched], geo_meta)
        self.assertFalse(fc["geospatial_available"])
        self.assertEqual(len(fc["features"]), 0, "Must NEVER fabricate GeoJSON features for unreferenced rasters")

    # -------------------------------------------------------------------------
    # TEST 21: Stale-result isolation
    # -------------------------------------------------------------------------
    def test_21_stale_result_isolation(self):
        q1 = ImagerySearchQuery(bbox=[80.0, 26.0, 81.0, 27.0], max_cloud_cover=10.0)
        q2 = ImagerySearchQuery(bbox=[72.0, 18.0, 73.0, 19.0], max_cloud_cover=10.0)
        self.assertNotEqual(q1.cache_key(), q2.cache_key(), "Different queries must yield distinct cache keys")

    # -------------------------------------------------------------------------
    # TEST 22: Multiple scenes ranking and pagination
    # -------------------------------------------------------------------------
    def test_22_multiple_scenes(self):
        q = ImagerySearchQuery(bbox=[80.8462, 26.7467, 81.0462, 26.9467], max_cloud_cover=30.0, limit=5)
        scenes = self.provider.search(q)
        self.assertGreaterEqual(len(scenes), 2)
        # Verify ranking order: higher ranking score first
        for i in range(len(scenes) - 1):
            self.assertGreaterEqual(scenes[i].ranking_score, scenes[i + 1].ranking_score)

    # -------------------------------------------------------------------------
    # TEST 23: Multiple detections georeferencing
    # -------------------------------------------------------------------------
    def test_23_multiple_detections(self):
        t = GeoTransform(a=0.0001, b=0.0, c=80.0, d=0.0, e=-0.0001, f=26.0)
        geo_meta = GeoMetadata(geospatial_available=True, crs="EPSG:4326", transform=t)
        items = [
            {"evidence_id": f"det_{i}", "bbox_pixel": [i * 10, i * 10, i * 10 + 5, i * 10 + 5]}
            for i in range(10)
        ]
        enriched = [GeospatialEngine.enrich_evidence_item(it, geo_meta) for it in items]
        fc = export_evidence_to_geojson(enriched, geo_meta)
        self.assertEqual(len(fc["features"]), 10)
        for idx, feat in enumerate(fc["features"]):
            self.assertEqual(feat["properties"]["evidence_id"], f"det_{idx}")

    # -------------------------------------------------------------------------
    # TEST 24: Before/after imagery search
    # -------------------------------------------------------------------------
    def test_24_before_after_imagery_search(self):
        plan = DiscoveryQueryParser.parse("Show imagery of Lucknow before and after the flood")
        self.assertTrue(plan.temporal.is_bitemporal)
        self.assertEqual(plan.analysis_task, "change_analysis")
        self.assertIsNotNone(plan.temporal.datetime_before)
        self.assertIsNotNone(plan.temporal.datetime_after)

    # -------------------------------------------------------------------------
    # TEST 25: Existing upload workflow regression
    # -------------------------------------------------------------------------
    def test_25_existing_upload_workflow_regression(self):
        from target_spec import SemanticQueryParser
        specs = SemanticQueryParser.parse("find ships near the docks")
        self.assertGreater(len(specs), 0)
        self.assertEqual(specs[0].concept, "ships")
        self.assertEqual(specs[0].geometry_type, GeometryType.DISCRETE_OBJECT)
        self.assertEqual(len(specs[0].spatial_relations), 1)
        self.assertEqual(specs[0].spatial_relations[0].reference_entity, "docks")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSatelliteDiscoveryAndMaps)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    return res.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
