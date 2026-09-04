"""
test_step9_geospatial.py
------------------------
Comprehensive test suite for STEP 9:
Real Geospatial Intelligence, GeoTIFF / CRS / GSD Support & Pixel-to-World Coordinate Engine.
"""

import sys
import os
import unittest
import math
import numpy as np
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

import tifffile
from geospatial import (
    GeospatialEngine,
    GeoMetadata,
    GeoTransform,
    TAG_MODEL_PIXEL_SCALE,
    TAG_MODEL_TIEPOINT,
    TAG_GEO_KEY_DIRECTORY,
)
from geojson_export import export_evidence_to_geojson
from evidence_graph import EvidenceGraph, EvidenceNode
from anomaly_engine import AnomalyEngine, anomaly_engine
from orchestrator import GeoVLMController
from server import app, encode_image_b64
from fastapi.testclient import TestClient


def create_synthetic_geotiff(
    filename: str,
    width: int = 100,
    height: int = 100,
    origin_lon: float = 13.4050,
    origin_lat: float = 52.5200,
    pixel_scale_lon: float = 0.0001,  # ~11m
    pixel_scale_lat: float = 0.0001,
    epsg_code: int = 4326,
) -> str:
    """
    Creates a tiny synthetic GeoTIFF mathematical test fixture with genuine tags.
    Used strictly to verify coordinate transform mathematics.
    """
    path = os.path.join(_dir, filename)
    data = np.zeros((height, width, 3), dtype=np.uint8)
    data[20:40, 30:60, :] = 255  # White rectangle

    # ModelTiepointTag: (I, J, K, X, Y, Z) -> (0, 0, 0, origin_lon, origin_lat, 0)
    tiepoint = (0.0, 0.0, 0.0, origin_lon, origin_lat, 0.0)
    # ModelPixelScaleTag: (ScaleX, ScaleY, ScaleZ)
    pixel_scale = (pixel_scale_lon, pixel_scale_lat, 0.0)
    # GeoKeyDirectory: standard GeoKey format for EPSG:4326 or projected EPSG
    geo_keys = (
        1, 1, 0, 2,           # Header: DirectoryVersion=1, Revision=1, MinorRevision=0, NumberOfKeys=2
        1024, 0, 1, 2 if epsg_code == 4326 else 1,  # GTModelType: 2=Geographic, 1=Projected
        2048 if epsg_code == 4326 else 3072, 0, 1, epsg_code,  # GeographicTypeGeoKey / ProjectedCSTypeGeoKey
    )

    extratags = [
        (TAG_MODEL_PIXEL_SCALE, "d", 3, pixel_scale, True),
        (TAG_MODEL_TIEPOINT, "d", 6, tiepoint, True),
        (TAG_GEO_KEY_DIRECTORY, "H", len(geo_keys), geo_keys, True),
    ]

    tifffile.imwrite(path, data, extratags=extratags)
    return path


class TestStep9GeospatialIntelligence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.controller = GeoVLMController()
        cls.client = TestClient(app)

        # 1. Create synthetic GeoTIFF fixtures for mathematical unit tests
        cls.geotiff_wgs84_path = create_synthetic_geotiff(
            "test_fixture_wgs84.tif",
            width=200,
            height=200,
            origin_lon=13.4050,
            origin_lat=52.5200,
            pixel_scale_lon=0.0001,
            pixel_scale_lat=0.0001,
            epsg_code=4326,
        )

        cls.geotiff_utm_path = create_synthetic_geotiff(
            "test_fixture_utm.tif",
            width=200,
            height=200,
            origin_lon=500000.0,  # 500,000 m Easting
            origin_lat=5800000.0,  # 5,800,000 m Northing
            pixel_scale_lon=0.5,   # 0.5m GSD
            pixel_scale_lat=0.5,
            epsg_code=32633,       # WGS 84 / UTM zone 33N
        )

        # 2. Standard non-georeferenced images
        cls.plain_png = Image.new("RGB", (200, 200), (34, 139, 34))
        cls.plain_png_path = os.path.join(_dir, "test_plain.png")
        cls.plain_png.save(cls.plain_png_path)

    @classmethod
    def tearDownClass(cls):
        for p in [cls.geotiff_wgs84_path, cls.geotiff_utm_path, cls.plain_png_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    # 1. GeoTIFF Metadata Extraction
    def test_geotiff_metadata_extraction(self):
        print("\n--- [TEST 1] GeoTIFF Metadata Extraction ---")
        meta = GeospatialEngine.extract_metadata(self.geotiff_wgs84_path)
        self.assertTrue(meta.geospatial_available)
        self.assertEqual(meta.crs, "EPSG:4326")
        self.assertEqual(meta.crs_type, "geographic")
        self.assertEqual(meta.crs_units, "degree")
        self.assertIsNotNone(meta.transform)
        self.assertAlmostEqual(meta.transform.c, 13.4050, places=4)
        self.assertAlmostEqual(meta.transform.f, 52.5200, places=4)
        print(f"Extracted WGS84 GeoMetadata: {meta.to_dict()}")

    # 2. CRS Detection (Projected UTM)
    def test_crs_detection_projected_utm(self):
        print("\n--- [TEST 2] Projected UTM CRS Detection ---")
        meta = GeospatialEngine.extract_metadata(self.geotiff_utm_path)
        self.assertTrue(meta.geospatial_available)
        self.assertEqual(meta.crs, "EPSG:32633")
        self.assertEqual(meta.crs_type, "projected")
        self.assertEqual(meta.crs_units, "metre")
        self.assertEqual(meta.resolution["unit"], "metre")
        self.assertAlmostEqual(meta.resolution["x"], 0.5, places=2)
        print(f"Extracted UTM GeoMetadata: {meta.to_dict()}")

    # 3. Affine Transform Validation
    def test_affine_transform_validation(self):
        print("\n--- [TEST 3] Affine Transform Mathematical Inversion ---")
        transform = GeoTransform(a=0.5, b=0.0, c=500000.0, d=0.0, e=-0.5, f=5800000.0)
        # Test Pixel (100, 50) -> World
        X, Y = transform.pixel_to_world(100.0, 50.0)
        self.assertAlmostEqual(X, 500050.0)
        self.assertAlmostEqual(Y, 5799975.0)

        # Invert World (500050.0, 5799975.0) -> Pixel
        px, py = transform.world_to_pixel(X, Y)
        self.assertAlmostEqual(px, 100.0)
        self.assertAlmostEqual(py, 50.0)
        print(f"Pixel (100, 50) <-> World ({X}, {Y}) matches perfectly.")

    # 4. Pixel -> World Conversion
    def test_pixel_to_world_conversion(self):
        print("\n--- [TEST 4] Pixel to World Forward Mapping ---")
        meta = GeospatialEngine.extract_metadata(self.geotiff_wgs84_path)
        lon, lat = meta.transform.pixel_to_world(0.0, 0.0)
        self.assertAlmostEqual(lon, 13.4050, places=4)
        self.assertAlmostEqual(lat, 52.5200, places=4)

        lon_end, lat_end = meta.transform.pixel_to_world(200.0, 200.0)
        self.assertAlmostEqual(lon_end, 13.4050 + 200 * 0.0001, places=4)
        self.assertAlmostEqual(lat_end, 52.5200 - 200 * 0.0001, places=4)
        print(f"Top-Left: ({lon}, {lat}), Bottom-Right: ({lon_end}, {lat_end})")

    # 5. World -> Pixel Conversion
    def test_world_to_pixel_conversion(self):
        print("\n--- [TEST 5] World to Pixel Inverse Mapping ---")
        meta = GeospatialEngine.extract_metadata(self.geotiff_wgs84_path)
        px, py = meta.transform.world_to_pixel(13.4150, 52.5100)
        self.assertAlmostEqual(px, 100.0, places=1)
        self.assertAlmostEqual(py, 100.0, places=1)
        print(f"World (13.4150, 52.5100) maps back to Pixel ({px}, {py})")

    # 6. BBox Geographic Footprint Conversion
    def test_bbox_geographic_footprint(self):
        print("\n--- [TEST 6] BBox Geographic Conversion ---")
        meta = GeospatialEngine.extract_metadata(self.geotiff_wgs84_path)
        bbox_px = [30.0, 20.0, 60.0, 40.0]
        footprint = meta.transform.bbox_pixel_to_world(bbox_px, crs=meta.crs)
        self.assertIn("min_x", footprint)
        self.assertIn("min_y", footprint)
        self.assertIn("max_x", footprint)
        self.assertIn("max_y", footprint)
        self.assertIn("polygon_world", footprint)
        self.assertAlmostEqual(footprint["min_x"], 13.4050 + 30 * 0.0001, places=4)
        self.assertAlmostEqual(footprint["max_y"], 52.5200 - 20 * 0.0001, places=4)
        print(f"BBox World Footprint: {footprint}")

    # 7. Ground Area Calculation
    def test_ground_area_calculation(self):
        print("\n--- [TEST 7] Physical Ground Area Calculation ---")
        # 1. Projected UTM: 100 pixels at 0.5m x 0.5m = 25.0 m^2
        meta_utm = GeospatialEngine.extract_metadata(self.geotiff_utm_path)
        area_res = GeospatialEngine.calculate_ground_area(100.0, meta_utm)
        self.assertTrue(area_res["geospatial_available"])
        self.assertEqual(area_res["ground_area"], 25.0)
        self.assertEqual(area_res["ground_area_unit"], "m2")

        # 2. Plain image with no metadata: ground_area is None
        meta_plain = GeospatialEngine.extract_metadata(self.plain_png)
        area_plain = GeospatialEngine.calculate_ground_area(100.0, meta_plain)
        self.assertFalse(area_plain["geospatial_available"])
        self.assertIsNone(area_plain["ground_area"])
        print(f"Calculated UTM Ground Area: {area_res}")

    # 8. Distance Measurement
    def test_ground_distance_calculation(self):
        print("\n--- [TEST 8] Ground Distance Calculation ---")
        meta_wgs84 = GeospatialEngine.extract_metadata(self.geotiff_wgs84_path)
        p1 = (13.4050, 52.5200)
        p2 = (13.4060, 52.5200)  # ~67 meters east at 52.5 deg latitude
        dist_res = GeospatialEngine.calculate_ground_distance(p1, p2, meta_wgs84)
        self.assertIn(dist_res["method"], ("wgs84_geodesic", "haversine_great_circle"))
        self.assertGreater(dist_res["distance"], 60.0)
        self.assertLess(dist_res["distance"], 75.0)
        print(f"Measured Geodesic Distance: {dist_res['distance']} {dist_res['unit']} ({dist_res['method']})")

    # 9. Missing Metadata Handling (Ordinary PNG/JPEG)
    def test_missing_metadata_handling(self):
        print("\n--- [TEST 9] Missing Metadata Handling ---")
        meta = GeospatialEngine.extract_metadata(self.plain_png)
        self.assertFalse(meta.geospatial_available)
        self.assertIsNone(meta.crs)
        self.assertIsNone(meta.transform)
        self.assertIn("No geospatial metadata available", meta.reason)
        print("Non-georeferenced image handled cleanly without inventing CRS.")

    # 10. Malformed Metadata Rejection
    def test_malformed_metadata_rejection(self):
        print("\n--- [TEST 10] Malformed Metadata Handling ---")
        # Null bytes / non-image data
        meta = GeospatialEngine.extract_metadata(b"malformed binary header")
        self.assertFalse(meta.geospatial_available)
        self.assertIsNone(meta.crs)
        print("Malformed binary stream handled safely without crashing.")

    # 11. Anomaly Geospatial Enrichment
    def test_anomaly_geospatial_enrichment(self):
        print("\n--- [TEST 11] Anomaly Geospatial Enrichment ---")
        img_a = Image.open(self.geotiff_utm_path)
        img_b = img_a.copy()
        draw = ImageDraw.Draw(img_b)
        draw.rectangle([50, 50, 90, 90], fill=(255, 0, 0))

        anom = anomaly_engine.extract_change_anomalies(img_a, img_b)
        self.assertTrue(anom["geospatial_coordinates_available"])
        self.assertGreater(len(anom["regions"]), 0)
        r0 = anom["regions"][0]
        self.assertTrue(r0["geospatial_coordinates_available"])
        self.assertIn("bbox_world", r0)
        self.assertIn("ground_area", r0)
        self.assertEqual(r0["crs"], "EPSG:32633")
        print(f"Enriched Anomaly Region: BBox World={r0['bbox_world']}, Ground Area={r0['ground_area']} m2")

    # 12. Grounding Geospatial Enrichment
    def test_grounding_geospatial_enrichment(self):
        print("\n--- [TEST 12] Grounding Geospatial Enrichment ---")
        meta = GeospatialEngine.extract_metadata(self.geotiff_wgs84_path)
        detection = {
            "type": "object_detection",
            "label": "building",
            "bbox_pixel": [20.0, 30.0, 50.0, 60.0],
            "score": 0.88,
            "source": "Grounding_DINO",
        }
        enriched = GeospatialEngine.enrich_evidence_item(detection, meta)
        self.assertTrue(enriched["geospatial_coordinates_available"])
        self.assertIn("bbox_world", enriched)
        self.assertEqual(enriched["crs"], "EPSG:4326")
        print(f"Enriched Grounding Detection: {enriched}")

    # 13. GeoJSON Export
    def test_geojson_export(self):
        print("\n--- [TEST 13] GeoJSON Export ---")
        meta = GeospatialEngine.extract_metadata(self.geotiff_wgs84_path)
        evidence = [
            {
                "id": "ev_001",
                "type": "object_detection",
                "label": "building",
                "bbox_pixel": [10.0, 10.0, 40.0, 40.0],
                "bbox_world": meta.transform.bbox_pixel_to_world([10.0, 10.0, 40.0, 40.0], "EPSG:4326"),
                "ground_area": 120.5,
                "score": 0.92,
            }
        ]
        geojson = export_evidence_to_geojson(evidence, meta)
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertTrue(geojson["geospatial_available"])
        self.assertEqual(len(geojson["features"]), 1)
        feat = geojson["features"][0]
        self.assertEqual(feat["geometry"]["type"], "Polygon")
        self.assertEqual(feat["properties"]["label"], "building")
        print(f"Generated GeoJSON: {geojson}")

    # 14. Evidence Graph Integration
    def test_evidence_graph_with_world_coordinates(self):
        print("\n--- [TEST 14] Evidence Graph with World Bounding Boxes ---")
        graph = EvidenceGraph(query_id="query_geo_001")
        meta = GeospatialEngine.extract_metadata(self.geotiff_utm_path)
        footprint = meta.transform.bbox_pixel_to_world([10.0, 20.0, 80.0, 90.0], "EPSG:32633")

        node = graph.create_and_add(
            type="change_region",
            source_tool="Change_Analysis",
            source_model="classical_change_analysis",
            payload={
                "bbox_pixel": [10.0, 20.0, 80.0, 90.0],
                "bbox_world": footprint,
                "ground_area": 1225.0,
                "crs": "EPSG:32633",
            },
            confidence=None,
            confidence_type="heuristic",
        )
        self.assertIsNotNone(node)
        self.assertEqual(node.validation_status, "valid")
        print(f"Validated Node in Evidence Graph: {node.to_dict()}")

    # 15. Controller Routing & Spatial Summary
    def test_controller_spatial_summary(self):
        print("\n--- [TEST 15] Controller Spatial Summary ---")
        img_a = Image.open(self.geotiff_utm_path)
        img_b = img_a.copy()
        draw = ImageDraw.Draw(img_b)
        draw.rectangle([40, 40, 80, 80], fill=(255, 255, 255))

        res = self.controller.handle_request(
            query="Where is the changed region and how large is it?",
            change_image_a=img_a,
            change_image_b=img_b,
        )
        self.assertEqual(res["status"], "completed")
        report = res["investigation_report"]
        self.assertIn("spatial_summary", report)
        spatial = report["spatial_summary"]
        self.assertTrue(spatial["geospatial_available"])
        self.assertEqual(spatial["crs"], "EPSG:32633")
        self.assertGreater(spatial["total_ground_area"], 0.0)
        print(f"Investigation Report Spatial Summary: {spatial}")

    # 16. API Compatibility (POST /api/query & POST /api/change-analysis & POST /api/geojson)
    def test_api_compatibility_geospatial(self):
        print("\n--- [TEST 16] API Compatibility: GeoJSON & Metadata ---")
        img_a = Image.open(self.geotiff_utm_path)
        img_b = img_a.copy()
        b64_a = encode_image_b64(img_a)
        b64_b = encode_image_b64(img_b)

        # 1. /api/change-analysis
        res_chg = self.client.post("/api/change-analysis", json={"image_a": b64_a, "image_b": b64_b})
        self.assertEqual(res_chg.status_code, 200)
        data_chg = res_chg.json()
        self.assertIn("geojson", data_chg)
        self.assertIn("geospatial_metadata", data_chg)

        # 2. /api/geojson
        res_geo = self.client.post("/api/geojson", json={"evidence": data_chg.get("evidence", [])})
        self.assertEqual(res_geo.status_code, 200)
        self.assertEqual(res_geo.json()["type"], "FeatureCollection")
        print("API endpoints returned valid geospatial & GeoJSON payloads.")

    # 17. No Fabricated Coordinates for Plain Images
    def test_no_fabricated_coordinates(self):
        print("\n--- [TEST 17] Zero Fabricated Coordinates for Non-Geo Imagery ---")
        res = self.controller.handle_request(
            query="Where is the building located?",
            optical_image=self.plain_png,
        )
        report = res["investigation_report"]
        self.assertFalse(report["spatial_summary"]["geospatial_available"])
        self.assertIsNone(report["spatial_summary"]["crs"])
        for ev in report["evidence"]:
            payload = ev.get("payload", {})
            self.assertFalse(payload.get("geospatial_coordinates_available", False))
            self.assertNotIn("bbox_world", payload)
        print("Plain imagery strictly remained in image-coordinate mode without fabricating lat/lon.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
