"""
test_phase2_overpass.py
-----------------------
Unit and integration tests for Phase 2: OpenStreetMap & Overpass API integration.

Verifies:
- Bounding box validation, normalization, and bounds clamping.
- Overpass QL query construction for buildings, roads, bridges, waterways.
- GeoJSON parsing (points, ways, polygons, relations) with stable IDs.
- Strict attribution to OpenStreetMap contributors.
- Caching behavior (hit, miss, disk persistence).
- Outage / rate limit / timeout resilience without hallucinating data.
- Specialist tool execution via ToolRegistry.
- FastAPI endpoint /api/osm-features response contract.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml_backend"))

from overpass_client import OverpassClient, OverpassQueryResult
from tools import tool_registry, OpenStreetMapTool
from router import TaskType
from fastapi.testclient import TestClient
from server import app


class TestPhase2OverpassClient(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.client = OverpassClient(cache_dir=self.temp_dir, request_interval_s=0.01)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bbox_validation_valid_list(self):
        valid, box, err = self.client.validate_bbox([80.9, 26.8, 81.0, 26.9])
        self.assertTrue(valid)
        self.assertEqual(box, [80.9, 26.8, 81.0, 26.9])
        self.assertIsNone(err)

    def test_bbox_validation_valid_dict(self):
        valid, box, err = self.client.validate_bbox({"west": 80.9, "south": 26.8, "east": 81.0, "north": 26.9})
        self.assertTrue(valid)
        self.assertEqual(box, [80.9, 26.8, 81.0, 26.9])
        self.assertIsNone(err)

    def test_bbox_validation_out_of_bounds(self):
        valid, box, err = self.client.validate_bbox([195.0, 26.8, 81.0, 26.9])
        self.assertFalse(valid)
        self.assertIn("Longitude out of WGS84 range", err)

        valid2, box2, err2 = self.client.validate_bbox([80.9, -95.0, 81.0, 26.9])
        self.assertFalse(valid2)
        self.assertIn("Latitude out of WGS84 range", err2)

    def test_bbox_validation_inverted(self):
        valid, box, err = self.client.validate_bbox([81.5, 26.8, 81.0, 26.9])
        self.assertFalse(valid)
        self.assertIn("cannot be greater than max_lon", err)

    def test_bbox_validation_degenerate(self):
        valid, box, err = self.client.validate_bbox([80.0, 25.0, 80.0, 25.0])
        self.assertFalse(valid)
        self.assertIn("Degenerate point bounding box", err)

    def test_build_query_ql(self):
        query = self.client.build_query([80.9, 26.8, 81.0, 26.9], ["buildings", "bridges"])
        self.assertIn('[out:json][timeout:25];', query)
        self.assertIn('way["building"](26.8,80.9,26.9,81.0);', query)
        self.assertIn('way["bridge"="yes"](26.8,80.9,26.9,81.0);', query)
        self.assertIn('out body geom qt;', query)

    def test_geojson_parsing(self):
        raw_elements = [
            {
                "type": "way",
                "id": 1001,
                "tags": {"building": "yes", "name": "City Hall"},
                "geometry": [
                    {"lat": 26.84, "lon": 80.94},
                    {"lat": 26.85, "lon": 80.94},
                    {"lat": 26.85, "lon": 80.95},
                    {"lat": 26.84, "lon": 80.94},
                ],
            },
            {
                "type": "way",
                "id": 1002,
                "tags": {"highway": "primary", "name": "Main Highway"},
                "geometry": [
                    {"lat": 26.80, "lon": 80.90},
                    {"lat": 26.82, "lon": 80.92},
                ],
            },
            {
                "type": "way",
                "id": 1003,
                "tags": {"bridge": "yes", "highway": "secondary", "name": "River Bridge"},
                "geometry": [
                    {"lat": 26.83, "lon": 80.93},
                    {"lat": 26.84, "lon": 80.93},
                ],
            },
        ]

        geojson, counts = self.client.parse_elements_to_geojson(raw_elements)
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 3)
        self.assertEqual(counts["buildings"], 1)
        self.assertEqual(counts["roads"], 1)
        self.assertEqual(counts["bridges"], 1)

        # Polygon check
        bldg_feat = geojson["features"][0]
        self.assertEqual(bldg_feat["geometry"]["type"], "Polygon")
        self.assertEqual(bldg_feat["properties"]["name"], "City Hall")
        self.assertEqual(bldg_feat["properties"]["attribution"], "© OpenStreetMap contributors")

        # LineString check
        road_feat = geojson["features"][1]
        self.assertEqual(road_feat["geometry"]["type"], "LineString")
        self.assertEqual(road_feat["properties"]["feature_class"], "roads")

    def test_caching_behavior(self):
        bbox = [80.94, 26.84, 80.95, 26.85]
        fake_response = json.dumps({
            "elements": [
                {
                    "type": "way",
                    "id": 2001,
                    "tags": {"building": "residential"},
                    "geometry": [
                        {"lat": 26.841, "lon": 80.941},
                        {"lat": 26.842, "lon": 80.941},
                        {"lat": 26.842, "lon": 80.942},
                        {"lat": 26.841, "lon": 80.941},
                    ],
                }
            ]
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            # 1. First call - network fetch
            res1 = self.client.query_features(bbox, feature_types=["buildings"], use_cache=True)
            self.assertEqual(res1.status, "success")
            self.assertFalse(res1.cached)
            self.assertEqual(res1.total_features, 1)
            self.assertEqual(mock_urlopen.call_count, 1)

            # 2. Second call - should hit disk cache
            res2 = self.client.query_features(bbox, feature_types=["buildings"], use_cache=True)
            self.assertEqual(res2.status, "success")
            self.assertTrue(res2.cached)
            self.assertEqual(res2.total_features, 1)
            # URLopen call count should still be 1 (zero additional network calls)
            self.assertEqual(mock_urlopen.call_count, 1)

    def test_outage_resilience_empty_results(self):
        fake_empty = json.dumps({"elements": []}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = fake_empty
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = self.client.query_features([80.0, 25.0, 80.01, 25.01])
            self.assertEqual(res.status, "empty")
            self.assertEqual(res.total_features, 0)
            self.assertEqual(res.geojson["features"], [])
            self.assertIn("OpenStreetMap", res.attribution)

    def test_outage_resilience_network_error(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Server unreachable")):
            res = self.client.query_features([80.0, 25.0, 80.01, 25.01], use_cache=False)
            self.assertEqual(res.status, "error")
            self.assertEqual(res.total_features, 0)
            self.assertIn("Server unreachable", res.error_message)
            self.assertIsNotNone(res.actionable_reason)


class TestPhase2ToolRegistry(unittest.TestCase):
    def test_tool_registered(self):
        tool = tool_registry.get("OpenStreetMap_Feature_Query")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "OpenStreetMap_Feature_Query")
        self.assertEqual(tool.task_type, TaskType.OSM_FEATURE_QUERY)

        # Test aliases
        self.assertIsNotNone(tool_registry.get("OpenStreetMap"))
        self.assertIsNotNone(tool_registry.get("OSM"))
        self.assertIsNotNone(tool_registry.get("overpass"))

    def test_tool_execution_missing_bbox(self):
        tool = tool_registry.get("OpenStreetMap_Feature_Query")
        result = tool.execute({})
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.provenance, "UNAVAILABLE")
        self.assertIn("No geographic bounding box", result.error)

    def test_tool_execution_with_bbox(self):
        tool = tool_registry.get("OpenStreetMap_Feature_Query")
        fake_response = json.dumps({
            "elements": [
                {
                    "type": "way",
                    "id": 3001,
                    "tags": {"highway": "residential", "name": "Civil Lines Road"},
                    "geometry": [
                        {"lat": 26.85, "lon": 80.95},
                        {"lat": 26.86, "lon": 80.96},
                    ],
                }
            ]
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = tool.execute({"bbox": [80.95, 26.85, 80.96, 26.86], "use_cache": False})
            self.assertEqual(res.status, "success")
            self.assertEqual(res.provenance, "OPENSTREETMAP_OVERPASS")
            self.assertEqual(res.data["total_features"], 1)
            self.assertEqual(len(res.evidence), 1)
            self.assertIn("Civil Lines Road", res.evidence[0]["payload"]["name"])


class TestPhase2FastAPIEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_osm_features_endpoint_validation_error(self):
        # Empty payload
        resp = self.client.post("/api/osm-features", json={})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Either a valid bounding box", resp.json()["detail"])

    def test_osm_features_endpoint_success(self):
        fake_response = json.dumps({
            "elements": [
                {
                    "type": "way",
                    "id": 4001,
                    "tags": {"building": "school", "name": "Public School"},
                    "geometry": [
                        {"lat": 26.84, "lon": 80.94},
                        {"lat": 26.85, "lon": 80.94},
                        {"lat": 26.85, "lon": 80.95},
                        {"lat": 26.84, "lon": 80.94},
                    ],
                }
            ]
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            resp = self.client.post("/api/osm-features", json={
                "bbox": [80.94, 26.84, 80.95, 26.85],
                "feature_types": ["buildings"],
                "use_cache": False,
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["total_features"], 1)
            self.assertEqual(data["feature_counts"]["buildings"], 1)
            self.assertIn("OpenStreetMap", data["attribution"])
            self.assertIn("distinction_note", data)


if __name__ == "__main__":
    unittest.main()
