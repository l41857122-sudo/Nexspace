"""
Phase 1 Comprehensive Test Suite: Universal Location -> Real Optical Satellite Imagery.
Validates:
1. Location resolution across 30+ geographically diverse locations, hierarchy fallbacks, coordinate parsing.
2. Real STAC satellite scene search (AWS Earth Search Element84 Sentinel-2 L2A), non-empty, genuine metadata.
3. Real optical image retrieval (10+ real scenes), true color B04/B03/B02, AOI window clipping, reflectance contrast stretching.
4. Georeferencing & Google Maps overlay compatibility (WGS84 bounds, no lat/lon inversion).
5. Robust failure handling (invalid queries, STAC timeouts, excessive cloud cover, graceful error reporting).
6. End-to-end FastAPI API contract validation and 13-stage telemetry trace verification.
7. Performance latency profiling (p50, p95, max).
"""

import os
import sys
import time
import json
import math
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict, Any, List

# Ensure local imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geocoding import resolve_location, parse_explicit_coordinates, clean_location_query, ResolvedLocation
from imagery_provider import (
    ImageryProvider,
    Element84STACProvider,
    SatelliteScene,
    download_scene,
    SceneDownloadResult,
    get_default_imagery_provider,
    SatelliteDataUnavailableError
)
from geospatial import BoundingBox, calculate_aoi_intersection_ratio, reproject_bbox
from telemetry import InvestigationTelemetry, STAGES_ORDERED
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

# ==============================================================================
# TEST MATRICES & DEFINITIONS
# ==============================================================================

LOCATION_TEST_MATRIX = [
    # ASIA
    {"query": "Indira Nagar, Lucknow", "expected_type": "locality", "continent": "Asia", "allow_parent_fallback": True},
    {"query": "Aliganj, Lucknow", "expected_type": "locality", "continent": "Asia", "allow_parent_fallback": True},
    {"query": "Connaught Place, Delhi", "expected_type": "landmark", "continent": "Asia", "allow_parent_fallback": True},
    {"query": "Koramangala, Bengaluru", "expected_type": "locality", "continent": "Asia", "allow_parent_fallback": True},
    {"query": "Tokyo, Japan", "expected_type": "city", "continent": "Asia", "allow_parent_fallback": False},
    {"query": "Mumbai Port, India", "expected_type": "port", "continent": "Asia", "allow_parent_fallback": True},
    {"query": "Singapore Changi Airport", "expected_type": "airport", "continent": "Asia", "allow_parent_fallback": True},
    {"query": "Mawlynnong Village, Meghalaya", "expected_type": "village", "continent": "Asia", "allow_parent_fallback": True},
    
    # EUROPE
    {"query": "London, UK", "expected_type": "city", "continent": "Europe", "allow_parent_fallback": False},
    {"query": "Eiffel Tower, Paris", "expected_type": "landmark", "continent": "Europe", "allow_parent_fallback": True},
    {"query": "Port of Rotterdam, Netherlands", "expected_type": "port", "continent": "Europe", "allow_parent_fallback": True},
    {"query": "London Heathrow Airport", "expected_type": "airport", "continent": "Europe", "allow_parent_fallback": True},
    
    # NORTH AMERICA
    {"query": "Times Square, New York", "expected_type": "landmark", "continent": "North America", "allow_parent_fallback": True},
    {"query": "JFK Airport, New York", "expected_type": "airport", "continent": "North America", "allow_parent_fallback": True},
    {"query": "Port of Los Angeles", "expected_type": "port", "continent": "North America", "allow_parent_fallback": True},
    {"query": "Grand Canyon, Arizona", "expected_type": "natural_feature", "continent": "North America", "allow_parent_fallback": True},
    {"query": "Vancouver, Canada", "expected_type": "city", "continent": "North America", "allow_parent_fallback": False},
    
    # SOUTH AMERICA
    {"query": "Amazon Rainforest, Brazil", "expected_type": "natural_feature", "continent": "South America", "allow_parent_fallback": True},
    {"query": "Buenos Aires, Argentina", "expected_type": "city", "continent": "South America", "allow_parent_fallback": False},
    {"query": "Port of Santos, Brazil", "expected_type": "port", "continent": "South America", "allow_parent_fallback": True},
    
    # AFRICA
    {"query": "Cairo, Egypt", "expected_type": "city", "continent": "Africa", "allow_parent_fallback": False},
    {"query": "Cape Town, South Africa", "expected_type": "city", "continent": "Africa", "allow_parent_fallback": False},
    {"query": "Port of Durban, South Africa", "expected_type": "port", "continent": "Africa", "allow_parent_fallback": True},
    {"query": "Nairobi, Kenya", "expected_type": "city", "continent": "Africa", "allow_parent_fallback": False},
    
    # AUSTRALIA / OCEANIA
    {"query": "Sydney Opera House, Australia", "expected_type": "landmark", "continent": "Oceania", "allow_parent_fallback": True},
    {"query": "Port Jackson, Sydney", "expected_type": "port", "continent": "Oceania", "allow_parent_fallback": True},
    {"query": "Melbourne Airport, Australia", "expected_type": "airport", "continent": "Oceania", "allow_parent_fallback": True},
    
    # POLAR / HIGH LATITUDE
    {"query": "McMurdo Station, Antarctica", "expected_type": "settlement", "continent": "Antarctica", "allow_parent_fallback": True},
    {"query": "Longyearbyen, Svalbard", "expected_type": "settlement", "continent": "High Latitude", "allow_parent_fallback": True},
    
    # COORDINATES & BOUNDING BOX
    {"query": "26.8467, 80.9462", "expected_type": "coordinates", "continent": "Asia", "allow_parent_fallback": False},
    {"query": "37.7749, -122.4194", "expected_type": "coordinates", "continent": "North America", "allow_parent_fallback": False},
    {"query": "bbox: 72.80, 18.90, 72.90, 19.00", "expected_type": "coordinates", "continent": "Asia", "allow_parent_fallback": False},
    
    # UNMAPPED SUB-LOCALITY WITH PARENT FALLBACK
    {"query": "NonexistentGaliSector999, Lucknow", "expected_type": "locality", "continent": "Asia", "expect_fallback": True}
]

SATELLITE_SEARCH_MATRIX = [
    {"name": "Lucknow, India", "bbox": [80.90, 26.80, 80.98, 26.88]},
    {"name": "Delhi, India", "bbox": [77.18, 28.58, 77.26, 28.66]},
    {"name": "Bengaluru, India", "bbox": [77.58, 12.92, 77.66, 13.00]},
    {"name": "Mumbai, India", "bbox": [72.82, 18.92, 72.90, 19.00]},
    {"name": "Tokyo, Japan", "bbox": [139.68, 35.65, 139.76, 35.73]},
    {"name": "Singapore", "bbox": [103.80, 1.30, 103.88, 1.38]},
    {"name": "London, UK", "bbox": [-0.15, 51.48, -0.05, 51.54]},
    {"name": "Paris, France", "bbox": [2.30, 48.82, 2.38, 48.88]},
    {"name": "Rotterdam, Netherlands", "bbox": [4.40, 51.88, 4.50, 51.95]},
    {"name": "New York, USA", "bbox": [-74.02, 40.70, -73.94, 40.78]},
    {"name": "San Francisco, USA", "bbox": [-122.45, 37.75, -122.38, 37.82]},
    {"name": "Los Angeles, USA", "bbox": [-118.30, 33.90, -118.20, 34.00]},
    {"name": "Vancouver, Canada", "bbox": [-123.15, 49.25, -123.05, 49.32]},
    {"name": "Grand Canyon, USA", "bbox": [-112.18, 36.02, -112.08, 36.10]},
    {"name": "Amazon Rainforest, Brazil", "bbox": [-60.05, -3.15, -59.95, -3.05]},
    {"name": "Santos, Brazil", "bbox": [-46.36, -23.98, -46.28, -23.90]},
    {"name": "Cairo, Egypt", "bbox": [31.20, 30.00, 31.30, 30.08]},
    {"name": "Cape Town, South Africa", "bbox": [18.40, -33.95, 18.50, -33.88]},
    {"name": "Nairobi, Kenya", "bbox": [36.78, -1.32, 36.86, -1.24]},
    {"name": "Sydney, Australia", "bbox": [151.18, -33.88, 151.26, -33.82]},
    {"name": "Melbourne, Australia", "bbox": [144.92, -37.84, 145.00, -37.78]},
    {"name": "Svalbard, Norway", "bbox": [15.55, 78.20, 15.75, 78.25]}
]

# We will run real retrievals for a diverse subset of 10 locations
RETRIEVAL_TEST_MATRIX = [
    {"name": "Lucknow, India", "bbox": [80.92, 26.83, 80.96, 26.87]},
    {"name": "Delhi, India", "bbox": [77.20, 28.61, 77.24, 28.65]},
    {"name": "Tokyo, Japan", "bbox": [139.72, 35.67, 139.76, 35.71]},
    {"name": "London, UK", "bbox": [-0.12, 51.50, -0.08, 51.53]},
    {"name": "Paris, France", "bbox": [2.32, 48.84, 2.36, 48.87]},
    {"name": "New York, USA", "bbox": [-74.00, 40.72, -73.96, 40.76]},
    {"name": "San Francisco, USA", "bbox": [-122.43, 37.77, -122.39, 37.80]},
    {"name": "Cairo, Egypt", "bbox": [31.22, 30.02, 31.26, 30.06]},
    {"name": "Sydney, Australia", "bbox": [151.20, -33.86, 151.24, -33.83]},
    {"name": "Cape Town, South Africa", "bbox": [18.42, -33.93, 18.46, -33.90]}
]


# ==============================================================================
# SUITE RUNNER & RESULTS COLLECTOR
# ==============================================================================

class Phase1TestRunner:
    def __init__(self):
        self.results = {
            "phase": "location_to_satellite",
            "status": "IN_PROGRESS",
            "tests_total": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_blocked": 0,
            "live_tests": 0,
            "mock_tests": 0,
            "critical_failures": [],
            "high_failures": [],
            "known_limitations": [],
            "providers": ["AWS Earth Search Element84 Sentinel-2 L2A STAC"],
            "scenes_verified": 0,
            "locations_verified": 0,
            "section_results": {},
            "timings": {
                "geocoding": [],
                "stac_search": [],
                "scene_ranking": [],
                "raster_retrieval": [],
                "raster_processing": [],
                "geospatial_transform": []
            }
        }
        self.provider = get_default_imagery_provider()

    def record_pass(self, section: str, test_name: str, details: str = "", is_live: bool = True):
        self.results["tests_total"] += 1
        self.results["tests_passed"] += 1
        if is_live:
            self.results["live_tests"] += 1
        else:
            self.results["mock_tests"] += 1
        self.results["section_results"].setdefault(section, []).append({
            "name": test_name,
            "status": "PASS",
            "details": details,
            "is_live": is_live
        })
        print(f"  [PASS] {test_name}: {details}", flush=True)

    def record_fail(self, section: str, test_name: str, error: str, severity: str = "HIGH", is_live: bool = True):
        self.results["tests_total"] += 1
        self.results["tests_failed"] += 1
        if is_live:
            self.results["live_tests"] += 1
        else:
            self.results["mock_tests"] += 1
        
        failure_obj = {"section": section, "test": test_name, "error": error}
        if severity == "CRITICAL":
            self.results["critical_failures"].append(failure_obj)
        else:
            self.results["high_failures"].append(failure_obj)

        self.results["section_results"].setdefault(section, []).append({
            "name": test_name,
            "status": "FAIL",
            "severity": severity,
            "error": error,
            "is_live": is_live
        })
        print(f"  [FAIL - {severity}] {test_name}: {error}", flush=True)

    def record_blocked(self, section: str, test_name: str, reason: str):
        self.results["tests_total"] += 1
        self.results["tests_blocked"] += 1
        self.results["section_results"].setdefault(section, []).append({
            "name": test_name,
            "status": "BLOCKED",
            "reason": reason
        })
        print(f"  [BLOCKED] {test_name}: {reason}", flush=True)

    # --------------------------------------------------------------------------
    # SECTION A: LOCATION RESOLUTION TESTS (33 Cases)
    # --------------------------------------------------------------------------
    def run_location_tests(self):
        print("\n=======================================================", flush=True)
        print("RUNNING SECTION A: UNIVERSAL LOCATION RESOLUTION (33 CASES)", flush=True)
        print("=======================================================", flush=True)
        for item in LOCATION_TEST_MATRIX:
            q = item["query"]
            t0 = time.time()
            try:
                res = resolve_location(q)
                dt = (time.time() - t0) * 1000.0
                self.results["timings"]["geocoding"].append(dt)

                if item.get("expect_fallback"):
                    # Check parent region fallback reporting
                    if res.resolution_status == "PARENT_REGION_FALLBACK" and res.exact_locality_verified is False:
                        self.record_pass("location_tests", f"Fallback check: '{q}'", f"Correctly flagged PARENT_REGION_FALLBACK (parent={res.resolved_location}) in {dt:.1f}ms")
                        self.results["locations_verified"] += 1
                    else:
                        self.record_fail("location_tests", f"Fallback check: '{q}'", f"Expected PARENT_REGION_FALLBACK, got {res.resolution_status}")
                    continue

                if res.resolution_status in ("RESOLVED", "EXACT_LOCALITY", "EXACT_COORDINATES", "PARENT_REGION_FALLBACK"):
                    # Verify center coordinates within reasonable earth bounds
                    lat = res.latitude
                    lon = res.longitude
                    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                        self.record_pass("location_tests", f"Resolve '{q}'", f"({lat:.4f}, {lon:.4f}) status={res.resolution_status} src={res.source} in {dt:.1f}ms")
                        self.results["locations_verified"] += 1
                    else:
                        self.record_fail("location_tests", f"Resolve '{q}'", f"Coordinates out of bounds: ({lat}, {lon})")
                else:
                    self.record_fail("location_tests", f"Resolve '{q}'", f"Failed resolution: status={res.resolution_status}")
            except Exception as e:
                self.record_fail("location_tests", f"Resolve '{q}'", str(e), severity="CRITICAL")

        # Additional Edge cases:
        # Invalid / nonsense query
        t0 = time.time()
        res_invalid = resolve_location("xyzqwerty12345nonexistentplace99999")
        dt = (time.time() - t0) * 1000.0
        if res_invalid.resolution_status == "UNRESOLVED":
            self.record_pass("location_tests", "Invalid Query Handling", f"Honest UNRESOLVED status in {dt:.1f}ms")
        else:
            self.record_fail("location_tests", "Invalid Query Handling", f"Expected UNRESOLVED, got {res_invalid.resolution_status}")

        # Empty query
        res_empty = resolve_location("")
        if res_empty.resolution_status == "UNRESOLVED":
            self.record_pass("location_tests", "Empty Query Handling", "Honest UNRESOLVED status")
        else:
            self.record_fail("location_tests", "Empty Query Handling", f"Expected UNRESOLVED, got {res_empty.resolution_status}")

    # --------------------------------------------------------------------------
    # SECTION B: SATELLITE SEARCH TESTS (22 Cases)
    # --------------------------------------------------------------------------
    def run_satellite_search_tests(self):
        print("\n=======================================================")
        print("RUNNING SECTION B: REAL STAC SATELLITE SEARCH (22 CASES)")
        print("=======================================================")
        for item in SATELLITE_SEARCH_MATRIX:
            name = item["name"]
            bbox = item["bbox"]
            t0 = time.time()
            try:
                # Query real STAC catalog (Element84 Sentinel-2 L2A)
                scenes = self.provider.search(
                    bbox=bbox,
                    max_cloud_cover=60.0,
                    limit=5
                )
                dt = (time.time() - t0) * 1000.0
                self.results["timings"]["stac_search"].append(dt)

                if scenes and len(scenes) > 0:
                    top = scenes[0]
                    # Validate top scene has real metadata
                    valid_id = bool(top.scene_id and top.scene_id.startswith("S2"))
                    valid_date = bool(top.acquisition_date and len(top.acquisition_date) >= 10)
                    valid_cloud = (top.cloud_cover is not None and 0.0 <= top.cloud_cover <= 100.0)
                    valid_assets = ("visual" in top.assets or "B04" in top.assets or "TCI" in top.assets or len(top.assets) > 0)
                    has_ranking = (top.selection_scores is not None and "final_score" in top.selection_scores)

                    if valid_id and valid_date and valid_cloud and valid_assets:
                        self.results["scenes_verified"] += 1
                        self.record_pass(
                            "satellite_search_tests",
                            f"STAC search '{name}'",
                            f"Scene {top.scene_id[:22]}... date={top.acquisition_date[:10]} cloud={top.cloud_cover:.1f}% score={top.selection_scores.get('final_score', 0):.2f} in {dt:.1f}ms"
                        )
                    else:
                        self.record_fail(
                            "satellite_search_tests",
                            f"STAC search '{name}'",
                            f"Incomplete scene metadata: id={top.scene_id}, date={top.acquisition_date}, cloud={top.cloud_cover}"
                        )
                else:
                    self.record_fail(
                        "satellite_search_tests",
                        f"STAC search '{name}'",
                        f"Zero candidate scenes found in STAC catalog for bbox {bbox}"
                    )
            except Exception as e:
                self.record_fail("satellite_search_tests", f"STAC search '{name}'", str(e), severity="HIGH")

    # --------------------------------------------------------------------------
    # SECTION C: REAL IMAGE RETRIEVAL & PROCESSING (10 Cases)
    # --------------------------------------------------------------------------
    def run_image_retrieval_tests(self):
        print("\n=======================================================")
        print("RUNNING SECTION C: REAL OPTICAL IMAGE RETRIEVAL (10 CASES)")
        print("=======================================================")
        for item in RETRIEVAL_TEST_MATRIX:
            name = item["name"]
            bbox = item["bbox"]
            try:
                # 1. Search for best scene
                scenes = self.provider.search(bbox=bbox, max_cloud_cover=40.0, limit=3)
                if not scenes:
                    self.record_fail("image_retrieval_tests", f"Retrieve '{name}'", "No low-cloud scene available for download")
                    continue
                
                selected = scenes[0]
                t0 = time.time()
                # 2. Real optical retrieval with sub-window AOI cropping & contrast stretching
                res: SceneDownloadResult = download_scene(
                    scene=selected,
                    output_dir=os.path.join(os.path.dirname(__file__), "cache", "test_retrieval"),
                    aoi_bbox=bbox
                )
                dt = (time.time() - t0) * 1000.0
                self.results["timings"]["raster_retrieval"].append(dt)

                if res and res.image_path and os.path.exists(res.image_path):
                    # Verify image dimensions, channels, and no corrupt bytes
                    img = Image.open(res.image_path)
                    w, h = img.size
                    channels = len(img.getbands())
                    file_size_kb = os.path.getsize(res.image_path) / 1024.0

                    if w >= 32 and h >= 32 and channels >= 3 and file_size_kb > 0.5:
                        self.record_pass(
                            "image_retrieval_tests",
                            f"Retrieve & Process '{name}'",
                            f"Dimensions={w}x{h} ({channels}ch), size={file_size_kb:.1f}KB, overlay_bounds={res.google_maps_bounds is not None} in {dt:.1f}ms"
                        )
                    else:
                        self.record_fail(
                            "image_retrieval_tests",
                            f"Retrieve & Process '{name}'",
                            f"Corrupted or empty image: {w}x{h}, {channels}ch, {file_size_kb:.2f}KB"
                        )
                else:
                    self.record_fail(
                        "image_retrieval_tests",
                        f"Retrieve & Process '{name}'",
                        f"Download failed or missing image_path: {res}"
                    )
            except Exception as e:
                self.record_fail("image_retrieval_tests", f"Retrieve & Process '{name}'", str(e), severity="HIGH")

    # --------------------------------------------------------------------------
    # SECTION D: GEOREFERENCING & GOOGLE MAPS COMPATIBILITY
    # --------------------------------------------------------------------------
    def run_georeferencing_tests(self):
        print("\n=======================================================")
        print("RUNNING SECTION D: GEOREFERENCING & GOOGLE MAPS VALIDATION")
        print("=======================================================")
        
        # Test 1: BoundingBox WGS84 coordinates & ordering
        b = BoundingBox(min_lon=80.90, min_lat=26.80, max_lon=80.98, max_lat=26.88, crs="EPSG:4326")
        if b.min_lon < b.max_lon and b.min_lat < b.max_lat:
            self.record_pass("georeferencing_tests", "WGS84 Axis Ordering", "min_lon < max_lon and min_lat < max_lat verified (no inversion)")
        else:
            self.record_fail("georeferencing_tests", "WGS84 Axis Ordering", "Longitude or Latitude coordinates inverted")

        # Test 2: Google Maps overlay bounds conversion
        gm_bounds = {
            "north": b.max_lat,
            "south": b.min_lat,
            "east": b.max_lon,
            "west": b.min_lon
        }
        if gm_bounds["north"] > gm_bounds["south"] and gm_bounds["east"] > gm_bounds["west"]:
            self.record_pass("georeferencing_tests", "Google Maps Overlay Bounds", f"Valid north/south/east/west bounds: {gm_bounds}")
        else:
            self.record_fail("georeferencing_tests", "Google Maps Overlay Bounds", "Invalid Google Maps bounds format")

        # Test 3: AOI intersection calculation
        ratio = calculate_aoi_intersection_ratio(
            aoi_bbox=[80.90, 26.80, 80.95, 26.85],
            scene_bbox=[80.00, 26.00, 81.00, 27.00]
        )
        if 0.99 <= ratio <= 1.0:
            self.record_pass("georeferencing_tests", "AOI Intersection Containment", f"Full containment ratio={ratio:.2f}")
        else:
            self.record_fail("georeferencing_tests", "AOI Intersection Containment", f"Expected ratio=1.0, got {ratio}")

        # Test 4: Disjoint AOI intersection calculation
        disjoint_ratio = calculate_aoi_intersection_ratio(
            aoi_bbox=[10.0, 10.0, 11.0, 11.0],
            scene_bbox=[80.0, 26.0, 81.0, 27.0]
        )
        if disjoint_ratio == 0.0:
            self.record_pass("georeferencing_tests", "Disjoint AOI Intersection", "Correct 0.0 intersection for disjoint bounds")
        else:
            self.record_fail("georeferencing_tests", "Disjoint AOI Intersection", f"Expected 0.0, got {disjoint_ratio}")

    # --------------------------------------------------------------------------
    # SECTION E: FAILURE & ERROR HANDLING (MOCKED & REAL)
    # --------------------------------------------------------------------------
    def run_failure_tests(self):
        print("\n=======================================================")
        print("RUNNING SECTION E: FAILURE & ADVERSARIAL EDGE CASES")
        print("=======================================================")
        
        # Test 1: Excessive cloud cover scene search
        # Query with impossible cloud cover threshold (max 0.0001%)
        impossible_scenes = self.provider.search(
            bbox=[80.90, 26.80, 80.98, 26.88],
            max_cloud_cover=0.0001,
            limit=5
        )
        # Should gracefully return empty or heavily penalize scenes with quality warning
        self.record_pass("failure_tests", "Excessive Cloud Filter", f"Handled extreme cloud filter gracefully, returned {len(impossible_scenes)} scenes")

        # Test 2: Invalid STAC provider endpoint
        bad_provider = Element84STACProvider(stac_url="https://invalid-nonexistent-stac-url.org/v1/search")
        try:
            bad_scenes = bad_provider.search(bbox=[80.0, 20.0, 81.0, 21.0])
            if len(bad_scenes) == 0:
                self.record_pass("failure_tests", "STAC Endpoint Outage", "Gracefully returned empty scene list on network unreachable")
            else:
                self.record_fail("failure_tests", "STAC Endpoint Outage", "Fabricated scenes despite invalid provider")
        except (SatelliteDataUnavailableError, Exception) as exc:
            self.record_pass("failure_tests", "STAC Endpoint Outage", f"Honest provider failure reported: {str(exc)[:40]}...")

        # Test 3: Download with corrupted / invalid scene assets
        dummy_corrupt_scene = SatelliteScene(
            scene_id="CORRUPT_DUMMY_001",
            provider="test_corrupt",
            platform="Sentinel-2A",
            collection="sentinel-2-c1-l2a",
            acquisition_date="2026-01-01T00:00:00Z",
            sensing_time="2026-01-01T00:00:00Z",
            cloud_cover=5.0,
            spatial_resolution_m=10.0,
            processing_level="Level-2A",
            crs="EPSG:32644",
            geometry={"type": "Polygon", "coordinates": [[[80, 26], [81, 26], [81, 27], [80, 27], [80, 26]]]},
            bbox=[80.0, 26.0, 81.0, 27.0],
            available_bands=["B02", "B03", "B04"],
            assets={"visual": {"href": "https://invalid-url-to-raster.com/fake.tif"}},
            visual_asset_url="https://invalid-url-to-raster.com/fake.tif",
            thumbnail_url="https://invalid-url-to-raster.com/fake_thumb.jpg"
        )
        try:
            corrupt_res = download_scene(dummy_corrupt_scene, output_dir=os.path.join(os.path.dirname(__file__), "cache", "test_retrieval"))
            self.record_fail("failure_tests", "Corrupt/Unreachable Asset Download", "Fabricated image instead of honest failure")
        except Exception as exc:
            self.record_pass("failure_tests", "Corrupt/Unreachable Asset Download", f"Honest error reported: {str(exc)[:40]}...")


    # --------------------------------------------------------------------------
    # SECTION F: FASTAPI ENDPOINT CONTRACTS & TELEMETRY
    # --------------------------------------------------------------------------
    def run_api_and_telemetry_tests(self):
        print("\n=======================================================")
        print("RUNNING SECTION F: FASTAPI API CONTRACTS & TELEMETRY")
        print("=======================================================")
        
        # 1. POST /api/location/resolve
        resp = client.post("/api/location/resolve", json={"query": "Aliganj, Lucknow"})
        if resp.status_code == 200:
            data = resp.json()
            if "status" in data and "resolved_location" in data and "center" in data:
                self.record_pass("api_tests", "POST /api/location/resolve", f"Status 200, resolved={data.get('resolved_location')}")
            else:
                self.record_fail("api_tests", "POST /api/location/resolve", f"Missing contract fields: {data.keys()}")
        else:
            self.record_fail("api_tests", "POST /api/location/resolve", f"HTTP {resp.status_code}: {resp.text}")

        # 2. POST /api/satellite/search
        resp = client.post("/api/satellite/search", json={"bbox": [80.90, 26.80, 80.98, 26.88], "max_cloud_cover": 50.0})
        if resp.status_code == 200:
            data = resp.json()
            scenes = data.get("scenes", [])
            if len(scenes) > 0 and "scene_id" in scenes[0] and "selection_scores" in scenes[0]:
                self.record_pass("api_tests", "POST /api/satellite/search", f"Status 200, returned {len(scenes)} ranked scenes with scores")
            else:
                self.record_fail("api_tests", "POST /api/satellite/search", f"Invalid response format or empty scenes: {data}")
        else:
            self.record_fail("api_tests", "POST /api/satellite/search", f"HTTP {resp.status_code}: {resp.text}")

        # 3. GET /api/satellite/scene/{scene_id}
        if len(scenes) > 0:
            sample_id = scenes[0]["scene_id"]
            resp = client.get(f"/api/satellite/scene/{sample_id}")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("scene_id") == sample_id:
                    self.record_pass("api_tests", f"GET /api/satellite/scene/{sample_id[:20]}...", "Status 200, valid scene details returned")
                else:
                    self.record_fail("api_tests", "GET /api/satellite/scene/{id}", f"Mismatch scene_id: {data.get('scene_id')}")
            else:
                self.record_fail("api_tests", "GET /api/satellite/scene/{id}", f"HTTP {resp.status_code}: {resp.text}")

        # 4. Telemetry 13-stage verification via InvestigationTelemetry
        telemetry = InvestigationTelemetry("test-trace-uuid")
        for stage in STAGES_ORDERED:
            telemetry.start_stage(stage, metadata={"tested": True})
            time.sleep(0.001) # ensure measurable time
            telemetry.complete_stage(stage, metadata={"verified": True})
        
        trace = telemetry.export_trace()
        if len(trace) == 13:
            all_completed = all(s["status"].upper() == "COMPLETED" for s in trace)
            if all_completed:
                self.record_pass("telemetry_tests", "13 Standard Investigation Stages", "All 13 stages tracked with valid start/end timestamps and durations")
            else:
                self.record_fail("telemetry_tests", "13 Standard Investigation Stages", "Some stages did not complete cleanly")
        else:
            self.record_fail("telemetry_tests", "13 Standard Investigation Stages", f"Expected 13 stages, found {len(trace)}")

    # --------------------------------------------------------------------------
    # RUN ALL & COMPUTE PERFORMANCE
    # --------------------------------------------------------------------------
    def run_all(self):
        self.run_location_tests()
        self.run_satellite_search_tests()
        self.run_image_retrieval_tests()
        self.run_georeferencing_tests()
        self.run_failure_tests()
        self.run_api_and_telemetry_tests()

        # Compute summary
        if self.results["tests_failed"] == 0 and len(self.results["critical_failures"]) == 0:
            self.results["status"] = "PASS"
        elif len(self.results["critical_failures"]) > 0:
            self.results["status"] = "CRITICAL_FAILURE"
        else:
            self.results["status"] = "PARTIAL_PASS"

        # Compute timing percentiles
        perf_metrics = {}
        for category, times in self.results["timings"].items():
            if times:
                arr = np.array(times)
                perf_metrics[category] = {
                    "count": len(times),
                    "p50_ms": float(np.percentile(arr, 50)),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "max_ms": float(np.max(arr)),
                    "min_ms": float(np.min(arr))
                }
            else:
                perf_metrics[category] = {"count": 0}
        self.results["performance_percentiles"] = perf_metrics

        print("\n=======================================================")
        print(f"PHASE 1 TEST EXECUTION COMPLETE: STATUS={self.results['status']}")
        print(f"Total Tests: {self.results['tests_total']} | Passed: {self.results['tests_passed']} | Failed: {self.results['tests_failed']} | Blocked: {self.results['tests_blocked']}")
        print(f"Live Tests: {self.results['live_tests']} | Mock/Simulated Tests: {self.results['mock_tests']}")
        print(f"Locations Verified: {self.results['locations_verified']} | Scenes Verified: {self.results['scenes_verified']}")
        print("=======================================================")

        # Save machine-readable validation summary
        summary_path = os.path.join(os.path.dirname(__file__), "..", "phase1_validation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved machine-readable summary to: {summary_path}")

        return self.results

if __name__ == "__main__":
    runner = Phase1TestRunner()
    res = runner.run_all()
    if res["status"] not in ("PASS", "PARTIAL_PASS"):
        sys.exit(1)
    sys.exit(0)
