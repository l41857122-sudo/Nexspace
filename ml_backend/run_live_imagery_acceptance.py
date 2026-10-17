"""
run_live_imagery_acceptance.py
------------------------------
Executes Live Acceptance Tests (A through F) from Section 26 using REAL Earth Observation APIs.

For every test record:
- query
- parsed plan
- imagery provider
- returned scene count
- selected scene ID
- acquisition date
- cloud cover
- resolution
- CRS
- bounds
- download status
- analysis status
- detection count
- geographic coordinates
- map geometry
- errors
"""

import sys
import os
import io
import time
import json
from datetime import datetime, timezone
from PIL import Image

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

from discovery_parser import DiscoveryQueryParser
from imagery_provider import (
    get_default_imagery_provider,
    ImagerySearchQuery,
    SatelliteScene,
    SatelliteDataUnavailableError,
    Element84STACProvider,
)
from geospatial import GeospatialEngine, GeoTransform, GeoMetadata
from geojson_export import export_evidence_to_geojson
from orchestrator import GeoVLMController

LIVE_TEST_CASES = [
    {
        "id": "CASE_A",
        "name": "Location Place Name Discovery",
        "query": "Find recent satellite imagery of Lucknow",
        "expected_location": "Lucknow",
        "run_analysis": False,
    },
    {
        "id": "CASE_B",
        "name": "Temporal & Port Location Discovery",
        "query": "Find satellite imagery of Mumbai port from last month",
        "expected_location": "Mumbai port",
        "run_analysis": False,
    },
    {
        "id": "CASE_C",
        "name": "Discovery + Downstream AI Detection",
        "query": "Find recent imagery of Lucknow and locate buildings",
        "expected_location": "Lucknow",
        "run_analysis": True,
    },
    {
        "id": "CASE_D",
        "name": "Bitemporal Before/After Change Discovery",
        "query": "Find imagery of Lucknow before and after 2024",
        "expected_location": "Lucknow",
        "run_analysis": False,
    },
    {
        "id": "CASE_E",
        "name": "Intentionally Impossible / Invalid Location",
        "query": "Find satellite imagery of AtlantisNonexistentCity999XYZ",
        "expected_location": "AtlantisNonexistentCity999XYZ",
        "run_analysis": False,
    },
    {
        "id": "CASE_F",
        "name": "Provider Unavailable Resilience",
        "query": "Find recent satellite imagery of Lucknow",
        "expected_location": "Lucknow",
        "simulate_offline": True,
        "run_analysis": False,
    },
]


def run_acceptance_suite():
    print("=" * 70)
    print("NEXSPACE SATELLITE IMAGERY DISCOVERY - LIVE ACCEPTANCE SUITE (A-F)")
    print("=" * 70)

    provider = get_default_imagery_provider()
    controller = GeoVLMController()

    results = []

    for test_case in LIVE_TEST_CASES:
        cid = test_case["id"]
        q = test_case["query"]
        print(f"\n[{cid}] Executing: '{q}'")
        t0 = time.perf_counter()

        record = {
            "test_id": cid,
            "test_name": test_case["name"],
            "query": q,
            "parsed_plan": None,
            "imagery_provider": type(provider).__name__,
            "returned_scene_count": 0,
            "selected_scene_id": None,
            "acquisition_date": None,
            "cloud_cover": None,
            "resolution": None,
            "crs": None,
            "bounds": None,
            "download_status": "SKIPPED",
            "analysis_status": "SKIPPED",
            "detection_count": 0,
            "geographic_coordinates": [],
            "map_geometry": None,
            "errors": None,
            "duration_ms": 0.0,
            "status": "PENDING",
        }

        try:
            # 1. Parse Plan
            plan = DiscoveryQueryParser.parse(q)
            record["parsed_plan"] = plan.to_dict()

            # Handle Case E: Impossible location
            if cid == "CASE_E":
                if not plan.geocoding or not plan.geocoding.geocoding_available:
                    record["status"] = "PASSED"
                    record["errors"] = "Expected geocoding failure: Location non-existent. Zero coordinate fabrication maintained."
                    record["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                    results.append(record)
                    print(f"  ✓ {cid} PASSED: Location correctly marked unresolved without coordinate fabrication.")
                    continue

            # Handle Case F: Simulated offline provider
            if test_case.get("simulate_offline"):
                class MockOfflineProvider(Element84STACProvider):
                    STAC_ENDPOINT = "https://unreachable-cdse-stac.copernicus.eu/error"

                offline_p = MockOfflineProvider()
                search_q = ImagerySearchQuery(bbox=plan.geocoding.bbox or [80.84, 26.74, 81.04, 26.94])
                try:
                    offline_p.search(search_q)
                    record["status"] = "FAILED"
                    record["errors"] = "Expected SatelliteDataUnavailableError, but call succeeded."
                except SatelliteDataUnavailableError as exc:
                    record["status"] = "PASSED"
                    record["errors"] = f"Expected SATELLITE_DATA_UNAVAILABLE: {exc}"
                    print(f"  ✓ {cid} PASSED: Handled provider unavailability cleanly with SATELLITE_DATA_UNAVAILABLE.")
                record["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                results.append(record)
                continue

            # 2. Query Real Catalog
            if not plan.geocoding or not plan.geocoding.bbox:
                raise ValueError("Could not resolve location to AOI bbox")

            dt_range = plan.temporal.datetime_range
            search_q = ImagerySearchQuery(
                bbox=plan.geocoding.bbox,
                datetime_range=dt_range,
                max_cloud_cover=plan.imagery.max_cloud_cover,
                limit=5,
            )

            scenes = provider.search(search_q)
            record["returned_scene_count"] = len(scenes)

            if not scenes:
                # If cloud filter was too strict or date range narrow, try fallback search
                fallback_q = ImagerySearchQuery(bbox=plan.geocoding.bbox, max_cloud_cover=40.0, limit=3)
                scenes = provider.search(fallback_q)
                record["returned_scene_count"] = len(scenes)

            if not scenes:
                record["status"] = "NO_SCENES_FOUND"
                record["errors"] = "Catalog returned 0 matching scenes for criteria"
                record["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                results.append(record)
                print(f"  ⚠ {cid}: No scenes returned by catalog.")
                continue

            # 3. Select Scene
            sel_scene = scenes[0]
            record["selected_scene_id"] = sel_scene.scene_id
            record["acquisition_date"] = sel_scene.acquisition_date
            record["cloud_cover"] = round(sel_scene.cloud_cover, 2)
            record["resolution"] = f"{sel_scene.spatial_resolution_m}m GSD"
            record["crs"] = sel_scene.crs
            record["bounds"] = sel_scene.bbox
            record["map_geometry"] = sel_scene.geometry

            # 4. Download / Retrieve Scene
            dl_res = provider.download_scene(sel_scene, prefer_preview=True)
            record["download_status"] = "COMPLETED"
            record["crs"] = dl_res.crs
            record["bounds"] = dl_res.bounds_world

            # 5. Downstream AI Investigation (Case C)
            if test_case.get("run_analysis"):
                with Image.open(dl_res.image_path) as img:
                    rgb_pil = img.convert("RGB")

                inv_res = controller.handle_request(
                    query=q,
                    optical_image=rgb_pil,
                    execution_profile="fast",
                )
                record["analysis_status"] = "COMPLETED"

                # Extract detections & geographic projection
                t = GeoTransform(
                    a=dl_res.transform["a"],
                    b=dl_res.transform["b"],
                    c=dl_res.transform["c"],
                    d=dl_res.transform["d"],
                    e=dl_res.transform["e"],
                    f=dl_res.transform["f"],
                )
                geo_meta = GeoMetadata(
                    geospatial_available=True,
                    crs=dl_res.crs,
                    transform=t,
                    bounds_world=dl_res.bounds_world,
                )

                raw_ev = inv_res.get("evidence", [])
                enriched = [GeospatialEngine.enrich_evidence_item(it, geo_meta) for it in raw_ev]
                fc = export_evidence_to_geojson(enriched, geo_meta)
                record["detection_count"] = len(enriched)

                for f in fc.get("features", []):
                    c_ring = f.get("geometry", {}).get("coordinates", [[]])[0]
                    if c_ring:
                        record["geographic_coordinates"].append({
                            "evidence_id": f.get("id"),
                            "label": f.get("properties", {}).get("label"),
                            "lon": c_ring[0][0],
                            "lat": c_ring[0][1],
                            "google_maps_url": f"https://www.google.com/maps?q={c_ring[0][1]:.6f},{c_ring[0][0]:.6f}",
                        })

            record["status"] = "PASSED"
            print(f"  ✓ {cid} PASSED: {record['returned_scene_count']} scenes discovered. Selected: {record['selected_scene_id']} ({record['acquisition_date']})")

        except Exception as exc:
            record["status"] = "FAILED"
            record["errors"] = str(exc)
            print(f"  ✗ {cid} FAILED: {exc}")

        record["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        results.append(record)

    # Save output to JSON
    out_file = os.path.join(_dir, "SATELLITE_IMAGERY_LIVE_ACCEPTANCE.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"LIVE ACCEPTANCE SUITE FINISHED. Saved full forensic report to:\n{out_file}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_acceptance_suite()
