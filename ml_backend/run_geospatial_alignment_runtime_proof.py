"""
NexSpace Gate 3 Step 4: Geospatial Alignment Runtime Proof Engine
Executes Phase 15 (Synthetic Proof), Phase 16 (Real S2 Runtime Proof),
Phase 17 (Optical/SAR Proof), and Phase 18 (Negative Proofs).
"""

import os
import sys
import json
import tempfile
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

sys.path.insert(0, os.path.dirname(__file__))

from co_registration import (
    RasterGeometry,
    GeospatialCoRegistrationEngine,
    CoRegistrationResult,
    AlignmentStatus,
    OverlapStatus,
    TargetGrid,
)
from imagery_provider import Element84STACProvider, ImagerySearchQuery
from multispectral import MultispectralScene, SpectralEngine, SpectralBand
from tools import OpticalSARAnalysisTool, MultispectralAnalysisTool


def run_proofs():
    print("=" * 80)
    print("NEXSPACE GATE 3 STEP 4 — GEOSPATIAL CO-REGISTRATION & ALIGNMENT RUNTIME PROOFS")
    print("=" * 80)

    results = {}

    # -------------------------------------------------------------------------
    # PHASE 15: SYNTHETIC GEOSPATIAL PROOF (Small GeoTIFFs with known metadata)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 15: SYNTHETIC GEOSPATIAL PROOF ---")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_a = os.path.join(tmp_dir, "raster_a_utm_10m.tif")
        path_b = os.path.join(tmp_dir, "raster_b_utm_20m.tif")
        path_c = os.path.join(tmp_dir, "raster_c_wgs84_geo.tif")
        path_d = os.path.join(tmp_dir, "raster_d_utm_disjoint.tif")

        # Raster A: EPSG:32644, 256x256, 10m GSD, Origin=(500000, 2000000)
        transform_a = from_origin(500000.0, 2000000.0, 10.0, 10.0)
        data_a = np.linspace(0.1, 0.9, 256 * 256, dtype=np.float32).reshape(256, 256)
        with rasterio.open(
            path_a, "w", driver="GTiff", height=256, width=256, count=1,
            dtype="float32", crs="EPSG:32644", transform=transform_a, nodata=-9999.0
        ) as dst:
            dst.write(data_a, 1)

        # Raster B: EPSG:32644, 128x128, 20m GSD, Origin=(500000, 2000000) (identical geographic footprint)
        transform_b = from_origin(500000.0, 2000000.0, 20.0, 20.0)
        data_b = np.full((128, 128), 0.45, dtype=np.float32)
        with rasterio.open(
            path_b, "w", driver="GTiff", height=128, width=128, count=1,
            dtype="float32", crs="EPSG:32644", transform=transform_b, nodata=-9999.0
        ) as dst:
            dst.write(data_b, 1)

        # Raster C: EPSG:4326, overlapping footprint (around 78.99 E, 18.08 N)
        # Transform corners of Raster A to WGS84
        bounds_a_dict = {
            "min_x": 500000.0,
            "min_y": 2000000.0 - 256 * 10.0,
            "max_x": 500000.0 + 256 * 10.0,
            "max_y": 2000000.0,
        }
        bounds_c_geo = GeospatialCoRegistrationEngine.transform_bounds(
            bounds_a_dict, "EPSG:32644", "EPSG:4326"
        )
        res_x_geo = (bounds_c_geo["max_x"] - bounds_c_geo["min_x"]) / 100.0
        res_y_geo = (bounds_c_geo["max_y"] - bounds_c_geo["min_y"]) / 100.0
        transform_c = from_origin(bounds_c_geo["min_x"], bounds_c_geo["max_y"], res_x_geo, res_y_geo)
        data_c = np.full((100, 100), 0.70, dtype=np.float32)
        with rasterio.open(
            path_c, "w", driver="GTiff", height=100, width=100, count=1,
            dtype="float32", crs="EPSG:4326", transform=transform_c, nodata=-9999.0
        ) as dst:
            dst.write(data_c, 1)

        # Raster D: EPSG:32644, same dimensions 256x256 and same 10m GSD as A, but 100km away (Origin: 600000, 2100000)
        transform_d = from_origin(600000.0, 2100000.0, 10.0, 10.0)
        data_d = np.full((256, 256), 0.33, dtype=np.float32)
        with rasterio.open(
            path_d, "w", driver="GTiff", height=256, width=256, count=1,
            dtype="float32", crs="EPSG:32644", transform=transform_d, nodata=-9999.0
        ) as dst:
            dst.write(data_d, 1)

        geom_a = RasterGeometry.from_geotiff(path_a)
        geom_b = RasterGeometry.from_geotiff(path_b)
        geom_c = RasterGeometry.from_geotiff(path_c)
        geom_d = RasterGeometry.from_geotiff(path_d)

        print(f"Raster A: CRS={geom_a.crs}, Dim={geom_a.width}x{geom_a.height}, GSD={geom_a.native_gsd}m")
        print(f"Raster B: CRS={geom_b.crs}, Dim={geom_b.width}x{geom_b.height}, GSD={geom_b.native_gsd}m (same footprint)")
        print(f"Raster C: CRS={geom_c.crs}, Dim={geom_c.width}x{geom_c.height}, GSD={geom_c.native_gsd} (overlapping WGS84)")
        print(f"Raster D: CRS={geom_d.crs}, Dim={geom_d.width}x{geom_d.height}, GSD={geom_d.native_gsd}m (distant footprint)")

        # Test Pair A & B (GSD harmonization: 20m -> 10m)
        target_grid_ab, _ = GeospatialCoRegistrationEngine.establish_target_grid(geom_a, geom_b)
        overlap_ab, ratio_ab, _ = GeospatialCoRegistrationEngine.calculate_spatial_overlap(geom_a, geom_b)
        res_b_on_a = GeospatialCoRegistrationEngine.reproject_raster(data_b, geom_b, target_grid_ab)
        print(f"\n[Pair A & B Alignment]")
        print(f"Overlap: {overlap_ab.value} (ratio={ratio_ab:.2f})")
        print(f"Target Grid: {target_grid_ab.crs}, Dim={target_grid_ab.width}x{target_grid_ab.height}, Res={target_grid_ab.resolution_x}m")
        print(f"Resampling Applied: {res_b_on_a.resampling_applied}, Method: {res_b_on_a.resampling_method}")
        print(f"Aligned Data Shape: {res_b_on_a.aligned_raster.shape}")
        assert res_b_on_a.aligned_raster.shape == (256, 256)
        assert overlap_ab == OverlapStatus.OVERLAP_CONFIRMED

        # Test Pair A & C (Cross-CRS Reprojection: WGS84 -> UTM)
        target_grid_ac, _ = GeospatialCoRegistrationEngine.establish_target_grid(geom_a, geom_c)
        overlap_ac, ratio_ac, _ = GeospatialCoRegistrationEngine.calculate_spatial_overlap(geom_a, geom_c)
        res_c_on_a = GeospatialCoRegistrationEngine.reproject_raster(data_c, geom_c, target_grid_ac)
        print(f"\n[Pair A & C Alignment (Cross-CRS: WGS84 -> UTM)]")
        print(f"Overlap: {overlap_ac.value} (ratio={ratio_ac:.2f})")
        print(f"Target Grid CRS: {target_grid_ac.crs}, Reason: {target_grid_ac.target_grid_reason}")
        print(f"Reprojection Status: {res_c_on_a.status.value}")
        print(f"Aligned Data Shape: {res_c_on_a.aligned_raster.shape}")
        assert res_c_on_a.status == AlignmentStatus.REPROJECTED
        assert res_c_on_a.aligned_raster.shape == (256, 256)

        # Test Pair A & D (Same Dimensions 256x256, Same CRS, but Disjoint Bounds)
        overlap_ad, ratio_ad, _ = GeospatialCoRegistrationEngine.calculate_spatial_overlap(geom_a, geom_d)
        print(f"\n[Pair A & D Alignment (Equal Dimensions, Geographically Disjoint)]")
        print(f"Dimensions: A={geom_a.width}x{geom_a.height}, D={geom_d.width}x{geom_d.height}")
        print(f"Overlap Status: {overlap_ad.value} (ratio={ratio_ad:.2f})")
        assert overlap_ad == OverlapStatus.NO_OVERLAP

        results["phase_15_synthetic_proof"] = {
            "status": "PASSED",
            "pair_ab_gsd_harmonization": {
                "source_resolutions": [geom_a.native_gsd, geom_b.native_gsd],
                "target_resolution": target_grid_ab.resolution_x,
                "aligned_shape": list(res_b_on_a.aligned_raster.shape),
                "resampling_method": res_b_on_a.resampling_method,
            },
            "pair_ac_reprojection": {
                "source_crs": geom_c.crs,
                "target_crs": target_grid_ac.crs,
                "status": res_c_on_a.status.value,
                "overlap_ratio": ratio_ac,
            },
            "pair_ad_disjoint_check": {
                "dimensions": "identical_256x256",
                "overlap_status": overlap_ad.value,
                "geospatially_rejected": True,
            },
        }

    # -------------------------------------------------------------------------
    # PHASE 16: REAL SENTINEL-2 RUNTIME PROOF
    # -------------------------------------------------------------------------
    print("\n--- PHASE 16: REAL SENTINEL-2 RUNTIME PROOF ---")
    provider = Element84STACProvider()
    lucknow_bbox = [80.90, 26.80, 81.00, 26.90]
    search_q = ImagerySearchQuery(bbox=lucknow_bbox, max_cloud_cover=20.0, limit=5)
    scenes = provider.search(search_q)
    if not scenes:
        search_q.bbox = [77.10, 28.50, 77.30, 28.70]
        scenes = provider.search(search_q)

    assert len(scenes) > 0, "No live Sentinel-2 scenes found!"
    target_scene = scenes[0]
    print(f"Target Live Scene: {target_scene.scene_id}")

    # Ingest bands B04 (10m Red) and B11 (20m SWIR-1)
    ms_scene = provider.download_multispectral_scene(
        scene=target_scene,
        bands=["B04", "B11"],
        window_size=256,
    )
    b04 = ms_scene.bands["B04"]
    b11 = ms_scene.bands["B11"]

    geom_b04 = RasterGeometry.from_spectral_band(b04)
    geom_b11 = RasterGeometry.from_spectral_band(b11)

    print(f"Band B04: CRS={geom_b04.crs}, GSD={geom_b04.native_gsd}m, Res={geom_b04.resolution_x}m, Shape=({geom_b04.height}, {geom_b04.width})")
    print(f"Band B11: CRS={geom_b11.crs}, GSD={geom_b11.native_gsd}m, Res={geom_b11.resolution_x}m, Shape=({geom_b11.height}, {geom_b11.width})")
    print(f"Transform B04: {geom_b04.transform}")
    print(f"Transform B11: {geom_b11.transform}")

    # Establish target grid and reproject B11 (20m) onto 10m grid
    target_grid_s2, grid_reason = GeospatialCoRegistrationEngine.establish_target_grid(geom_b04, geom_b11)
    overlap_s2, ratio_s2, _ = GeospatialCoRegistrationEngine.calculate_spatial_overlap(geom_b04, geom_b11)

    print(f"Target CRS: {target_grid_s2.crs} ({grid_reason})")
    print(f"Target GSD: {target_grid_s2.resolution_x}m")
    print(f"Overlap Status: {overlap_s2.value} ({ratio_s2*100.0:.1f}%)")

    res_b11_aligned = GeospatialCoRegistrationEngine.reproject_raster(
        b11.data, geom_b11, target_grid_s2, resampling_method="bilinear"
    )
    print(f"Resampling Applied: {res_b11_aligned.resampling_applied} ({res_b11_aligned.resampling_method})")
    print(f"B11 Aligned Shape: {res_b11_aligned.aligned_raster.shape}")
    print(f"Valid Pixels: {res_b11_aligned.valid_pixel_count}, Nodata: {res_b11_aligned.nodata_pixel_count}")
    print(f"Alignment Status: {res_b11_aligned.status.value}")

    assert res_b11_aligned.status in (AlignmentStatus.ALIGNED, AlignmentStatus.RESAMPLED)
    assert res_b11_aligned.aligned_raster.shape == (target_grid_s2.height, target_grid_s2.width)

    results["phase_16_real_runtime_proof"] = {
        "status": "PASSED",
        "scene_id": ms_scene.scene_id,
        "source_crs": geom_b04.crs,
        "source_gsd_b04": geom_b04.native_gsd,
        "source_gsd_b11": geom_b11.native_gsd,
        "target_crs": target_grid_s2.crs,
        "target_gsd": target_grid_s2.resolution_x,
        "overlap_status": overlap_s2.value,
        "overlap_ratio": ratio_s2,
        "resampling_applied": res_b11_aligned.resampling_applied,
        "resampling_method": res_b11_aligned.resampling_method,
        "aligned_dimensions": [target_grid_s2.width, target_grid_s2.height],
        "alignment_status": res_b11_aligned.status.value,
        "valid_pixels": res_b11_aligned.valid_pixel_count,
        "nodata_pixels": res_b11_aligned.nodata_pixel_count,
    }

    # -------------------------------------------------------------------------
    # PHASE 17: LIVE OPTICAL / SAR INTEGRATION PROOF
    # -------------------------------------------------------------------------
    print("\n--- PHASE 17: LIVE OPTICAL / SAR INTEGRATION PROOF ---")
    opt_sar_tool = OpticalSARAnalysisTool()
    img_opt = Image.new("RGB", (256, 256), (120, 160, 200))
    img_sar = Image.new("RGB", (256, 256), (40, 40, 40))

    # Test 1: Independent inputs with matching CRS and overlapping bounds
    exec_res_valid = opt_sar_tool.execute({
        "optical_image": img_opt,
        "sar_image": img_sar,
        "optical_crs": "EPSG:32644",
        "sar_crs": "EPSG:32644",
        "optical_bounds": bounds_a_dict,
        "sar_bounds": bounds_a_dict,
        "optical_transform": (10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0),
        "sar_transform": (10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0),
    })
    res_dict_valid = exec_res_valid.to_dict()
    print(f"Independent Optical+SAR Execution: status={exec_res_valid.status}, fusion_executed={res_dict_valid['data'].get('fusion_executed')}")
    print(f"Geospatial Valid: {res_dict_valid['data'].get('geospatial_valid')}")
    print(f"Alignment Status: {res_dict_valid['data'].get('alignment_status')}")
    print(f"Alignment Telemetry: {res_dict_valid['data'].get('alignment', {}).get('status')}")
    assert exec_res_valid.status == "success"
    assert res_dict_valid["data"]["fusion_executed"] is True
    assert res_dict_valid["data"]["geospatial_valid"] is True
    assert res_dict_valid["data"]["alignment"]["status"] == "ALIGNED"

    # Test 2: Same image supplied twice (Gate 3 Step 2 invariant must reject before fusion)
    exec_res_same = opt_sar_tool.execute({
        "optical_image": img_opt,
        "sar_image": img_opt,
        "optical_crs": "EPSG:32644",
        "sar_crs": "EPSG:32644",
        "optical_bounds": bounds_a_dict,
        "sar_bounds": bounds_a_dict,
    })
    print(f"Same-Image Rejection: status={exec_res_same.status}, provenance={exec_res_same.provenance}")
    assert exec_res_same.status == "failed"
    assert exec_res_same.provenance == "UNAVAILABLE"
    assert exec_res_same.to_dict()["data"]["fusion_executed"] is False

    results["phase_17_optical_sar_proof"] = {
        "status": "PASSED",
        "independent_inputs_executed": True,
        "geospatially_coregistered": True,
        "fusion_executed": True,
        "same_image_rejected_before_fusion": True,
    }

    # -------------------------------------------------------------------------
    # PHASE 18: NEGATIVE PROOFS (10 Mandatory Cases)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 18: NEGATIVE PROOFS ---")
    neg_results = {}

    # 1. Same dimensions, different geographic locations
    overlap_1, _, _ = GeospatialCoRegistrationEngine.calculate_spatial_overlap(geom_a, geom_d)
    neg_results["case_1_same_dims_diff_locations"] = (overlap_1 == OverlapStatus.NO_OVERLAP)

    # 2. Different CRS, non-overlapping areas
    bounds_far_geo = {"min_x": -10.0, "min_y": 50.0, "max_x": -9.0, "max_y": 51.0}
    geom_far_wgs84 = RasterGeometry.from_image_or_metadata(
        np.zeros((100, 100)), crs="EPSG:4326", bounds=bounds_far_geo,
        transform=from_origin(-10.0, 51.0, 0.01, 0.01)
    )
    overlap_2, _, _ = GeospatialCoRegistrationEngine.calculate_spatial_overlap(geom_a, geom_far_wgs84)
    neg_results["case_2_diff_crs_non_overlapping"] = (overlap_2 == OverlapStatus.NO_OVERLAP)

    # 3. Missing CRS
    geom_no_crs = RasterGeometry.from_image_or_metadata(data_a, crs=None, bounds=bounds_a_dict, transform=transform_a)
    neg_results["case_3_missing_crs"] = (not geom_no_crs.is_valid_geospatial())

    # 4. Missing transform
    geom_no_trans = RasterGeometry.from_image_or_metadata(data_a, crs="EPSG:32644", bounds=bounds_a_dict, transform=None)
    neg_results["case_4_missing_transform"] = (not geom_no_trans.is_valid_geospatial())

    # 5. Invalid CRS
    is_valid_crs, _ = GeospatialCoRegistrationEngine.validate_crs("NOT_A_VALID_CRS_99999")
    neg_results["case_5_invalid_crs"] = (not is_valid_crs)

    # 6. Missing bounds (no bounds and no transform to infer bounds from)
    geom_no_bounds = RasterGeometry.from_image_or_metadata(data_a, crs="EPSG:32644", bounds=None, transform=None)
    neg_results["case_6_missing_bounds"] = (not geom_no_bounds.is_valid_geospatial())

    # 7. Non-overlapping Optical / SAR
    bounds_disjoint_dict = {"min_x": 600000.0, "min_y": 2097440.0, "max_x": 602560.0, "max_y": 2100000.0}
    exec_res_non_overlap = opt_sar_tool.execute({
        "optical_image": img_opt,
        "sar_image": img_sar,
        "optical_crs": "EPSG:32644",
        "sar_crs": "EPSG:32644",
        "optical_bounds": bounds_a_dict,
        "sar_bounds": bounds_disjoint_dict,
        "optical_transform": transform_a,
        "sar_transform": transform_d,
    })
    neg_results["case_7_non_overlapping_opt_sar"] = (
        exec_res_non_overlap.status == "failed" and
        exec_res_non_overlap.to_dict()["data"]["fusion_executed"] is False
    )

    # 8. Optical + SAR with different GSD
    geom_sar_20m = RasterGeometry.from_image_or_metadata(
        np.zeros((128, 128)), crs="EPSG:32644", bounds=bounds_a_dict, transform=transform_b
    )
    grid_diff_gsd, _ = GeospatialCoRegistrationEngine.establish_target_grid(geom_a, geom_sar_20m)
    neg_results["case_8_diff_gsd_harmonization"] = (grid_diff_gsd.resolution_x == 10.0)

    # 9. RGB-only image for multispectral
    ms_tool = MultispectralAnalysisTool()
    res_rgb_only = ms_tool.execute({"image": img_opt, "indices": ["NDVI"]})
    neg_results["case_9_rgb_only_multispectral"] = (
        res_rgb_only.status == "unavailable" and
        res_rgb_only.provenance == "UNAVAILABLE"
    )

    # 10. Same-image optical self-substitution attack
    neg_results["case_10_same_image_attack"] = (
        exec_res_same.status == "failed" and
        exec_res_same.to_dict()["data"]["fusion_executed"] is False
    )

    for k, v in neg_results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
        assert v is True, f"Negative proof failed: {k}"

    results["phase_18_negative_proofs"] = {
        "status": "PASSED",
        "checks": neg_results,
    }

    print("\nALL RUNTIME PROOFS COMPLETED SUCCESSFULLY!")
    return results


if __name__ == "__main__":
    res = run_proofs()
    with open("GATE3_STEP4_RUNTIME_PROOF_OUTPUT.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved output to GATE3_STEP4_RUNTIME_PROOF_OUTPUT.json")
