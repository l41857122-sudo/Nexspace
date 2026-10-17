"""
test_gate3_step4_geospatial_alignment.py
-----------------------------------------
Gate 3 Step 4: True Geospatial Co-Registration, CRS Reprojection & GSD Harmonization Test Suite.

Validates the complete 25-case test matrix:
  1. Same CRS + same grid -> ALIGNED
  2. Same CRS + different resolution -> RESAMPLED
  3. Same CRS + different dimensions but overlapping bounds -> geospatially aligned
  4. Different CRS + overlapping geographic area -> REPROJECTED
  5. Different CRS + non-overlapping area -> FAILED / NO_OVERLAP
  6. Missing CRS -> UNAVAILABLE / honest failure
  7. Invalid CRS -> FAILED
  8. Missing transform -> UNAVAILABLE
  9. Different GSD -> explicit harmonization
  10. Non-overlapping bounds -> rejection
  11. Identical dimensions but different geographic locations -> rejection / not aligned
  12. Same dimensions + same geographic grid -> ALIGNED
  13. Continuous raster -> valid continuous bilinear resampling
  14. Categorical mask -> nearest-neighbor resampling
  15. Nodata preservation -> nodata remains nodata
  16. CRS reprojection preserves geographic footprint
  17. Target transform is deterministic
  18. Target grid dimensions are deterministic
  19. Optical + independent SAR integration
  20. Same-image Optical + SAR still rejected
  21. Multispectral B11 20m -> 10m alignment
  22. NDVI still works after alignment
  23. NDWI still works after alignment
  24. NDBI still works after alignment
  25. RGB-only NDVI remains unavailable
"""

from __future__ import annotations
import os
import sys
import unittest
import math
import numpy as np
from PIL import Image
from rasterio.warp import Resampling

_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(_dir)
ml_dir = os.path.join(root_dir, "ml_backend")
if ml_dir not in sys.path:
    sys.path.insert(0, ml_dir)

from co_registration import (
    RasterGeometry,
    TargetGrid,
    CoRegistrationResult,
    GeospatialCoRegistrationEngine,
    AlignmentStatus,
    OverlapStatus,
)
from tools import OpticalSARAnalysisTool, MultispectralAnalysisTool
from multispectral import SpectralBand, MultispectralScene, SpectralEngine


class TestGate3Step4GeospatialAlignment(unittest.TestCase):

    def setUp(self):
        # Base synthetic UTM 44N grid (e.g. central India / Hyderabad region)
        # Origin: (500000.0, 2000000.0)
        self.crs_utm = "EPSG:32644"
        self.transform_10m = (10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)
        self.width_256 = 256
        self.height_256 = 256
        # Bounds: [500000, 2000000 - 2560, 500000 + 2560, 2000000] -> [500000, 1997440, 502560, 2000000]
        self.bounds_utm_a = {
            "min_x": 500000.0,
            "min_y": 1997440.0,
            "max_x": 502560.0,
            "max_y": 2000000.0,
        }

        # Continuous test raster: gradient
        y, x = np.mgrid[0:256, 0:256]
        self.raster_continuous = (x + y).astype(np.float32)

        # Categorical mask test raster: 3 distinct classes (10, 20, 30)
        self.raster_categorical = np.zeros((256, 256), dtype=np.uint8)
        self.raster_categorical[:80, :] = 10
        self.raster_categorical[80:160, :] = 20
        self.raster_categorical[160:, :] = 30

    # ------------------------------------------------------------------
    # 1. Same CRS + same grid -> ALIGNED
    # ------------------------------------------------------------------
    def test_01_same_crs_same_grid(self):
        geom_a = RasterGeometry.from_image_or_metadata(
            self.raster_continuous,
            crs=self.crs_utm,
            transform=self.transform_10m,
            bounds=self.bounds_utm_a,
        )
        geom_b = RasterGeometry.from_image_or_metadata(
            self.raster_continuous * 1.5,
            crs=self.crs_utm,
            transform=self.transform_10m,
            bounds=self.bounds_utm_a,
        )
        res_a, res_b = GeospatialCoRegistrationEngine.co_register_pair(
            self.raster_continuous,
            self.raster_continuous * 1.5,
            geom_a,
            geom_b,
        )
        self.assertEqual(res_a.status, AlignmentStatus.ALIGNED)
        self.assertEqual(res_b.status, AlignmentStatus.ALIGNED)
        self.assertEqual(res_a.overlap_status, OverlapStatus.OVERLAP_CONFIRMED)
        self.assertAlmostEqual(res_a.overlap_ratio, 1.0, places=3)
        self.assertFalse(res_a.resampling_applied)

    # ------------------------------------------------------------------
    # 2. Same CRS + different resolution -> RESAMPLED
    # ------------------------------------------------------------------
    def test_02_same_crs_different_resolution(self):
        # Raster B is 20m native resolution covering the exact same area (128x128)
        transform_20m = (20.0, 0.0, 500000.0, 0.0, -20.0, 2000000.0)
        raster_b = np.full((128, 128), 42.0, dtype=np.float32)

        geom_a = RasterGeometry.from_image_or_metadata(
            self.raster_continuous,
            crs=self.crs_utm,
            transform=self.transform_10m,
            bounds=self.bounds_utm_a,
        )
        geom_b = RasterGeometry.from_image_or_metadata(
            raster_b,
            crs=self.crs_utm,
            transform=transform_20m,
            bounds=self.bounds_utm_a,
        )
        res_a, res_b = GeospatialCoRegistrationEngine.co_register_pair(
            self.raster_continuous,
            raster_b,
            geom_a,
            geom_b,
        )
        self.assertEqual(res_b.status, AlignmentStatus.RESAMPLED)
        self.assertTrue(res_b.resampling_applied)
        self.assertEqual(res_b.aligned_raster.shape, (256, 256))
        self.assertEqual(res_b.target_resolution["x"], 10.0)

    # ------------------------------------------------------------------
    # 3. Same CRS + different dimensions but overlapping bounds
    # ------------------------------------------------------------------
    def test_03_same_crs_different_dimensions_overlapping(self):
        # Raster B offset by 500m east, size 200x200 at 10m
        transform_b = (10.0, 0.0, 500500.0, 0.0, -10.0, 2000000.0)
        raster_b = np.ones((200, 200), dtype=np.float32)
        bounds_b = {"min_x": 500500.0, "min_y": 1998000.0, "max_x": 502500.0, "max_y": 2000000.0}

        geom_a = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        geom_b = RasterGeometry.from_image_or_metadata(
            raster_b, crs=self.crs_utm, transform=transform_b, bounds=bounds_b
        )
        res_a, res_b = GeospatialCoRegistrationEngine.co_register_pair(
            self.raster_continuous, raster_b, geom_a, geom_b
        )
        self.assertEqual(res_a.overlap_status, OverlapStatus.OVERLAP_CONFIRMED)
        self.assertGreater(res_a.overlap_ratio, 0.0)
        self.assertEqual(res_a.aligned_raster.shape, res_b.aligned_raster.shape)

    # ------------------------------------------------------------------
    # 4. Different CRS + overlapping geographic area -> REPROJECTED
    # ------------------------------------------------------------------
    def test_04_different_crs_overlapping(self):
        # Reproject bounds of A to EPSG:4326
        import pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)
        w, s = transformer.transform(500000.0, 1997440.0)
        e, n = transformer.transform(502560.0, 2000000.0)

        bounds_geo = {"min_x": min(w, e), "min_y": min(s, n), "max_x": max(w, e), "max_y": max(s, n)}
        transform_geo = ((max(w, e) - min(w, e)) / 100.0, 0.0, min(w, e), 0.0, -(max(s, n) - min(s, n)) / 100.0, max(s, n))
        raster_geo = np.full((100, 100), 55.0, dtype=np.float32)

        geom_a = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs="EPSG:32644", transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        geom_geo = RasterGeometry.from_image_or_metadata(
            raster_geo, crs="EPSG:4326", transform=transform_geo, bounds=bounds_geo
        )

        res_a, res_geo = GeospatialCoRegistrationEngine.co_register_pair(
            self.raster_continuous, raster_geo, geom_a, geom_geo
        )
        self.assertEqual(res_geo.status, AlignmentStatus.REPROJECTED)
        self.assertTrue(res_geo.resampling_applied)
        self.assertEqual(res_geo.target_crs, "EPSG:32644")
        self.assertEqual(res_a.aligned_raster.shape, res_geo.aligned_raster.shape)

    # ------------------------------------------------------------------
    # 5. Different CRS + non-overlapping area -> FAILED
    # ------------------------------------------------------------------
    def test_05_different_crs_non_overlapping(self):
        # Raster in London (EPSG:4326) vs Raster in India (EPSG:32644)
        bounds_london = {"min_x": -0.15, "min_y": 51.48, "max_x": -0.10, "max_y": 51.52}
        transform_london = (0.0005, 0.0, -0.15, 0.0, -0.0004, 51.52)
        raster_london = np.zeros((100, 100), dtype=np.float32)

        geom_a = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs="EPSG:32644", transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        geom_london = RasterGeometry.from_image_or_metadata(
            raster_london, crs="EPSG:4326", transform=transform_london, bounds=bounds_london
        )

        res_a, res_london = GeospatialCoRegistrationEngine.co_register_pair(
            self.raster_continuous, raster_london, geom_a, geom_london
        )
        self.assertEqual(res_a.status, AlignmentStatus.FAILED)
        self.assertEqual(res_a.overlap_status, OverlapStatus.NO_OVERLAP)

    # ------------------------------------------------------------------
    # 6. Missing CRS -> UNAVAILABLE
    # ------------------------------------------------------------------
    def test_06_missing_crs(self):
        geom_a = RasterGeometry.from_image_or_metadata(self.raster_continuous, crs=None)
        geom_b = RasterGeometry.from_image_or_metadata(self.raster_continuous, crs="EPSG:32644")

        res_a, res_b = GeospatialCoRegistrationEngine.co_register_pair(
            self.raster_continuous, self.raster_continuous, geom_a, geom_b
        )
        self.assertEqual(res_a.status, AlignmentStatus.UNAVAILABLE)
        self.assertIn("requires genuine CRS", res_a.error)

    # ------------------------------------------------------------------
    # 7. Invalid CRS -> FAILED
    # ------------------------------------------------------------------
    def test_07_invalid_crs(self):
        valid, err = GeospatialCoRegistrationEngine.validate_crs("NOT_A_VALID_CRS_CODE_XYZ_9999")
        self.assertFalse(valid)
        self.assertIn("Invalid CRS", err)

    # ------------------------------------------------------------------
    # 8. Missing transform -> UNAVAILABLE
    # ------------------------------------------------------------------
    def test_08_missing_transform(self):
        geom_a = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs="EPSG:32644", transform=None, bounds=self.bounds_utm_a
        )
        self.assertFalse(geom_a.is_valid_geospatial())

    # ------------------------------------------------------------------
    # 9. Different GSD -> explicit harmonization
    # ------------------------------------------------------------------
    def test_09_different_gsd_harmonization(self):
        # 10m vs 30m raster
        transform_30m = (30.0, 0.0, 500000.0, 0.0, -30.0, 2000000.0)
        geom_10m = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        geom_30m = RasterGeometry.from_image_or_metadata(
            np.ones((64, 64), dtype=np.float32), crs=self.crs_utm, transform=transform_30m, bounds=self.bounds_utm_a
        )
        target_grid, _ = GeospatialCoRegistrationEngine.establish_target_grid(geom_10m, geom_30m)
        self.assertIsNotNone(target_grid)
        self.assertEqual(target_grid.resolution_x, 10.0)  # Harmonized to finest GSD

    # ------------------------------------------------------------------
    # 10. Non-overlapping bounds -> rejection
    # ------------------------------------------------------------------
    def test_10_non_overlapping_bounds_rejected(self):
        bounds_disjoint = {"min_x": 600000.0, "min_y": 2097440.0, "max_x": 602560.0, "max_y": 2100000.0}
        transform_disjoint = (10.0, 0.0, 600000.0, 0.0, -10.0, 2100000.0)

        geom_a = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        geom_b = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=transform_disjoint, bounds=bounds_disjoint
        )
        status, ratio, _ = GeospatialCoRegistrationEngine.calculate_spatial_overlap(geom_a, geom_b)
        self.assertEqual(status, OverlapStatus.NO_OVERLAP)
        self.assertEqual(ratio, 0.0)

    # ------------------------------------------------------------------
    # 11. Identical dimensions but different geographic locations
    # ------------------------------------------------------------------
    def test_11_identical_dimensions_different_locations(self):
        # Both are 256x256, but B is located in a different UTM coordinate block
        bounds_b = {"min_x": 700000.0, "min_y": 3000000.0, "max_x": 702560.0, "max_y": 3002560.0}
        transform_b = (10.0, 0.0, 700000.0, 0.0, -10.0, 3002560.0)

        geom_a = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        geom_b = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=transform_b, bounds=bounds_b
        )
        res_a, res_b = GeospatialCoRegistrationEngine.co_register_pair(
            self.raster_continuous, self.raster_continuous, geom_a, geom_b
        )
        self.assertEqual(res_a.status, AlignmentStatus.FAILED)
        self.assertEqual(res_a.overlap_status, OverlapStatus.NO_OVERLAP)

    # ------------------------------------------------------------------
    # 12. Same dimensions + same geographic grid -> ALIGNED
    # ------------------------------------------------------------------
    def test_12_same_dimensions_same_grid(self):
        geom_a = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        geom_b = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        res_a, res_b = GeospatialCoRegistrationEngine.co_register_pair(
            self.raster_continuous, self.raster_continuous, geom_a, geom_b
        )
        self.assertEqual(res_a.status, AlignmentStatus.ALIGNED)
        self.assertFalse(res_a.resampling_applied)

    # ------------------------------------------------------------------
    # 13. Continuous raster -> valid continuous resampling (bilinear)
    # ------------------------------------------------------------------
    def test_13_continuous_raster_bilinear(self):
        # 2x upsampling
        geom_src = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        target_grid = TargetGrid(
            crs=self.crs_utm,
            bounds=self.bounds_utm_a,
            transform=(5.0, 0.0, 500000.0, 0.0, -5.0, 2000000.0),
            width=512,
            height=512,
            resolution_x=5.0,
            resolution_y=5.0,
            target_grid_reason="test_upsampling",
        )
        res = GeospatialCoRegistrationEngine.reproject_raster(
            self.raster_continuous, geom_src, target_grid, resampling=Resampling.bilinear
        )
        self.assertEqual(res.aligned_raster.shape, (512, 512))
        self.assertEqual(res.resampling_method, "bilinear")
        self.assertTrue(np.all(res.aligned_raster >= 0.0))

    # ------------------------------------------------------------------
    # 14. Categorical mask -> nearest-neighbor resampling
    # ------------------------------------------------------------------
    def test_14_categorical_mask_nearest(self):
        geom_src = RasterGeometry.from_image_or_metadata(
            self.raster_categorical, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        target_grid = TargetGrid(
            crs=self.crs_utm,
            bounds=self.bounds_utm_a,
            transform=(20.0, 0.0, 500000.0, 0.0, -20.0, 2000000.0),
            width=128,
            height=128,
            resolution_x=20.0,
            resolution_y=20.0,
            target_grid_reason="test_downsampling",
        )
        res = GeospatialCoRegistrationEngine.reproject_raster(
            self.raster_categorical, geom_src, target_grid, resampling=Resampling.nearest
        )
        # All pixel values in resampled categorical array must strictly be in {10, 20, 30}
        unique_vals = set(np.unique(res.aligned_raster))
        self.assertTrue(unique_vals.issubset({10, 20, 30}), f"Categorical values violated: {unique_vals}")

    # ------------------------------------------------------------------
    # 15. Nodata preservation
    # ------------------------------------------------------------------
    def test_15_nodata_preservation(self):
        raster_with_nan = self.raster_continuous.copy()
        raster_with_nan[:50, :50] = np.nan

        geom_src = RasterGeometry.from_image_or_metadata(
            raster_with_nan, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a, nodata=np.nan
        )
        target_grid = TargetGrid(
            crs=self.crs_utm,
            bounds=self.bounds_utm_a,
            transform=self.transform_10m,
            width=256,
            height=256,
            resolution_x=10.0,
            resolution_y=10.0,
            target_grid_reason="test_nodata",
        )
        res = GeospatialCoRegistrationEngine.reproject_raster(
            raster_with_nan, geom_src, target_grid, dst_nodata=np.nan
        )
        self.assertGreater(res.nodata_pixel_count, 0)
        self.assertTrue(np.isnan(res.aligned_raster[0, 0]))

    # ------------------------------------------------------------------
    # 16. CRS reprojection preserves geographic footprint
    # ------------------------------------------------------------------
    def test_16_reprojection_preserves_footprint(self):
        # Reproject from UTM to WGS84 and check transformed bounds encompass Hyderabad
        bounds_geo = GeospatialCoRegistrationEngine.transform_bounds(
            self.bounds_utm_a, "EPSG:32644", "EPSG:4326"
        )
        self.assertGreater(bounds_geo["min_x"], 70.0)
        self.assertLess(bounds_geo["max_x"], 85.0)
        self.assertGreater(bounds_geo["min_y"], 15.0)
        self.assertLess(bounds_geo["max_y"], 25.0)

    # ------------------------------------------------------------------
    # 17. Target transform is deterministic
    # ------------------------------------------------------------------
    def test_17_target_transform_deterministic(self):
        geom_a = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        geom_b = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        g1, _ = GeospatialCoRegistrationEngine.establish_target_grid(geom_a, geom_b)
        g2, _ = GeospatialCoRegistrationEngine.establish_target_grid(geom_a, geom_b)
        self.assertEqual(g1.transform, g2.transform)
        self.assertEqual(g1.width, g2.width)
        self.assertEqual(g1.height, g2.height)

    # ------------------------------------------------------------------
    # 18. Target grid dimensions are deterministic
    # ------------------------------------------------------------------
    def test_18_target_grid_dimensions_deterministic(self):
        geom_a = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        geom_b = RasterGeometry.from_image_or_metadata(
            self.raster_continuous, crs=self.crs_utm, transform=self.transform_10m, bounds=self.bounds_utm_a
        )
        g, _ = GeospatialCoRegistrationEngine.establish_target_grid(geom_a, geom_b)
        self.assertEqual(g.width, 256)
        self.assertEqual(g.height, 256)

    # ------------------------------------------------------------------
    # 19. Optical + independent SAR integration
    # ------------------------------------------------------------------
    def test_19_optical_sar_tool_integration(self):
        tool = OpticalSARAnalysisTool()
        img_opt = Image.new("RGB", (256, 256), (100, 150, 200))
        img_sar = Image.new("RGB", (256, 256), (30, 30, 30))

        res = tool.execute({
            "optical_image": img_opt,
            "sar_image": img_sar,
            "optical_crs": self.crs_utm,
            "sar_crs": self.crs_utm,
            "optical_bounds": self.bounds_utm_a,
            "sar_bounds": self.bounds_utm_a,
            "optical_transform": self.transform_10m,
            "sar_transform": self.transform_10m,
        })
        d = res.to_dict()
        self.assertEqual(res.status, "success")
        self.assertTrue(d["data"]["fusion_executed"])
        self.assertTrue(d["data"]["geospatial_valid"])
        self.assertIsNotNone(d["data"].get("alignment"))
        self.assertEqual(d["data"]["alignment"]["status"], "ALIGNED")

    # ------------------------------------------------------------------
    # 20. Same-image Optical + SAR still rejected (Gate 3 Step 2 invariant)
    # ------------------------------------------------------------------
    def test_20_same_image_optical_sar_rejected(self):
        tool = OpticalSARAnalysisTool()
        img_opt = Image.new("RGB", (256, 256), (100, 150, 200))
        res = tool.execute({
            "optical_image": img_opt,
            "sar_image": img_opt,
            "optical_crs": self.crs_utm,
            "sar_crs": self.crs_utm,
        })
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.provenance, "UNAVAILABLE")

    def _make_scene(self, scene_id: str, bands: Dict[str, SpectralBand]) -> MultispectralScene:
        return MultispectralScene(
            scene_id=scene_id,
            bands=bands,
            loaded_bands=list(bands.keys()),
            available_bands=list(bands.keys()),
            width=256,
            height=256,
            crs=self.crs_utm,
            transform=self.transform_10m,
            bounds=self.bounds_utm_a,
            resolution_m=10.0,
            acquisition_time="2026-09-26T00:00:00Z",
            source="Sentinel-2 L2A",
            spectral_source="REAL_MULTISPECTRAL",
        )

    # ------------------------------------------------------------------
    # 21. Multispectral B11 20m -> 10m alignment
    # ------------------------------------------------------------------
    def test_21_multispectral_b11_geospatial_alignment(self):
        b04 = SpectralBand(
            name="B04", common_name="red", data=np.full((256, 256), 0.15, dtype=np.float32),
            native_width=256, native_height=256, native_resolution_m=10.0,
            crs=self.crs_utm, transform=self.transform_10m, wavelength_um=0.665,
        )
        b08 = SpectralBand(
            name="B08", common_name="nir", data=np.full((256, 256), 0.45, dtype=np.float32),
            native_width=256, native_height=256, native_resolution_m=10.0,
            crs=self.crs_utm, transform=self.transform_10m, wavelength_um=0.842,
        )
        t_20m = (20.0, 0.0, 500000.0, 0.0, -20.0, 2000000.0)
        b11 = SpectralBand(
            name="B11", common_name="swir1", data=np.full((128, 128), 0.25, dtype=np.float32),
            native_width=128, native_height=128, native_resolution_m=20.0,
            crs=self.crs_utm, transform=t_20m, wavelength_um=1.610,
        )

        scene = self._make_scene("S2B_TEST_GEOSPATIAL", {"B04": b04, "B08": b08, "B11": b11})
        aligned_b11 = scene.align_band_to_grid(b11, (256, 256))
        self.assertEqual(aligned_b11.shape, (256, 256))
        self.assertAlmostEqual(float(np.mean(aligned_b11)), 0.25, places=3)

    # ------------------------------------------------------------------
    # 22. NDVI still works after alignment
    # ------------------------------------------------------------------
    def test_22_ndvi_still_works(self):
        b04 = SpectralBand(
            name="B04", common_name="red", data=np.full((256, 256), 0.10, dtype=np.float32),
            native_width=256, native_height=256, native_resolution_m=10.0,
            crs=self.crs_utm, transform=self.transform_10m, wavelength_um=0.665,
        )
        b08 = SpectralBand(
            name="B08", common_name="nir", data=np.full((256, 256), 0.50, dtype=np.float32),
            native_width=256, native_height=256, native_resolution_m=10.0,
            crs=self.crs_utm, transform=self.transform_10m, wavelength_um=0.842,
        )
        scene = self._make_scene("S2B_NDVI_TEST", {"B04": b04, "B08": b08})
        res = SpectralEngine.compute_ndvi(scene)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.provenance, "REAL_MULTISPECTRAL")
        expected_val = (0.50 - 0.10) / (0.50 + 0.10)
        self.assertAlmostEqual(res.stats.mean, expected_val, places=4)

    # ------------------------------------------------------------------
    # 23. NDWI still works after alignment
    # ------------------------------------------------------------------
    def test_23_ndwi_still_works(self):
        b03 = SpectralBand(
            name="B03", common_name="green", data=np.full((256, 256), 0.40, dtype=np.float32),
            native_width=256, native_height=256, native_resolution_m=10.0,
            crs=self.crs_utm, transform=self.transform_10m, wavelength_um=0.560,
        )
        b08 = SpectralBand(
            name="B08", common_name="nir", data=np.full((256, 256), 0.20, dtype=np.float32),
            native_width=256, native_height=256, native_resolution_m=10.0,
            crs=self.crs_utm, transform=self.transform_10m, wavelength_um=0.842,
        )
        scene = self._make_scene("S2B_NDWI_TEST", {"B03": b03, "B08": b08})
        res = SpectralEngine.compute_ndwi(scene)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.provenance, "REAL_MULTISPECTRAL")
        expected_val = (0.40 - 0.20) / (0.40 + 0.20)
        self.assertAlmostEqual(res.stats.mean, expected_val, places=4)

    # ------------------------------------------------------------------
    # 24. NDBI still works after alignment
    # ------------------------------------------------------------------
    def test_24_ndbi_still_works(self):
        b08 = SpectralBand(
            name="B08", common_name="nir", data=np.full((256, 256), 0.30, dtype=np.float32),
            native_width=256, native_height=256, native_resolution_m=10.0,
            crs=self.crs_utm, transform=self.transform_10m, wavelength_um=0.842,
        )
        t_20m = (20.0, 0.0, 500000.0, 0.0, -20.0, 2000000.0)
        b11 = SpectralBand(
            name="B11", common_name="swir1", data=np.full((128, 128), 0.50, dtype=np.float32),
            native_width=128, native_height=128, native_resolution_m=20.0,
            crs=self.crs_utm, transform=t_20m, wavelength_um=1.610,
        )
        scene = self._make_scene("S2B_NDBI_TEST", {"B08": b08, "B11": b11})
        res = SpectralEngine.compute_ndbi(scene)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.provenance, "REAL_MULTISPECTRAL")
        self.assertTrue(res.resampling_applied)
        expected_val = (0.50 - 0.30) / (0.50 + 0.30)
        self.assertAlmostEqual(res.stats.mean, expected_val, places=4)

    # ------------------------------------------------------------------
    # 25. RGB-only NDVI remains unavailable
    # ------------------------------------------------------------------
    def test_25_rgb_only_ndvi_unavailable(self):
        tool = MultispectralAnalysisTool()
        rgb_img = Image.new("RGB", (256, 256), (80, 140, 60))
        res = tool.execute({"image": rgb_img, "indices": ["NDVI"]})
        self.assertEqual(res.status, "unavailable")
        self.assertEqual(res.provenance, "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
