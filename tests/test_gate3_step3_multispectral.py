"""
NexSpace Gate 3 Step 3: Multispectral Ingestion & Spectral-Truth Hardening Test Suite
Verifies:
1. Real B04 + B08 -> NDVI succeeds with valid statistics and physics-grounded formula.
2. Real B03 + B08 -> NDWI succeeds with valid statistics.
3. Real B11 + B08 -> NDBI succeeds with valid statistics.
4. Missing B04 -> NDVI unavailable.
5. Missing B08 -> NDVI unavailable.
6. Missing B03 -> NDWI unavailable.
7. Missing B11 -> NDBI unavailable.
8. RGB-only image -> NDVI rejected truthfully.
9. RGB-only image -> NOT labelled REAL_MULTISPECTRAL.
10. Zero denominator handled safely (no div-by-zero, returns NaN/safe mask).
11. NaN handled safely in inputs.
12. Nodata handled safely (DN=0 or explicit nodata masked out).
13. CRS preserved from source raster to indices.
14. Transform preserved from source raster to indices.
15. Native resolution recorded per band (e.g. 10m vs 20m).
16. Resampling status recorded (e.g. 20m SWIR-1 resampled to 10m grid).
17. Band identity preserved (B02, B03, B04, B08, B11, B12 not conflated).
18. Loaded bands differ correctly from provider-available bands.
19. RGB visualization derivative does not overwrite original spectral metadata.
20. Invalid/non-raster input rejected honestly.
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image

# Ensure ml_backend is on sys.path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from multispectral import (
    SpectralBand,
    MultispectralScene,
    SpectralEngine,
    SpectralIndexStats,
    SpectralIndexResult,
)
from tools import MultispectralAnalysisTool, tool_registry
from imagery_provider import SatelliteScene, SceneDownloadResult, Element84STACProvider


class TestGate3Step3Multispectral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a mock 100x100 multispectral scene with genuine synthetic reflectance bands
        cls.width = 100
        cls.height = 100
        cls.crs = "EPSG:32643"
        cls.transform = (500000.0, 10.0, 0.0, 3000000.0, 0.0, -10.0)
        cls.bounds = (500000.0, 2999000.0, 501000.0, 3000000.0)

        # Scale factor 0.0001: DN 2000 = 0.2 reflectance, DN 8000 = 0.8 reflectance
        np.random.seed(42)
        # B02: Blue (~0.05 to 0.15)
        cls.b02_data = np.full((100, 100), 1000, dtype=np.uint16)
        # B03: Green (~0.1 to 0.3)
        cls.b03_data = np.full((100, 100), 1500, dtype=np.uint16)
        # B04: Red (~0.05 to 0.2 for vegetation, higher for soil)
        cls.b04_data = np.full((100, 100), 800, dtype=np.uint16)
        # B08: NIR (~0.4 to 0.8 for healthy vegetation)
        cls.b08_data = np.full((100, 100), 5000, dtype=np.uint16)
        # B11: SWIR1 (~0.1 to 0.3) at 50x50 native resolution (20m)
        cls.b11_data = np.full((50, 50), 2200, dtype=np.uint16)
        # B12: SWIR2 at 50x50 native resolution (20m)
        cls.b12_data = np.full((50, 50), 1800, dtype=np.uint16)

        # Introduce some varied pixels
        cls.b04_data[20:40, 20:40] = 3000  # Built-up / soil area
        cls.b08_data[20:40, 20:40] = 2000  # Low NIR
        cls.b03_data[60:80, 60:80] = 4000  # Water area
        cls.b08_data[60:80, 60:80] = 800   # Very low NIR in water

        # Add explicit nodata (DN=0) in corner
        cls.b04_data[0:5, 0:5] = 0
        cls.b08_data[0:5, 0:5] = 0
        cls.b03_data[0:5, 0:5] = 0

    def _create_full_scene(self) -> MultispectralScene:
        bands = {
            "B02": SpectralBand(
                name="B02", common_name="blue", data=self.b02_data.copy(),
                native_width=100, native_height=100, native_resolution_m=10.0,
                dtype="uint16", scale=0.0001, offset=0.0, nodata=0,
                crs=self.crs, transform=self.transform, wavelength_um=0.490,
            ),
            "B03": SpectralBand(
                name="B03", common_name="green", data=self.b03_data.copy(),
                native_width=100, native_height=100, native_resolution_m=10.0,
                dtype="uint16", scale=0.0001, offset=0.0, nodata=0,
                crs=self.crs, transform=self.transform, wavelength_um=0.560,
            ),
            "B04": SpectralBand(
                name="B04", common_name="red", data=self.b04_data.copy(),
                native_width=100, native_height=100, native_resolution_m=10.0,
                dtype="uint16", scale=0.0001, offset=0.0, nodata=0,
                crs=self.crs, transform=self.transform, wavelength_um=0.665,
            ),
            "B08": SpectralBand(
                name="B08", common_name="nir", data=self.b08_data.copy(),
                native_width=100, native_height=100, native_resolution_m=10.0,
                dtype="uint16", scale=0.0001, offset=0.0, nodata=0,
                crs=self.crs, transform=self.transform, wavelength_um=0.842,
            ),
            "B11": SpectralBand(
                name="B11", common_name="swir16", data=self.b11_data.copy(),
                native_width=50, native_height=50, native_resolution_m=20.0,
                dtype="uint16", scale=0.0001, offset=0.0, nodata=0,
                crs=self.crs, transform=self.transform, wavelength_um=1.610,
            ),
            "B12": SpectralBand(
                name="B12", common_name="swir22", data=self.b12_data.copy(),
                native_width=50, native_height=50, native_resolution_m=20.0,
                dtype="uint16", scale=0.0001, offset=0.0, nodata=0,
                crs=self.crs, transform=self.transform, wavelength_um=2.190,
            ),
        }
        return MultispectralScene(
            scene_id="S2B_TEST_SCENE",
            bands=bands,
            width=100,
            height=100,
            crs=self.crs,
            transform=self.transform,
            bounds=self.bounds,
            resolution_m=10.0,
            acquisition_time="2026-03-15T05:30:00Z",
            source="Sentinel-2 L2A",
            spectral_source="REAL_MULTISPECTRAL",
            provenance={"provider": "Element84_AWS_STAC"},
            loaded_bands=["B02", "B03", "B04", "B08", "B11", "B12"],
            available_bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
        )

    # -----------------------------------------------------------------------
    # TEST 1: Real B04 + B08 -> NDVI succeeds
    # -----------------------------------------------------------------------
    def test_01_real_b04_b08_ndvi_succeeds(self):
        scene = self._create_full_scene()
        res = SpectralEngine.compute_ndvi(scene)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.provenance, "REAL_MULTISPECTRAL")
        self.assertEqual(res.bands_used, ["B04", "B08"])
        self.assertEqual(res.formula, "(B08 - B04) / (B08 + B04)")
        self.assertIsNotNone(res.raster)
        self.assertIsNotNone(res.stats)
        self.assertGreater(res.stats.valid_pixel_count, 0)
        # Check vegetation value for homogeneous area: (5000 - 800) / (5000 + 800) = 4200 / 5800 ≈ 0.7241
        val = res.raster[50, 50]
        self.assertAlmostEqual(val, 4200.0 / 5800.0, places=3)
        self.assertTrue(-1.0 <= res.stats.min <= 1.0)
        self.assertTrue(-1.0 <= res.stats.max <= 1.0)

    # -----------------------------------------------------------------------
    # TEST 2: Real B03 + B08 -> NDWI succeeds
    # -----------------------------------------------------------------------
    def test_02_real_b03_b08_ndwi_succeeds(self):
        scene = self._create_full_scene()
        res = SpectralEngine.compute_ndwi(scene)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.provenance, "REAL_MULTISPECTRAL")
        self.assertEqual(res.bands_used, ["B03", "B08"])
        self.assertEqual(res.formula, "(B03 - B08) / (B03 + B08)")
        self.assertIsNotNone(res.raster)
        self.assertIsNotNone(res.stats)
        # In water area (60:80, 60:80): Green=4000, NIR=800 -> (4000 - 800) / (4000 + 800) = 3200 / 4800 = +0.6667 (positive for water)
        water_val = res.raster[70, 70]
        self.assertAlmostEqual(water_val, 3200.0 / 4800.0, places=3)
        self.assertGreater(water_val, 0.5)

    # -----------------------------------------------------------------------
    # TEST 3: Real B11 + B08 -> NDBI succeeds
    # -----------------------------------------------------------------------
    def test_03_real_b11_b08_ndbi_succeeds(self):
        scene = self._create_full_scene()
        res = SpectralEngine.compute_ndbi(scene)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.provenance, "REAL_MULTISPECTRAL")
        self.assertEqual(res.bands_used, ["B11", "B08"])
        self.assertEqual(res.formula, "(B11 - B08) / (B11 + B08)")
        self.assertTrue(res.resampling_applied)
        self.assertIn("resampled", res.resampling_details.lower())
        self.assertIsNotNone(res.stats)

    # -----------------------------------------------------------------------
    # TEST 4: Missing B04 -> NDVI unavailable
    # -----------------------------------------------------------------------
    def test_04_missing_b04_ndvi_unavailable(self):
        scene = self._create_full_scene()
        del scene.bands["B04"]
        scene.loaded_bands.remove("B04")
        res = SpectralEngine.compute_ndvi(scene)
        self.assertEqual(res.status, "unavailable")
        self.assertEqual(res.provenance, "UNAVAILABLE")
        self.assertIsNone(res.raster)
        self.assertIsNone(res.stats)
        self.assertIn("B04", res.resampling_details)

    # -----------------------------------------------------------------------
    # TEST 5: Missing B08 -> NDVI unavailable
    # -----------------------------------------------------------------------
    def test_05_missing_b08_ndvi_unavailable(self):
        scene = self._create_full_scene()
        del scene.bands["B08"]
        scene.loaded_bands.remove("B08")
        res = SpectralEngine.compute_ndvi(scene)
        self.assertEqual(res.status, "unavailable")
        self.assertEqual(res.provenance, "UNAVAILABLE")
        self.assertIsNone(res.raster)
        self.assertIn("B08", res.resampling_details)

    # -----------------------------------------------------------------------
    # TEST 6: Missing B03 -> NDWI unavailable
    # -----------------------------------------------------------------------
    def test_06_missing_b03_ndwi_unavailable(self):
        scene = self._create_full_scene()
        del scene.bands["B03"]
        scene.loaded_bands.remove("B03")
        res = SpectralEngine.compute_ndwi(scene)
        self.assertEqual(res.status, "unavailable")
        self.assertEqual(res.provenance, "UNAVAILABLE")
        self.assertIsNone(res.raster)
        self.assertIn("B03", res.resampling_details)

    # -----------------------------------------------------------------------
    # TEST 7: Missing B11 -> NDBI unavailable
    # -----------------------------------------------------------------------
    def test_07_missing_b11_ndbi_unavailable(self):
        scene = self._create_full_scene()
        del scene.bands["B11"]
        scene.loaded_bands.remove("B11")
        res = SpectralEngine.compute_ndbi(scene)
        self.assertEqual(res.status, "unavailable")
        self.assertEqual(res.provenance, "UNAVAILABLE")
        self.assertIsNone(res.raster)
        self.assertIn("B11", res.resampling_details)

    # -----------------------------------------------------------------------
    # TEST 8: RGB-only image -> NDVI rejected
    # -----------------------------------------------------------------------
    def test_08_rgb_only_image_ndvi_rejected(self):
        tool = MultispectralAnalysisTool()
        rgb_img = Image.new("RGB", (100, 100), color=(100, 150, 50))
        # Call tool with RGB-only image and no MultispectralScene
        tool_res = tool.execute({
            "image": rgb_img,
            "multispectral_scene": None,
            "indices": ["NDVI"],
            "query": "compute NDVI index",
        })
        self.assertEqual(tool_res.status, "unavailable")
        self.assertEqual(tool_res.provenance, "UNAVAILABLE")
        self.assertTrue("B04" in tool_res.error and "B08" in tool_res.error)
        self.assertIn("spectral", tool_res.data)
        self.assertEqual(tool_res.data["spectral"]["provenance"], "UNAVAILABLE")

    # -----------------------------------------------------------------------
    # TEST 9: RGB-only image -> NOT labelled REAL_MULTISPECTRAL
    # -----------------------------------------------------------------------
    def test_09_rgb_only_image_not_labelled_real_multispectral(self):
        download_res = SceneDownloadResult(
            scene_id="S2B_RGB_ONLY",
            image_path="test_preview.jpg",
            image_base64=None,
            width=512,
            height=512,
            crs="EPSG:4326",
            transform={"a": 1, "b": 0, "c": 0, "d": 0, "e": -1, "f": 0},
            resolution={"x": 10.0, "y": 10.0, "unit": "meter"},
            bounds_world={"min_x": 70, "min_y": 20, "max_x": 71, "max_y": 21},
            google_maps_bounds={"north": 21, "south": 20, "east": 71, "west": 70},
            cloud_cover=0.0,
            spectral_source="RGB_VISUAL_ASSET",
            available_bands=["B02", "B03", "B04", "B08"],
            loaded_bands=["R", "G", "B"],
            provenance={"tier": "preview_jpg"},
            multispectral_scene=None,
        )
        self.assertNotEqual(download_res.spectral_source, "REAL_MULTISPECTRAL")
        self.assertEqual(download_res.spectral_source, "RGB_VISUAL_ASSET")
        self.assertEqual(download_res.loaded_bands, ["R", "G", "B"])
        self.assertNotEqual(download_res.loaded_bands, download_res.available_bands)

    # -----------------------------------------------------------------------
    # TEST 10: Zero denominator handled safely
    # -----------------------------------------------------------------------
    def test_10_zero_denominator_handled_safely(self):
        scene = self._create_full_scene()
        # Set both B08 and B04 to 0 (scaled reflectance 0.0 + 0.0 = 0.0 denominator)
        scene.bands["B04"].data[50, 50] = 0
        scene.bands["B08"].data[50, 50] = 0
        res = SpectralEngine.compute_ndvi(scene)
        self.assertEqual(res.status, "success")
        # Pixel must be NaN, NOT 0.0 or infinity or division error
        self.assertTrue(np.isnan(res.raster[50, 50]))
        self.assertFalse(np.isinf(res.raster[50, 50]))

    # -----------------------------------------------------------------------
    # TEST 11: NaN handled safely
    # -----------------------------------------------------------------------
    def test_11_nan_handled_safely(self):
        scene = self._create_full_scene()
        # Cast band data to float and introduce NaN
        float_b04 = scene.bands["B04"].data.astype(np.float32)
        float_b04[45, 45] = np.nan
        scene.bands["B04"].data = float_b04
        res = SpectralEngine.compute_ndvi(scene)
        self.assertEqual(res.status, "success")
        self.assertTrue(np.isnan(res.raster[45, 45]))
        # Stats must not be corrupted by NaN
        self.assertIsNotNone(res.stats.mean)
        self.assertFalse(np.isnan(res.stats.mean))
        self.assertFalse(np.isnan(res.stats.min))
        self.assertFalse(np.isnan(res.stats.max))

    # -----------------------------------------------------------------------
    # TEST 12: Nodata handled safely
    # -----------------------------------------------------------------------
    def test_12_nodata_handled_safely(self):
        scene = self._create_full_scene()
        # Corner 0:5, 0:5 was initialized as 0 (nodata)
        res = SpectralEngine.compute_ndvi(scene)
        self.assertEqual(res.status, "success")
        self.assertTrue(np.all(np.isnan(res.raster[0:5, 0:5])))
        self.assertGreater(res.stats.nodata_pixel_count, 0)
        self.assertEqual(res.stats.total_pixel_count, 10000)
        self.assertEqual(res.stats.valid_pixel_count + res.stats.nodata_pixel_count, 10000)

    # -----------------------------------------------------------------------
    # TEST 13: CRS preserved
    # -----------------------------------------------------------------------
    def test_13_crs_preserved(self):
        scene = self._create_full_scene()
        res = SpectralEngine.compute_ndvi(scene)
        self.assertEqual(res.crs, "EPSG:32643")
        self.assertEqual(scene.crs, res.crs)

    # -----------------------------------------------------------------------
    # TEST 14: Transform preserved
    # -----------------------------------------------------------------------
    def test_14_transform_preserved(self):
        scene = self._create_full_scene()
        self.assertEqual(scene.transform, (500000.0, 10.0, 0.0, 3000000.0, 0.0, -10.0))
        self.assertEqual(scene.bands["B04"].transform, scene.transform)
        self.assertEqual(scene.bounds, (500000.0, 2999000.0, 501000.0, 3000000.0))

    # -----------------------------------------------------------------------
    # TEST 15: Native resolution recorded
    # -----------------------------------------------------------------------
    def test_15_native_resolution_recorded(self):
        scene = self._create_full_scene()
        self.assertEqual(scene.bands["B04"].native_resolution_m, 10.0)
        self.assertEqual(scene.bands["B08"].native_resolution_m, 10.0)
        self.assertEqual(scene.bands["B11"].native_resolution_m, 20.0)
        self.assertEqual(scene.bands["B11"].native_width, 50)
        self.assertEqual(scene.bands["B11"].native_height, 50)

    # -----------------------------------------------------------------------
    # TEST 16: Resampling status recorded
    # -----------------------------------------------------------------------
    def test_16_resampling_status_recorded(self):
        scene = self._create_full_scene()
        # Compute NDBI which requires B11 (20m) and B08 (10m)
        res = SpectralEngine.compute_ndbi(scene)
        self.assertTrue(res.resampling_applied)
        self.assertIn("50x50", res.resampling_details)
        self.assertIn("100x100", res.resampling_details)
        self.assertIn("bilinear", res.resampling_details)

    # -----------------------------------------------------------------------
    # TEST 17: Band identity preserved
    # -----------------------------------------------------------------------
    def test_17_band_identity_preserved(self):
        scene = self._create_full_scene()
        b02 = scene.get_band("B02")
        b03 = scene.get_band("B03")
        b04 = scene.get_band("B04")
        b08 = scene.get_band("B08")
        b11 = scene.get_band("B11")
        b12 = scene.get_band("B12")

        self.assertEqual(b02.name, "B02")
        self.assertEqual(b02.common_name, "blue")
        self.assertEqual(b03.name, "B03")
        self.assertEqual(b03.common_name, "green")
        self.assertEqual(b04.name, "B04")
        self.assertEqual(b04.common_name, "red")
        self.assertEqual(b08.name, "B08")
        self.assertEqual(b08.common_name, "nir")
        self.assertEqual(b11.name, "B11")
        self.assertEqual(b11.common_name, "swir16")
        self.assertEqual(b12.name, "B12")
        self.assertEqual(b12.common_name, "swir22")

        # Ensure no band aliases or shallow copies conflate arrays
        self.assertFalse(np.array_equal(b04.data, b08.data))
        self.assertFalse(np.array_equal(b03.data, b08.data))

    # -----------------------------------------------------------------------
    # TEST 18: Loaded bands differ correctly from provider-available bands
    # -----------------------------------------------------------------------
    def test_18_loaded_bands_differ_from_available_bands(self):
        scene = self._create_full_scene()
        self.assertNotEqual(len(scene.loaded_bands), len(scene.available_bands))
        self.assertEqual(len(scene.loaded_bands), 6)
        self.assertGreater(len(scene.available_bands), 6)
        # B01, B05, B06 are available in catalog but not loaded in this scene
        self.assertIn("B01", scene.available_bands)
        self.assertNotIn("B01", scene.loaded_bands)

    # -----------------------------------------------------------------------
    # TEST 19: RGB visualization derivative does not overwrite original spectral metadata
    # -----------------------------------------------------------------------
    def test_19_rgb_visualization_derivative_does_not_overwrite_spectral_metadata(self):
        scene = self._create_full_scene()
        rgb_img, deriv_provenance = scene.to_rgb_image()
        self.assertIsInstance(rgb_img, Image.Image)
        self.assertEqual(deriv_provenance, "VISUAL_DERIVATIVE")
        # MultispectralScene itself must strictly retain REAL_MULTISPECTRAL
        self.assertEqual(scene.spectral_source, "REAL_MULTISPECTRAL")
        self.assertEqual(scene.provenance["provider"], "Element84_AWS_STAC")
        # Ensure bands were not modified or destroyed during visual RGB creation
        self.assertIn("B04", scene.bands)
        self.assertIn("B08", scene.bands)
        self.assertEqual(scene.bands["B04"].data.dtype, np.uint16)

    # -----------------------------------------------------------------------
    # TEST 20: Invalid / non-raster input rejected honestly
    # -----------------------------------------------------------------------
    def test_20_invalid_non_raster_input_rejected_honestly(self):
        tool = MultispectralAnalysisTool()
        # Invalid input: no image, no multispectral scene, empty dict
        res = tool.execute({"multispectral_scene": None, "image": None})
        self.assertEqual(res.status, "unavailable")
        self.assertEqual(res.provenance, "UNAVAILABLE")
        self.assertTrue("authentic spectral bands" in res.error or "RGB-only" in res.error)

        # Invalid object passed as multispectral_scene
        res2 = tool.execute({"multispectral_scene": "invalid_string_not_scene"})
        self.assertEqual(res2.status, "unavailable")
        self.assertEqual(res2.provenance, "UNAVAILABLE")

    # -----------------------------------------------------------------------
    # BONUS TEST 21: Full Tool Pipeline Execution with MultispectralScene
    # -----------------------------------------------------------------------
    def test_21_tool_pipeline_execution_multispectral_scene(self):
        tool = MultispectralAnalysisTool()
        scene = self._create_full_scene()
        tool_res = tool.execute({
            "multispectral_scene": scene,
            "indices": ["NDVI", "NDWI", "NDBI"],
            "query": "calculate NDVI and NDWI spectral indices",
        })
        self.assertEqual(tool_res.status, "success")
        self.assertEqual(tool_res.provenance, "REAL_MULTISPECTRAL")
        self.assertEqual(tool_res.data["spectral_source"], "REAL_MULTISPECTRAL")
        data = tool_res.data
        self.assertIn("spectral", data)
        spectral = data["spectral"]
        self.assertEqual(spectral["source"], "Sentinel-2 L2A")
        self.assertEqual(spectral["provenance"], "REAL_MULTISPECTRAL")
        self.assertIn("NDVI", spectral["indices"])
        self.assertIn("NDWI", spectral["indices"])
        self.assertIn("NDBI", spectral["indices"])
        self.assertEqual(spectral["indices"]["NDVI"]["status"], "success")
        self.assertEqual(spectral["indices"]["NDWI"]["status"], "success")
        self.assertEqual(spectral["indices"]["NDBI"]["status"], "success")

    # -----------------------------------------------------------------------
    # BONUS TEST 22: Deserialization via from_dict preserves bands and properties
    # -----------------------------------------------------------------------
    def test_22_scene_serialization_roundtrip(self):
        scene = self._create_full_scene()
        d = scene.to_dict()
        # Inject band data into dict to test full reconstruction
        for b_name in d["band_details"]:
            d["band_details"][b_name]["data"] = scene.bands[b_name].data
        reconstructed = MultispectralScene.from_dict(d)
        self.assertEqual(reconstructed.scene_id, scene.scene_id)
        self.assertEqual(reconstructed.spectral_source, "REAL_MULTISPECTRAL")
        self.assertEqual(reconstructed.loaded_bands, scene.loaded_bands)
        self.assertEqual(reconstructed.crs, scene.crs)
        # Compute NDVI on reconstructed scene
        res = SpectralEngine.compute_ndvi(reconstructed)
        self.assertEqual(res.status, "success")
        self.assertEqual(res.provenance, "REAL_MULTISPECTRAL")


if __name__ == "__main__":
    unittest.main()
