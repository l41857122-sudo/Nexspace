"""
test_phase2_pipeline.py
=======================
Automated unit & regression tests for Phase 2:
Photo -> Location -> Satellite Context Investigation Engine.

Verifies:
1. Deterministic GeoTIFF raster parsing & CRS transformation.
2. EXIF GPS decoding, DMS-to-decimal conversion, hemisphere sign handling.
3. Quality gating (blur, featureless, small dimensions).
4. VPR landmark retrieval, ranking, and candidate structure.
5. Zero coordinate fabrication rule (regional estimates must not fake exact coordinates).
6. Evidence hierarchy classification fidelity (EXACT, GEOTAGGED, VERIFIED_LANDMARK, UNKNOWN).
7. Execution trace telemetry stages.
8. FastAPI endpoints compatibility (/api/image/metadata, /api/image/investigate).
"""

import os
import sys
import io
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image, ExifTags
import tifffile
from fastapi.testclient import TestClient

from image_forensics import ImageForensicsEngine
from vpr_engine import VisualPlaceRecognitionEngine, VERIFIED_REFERENCE_ENTITIES
from geolocation_pipeline import PhotoGeolocationPipeline
from server import app


class TestPhase2ForensicsAndPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.join(os.path.dirname(__file__), "scratch", "test_p2_tmp")
        os.makedirs(cls.temp_dir, exist_ok=True)
        cls.pipeline = PhotoGeolocationPipeline()
        cls.client = TestClient(app)

        # 1. Create a test GeoTIFF
        cls.geotiff_path = os.path.join(cls.temp_dir, "test_sample_geotiff.tif")
        data = np.zeros((128, 128, 3), dtype=np.uint8)
        data[:, :, 0] = 100
        data[:, :, 1] = 150
        data[:, :, 2] = 200
        # ModelPixelScaleTag (33550): [scaleX, scaleY, scaleZ]
        # ModelTiepointTag (33922): [i, j, k, x, y, z]
        extratags = [
            (33550, 'd', 3, (0.001, 0.001, 0.0), False),
            (33922, 'd', 6, (0.0, 0.0, 0.0, 2.2945, 48.8584, 0.0), False),
            (34735, 'H', 4, (1, 1, 0, 0), False)
        ]
        tifffile.imwrite(cls.geotiff_path, data, extratags=extratags)

        # 2. Create a test EXIF GPS photo
        cls.exif_path = os.path.join(cls.temp_dir, "test_sample_exif.jpg")
        img = Image.new("RGB", (200, 200), color=(120, 160, 200))
        exif = img.getexif()
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps_ifd[1] = "N"
        gps_ifd[2] = (48.0, 51.0, 30.2)  # 48.85838889
        gps_ifd[3] = "E"
        gps_ifd[4] = (2.0, 17.0, 40.2)   # 2.2945
        gps_ifd[6] = 35.0
        exif[ExifTags.Base.Make] = "Nikon"
        exif[ExifTags.Base.Model] = "D850"
        img.save(cls.exif_path, "JPEG", exif=exif)

        # 3. Create a blank featureless photo
        cls.blank_path = os.path.join(cls.temp_dir, "test_blank_gray.jpg")
        blank = Image.new("RGB", (200, 200), color=(128, 128, 128))
        blank.save(cls.blank_path, "JPEG")

    def test_01_geotiff_deterministic_extraction(self):
        """Test Case A: GeoTIFF metadata extraction and WGS84 coordinate calculation."""
        res = ImageForensicsEngine.inspect_file(self.geotiff_path)
        self.assertTrue(res["is_deterministic"])
        self.assertEqual(res["classification"], "EXACT")
        det = res["deterministic_geolocation"]
        self.assertIsNotNone(det)
        self.assertEqual(det["evidence_type"], "GEOREFERENCED_RASTER")
        coords = det["coordinates"]
        self.assertAlmostEqual(coords[0], 48.8584 - 0.064, delta=0.01)
        self.assertAlmostEqual(coords[1], 2.2945 + 0.064, delta=0.01)

    def test_02_exif_gps_extraction(self):
        """Test Case B: EXIF GPS decoding, DMS to decimal conversion, hemisphere refs."""
        res = ImageForensicsEngine.inspect_file(self.exif_path)
        self.assertTrue(res["is_deterministic"])
        self.assertEqual(res["classification"], "GEOTAGGED")
        det = res["deterministic_geolocation"]
        self.assertIsNotNone(det)
        coords = det["coordinates"]
        self.assertAlmostEqual(coords[0], 48.85838889, places=4)
        self.assertAlmostEqual(coords[1], 2.2945, places=4)

    def test_03_quality_gating_featureless_image(self):
        """Test quality gating rejects featureless/flat images."""
        res = ImageForensicsEngine.inspect_file(self.blank_path)
        q = res["quality"]
        self.assertFalse(q["quality_pass"])
        self.assertEqual(q["quality_status"], "REJECTED")
        self.assertTrue(any("featureless" in issue.lower() for issue in q["issues"]))

    def test_04_vpr_landmark_reference_db(self):
        """Test VPR catalog has coverage across global continents."""
        vpr = VisualPlaceRecognitionEngine.get_instance()
        self.assertGreaterEqual(len(VERIFIED_REFERENCE_ENTITIES), 30)
        continents = {e["continent"] for e in VERIFIED_REFERENCE_ENTITIES}
        self.assertIn("Europe", continents)
        self.assertIn("Asia", continents)
        self.assertIn("North America", continents)
        self.assertIn("South America", continents)
        self.assertIn("Africa", continents)
        self.assertIn("Oceania", continents)

    def test_05_vpr_candidate_retrieval(self):
        """Test VPR candidate retrieval returns ranked list with required fields."""
        vpr = VisualPlaceRecognitionEngine.get_instance()
        test_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        cands = vpr.retrieve_candidates(test_img, top_k=3, hint_text="Eiffel")
        self.assertIsInstance(cands, list)
        self.assertGreaterEqual(len(cands), 1)
        top = cands[0]
        self.assertIn("name", top)
        self.assertIn("coordinates", top)
        self.assertIn("similarity", top)
        self.assertIn("country", top)
        self.assertTrue(0.0 <= top["similarity"] <= 1.0)

    def test_06_no_coordinate_fabrication_on_unknown(self):
        """Rule 3, 9, 14: Insufficient visual evidence MUST return UNKNOWN with null coordinates."""
        res = self.pipeline.investigate(self.blank_path, retrieve_satellite=False)
        self.assertEqual(res["classification"], "UNKNOWN")
        self.assertIsNone(res["coordinates"])
        self.assertEqual(res["confidence"], 0.0)

    def test_07_telemetry_trace_recording(self):
        """Test execution trace records actual stages with duration and provenance."""
        res = self.pipeline.investigate(self.exif_path, retrieve_satellite=False)
        trace = res.get("trace", [])
        steps = trace if isinstance(trace, list) else trace.get("steps", [])
        self.assertGreaterEqual(len(steps), 4)
        stages = [s["stage"] for s in steps]
        self.assertIn("IMAGE_RECEIVED", stages)
        self.assertIn("FILE_VALIDATED", stages)
        self.assertIn("EXIF_CHECKED", stages)
        self.assertIn("FINAL_RESULT", stages)

    def test_08_api_metadata_endpoint(self):
        """Test POST /api/image/metadata via FastAPI TestClient."""
        with open(self.exif_path, "rb") as f:
            resp = self.client.post("/api/image/metadata", files={"image": ("test.jpg", f, "image/jpeg")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("is_deterministic"))
        self.assertEqual(data.get("classification"), "GEOTAGGED")

    def test_09_api_investigate_endpoint(self):
        """Test POST /api/image/investigate via FastAPI TestClient."""
        with open(self.exif_path, "rb") as f:
            resp = self.client.post("/api/image/investigate", files={"image": ("test.jpg", f, "image/jpeg")})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("classification"), "GEOTAGGED")
        self.assertIsNotNone(data.get("coordinates"))
        self.assertIn("trace", data)

    def test_10_adversarial_null_island_rejection(self):
        """Test suspicious coordinates (0.0, 0.0) produce warning or rejected status."""
        adv_path = os.path.join(self.temp_dir, "test_null_island.jpg")
        img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        exif = img.getexif()
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps[1] = "N"
        gps[2] = (0.0, 0.0, 0.001)
        gps[3] = "E"
        gps[4] = (0.0, 0.0, 0.001)
        img.save(adv_path, "JPEG", exif=exif)

        res = ImageForensicsEngine.inspect_file(adv_path)
        exif_info = res.get("exif_gps", {})
        warnings = exif_info.get("warnings", [])
        self.assertTrue(any("null island" in w.lower() for w in warnings))


if __name__ == "__main__":
    unittest.main(verbosity=2)
