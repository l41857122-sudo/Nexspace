"""
tests/test_phase4_change_analysis.py
-------------------------------------
Phase 4 Verification Suite:
Evidence-Based Temporal Change Analysis.

Acceptance Criteria:
 1. Unchanged scenes produce negligible change (< 2%)
 2. Genuine changed scenes produce accurate localized change metrics
 3. Acquisition dates and scene IDs are preserved in output
 4. Nodata / zero-padding margin pixels are masked out and do not cause false-positive spikes
 5. Cloud / high-saturation atmospheric conditions are detected and reported in limitations
 6. Disjoint / non-overlapping geographic bounds are rejected
 7. Structured limitations clearly state radiometric basis without claiming unverified physical construction
 8. Specialist tool ChangeAnalysisTool properly handles valid and invalid inputs
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_backend"))

import change_analysis
from tools import ChangeAnalysisTool


class TestPhase4ChangeAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tool = ChangeAnalysisTool()

        # Base scene: 200x200 urban/vegetation terrain
        cls.base_img = Image.new("RGB", (200, 200), (120, 120, 120))
        d_base = ImageDraw.Draw(cls.base_img)
        d_base.rectangle([20, 20, 80, 80], fill=(180, 50, 50))    # Structure
        d_base.rectangle([100, 20, 180, 80], fill=(40, 160, 40))  # Green plot

        # Changed scene: Structure removed, new structure built on green plot
        cls.changed_img = cls.base_img.copy()
        d_changed = ImageDraw.Draw(cls.changed_img)
        d_changed.rectangle([20, 20, 80, 80], fill=(120, 120, 120))  # Cleared
        d_changed.rectangle([110, 30, 170, 70], fill=(220, 220, 220)) # New bright roof

        # Geographic bounds
        cls.bounds_a = {"min_x": 500000.0, "min_y": 2000000.0, "max_x": 502000.0, "max_y": 2002000.0}
        cls.bounds_b = {"min_x": 500500.0, "min_y": 2000500.0, "max_x": 502500.0, "max_y": 2002500.0}
        cls.bounds_disjoint = {"min_x": 600000.0, "min_y": 3000000.0, "max_x": 602000.0, "max_y": 3002000.0}

    def test_01_unchanged_scenes(self):
        """Identical scenes produce 0.0 changed fraction and negligible change summary."""
        res = change_analysis.analyze(self.base_img, self.base_img)
        self.assertEqual(res.changed_fraction, 0.0)
        self.assertEqual(res.mean_intensity_delta, 0.0)
        self.assertIn("negligible change", res.summary)

    def test_02_genuine_changed_scenes(self):
        """Modified scene produces expected changed fraction and alpha overlay."""
        res = change_analysis.analyze(self.base_img, self.changed_img, change_threshold=0.15)
        self.assertGreater(res.changed_fraction, 0.05)
        self.assertLess(res.changed_fraction, 0.50)
        self.assertEqual(res.heatmap.size, (200, 200))
        self.assertEqual(res.overlay.size, (200, 200))
        self.assertEqual(res.overlay.mode, "RGBA")

    def test_03_temporal_metadata_preservation(self):
        """Scene IDs, acquisition dates, and sensor tags are accurately retained."""
        res = change_analysis.analyze(
            self.base_img,
            self.changed_img,
            scene_a_id="S2A_20240101",
            scene_b_id="S2B_20240601",
            acquisition_date_a="2024-01-01",
            acquisition_date_b="2024-06-01",
            sensor_a="Sentinel-2A",
            sensor_b="Sentinel-2B",
        )
        self.assertEqual(res.scene_a_id, "S2A_20240101")
        self.assertEqual(res.scene_b_id, "S2B_20240601")
        self.assertEqual(res.acquisition_date_a, "2024-01-01")
        self.assertEqual(res.acquisition_date_b, "2024-06-01")
        self.assertEqual(res.sensor_a, "Sentinel-2A")
        self.assertEqual(res.sensor_b, "Sentinel-2B")
        self.assertIn("2024-01-01", res.summary)
        self.assertIn("2024-06-01", res.summary)

    def test_04_nodata_margin_masking(self):
        """Zero-padded/nodata margins are excluded from changed fraction and made transparent."""
        # Create image with 25% black border (nodata)
        img_a_nodata = Image.new("RGB", (200, 200), (0, 0, 0))
        img_b_nodata = Image.new("RGB", (200, 200), (0, 0, 0))
        # Valid center (100x100) identical in both
        center = Image.new("RGB", (100, 100), (150, 150, 150))
        img_a_nodata.paste(center, (50, 50))
        img_b_nodata.paste(center, (50, 50))

        res = change_analysis.analyze(img_a_nodata, img_b_nodata)
        # Since center is unchanged, changed_fraction should be 0.0 despite large black margin
        self.assertEqual(res.changed_fraction, 0.0)
        self.assertGreater(res.nodata_pixels_count, 0)
        self.assertLess(res.valid_pixels_fraction, 1.0)
        # Verify any nodata note in limitations
        lim_str = " ".join(res.limitations)
        self.assertIn("nodata/border padding", lim_str)

    def test_05_cloud_contamination_detection(self):
        """Scenes with high cloud saturation trigger quality warning in limitations."""
        img_cloud = self.base_img.copy()
        d_cloud = ImageDraw.Draw(img_cloud)
        # Large white cloud covering 40% of scene
        d_cloud.rectangle([10, 10, 150, 150], fill=(245, 245, 245))

        res = change_analysis.analyze(self.base_img, img_cloud)
        self.assertTrue(res.cloud_contamination_detected)
        self.assertGreater(res.cloud_pixels_fraction, 0.10)
        lim_str = " ".join(res.limitations)
        self.assertIn("cloud/shadow contamination", lim_str)

    def test_06_non_overlapping_bounds_rejection(self):
        """Geographically disjoint observations raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            change_analysis.analyze(
                self.base_img,
                self.changed_img,
                bounds_a=self.bounds_a,
                bounds_b=self.bounds_disjoint,
            )
        self.assertIn("do not overlap", str(ctx.exception))

    def test_07_structured_limitations_integrity(self):
        """Limitations contain required transparency disclaimers."""
        res = change_analysis.analyze(self.base_img, self.changed_img)
        self.assertIsInstance(res.limitations, list)
        self.assertGreaterEqual(len(res.limitations), 2)
        lim_str = " ".join(res.limitations)
        self.assertIn("not an assumption of verified physical construction", lim_str)
        self.assertIn("seasonal vegetation phenology", lim_str)

    def test_08_change_analysis_specialist_tool(self):
        """ChangeAnalysisTool properly formats execution result and handles errors."""
        # Success case
        res_ok = self.tool.execute({
            "image_a": self.base_img,
            "image_b": self.changed_img,
            "scene_a_id": "S2A_001",
            "scene_b_id": "S2B_002",
        })
        self.assertEqual(res_ok.status, "success")
        self.assertIn("changed_fraction", res_ok.data)
        self.assertIn("limitations", res_ok.data)
        self.assertEqual(res_ok.data["scene_a_id"], "S2A_001")

        # Disjoint bounds rejection via tool
        res_fail = self.tool.execute({
            "image_a": self.base_img,
            "image_b": self.changed_img,
            "bounds_a": self.bounds_a,
            "bounds_b": self.bounds_disjoint,
        })
        self.assertEqual(res_fail.status, "failed")
        self.assertEqual(res_fail.provenance, "UNAVAILABLE")
        self.assertIn("do not overlap", res_fail.error)


if __name__ == "__main__":
    unittest.main()
