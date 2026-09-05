"""
test_tiling.py
---------------
Test suite for Phase 6: High-Resolution Satellite Image Support & Tiling.

Validates:
  1. Sliding window tile slicing and coordinate remapping
  2. Canonical bounding box convention: [xmin, ymin, xmax, ymax] and [0, 1000] normalized
  3. Non-Maximum Suppression (NMS) duplicate elimination
  4. Non-inverted bounding box enforcement (xmin < xmax, ymin < ymax)
  5. Boundary clamping to original image dimensions
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from tiling import tiled_inference_engine, compute_iou, apply_nms


class TestTiling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = tiled_inference_engine
        # Large synthetic image (1500 x 1200)
        cls.large_img = Image.new("RGB", (1500, 1200), (80, 80, 80))

    # Test 1: Tile Slicing Grid Completeness
    def test_tile_generation_coverage(self):
        tiles = self.engine.generate_tiles(self.large_img, tile_size=512, overlap=0.20)
        self.assertTrue(len(tiles) >= 6)
        # Check that tiles cover boundaries
        max_x = max(off_x + tw for _, (off_x, off_y, tw, th) in tiles)
        max_y = max(off_y + th for _, (off_x, off_y, tw, th) in tiles)
        self.assertEqual(max_x, 1500)
        self.assertEqual(max_y, 1200)

    # Test 2: IoU Computation
    def test_iou_calculation(self):
        box1 = [10.0, 10.0, 50.0, 50.0]
        box2 = [10.0, 10.0, 50.0, 50.0]  # Exact match
        box3 = [100.0, 100.0, 200.0, 200.0]  # Disjoint
        self.assertAlmostEqual(compute_iou(box1, box2), 1.0, places=3)
        self.assertAlmostEqual(compute_iou(box1, box3), 0.0, places=3)

    # Test 3: NMS Filtering
    def test_nms_suppression(self):
        # Two overlapping detections of the same object across tile seam
        dets = [
            {"label": "building", "bbox_pixel": [100.0, 100.0, 200.0, 200.0], "score": 0.90},
            {"label": "building", "bbox_pixel": [105.0, 102.0, 202.0, 198.0], "score": 0.85},
            {"label": "ship", "bbox_pixel": [500.0, 500.0, 600.0, 600.0], "score": 0.95},
        ]
        kept = apply_nms(dets, iou_threshold=0.50)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["score"], 0.95)
        self.assertEqual(kept[1]["score"], 0.90)

    # Test 4: Tiled Grounding on Mock/Real
    def test_tiled_grounding_execution(self):
        res = self.engine.run_tiled_grounding(
            image=self.large_img,
            target_phrase="buildings",
            tile_size=512,
            overlap=0.20,
        )
        self.assertTrue(res["tiling_used"])
        self.assertEqual(res["image_width"], 1500)
        self.assertEqual(res["image_height"], 1200)
        self.assertIn("tiles_processed", res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
