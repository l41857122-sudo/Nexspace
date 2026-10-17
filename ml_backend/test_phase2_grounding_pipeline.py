import sys
import os
import unittest
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from ml_backend.tiling import (
    TiledInferenceEngine,
    compute_iou,
    apply_nms,
    filter_boxes_by_semantics,
    remap_tile_boxes_to_global,
    normalize_bbox_1000
)
from ml_backend.router import (
    build_generalized_aerial_prompt,
    extract_detection_targets,
    is_macro_target,
    extract_grounding_target,
    route,
    TaskType
)


class TestPhase2GroundingPipeline(unittest.TestCase):
    def setUp(self):
        self.engine = TiledInferenceEngine(default_tile_size=512, default_overlap=0.20)

    def test_1_tile_generation_and_overlap(self):
        """1. Test tile generation produces valid sub-windows with ~20% overlap."""
        img = Image.new("RGB", (1000, 800))
        tiles = self.engine.generate_tiles(img, tile_size=512, overlap=0.20)
        self.assertTrue(len(tiles) > 1, "Expected multiple tiles for 1000x800 image")
        for tile_img, (x_off, y_off, tw, th) in tiles:
            self.assertLessEqual(tw, 512)
            self.assertLessEqual(th, 512)
            self.assertGreaterEqual(x_off, 0)
            self.assertGreaterEqual(y_off, 0)
            self.assertLessEqual(x_off + tw, 1000)
            self.assertLessEqual(y_off + th, 800)

    def test_2_edge_and_boundary_handling(self):
        """2. Test that tile generation correctly clamps to image boundaries with no gaps."""
        width, height = 750, 600
        img = Image.new("RGB", (width, height))
        tiles = self.engine.generate_tiles(img, tile_size=512, overlap=0.20)
        
        covered_x = max(x_off + tw for _, (x_off, y_off, tw, th) in tiles)
        covered_y = max(y_off + th for _, (x_off, y_off, tw, th) in tiles)
        self.assertEqual(covered_x, width)
        self.assertEqual(covered_y, height)
        
        small_img = Image.new("RGB", (400, 300))
        small_tiles = self.engine.generate_tiles(small_img, tile_size=512)
        self.assertEqual(len(small_tiles), 1)
        self.assertEqual(small_tiles[0][1], (0, 0, 400, 300))

    def test_3_tile_to_global_coordinate_remapping(self):
        """3. Test tile-local [0, 1000] coords are correctly mapped back to global [0, 1000]."""
        tile_boxes = [
            {"box": [250, 250, 750, 750], "score": 0.85, "label": "building"}
        ]
        remapped = remap_tile_boxes_to_global(
            boxes=tile_boxes,
            tile_x1=1000,
            tile_y1=0,
            tile_w=1000,
            tile_h=1000,
            image_w=2000,
            image_h=1000
        )
        self.assertEqual(len(remapped), 1)
        gx_min, gy_min, gx_max, gy_max = remapped[0]["box"]
        self.assertAlmostEqual(gx_min, 625, delta=1)
        self.assertAlmostEqual(gy_min, 250, delta=1)
        self.assertAlmostEqual(gx_max, 875, delta=1)
        self.assertAlmostEqual(gy_max, 750, delta=1)

    def test_4_class_aware_nms_deduplication(self):
        """4. Test NMS deduplicates overlapping boxes of the same class but preserves distinct classes."""
        boxes = [
            {"box": [100, 100, 200, 200], "score": 0.90, "label": "building"},
            {"box": [105, 105, 205, 205], "score": 0.75, "label": "building"},  # High overlap, lower score -> duplicate
            {"box": [102, 102, 198, 198], "score": 0.88, "label": "car"},       # Different class -> PRESERVE
            {"box": [500, 500, 600, 600], "score": 0.82, "label": "building"}   # Different location -> PRESERVE
        ]
        kept = apply_nms(boxes, iou_threshold=0.45, class_aware=True)
        self.assertEqual(len(kept), 3)
        scores = [k["score"] for k in kept]
        self.assertIn(0.90, scores)
        self.assertIn(0.88, scores)
        self.assertIn(0.82, scores)
        self.assertNotIn(0.75, scores)

    def test_5_semantic_box_filtering_compact_vs_macro(self):
        """5. Test semantic sanity checks reject pathological full-image boxes for compact objects."""
        img_w, img_h = 1000, 1000
        pathological_box = {"box": [10, 10, 990, 990], "score": 0.85, "label": "car"}
        valid_car_box = {"box": [100, 100, 150, 150], "score": 0.85, "label": "car"}
        macro_road_box = {"box": [10, 10, 990, 990], "score": 0.85, "label": "road"}
        degenerate_box = {"box": [100, 100, 101, 100], "score": 0.85, "label": "building"}

        car_results = filter_boxes_by_semantics([pathological_box, valid_car_box, degenerate_box], target_phrase="car", img_w=img_w, img_h=img_h)
        self.assertEqual(len(car_results), 1)
        self.assertEqual(car_results[0]["box"], [100, 100, 150, 150])

        road_results = filter_boxes_by_semantics([macro_road_box, valid_car_box], target_phrase="road", img_w=img_w, img_h=img_h)
        self.assertEqual(len(road_results), 2, "Macro target 'road' should allow spanning box")

    def test_6_prompt_generation_and_all_target_categories(self):
        """6. Test prompt generation for all mandatory categories."""
        targets = [
            ("buildings", ["building", "rooftop"]),
            ("houses", ["house", "residential"]),
            ("rooftops", ["building", "rooftop"]),
            ("warehouses", ["warehouse", "industrial"]),
            ("cars", ["car", "automobile"]),
            ("trucks", ["truck", "heavy vehicle"]),
            ("buses", ["bus", "vehicle"]),
            ("vehicles", ["car", "truck", "vehicle"]),
            ("ships", ["ship", "vessel"]),
            ("boats", ["boat", "vessel"]),
            ("vessels", ["vessel", "ship"]),
            ("bridges", ["bridge", "overpass"]),
            ("roads", ["road", "highway"]),
            ("highways", ["highway", "road"]),
            ("airplanes", ["airplane", "aircraft"]),
            ("airports", ["airport", "runway"]),
            ("towers", ["tower", "antenna", "mast"]),
            ("containers", ["shipping container", "cargo"]),
            ("agricultural structures", ["farm", "agricultural"]),
            ("trees / vegetation", ["tree", "vegetation"]),
        ]
        for query, expected_tokens in targets:
            prompt = build_generalized_aerial_prompt(query)
            for tok in expected_tokens:
                self.assertIn(tok, prompt)
            self.assertTrue(prompt.endswith("."))

    def test_7_arbitrary_target_phrases(self):
        """7. Test prompt generation for arbitrary novel user terms."""
        novel_phrases = [
            "wind turbine generator",
            "solar photovoltaic farm",
            "helipad marker",
            "quarry excavation pit",
            "floating oil platform"
        ]
        for p in novel_phrases:
            expanded = build_generalized_aerial_prompt(p)
            self.assertTrue(expanded.endswith("."))
            self.assertIn(p, expanded)

    def test_8_multi_target_queries(self):
        """8. Test multi-target query extraction and routing."""
        queries = [
            ("Locate all buildings and cars", ["buildings", "cars"]),
            ("Find bridges and roads", ["bridges", "roads"]),
            ("Detect ships, containers and cranes", ["ships", "containers", "cranes"]),
            ("Locate towers and buildings", ["towers", "buildings"]),
            ("Find all vehicles and count them", ["vehicles"])
        ]
        for q, expected in queries:
            targets = extract_detection_targets(q)
            for exp in expected:
                self.assertIn(exp, targets)

    def test_9_large_object_handling(self):
        """9. Verify large structures (bridges, runways, roads) are not discarded."""
        img_w, img_h = 1000, 1000
        bridge_box = {"box": [50, 400, 950, 600], "score": 0.88, "label": "bridge"}
        filtered = filter_boxes_by_semantics([bridge_box], target_phrase="bridge", img_w=img_w, img_h=img_h)
        self.assertEqual(len(filtered), 1)

    def test_10_small_object_handling(self):
        """10. Verify small compact objects (cars, boats) are properly normalized and retained."""
        img_w, img_h = 2000, 2000
        small_pixel_box = [500.0, 500.0, 530.0, 520.0]  # 30x20 pixels in 2000x2000 image
        norm = normalize_bbox_1000(small_pixel_box, img_w, img_h)
        self.assertEqual(norm, [250, 250, 265, 260])
        self.assertGreater(norm[2], norm[0])
        self.assertGreater(norm[3], norm[1])

    def test_11_coordinate_normalization(self):
        """11. Verify normalization clamp guarantees strictly valid [xmin, ymin, xmax, ymax] in [0, 1000]."""
        degenerate_boxes = [
            [-50.0, -20.0, 2000.0, 1500.0],
            [100.0, 100.0, 100.0, 100.0],  # zero extent
            [500.0, 500.0, 400.0, 400.0],  # inverted extent
        ]
        for b in degenerate_boxes:
            norm = normalize_bbox_1000(b, 1000, 1000)
            self.assertGreaterEqual(norm[0], 0)
            self.assertGreaterEqual(norm[1], 0)
            self.assertLessEqual(norm[2], 1000)
            self.assertLessEqual(norm[3], 1000)
            self.assertGreater(norm[2], norm[0])
            self.assertGreater(norm[3], norm[1])

    def test_12_regression_api_schema(self):
        """12. Verify output detections match NexSpace canonical schema."""
        mock_box = {"box": [100, 100, 200, 200], "score": 0.85, "label": "building"}
        remapped = remap_tile_boxes_to_global([mock_box], 0, 0, 500, 500, 500, 500)
        self.assertIn("box", remapped[0])
        self.assertIn("box_2d", remapped[0])
        self.assertIn("bbox_pixel", remapped[0])
        self.assertIn("bbox_normalized", remapped[0])
        self.assertIn("score", remapped[0])
        self.assertIn("label", remapped[0])

    def test_13_demo_image_independence(self):
        """13. Verify engine functions with arbitrary synthetic in-memory image."""
        synth_img = Image.new("RGB", (1280, 720), color=(128, 128, 128))
        tiles = self.engine.generate_tiles(synth_img, tile_size=512, overlap=0.20)
        self.assertGreaterEqual(len(tiles), 4)

    def test_14_router_backward_compatibility(self):
        """14. Verify router correctly supports existing non-grounding workflows."""
        d1 = route("What is the general land use in this area?", has_optical=True)
        self.assertIn("Optical_Caption", d1.target_tools)

        d2 = route("Compare optical and SAR radar passes", has_optical=True, has_sar=True)
        self.assertIn("Optical_SAR_Analysis", d2.target_tools)


if __name__ == "__main__":
    unittest.main(verbosity=2)
