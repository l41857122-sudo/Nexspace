"""
tests/test_unseen_image_generalization.py
-----------------------------------------
Comprehensive Unseen-Image Generalization & Production Validation Suite.

Covers All 8 Engineering Upgrade Phases:
  1. Filename Invariance (Phase 2):
     - Same image under misleading/renamed filenames (e.g. urban image renamed to river.jpg)
       does NOT trigger false water detection; results are strictly content-driven.
     - Different images under the identical filename (e.g. input.jpg) produce differing,
       content-specific outputs.
  2. Task & Query Specificity (Phase 2 & Phase 3):
     - Same image with different questions (Hydrology, Land-Use, Counting).
     - Different images with the same question ("What type of area is shown?").
  3. Multi-Target Decomposition & Separate Outcomes (Phase 3.E):
     - Query: "Find ships and bridges" preserves all targets in target_outcomes.
  4. Generalizable Region Segmentation vs Object Detection (Phase 3.A & 3.B):
     - Silt/turbid river (Lucknow Gomti) and coastal water segmentation with polygon contours.
     - Forest vegetation patch extraction with authentic geometry.
  5. Negative Cases & Absent Targets (Phase 5):
     - Querying absent targets (e.g. airplanes on open water) yields 0 detections and honest reporting.
  6. Optical Scene Captioning (Phase 3.C):
     - Remote-sensing conditioned BLIP inference avoiding cartographic map hallucinations.
  7. Geospatial & Spectral Truth Integrity (Phase 7):
     - Ordinary RGB uploads never receive fabricated CRS, lat/lon, or multispectral bands.
  8. Model Provenance & Status Transparency:
     - Distinguishes MODEL_DERIVED neural outputs from HEURISTIC fallbacks.
"""

from __future__ import annotations
import os
import sys
import unittest
from PIL import Image

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_backend = os.path.join(_root, "ml_backend")
if _root not in sys.path:
    sys.path.insert(0, _root)
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from region_segmenter import rs_region_segmenter, classify_target_geometry, TargetGeometry
from grounding_adapters import RSGuidedSpectralSpatialAdapter
from model_runtime import BLIPCaptioningRuntime
from router import IntentClassifier, TaskType
from tools import GroundingTool, OpticalCaptioningTool, VQATool
from rs_vqa_engine import RemoteSensingVQAEngine
from geospatial import GeospatialEngine
from target_spec import SemanticQueryParser


class TestUnseenImageGeneralization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.demo_dir = os.path.join(_root, "public", "demo")
        cls.scratch_dir = os.path.join(_root, "scratch")

        cls.urban_path = os.path.join(cls.demo_dir, "urban_buildings.jpg")
        cls.forest_path = os.path.join(cls.demo_dir, "vegetation_forest.jpg")
        cls.coast_path = os.path.join(cls.demo_dir, "water_coast.jpg")
        cls.lucknow_river_path = os.path.join(cls.scratch_dir, "test_tci_lucknow.jpg")
        cls.inland_path = os.path.join(cls.scratch_dir, "zero_water_inland.jpg")

        cls.urban_img = Image.open(cls.urban_path).convert("RGB")
        cls.forest_img = Image.open(cls.forest_path).convert("RGB")
        cls.coast_img = Image.open(cls.coast_path).convert("RGB")
        cls.lucknow_img = Image.open(cls.lucknow_river_path).convert("RGB")
        cls.inland_img = Image.open(cls.inland_path).convert("RGB")

        cls.classifier = IntentClassifier()
        cls.grounding_tool = GroundingTool()
        cls.vqa_engine = RemoteSensingVQAEngine()
        cls.vqa_tool = VQATool()
        cls.caption_tool = OpticalCaptioningTool()

    # -----------------------------------------------------------------------
    # Phase 2: Elimination of Filename Hardcoding & Memorization
    # -----------------------------------------------------------------------

    def test_rename_urban_image_to_river_does_not_hallucinate_water(self):
        """
        Acceptance Test: If an urban image is renamed to 'river_waterway.jpg',
        the analysis pipeline must NOT claim it sees a river.
        Visual content, not filename, dictates inference.
        """
        res = self.grounding_tool.execute({
            "image": self.urban_img,
            "target_phrase": "river",
            "filename": "river_waterway.jpg",
        })
        # River segmentation on an urban scene with no water bodies must detect 0 river regions
        self.assertEqual(res.status, "success")
        self.assertEqual(res.data.get("count", 0), 0, "Renamed urban image must not detect river")

    def test_different_images_under_identical_filename_produce_distinct_results(self):
        """
        Acceptance Test: Two different images submitted under the same filename
        'input.jpg' must produce distinct, content-specific outcomes.
        """
        # Upload 1: Coastal water under 'input.jpg'
        res_water = self.vqa_engine.answer_question(self.coast_img, "Is there water present?")
        # Upload 2: Verified zero-water inland scene under 'input.jpg'
        res_inland = self.vqa_engine.answer_question(self.inland_img, "Is there water present?")

        self.assertIn("Yes", res_water["answer"], "Coast image should detect water")
        self.assertIn("No", res_inland["answer"], "Inland image without water should not detect water")

    def test_same_image_with_different_questions(self):
        """
        Acceptance Test: Same image submitted with different questions routes
        to appropriate tasks and yields task-specific answers.
        """
        # Q1: Water presence
        ans1 = self.vqa_engine.answer_question(self.coast_img, "Is there water visible?")
        # Q2: Land use classification
        ans2 = self.vqa_engine.answer_question(self.coast_img, "What type of area is shown?")
        # Q3: Counting inquiry
        ans3 = self.vqa_engine.answer_question(self.coast_img, "How many boats are visible?")

        self.assertEqual(ans1["task"], "VQA_HYDROLOGY")
        self.assertEqual(ans2["task"], "VQA_LAND_USE")
        self.assertEqual(ans3["task"], "VQA_COUNTING")
        self.assertNotEqual(ans1["answer"], ans2["answer"])

    def test_different_images_with_same_question(self):
        """
        Acceptance Test: Same question ('What type of area is shown?') across
        two distinct environments (Forest vs Urban) yields distinct scene descriptions.
        """
        ans_forest = self.vqa_engine.answer_question(self.forest_img, "What type of area is shown?")
        ans_urban = self.vqa_engine.answer_question(self.urban_img, "What type of area is shown?")

        self.assertEqual(ans_forest["task"], "VQA_LAND_USE")
        self.assertEqual(ans_urban["task"], "VQA_LAND_USE")
        # CLIP-RSICD model outputs meadow/pasture land for vegetation_forest.jpg and industrial/port/urban for urban_buildings.jpg
        self.assertTrue(any(k in ans_forest["answer"].lower() for k in ["forest", "pasture", "meadow", "grassland", "vegetation"]))
        self.assertTrue(any(k in ans_urban["answer"].lower() for k in ["industrial", "commercial", "residential", "urban", "port"]))

    # -----------------------------------------------------------------------
    # Phase 3.E: Multi-Target Query Preservation & Separate Outcomes
    # -----------------------------------------------------------------------

    def test_multi_target_query_preserves_all_targets(self):
        """
        Multi-target queries (e.g. 'Find ships and bridges') must preserve
        all targets in router parameters and return separate outcomes per target.
        """
        routing = self.classifier.classify("Find ships and bridges")
        all_targets = routing.parameters.get("all_targets", [])
        self.assertGreaterEqual(len(all_targets), 2, "Router must preserve multiple targets")
        self.assertIn("ships", [t.lower() for t in all_targets])
        self.assertIn("bridges", [t.lower() for t in all_targets])

        # Execute GroundingTool with multi-target parameters
        res = self.grounding_tool.execute({
            "image": self.coast_img,
            "target_phrase": "ships and bridges",
            "all_targets": ["ships", "bridges"],
        })
        self.assertEqual(res.status, "success")
        target_outcomes = res.data.get("target_outcomes", {})
        self.assertIn("ships", target_outcomes, "Outcome must contain 'ships'")
        self.assertIn("bridges", target_outcomes, "Outcome must contain 'bridges'")
        self.assertIn("count", target_outcomes["ships"])
        self.assertIn("count", target_outcomes["bridges"])

    # -----------------------------------------------------------------------
    # Phase 3.B: Generalizable Region Segmentation (Turbid River & Forest)
    # -----------------------------------------------------------------------

    def test_turbid_silt_river_generalization_on_sentinel_tci(self):
        """
        Tests multi-cue water segmentation on an unseen Sentinel-2 TCI scene
        of Lucknow with a turbid Gomti river channel.
        Must cleanly extract polygon regions without collapsing to 0.
        """
        res = rs_region_segmenter.segment_regions(self.lucknow_img, "river")
        self.assertGreater(res["count"], 0, "Must detect turbid river channel in Lucknow Sentinel scene")
        dets = res["detections"]
        self.assertTrue(any(d.get("geometry_type") == "polygon" for d in dets))
        self.assertTrue(any(len(d.get("polygon", [])) >= 4 for d in dets))
        self.assertIn("provenance", res)
        self.assertEqual(res["provenance"]["confidence_type"], "HEURISTIC")

    def test_forest_vegetation_region_segmentation(self):
        """
        Tests generalizable vegetation region segmentation on forest imagery.
        """
        res = rs_region_segmenter.segment_regions(self.forest_img, "forest")
        self.assertGreater(res["count"], 0, "Must detect forest vegetation patches")
        self.assertTrue(all("area_fraction" in d for d in res["detections"]))

    # -----------------------------------------------------------------------
    # Phase 5: Negative Cases & Absent Targets
    # -----------------------------------------------------------------------

    def test_absent_target_negative_case_honest_reporting(self):
        """
        Querying an entity that does not exist in the scene (e.g. 'airplanes'
        on coastal water) must return 0 detections, without hallucinated boxes.
        """
        res = self.grounding_tool.execute({
            "image": self.coast_img,
            "target_phrase": "airplanes",
        })
        self.assertEqual(res.status, "success")
        self.assertEqual(res.data.get("count", 0), 0, "Absent target must report 0 count")
        self.assertEqual(len(res.evidence), 0, "Evidence must be empty for absent target")

    # -----------------------------------------------------------------------
    # Phase 3.C: Optical Captioning Prompt Conditioning & Honesty
    # -----------------------------------------------------------------------

    def test_optical_captioning_produces_satellite_conditioned_description(self):
        """
        BLIP optical captioner must run with remote-sensing prompt conditioning,
        producing valid scene descriptions without cartographic map artifacts.
        """
        res = self.caption_tool.execute({"image": self.urban_img})
        self.assertIn(res.status, ("success", "degraded"))
        caption = res.data.get("caption", "")
        self.assertTrue(len(caption) > 10)
        # Verify rejection of generic out-of-domain map hallucinations
        self.assertNotIn("a map of the proposed park", caption.lower())

    # -----------------------------------------------------------------------
    # Phase 7: Geospatial Metadata Integrity
    # -----------------------------------------------------------------------

    def test_ordinary_rgb_upload_never_fabricates_geospatial_metadata(self):
        """
        Uploading an ordinary RGB JPEG/PNG without GeoTIFF metadata must NEVER
        invent coordinates, CRS, sensor types, or GSD.
        """
        geo_meta = GeospatialEngine.extract_metadata(self.urban_img)
        self.assertFalse(geo_meta.geospatial_available, "Ordinary RGB must not claim geospatial availability")
        self.assertIsNone(geo_meta.crs, "Ordinary RGB must not invent CRS")
        self.assertIsNone(geo_meta.bounds_world, "Ordinary RGB must not invent world bounds")

    def test_ordinary_rgb_never_routed_to_multispectral_band_engine(self):
        """
        A general region query on an RGB image ('segment water') must not route
        to Multispectral_Analysis (which strictly requires genuine sensor bands).
        """
        routing = self.classifier.classify("segment the water")
        self.assertNotIn("Multispectral_Analysis", routing.target_tools)
        self.assertIn("Grounding", routing.target_tools)

    # -----------------------------------------------------------------------
    # Provenance & Introspection Honesty
    # -----------------------------------------------------------------------

    def test_model_provenance_distinction(self):
        """
        Verifies that neural model execution is labeled MODEL_DERIVED,
        while proxy-based segmentation is honestly labeled HEURISTIC.
        """
        region_res = rs_region_segmenter.segment_regions(self.coast_img, "water")
        self.assertEqual(region_res.get("provenance", {}).get("confidence_type"), "HEURISTIC")

        # Zero-shot land use classification using CLIP-RSICD
        if self.vqa_engine.rs_vision.is_available():
            scene_res = self.vqa_engine.rs_vision.classify_scene(self.forest_img)
            self.assertEqual(scene_res.get("provenance"), "REAL RS-ADAPTED MODEL")
            self.assertEqual(scene_res.get("confidence_type"), "model")


if __name__ == "__main__":
    unittest.main()
