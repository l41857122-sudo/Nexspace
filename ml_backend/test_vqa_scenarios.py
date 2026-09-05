"""
test_vqa_scenarios.py
----------------------
Comprehensive 22-Scenario RS-VQA Test Suite for Phase 2 validation.

Tests multi-tiered RS-VQA across diverse question families:
  1. Counting (buildings, ships, vehicles)
  2. Transport & Infrastructure (roads, highways, runways)
  3. Land-Use & Area Classification (dense residential, industrial, agricultural, forest)
  4. Maritime & Hydrology (ships, water bodies, rivers)
  5. Vegetation & Spectral Indices (woodland, crops, open terrain)
  6. Spatial Localization & Centroids (where are structures / vessels located)
  7. Comparative Reasoning (structures vs surrounding open land)
  8. Object Inventory & General Remote Sensing Inquiries
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from rs_vqa_engine import RemoteSensingVQAEngine, rs_vqa_engine
from tools import VQATool, tool_registry


class TestRSVQAScenarios(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = rs_vqa_engine
        cls.vqa_tool = VQATool()

        # 1. Synthetic Urban / Built-up Scene
        cls.urban_img = Image.new("RGB", (256, 256), (180, 180, 180))
        draw_u = ImageDraw.Draw(cls.urban_img)
        # Add road grid
        draw_u.line([(0, 128), (256, 128)], fill=(50, 50, 50), width=12)
        draw_u.line([(128, 0), (128, 256)], fill=(50, 50, 50), width=12)
        # Add building rooftops
        for x, y in [(30, 30), (70, 30), (30, 70), (160, 30), (200, 30), (160, 160), (200, 160), (30, 160)]:
            draw_u.rectangle([x, y, x + 28, y + 28], fill=(210, 60, 40), outline=(255, 255, 255))

        # 2. Synthetic Maritime / Port Scene
        cls.port_img = Image.new("RGB", (256, 256), (30, 80, 160))
        draw_p = ImageDraw.Draw(cls.port_img)
        # Add land/dock
        draw_p.rectangle([0, 0, 80, 256], fill=(160, 160, 160))
        # Add ship vessels in water
        draw_p.polygon([(120, 80), (160, 90), (160, 70)], fill=(240, 240, 240))
        draw_p.polygon([(180, 160), (220, 170), (220, 150)], fill=(240, 240, 240))

        # 3. Synthetic Vegetated / Agricultural Scene
        cls.forest_img = Image.new("RGB", (256, 256), (34, 139, 34))
        draw_f = ImageDraw.Draw(cls.forest_img)
        for i in range(10):
            draw_f.ellipse([i * 24, (i % 3) * 80, i * 24 + 30, (i % 3) * 80 + 30], fill=(20, 100, 20))

    def _verify_vqa_schema(self, res: dict):
        self.assertIn("answer", res)
        self.assertIn("question", res)
        self.assertIn("task", res)
        self.assertIn("model_id", res)
        self.assertIn("checkpoint", res)
        self.assertIn("inference_status", res)
        self.assertIn("confidence_source", res)
        self.assertIn("fallback_status", res)
        self.assertIn("evidence_references", res)
        self.assertTrue(len(res["answer"]) > 5, f"Answer too short or hardcoded: {res['answer']}")
        self.assertNotEqual(res["answer"].lower().strip(), "yes", "Answer must not be a hardcoded 'yes'")

    # Scenario 1: Counting Buildings
    def test_scenario_01_count_buildings(self):
        res = self.engine.answer_question(self.urban_img, "How many buildings are visible?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_COUNTING")
        self.assertTrue("building" in res["answer"].lower() or "candidate" in res["answer"].lower())

    # Scenario 2: Counting Ships
    def test_scenario_02_count_ships(self):
        res = self.engine.answer_question(self.port_img, "Count the ships in the water")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_COUNTING")

    # Scenario 3: Road Presence
    def test_scenario_03_road_presence(self):
        res = self.engine.answer_question(self.urban_img, "Is there a road?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_PRESENCE")
        self.assertTrue("road" in res["answer"].lower() or "transport" in res["answer"].lower() or "corridor" in res["answer"].lower())

    # Scenario 4: Highway Presence
    def test_scenario_04_highway_presence(self):
        res = self.engine.answer_question(self.urban_img, "Is there a highway visible?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_PRESENCE")

    # Scenario 5: Land Use Type
    def test_scenario_05_area_type(self):
        res = self.engine.answer_question(self.urban_img, "What type of area is shown?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_LAND_USE")

    # Scenario 6: Dominant Land-Use
    def test_scenario_06_dominant_land_use(self):
        res = self.engine.answer_question(self.forest_img, "What is the dominant land-use pattern?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_LAND_USE")

    # Scenario 7: Ships Presence
    def test_scenario_07_ships_presence(self):
        res = self.engine.answer_question(self.port_img, "Are there ships?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_MARITIME")

    # Scenario 8: Vessels in Harbor
    def test_scenario_08_vessels_in_harbor(self):
        res = self.engine.answer_question(self.port_img, "Are vessels docked in the harbor?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_MARITIME")

    # Scenario 9: Object Inventory
    def test_scenario_09_objects_present(self):
        res = self.engine.answer_question(self.urban_img, "What objects are present?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_INVENTORY")

    # Scenario 10: Elements Visible
    def test_scenario_10_elements_visible(self):
        res = self.engine.answer_question(self.port_img, "What elements can you see in this scene?")
        self._verify_vqa_schema(res)

    # Scenario 11: Vegetation Visible
    def test_scenario_11_vegetation_visible(self):
        res = self.engine.answer_question(self.forest_img, "Is vegetation visible?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_VEGETATION")
        self.assertTrue("vegetation" in res["answer"].lower() or "canopy" in res["answer"].lower())

    # Scenario 12: Trees and Forests
    def test_scenario_12_trees_and_forests(self):
        res = self.engine.answer_question(self.forest_img, "Are there trees and forests?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_VEGETATION")

    # Scenario 13: Water Body Presence
    def test_scenario_13_water_present(self):
        res = self.engine.answer_question(self.port_img, "Is there water present?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_HYDROLOGY")
        self.assertTrue("water" in res["answer"].lower())

    # Scenario 14: River or Coastal Body
    def test_scenario_14_river_or_coastal(self):
        res = self.engine.answer_question(self.port_img, "Can you see a river or coastal body?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_HYDROLOGY")

    # Scenario 15: Spatial Localization of Buildings
    def test_scenario_15_where_are_buildings(self):
        res = self.engine.answer_question(self.urban_img, "Where are the buildings?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_LOCALIZATION")

    # Scenario 16: Spatial Localization of Ships
    def test_scenario_16_where_are_ships(self):
        res = self.engine.answer_question(self.port_img, "Where are the ships located?")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_LOCALIZATION")

    # Scenario 17: Structural Comparison
    def test_scenario_17_compare_structures(self):
        res = self.engine.answer_question(self.urban_img, "Compare the visible structures with the surrounding area.")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_COMPARISON")

    # Scenario 18: Contrast Built-up vs Greenery
    def test_scenario_18_contrast_builtup_vegetation(self):
        res = self.engine.answer_question(self.forest_img, "Contrast the built-up area with the vegetation.")
        self._verify_vqa_schema(res)
        self.assertEqual(res["task"], "VQA_COMPARISON")

    # Scenario 19: Industrial vs Residential
    def test_scenario_19_industrial_vs_residential(self):
        res = self.engine.answer_question(self.urban_img, "What kind of residential or commercial zone is this?")
        self._verify_vqa_schema(res)

    # Scenario 20: Airport Facility
    def test_scenario_20_airport_facility(self):
        res = self.engine.answer_question(self.urban_img, "Is there an airport or runway facility?")
        self._verify_vqa_schema(res)

    # Scenario 21: Solar Energy Facility
    def test_scenario_21_solar_facility(self):
        res = self.engine.answer_question(self.forest_img, "Are there solar panels or energy facilities?")
        self._verify_vqa_schema(res)

    # Scenario 22: General Landscape Inspection
    def test_scenario_22_general_inspection(self):
        res = self.engine.answer_question(self.urban_img, "What is the general condition of this landscape?")
        self._verify_vqa_schema(res)

    # Tool Integration Check: VQATool delegates cleanly to RS-VQA engine
    def test_tool_registry_vqa_integration(self):
        tool_res = self.vqa_tool.execute({"image": self.urban_img, "question": "How many buildings are visible?"})
        self.assertEqual(tool_res.status, "success")
        self.assertIsNotNone(tool_res.data.get("primary_answer"))
        self.assertTrue(len(tool_res.evidence) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
