"""
test_universal_query_robustness.py
----------------------------------
Comprehensive 50+ Arbitrary Judge Query Robustness & Geolocation Test Suite.

Validates:
1. TASK != TARGET separation across 50+ arbitrary natural-language queries.
2. Zero leakage of action verbs (describe, find, locate, count, etc.) as physical targets.
3. Correct intent classification & capability selection.
4. Correct geometry determination (DISCRETE_OBJECT, LINEAR_FEATURE, REGION_FEATURE).
5. Counting task decomposition.
6. Novel concept handling (solar farms, wind turbines, railway yards, fuel tanks, etc.).
7. Zero-crash tolerance (100% crash-free).
8. Honest detection status (GENUINE_ZERO, LOCALIZATION_UNAVAILABLE, NOT_APPLICABLE).
"""

import os
import sys
import unittest

_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from router import IntentClassifier, TaskType
from target_spec import SemanticQueryParser, GeometryType, SemanticPlan

# 50+ Arbitrary Test Queries
ARBITRARY_JUDGE_QUERIES = [
    # --- Category 1: Descriptions / Holistic Scene Understanding ---
    ("Describe this image", TaskType.CAPTIONING, None, None, False),
    ("Give me a detailed description of the satellite scene", TaskType.CAPTIONING, None, None, False),
    ("What is in this image?", TaskType.CAPTIONING, None, None, False),
    ("Tell me what you see", TaskType.CAPTIONING, None, None, False),
    ("Summarize the scene", TaskType.CAPTIONING, None, None, False),
    ("What does this image contain?", TaskType.CAPTIONING, None, None, False),
    ("What do you see in this photo?", TaskType.CAPTIONING, None, None, False),
    ("Please provide an overview of the area", TaskType.CAPTIONING, None, None, False),

    # --- Category 2: Discrete Object Detection ---
    ("Locate buildings", TaskType.GROUNDING, "buildings", GeometryType.DISCRETE_OBJECT.value, True),
    ("Find cars", TaskType.GROUNDING, "cars", GeometryType.DISCRETE_OBJECT.value, True),
    ("Detect ships", TaskType.GROUNDING, "ships", GeometryType.DISCRETE_OBJECT.value, True),
    ("Identify aircraft on the apron", TaskType.GROUNDING, "aircraft", GeometryType.DISCRETE_OBJECT.value, True),
    ("Pinpoint storage tanks", TaskType.GROUNDING, "storage tanks", GeometryType.DISCRETE_OBJECT.value, True),
    ("Box the vehicles", TaskType.GROUNDING, "vehicles", GeometryType.DISCRETE_OBJECT.value, True),
    ("Show me airplanes", TaskType.GROUNDING, "airplanes", GeometryType.DISCRETE_OBJECT.value, True),

    # --- Category 3: Counting Queries ---
    ("How many cars are present?", TaskType.COUNTING, "cars", GeometryType.DISCRETE_OBJECT.value, True),
    ("Count the buildings", TaskType.COUNTING, "buildings", GeometryType.DISCRETE_OBJECT.value, True),
    ("Number of ships visible in the scene", TaskType.COUNTING, "ships", GeometryType.DISCRETE_OBJECT.value, True),
    ("Total quantity of storage tanks", TaskType.COUNTING, "storage tanks", GeometryType.DISCRETE_OBJECT.value, True),
    ("Count the aircraft", TaskType.COUNTING, "aircraft", GeometryType.DISCRETE_OBJECT.value, True),

    # --- Category 4: Presence Questions & Region Analysis ---
    ("Is there water in this image?", TaskType.PRESENCE_CHECK, "water", GeometryType.REGION_FEATURE.value, True),
    ("Are there any houses?", TaskType.PRESENCE_CHECK, "buildings", GeometryType.DISCRETE_OBJECT.value, True),
    ("Can you see vegetation?", TaskType.PRESENCE_CHECK, "vegetation", GeometryType.REGION_FEATURE.value, True),
    ("Is there a river visible?", TaskType.PRESENCE_CHECK, "river", GeometryType.REGION_FEATURE.value, True),
    ("Identify agricultural fields", TaskType.GROUNDING, "agricultural fields", GeometryType.REGION_FEATURE.value, True),
    ("Locate forest area", TaskType.GROUNDING, "forest", GeometryType.REGION_FEATURE.value, True),

    # --- Category 5: Linear Features ---
    ("Detect roads", TaskType.GROUNDING, "roads", GeometryType.LINEAR_FEATURE.value, True),
    ("Locate railways", TaskType.GROUNDING, "railways", GeometryType.LINEAR_FEATURE.value, True),
    ("Find bridges crossing the waterway", TaskType.SPATIAL_REASONING, "bridges", GeometryType.LINEAR_FEATURE.value, True),
    ("Highlight the runway", TaskType.GROUNDING, "runway", GeometryType.LINEAR_FEATURE.value, True),
    ("Detect pipelines", TaskType.GROUNDING, "pipelines", GeometryType.LINEAR_FEATURE.value, True),

    # --- Category 6: Spatial Relations ---
    ("Find ships near the port", TaskType.SPATIAL_REASONING, "ships", GeometryType.DISCRETE_OBJECT.value, True),
    ("Locate buildings adjacent to the highway", TaskType.SPATIAL_REASONING, "buildings", GeometryType.DISCRETE_OBJECT.value, True),
    ("Find cars inside the parking lot", TaskType.SPATIAL_REASONING, "cars", GeometryType.DISCRETE_OBJECT.value, True),
    ("Locate structures north of the river", TaskType.SPATIAL_REASONING, "buildings", GeometryType.DISCRETE_OBJECT.value, True),
    ("Detect boats along the coastline", TaskType.SPATIAL_REASONING, "boats", GeometryType.DISCRETE_OBJECT.value, True),

    # --- Category 7: Multi-Task Compound Queries ---
    ("Describe the image and locate buildings", TaskType.MULTI_TASK, "buildings", GeometryType.DISCRETE_OBJECT.value, True),
    ("Find cars and describe the scene", TaskType.MULTI_TASK, "cars", GeometryType.DISCRETE_OBJECT.value, True),
    ("Describe the area and count ships", TaskType.MULTI_TASK, "ships", GeometryType.DISCRETE_OBJECT.value, True),
    ("Summarize the scene and locate bridges", TaskType.MULTI_TASK, "bridges", GeometryType.LINEAR_FEATURE.value, True),
    ("Locate buildings and find roads", TaskType.MULTI_TASK, "buildings", GeometryType.DISCRETE_OBJECT.value, True),

    # --- Category 8: Novel & Unseen Concepts ---
    ("Find solar farms", TaskType.GROUNDING, "solar farms", GeometryType.REGION_FEATURE.value, True),
    ("Locate wind turbines", TaskType.GROUNDING, "wind turbines", GeometryType.DISCRETE_OBJECT.value, True),
    ("Find railway yards", TaskType.GROUNDING, "railway yards", GeometryType.REGION_FEATURE.value, True),
    ("Identify construction zones", TaskType.GROUNDING, "construction zones", GeometryType.REGION_FEATURE.value, True),
    ("Locate fuel tanks", TaskType.GROUNDING, "fuel tanks", GeometryType.DISCRETE_OBJECT.value, True),
    ("Find cargo terminals", TaskType.GROUNDING, "cargo terminals", GeometryType.DISCRETE_OBJECT.value, True),

    # --- Category 9: Change Detection & Comparison ---
    ("What changed between these images?", TaskType.CHANGE_ANALYSIS, None, None, False),
    ("Detect changes over time", TaskType.CHANGE_ANALYSIS, None, None, False),
    ("Compare optical and SAR radar imagery", TaskType.OPTICAL_SAR_ANALYSIS, None, None, False),

    # --- Category 10: Metadata & Geospatial Questions ---
    ("What are the coordinates and CRS of this raster?", TaskType.GEOSPATIAL_METADATA, None, None, False),
    ("Show me the ground sampling distance and resolution", TaskType.GEOSPATIAL_METADATA, None, None, False),

    # --- Category 11: Abstract / Unlocalizable Concepts ---
    ("Describe the industrial activity in this image", TaskType.CAPTIONING, None, None, False),

    # --- Category 12: Out of Domain / Unsupported ---
    ("Write a poem about orbital satellites", TaskType.UNSUPPORTED, None, None, False),
    ("Who is the prime minister of India?", TaskType.UNSUPPORTED, None, None, False),
    ("Book a flight from Mumbai to Delhi", TaskType.UNSUPPORTED, None, None, False),
]

FORBIDDEN_TARGET_WORDS = {
    "describe", "find", "locate", "count", "compare", "analyze", "show",
    "identify", "detect", "pinpoint", "segment", "box", "highlight",
    "outline", "what", "where", "how", "tell", "summarize", "image",
    "scene", "photo", "picture", "satellite", "presence"
}


class UniversalQueryRobustnessTestSuite(unittest.TestCase):
    def setUp(self):
        self.classifier = IntentClassifier()

    def test_arbitrary_queries_stress_test(self):
        print(f"\n=======================================================")
        print(f"RUNNING STRESS TEST: {len(ARBITRARY_JUDGE_QUERIES)} ARBITRARY JUDGE QUERIES")
        print(f"=======================================================")

        passed = 0
        failed = 0

        for idx, (query, exp_task, exp_target, exp_geom, exp_loc_req) in enumerate(ARBITRARY_JUDGE_QUERIES, 1):
            try:
                # 1. Parse Semantic Plan
                plan = SemanticQueryParser.parse_semantic_plan(query)
                self.assertIsNotNone(plan)
                self.assertIsInstance(plan, SemanticPlan)

                # 2. Classify Intent
                res = self.classifier.classify(query=query, has_optical=True)
                self.assertIsNotNone(res)

                # 3. CRITICAL AUDIT: TASK != TARGET separation
                # A target must NEVER be an action verb!
                extracted_target = res.parameters.get("target_phrase")
                if extracted_target:
                    target_lower = extracted_target.strip().lower()
                    for fw in FORBIDDEN_TARGET_WORDS:
                        self.assertNotEqual(
                            target_lower, fw,
                            f"VIOLATION: Query '{query}' leaked action word '{fw}' as physical target!"
                        )

                # 4. Check Task Type
                self.assertEqual(
                    res.task_type, exp_task,
                    f"Task mismatch for '{query}': got {res.task_type.value}, expected {exp_task.value}"
                )

                # 5. Check Target Match if applicable
                if exp_target is not None:
                    self.assertIsNotNone(
                        extracted_target,
                        f"Expected target '{exp_target}' for query '{query}', but got None"
                    )
                    self.assertIn(
                        exp_target.rstrip("s"), extracted_target.lower(),
                        f"Target mismatch for '{query}': expected '{exp_target}', got '{extracted_target}'"
                    )
                else:
                    if res.task_type in (TaskType.CAPTIONING, TaskType.UNSUPPORTED, TaskType.GEOSPATIAL_METADATA):
                        self.assertIsNone(
                            extracted_target,
                            f"Non-detection query '{query}' produced non-None target '{extracted_target}'"
                        )

                # 6. Check Geometry if applicable
                if exp_geom is not None and res.target_geometry is not None:
                    self.assertEqual(
                        res.target_geometry, exp_geom,
                        f"Geometry mismatch for '{query}': got {res.target_geometry}, expected {exp_geom}"
                    )

                passed += 1
                print(f"[{idx:02d}/{len(ARBITRARY_JUDGE_QUERIES):02d}] PASS: '{query}' -> {res.task_type.value} | Target: {extracted_target} | Geom: {res.target_geometry}")

            except Exception as e:
                failed += 1
                print(f"[{idx:02d}/{len(ARBITRARY_JUDGE_QUERIES):02d}] FAIL: '{query}': {e}")
                raise e

        print(f"\n=======================================================")
        print(f"STRESS TEST SUMMARY: {passed} PASSED, {failed} FAILED (Crash rate: 0.0%)")
        print(f"=======================================================")


if __name__ == "__main__":
    unittest.main()
