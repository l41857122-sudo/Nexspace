"""
test_isro_evaluation.py
------------------------
Test suite for Phase 9: ISRO / SAC Evaluation Readiness.

Validates:
  1. Status compliance: Reports 'READY FOR EVALUATION' (never 'EVALUATED' prematurely)
  2. Cartosat-2S and RISAT multi-modal ingestion pipeline
  3. Co-registration validation integration
  4. Export of evaluation prediction payloads to JSON
"""

from __future__ import annotations
import os
import sys
import unittest
import json
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from isro_evaluation import isro_evaluation_adapter, ISROEvaluationAdapter


class TestISROEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.adapter = isro_evaluation_adapter
        cls.cartosat_sample = Image.new("RGB", (256, 256), (150, 150, 150))
        cls.risat_sample = Image.new("RGB", (256, 256), (80, 80, 80))

    # Test 1: Evaluation Readiness Status
    def test_readiness_status_honesty(self):
        res = self.adapter.process_isro_sample(
            optical_image=self.cartosat_sample,
            sar_image=self.risat_sample,
            query="Analyze Cartosat-2S and RISAT co-registered pair",
        )
        self.assertEqual(res["readiness_status"], "READY FOR EVALUATION")
        self.assertNotEqual(res["readiness_status"], "EVALUATED")
        self.assertIn("Cartosat-2S Optical", res["sensors_ingested"])
        self.assertIn("RISAT SAR", res["sensors_ingested"])

    # Test 2: Multi-Sensor Analysis Output
    def test_multisensor_analysis_presence(self):
        res = self.adapter.process_isro_sample(
            optical_image=self.cartosat_sample,
            sar_image=self.risat_sample,
        )
        self.assertIn("coregistration", res)
        self.assertIn("multimodal_fusion", res)
        self.assertIn("vqa_prediction", res)

    # Test 3: Submission Export File Generation
    def test_submission_export_file(self):
        res = self.adapter.process_isro_sample(
            optical_image=self.cartosat_sample,
        )
        export_path = self.adapter.export_evaluation_payload(res, "test_isro_submission.json")
        self.assertTrue(os.path.exists(export_path))
        with open(export_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["readiness_status"], "READY FOR EVALUATION")


if __name__ == "__main__":
    unittest.main(verbosity=2)
