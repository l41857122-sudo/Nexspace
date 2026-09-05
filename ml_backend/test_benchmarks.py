"""
test_benchmarks.py
-------------------
Test suite for Phase 8: Benchmark Evaluation Framework.

Validates:
  1. Benchmark schema completeness for RSVQA, VRSBench, CDVQA, and BigEarthNet
  2. Truthful handling of missing datasets: reports 'NOT RUN — DATASET NOT AVAILABLE'
  3. No synthetic / fabricated scores generated
  4. JSON report generation and serialization
"""

from __future__ import annotations
import os
import sys
import unittest
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from benchmarks.evaluator import BenchmarkEvaluator, evaluate_benchmark


class TestBenchmarks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evaluator = BenchmarkEvaluator()

    # Test 1: Supported Benchmarks Coverage
    def test_supported_benchmarks_schemas(self):
        schemas = self.evaluator.BENCHMARK_SCHEMAS
        self.assertIn("RSVQA", schemas)
        self.assertIn("VRSBench", schemas)
        self.assertIn("CDVQA", schemas)
        self.assertIn("BigEarthNet", schemas)

    # Test 2: Honest Reporting on Absent Datasets
    def test_absent_dataset_reporting(self):
        res = self.evaluator.run_benchmark("RSVQA", custom_data_dir="./nonexistent_rsvqa_path")
        self.assertEqual(res["status"], "NOT RUN — DATASET NOT AVAILABLE")
        self.assertIn("In adherence to Scientific Honesty rules", res["note"])
        self.assertNotIn("accuracy_overall", res)  # Must NOT contain a fake accuracy score

    # Test 3: CDVQA Absent Dataset Reporting
    def test_cdvqa_absent_dataset(self):
        res = self.evaluator.run_benchmark("CDVQA", custom_data_dir="./nonexistent_cdvqa_path")
        self.assertEqual(res["status"], "NOT RUN — DATASET NOT AVAILABLE")
        self.assertIn("expected_files", res)

    # Test 4: Report Saved to Disk
    def test_report_file_generation(self):
        res = self.evaluator.run_benchmark("VRSBench", custom_data_dir="./nonexistent_vrsbench_path")
        report_path = os.path.join(self.evaluator.output_root, "vrsbench_evaluation_report.json")
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data["status"], "NOT RUN — DATASET NOT AVAILABLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
