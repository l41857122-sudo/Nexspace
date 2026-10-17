"""
test_gate3_step2_input_integrity.py
-----------------------------------
GATE 3 / STEP 2: Strict Optical-SAR Input Integrity Test Suite.

Validates the complete 10-case test matrix:
1. optical + valid independent SAR -> fusion may execute
2. optical only -> fusion MUST NOT execute
3. SAR only -> fusion MUST NOT execute
4. missing optical -> fusion MUST NOT execute
5. optical + same exact image as SAR -> fusion MUST NOT execute (rejected)
6. optical + invalid SAR (non-image) -> fusion MUST NOT execute
7. optical + SAR with different dimensions -> current resizing occurs without crash, dimension_mismatch_rescaled recorded
8. optical + SAR with CRS mismatch -> MUST NOT claim geospatially valid fusion
9. optical + SAR with non-overlapping bounds -> MUST NOT execute geospatially valid fusion
10. missing CRS/bounds -> MUST NOT fabricate metadata, records geospatial_validation_unavailable
"""

from __future__ import annotations
import os
import sys
import json
import unittest
import hashlib
from typing import Dict, Any, List
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir) if os.path.basename(_dir) == "tests" else _dir
ml_dir = os.path.join(_root, "ml_backend")
if ml_dir not in sys.path:
    sys.path.insert(0, ml_dir)

from tools import OpticalSARAnalysisTool
from model_runtime import OpticalSARFusionRuntime
from optical_sar_fusion import optical_sar_fusion_engine


class TestGate3Step2InputIntegrity(unittest.TestCase):
    test_results: List[Dict[str, Any]] = []

    @classmethod
    def setUpClass(cls):
        cls.tool = OpticalSARAnalysisTool()

        # 1. Independent Optical Image (Red roof + green vegetation)
        cls.opt_img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        d_opt = ImageDraw.Draw(cls.opt_img)
        d_opt.rectangle([20, 20, 80, 80], fill=(220, 40, 40))
        d_opt.rectangle([100, 100, 200, 200], fill=(30, 160, 30))

        # 2. Independent SAR Image (Dark terrain + bright corner reflector)
        cls.sar_img = Image.new("RGB", (256, 256), color=(50, 50, 50))
        d_sar = ImageDraw.Draw(cls.sar_img)
        d_sar.rectangle([20, 20, 80, 80], fill=(255, 255, 255))
        d_sar.rectangle([100, 100, 200, 200], fill=(90, 90, 90))

        # 3. Mismatched dimension SAR Image (128x128)
        cls.sar_mismatched_img = cls.sar_img.resize((128, 128), Image.Resampling.BILINEAR)

    @classmethod
    def tearDownClass(cls):
        # Save complete test matrix results to GATE3_STEP2_INPUT_INTEGRITY_RESULTS.json
        out_path = os.path.join(_dir, "GATE3_STEP2_INPUT_INTEGRITY_RESULTS.json")
        summary = {
            "gate": "GATE 3",
            "step": "STEP 2",
            "title": "Strict Optical-SAR Input Integrity Verification",
            "total_tests": len(cls.test_results),
            "passed_tests": sum(1 for r in cls.test_results if r.get("test_passed")),
            "test_matrix": cls.test_results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved test matrix results ({len(cls.test_results)} cases) to {out_path}")

    def _record_result(self, case_id: int, name: str, res_dict: Dict[str, Any], passed: bool):
        data = res_dict.get("data", {})
        record = {
            "case_id": case_id,
            "case_name": name,
            "test_passed": passed,
            "optical_present": data.get("optical_present", False),
            "sar_present": data.get("sar_present", False),
            "optical_hash": data.get("optical_hash"),
            "sar_hash": data.get("sar_hash"),
            "crs": data.get("crs"),
            "bounds": data.get("bounds"),
            "alignment_status": data.get("alignment_status"),
            "fusion_executed": data.get("fusion_executed", False),
            "status": res_dict.get("status"),
            "provenance": res_dict.get("provenance"),
            "confidence": res_dict.get("confidence"),
            "error": res_dict.get("error"),
        }
        self.test_results.append(record)

    # ------------------------------------------------------------------
    # CASE 1: Optical + valid independent SAR -> fusion may execute
    # ------------------------------------------------------------------
    def test_01_optical_and_independent_sar(self):
        print("\n--- [CASE 1] Optical + valid independent SAR ---")
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.sar_img,
        })
        d = res.to_dict()
        passed = (
            res.status == "success"
            and d["data"].get("fusion_executed") is True
            and d["data"].get("optical_present") is True
            and d["data"].get("sar_present") is True
            and d["data"].get("optical_hash") != d["data"].get("sar_hash")
            and res.error is None
        )
        self._record_result(1, "optical_and_independent_sar", d, passed)
        self.assertTrue(passed, f"Case 1 failed: {res.error}")

    # ------------------------------------------------------------------
    # CASE 2: Optical only -> fusion MUST NOT execute
    # ------------------------------------------------------------------
    def test_02_optical_only(self):
        print("\n--- [CASE 2] Optical only ---")
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": None,
        })
        d = res.to_dict()
        passed = (
            res.status == "failed"
            and res.provenance == "UNAVAILABLE"
            and res.confidence is None
            and d["data"].get("fusion_executed") is False
            and "independent SAR" in (res.error or "")
        )
        self._record_result(2, "optical_only", d, passed)
        self.assertTrue(passed, f"Case 2 failed: expected failure on missing SAR, got {res.status}")

    # ------------------------------------------------------------------
    # CASE 3: SAR only -> fusion MUST NOT execute
    # ------------------------------------------------------------------
    def test_03_sar_only(self):
        print("\n--- [CASE 3] SAR only ---")
        res = self.tool.execute({
            "optical_image": None,
            "sar_image": self.sar_img,
        })
        d = res.to_dict()
        passed = (
            res.status == "failed"
            and res.provenance == "UNAVAILABLE"
            and res.confidence is None
            and d["data"].get("fusion_executed") is False
            and "optical image" in (res.error or "").lower()
        )
        self._record_result(3, "sar_only", d, passed)
        self.assertTrue(passed, f"Case 3 failed: expected failure on missing optical, got {res.status}")

    # ------------------------------------------------------------------
    # CASE 4: Missing optical and SAR -> fusion MUST NOT execute
    # ------------------------------------------------------------------
    def test_04_missing_optical(self):
        print("\n--- [CASE 4] Missing optical and SAR ---")
        res = self.tool.execute({
            "optical_image": None,
            "sar_image": None,
        })
        d = res.to_dict()
        passed = (
            res.status == "failed"
            and res.provenance == "UNAVAILABLE"
            and res.confidence is None
            and d["data"].get("fusion_executed") is False
        )
        self._record_result(4, "missing_optical", d, passed)
        self.assertTrue(passed, f"Case 4 failed: {res.error}")

    # ------------------------------------------------------------------
    # CASE 5: Optical + same exact image as SAR -> MUST NOT execute
    # ------------------------------------------------------------------
    def test_05_optical_and_same_exact_image(self):
        print("\n--- [CASE 5] Optical + same exact image as SAR ---")
        # Test A: identical PIL object
        res_a = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.opt_img,
        })
        # Test B: cloned/identical pixel bytes
        opt_clone = self.opt_img.copy()
        res_b = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": opt_clone,
        })
        d = res_b.to_dict()
        passed = (
            res_a.status == "failed"
            and res_b.status == "failed"
            and res_b.provenance == "UNAVAILABLE"
            and res_b.confidence is None
            and d["data"].get("fusion_executed") is False
            and "independent" in (res_b.error or "").lower()
        )
        self._record_result(5, "optical_and_same_exact_image", d, passed)
        self.assertTrue(passed, f"Case 5 failed: self-fusion was not rejected: {res_b.error}")

    # ------------------------------------------------------------------
    # CASE 6: Optical + invalid SAR (non-image) -> MUST NOT execute
    # ------------------------------------------------------------------
    def test_06_optical_and_invalid_sar(self):
        print("\n--- [CASE 6] Optical + invalid SAR ---")
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": "not_an_image_instance",  # type: ignore
        })
        d = res.to_dict()
        passed = (
            res.status == "failed"
            and res.provenance == "UNAVAILABLE"
            and res.confidence is None
            and d["data"].get("fusion_executed") is False
            and "valid PIL Image" in (res.error or "")
        )
        self._record_result(6, "optical_and_invalid_sar", d, passed)
        self.assertTrue(passed, f"Case 6 failed: {res.error}")

    # ------------------------------------------------------------------
    # CASE 7: Optical + SAR with different dimensions -> resizing occurs, recorded
    # ------------------------------------------------------------------
    def test_07_different_dimensions_handled(self):
        print("\n--- [CASE 7] Different dimensions handled ---")
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.sar_mismatched_img,
        })
        d = res.to_dict()
        passed = (
            res.status == "success"
            and d["data"].get("fusion_executed") is True
            and "dimension_mismatch_rescaled" in d["data"].get("alignment_status", "")
            and d["data"].get("dimension_alignment") == "dimension_mismatch_rescaled"
        )
        self._record_result(7, "different_dimensions_handled", d, passed)
        self.assertTrue(passed, f"Case 7 failed: {res.error}")

    # ------------------------------------------------------------------
    # CASE 8: Optical + SAR with CRS mismatch -> MUST NOT claim geospatial validity
    # ------------------------------------------------------------------
    def test_08_crs_mismatch_not_geospatially_valid(self):
        print("\n--- [CASE 8] Optical + SAR with CRS mismatch ---")
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.sar_img,
            "optical_crs": "EPSG:32643",
            "sar_crs": "EPSG:32644",
        })
        d = res.to_dict()
        passed = (
            d["data"].get("geospatial_valid") is False
            and d["data"].get("geospatial_alignment_status") == "crs_mismatch"
            and "crs_mismatch" in d["data"].get("alignment_status", "")
        )
        self._record_result(8, "crs_mismatch_not_geospatially_valid", d, passed)
        self.assertTrue(passed, f"Case 8 failed: claimed geospatial validity despite CRS mismatch")

    # ------------------------------------------------------------------
    # CASE 9: Optical + SAR with non-overlapping bounds -> MUST NOT execute
    # ------------------------------------------------------------------
    def test_09_non_overlapping_bounds_rejected(self):
        print("\n--- [CASE 9] Optical + SAR with non-overlapping bounds ---")
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.sar_img,
            "optical_bounds": [72.80, 18.90, 72.85, 18.95],   # Mumbai
            "sar_bounds": [77.55, 12.95, 77.60, 13.00],       # Bengaluru (Disjoint)
        })
        d = res.to_dict()
        passed = (
            res.status == "failed"
            and res.provenance == "UNAVAILABLE"
            and d["data"].get("fusion_executed") is False
            and d["data"].get("geospatial_valid") is False
            and "overlap" in (res.error or "").lower()
        )
        self._record_result(9, "non_overlapping_bounds_rejected", d, passed)
        self.assertTrue(passed, f"Case 9 failed: did not reject disjoint bounds: {res.error}")

    # ------------------------------------------------------------------
    # CASE 10: Missing CRS/bounds -> MUST NOT fabricate metadata
    # ------------------------------------------------------------------
    def test_10_missing_geospatial_metadata_honest(self):
        print("\n--- [CASE 10] Missing CRS/bounds ---")
        # Ordinary synthetic PNG without any GeoTIFF tags
        res = self.tool.execute({
            "optical_image": self.opt_img,
            "sar_image": self.sar_img,
        })
        d = res.to_dict()
        crs_info = d["data"].get("crs", {})
        passed = (
            crs_info.get("optical") is None
            and crs_info.get("sar") is None
            and d["data"].get("geospatial_valid") is False
            and d["data"].get("geospatial_alignment_status") == "geospatial_validation_unavailable"
        )
        self._record_result(10, "missing_geospatial_metadata_honest", d, passed)
        self.assertTrue(passed, f"Case 10 failed: fabricated geospatial metadata: {crs_info}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
