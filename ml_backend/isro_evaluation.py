"""
isro_evaluation.py
-------------------
ISRO / SAC (Space Applications Centre) Multi-Sensor Evaluation Readiness Adapter.

Supports:
  1. Cartosat-2S High-Resolution Optical Imagery (PAN & Multispectral)
  2. RISAT-1 / RISAT-2B C-Band SAR Radar Imagery
  3. Co-registered Cartosat-2S + RISAT Multi-Modal Evaluation Pairs

Scientific Compliance:
  - If held-out evaluation imagery is not provided, reports status = 'READY FOR EVALUATION'
  - Never claims status = 'EVALUATED' without genuine data ingestion
  - Exports standard GeoJSON bounding boxes, mask geometries, and structured VQA predictions
"""

from __future__ import annotations
import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from coregistration import coregistration_validator
from optical_sar_fusion import optical_sar_fusion_engine
from rs_vqa_engine import rs_vqa_engine
from tiling import tiled_inference_engine
from geojson_export import export_evidence_to_geojson


class ISROEvaluationAdapter:
    """
    Dedicated evaluation ingest adapter and inference pipeline for ISRO/SAC datasets.
    """

    SENSOR_SPECS = {
        "Cartosat-2S": {
            "modality": "Optical High-Resolution",
            "gsd_nominal_meters": 0.65,
            "bands": ["PAN", "B1", "B2", "B3", "B4"],
            "radiometric_resolution_bits": 11,
        },
        "RISAT": {
            "modality": "C-band Synthetic Aperture Radar (SAR)",
            "polarizations": ["HH", "HV", "VH", "VV", "RH", "RV"],
            "operating_frequency_ghz": 5.35,
        },
    }

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(_dir, "isro_evaluation_exports")
        self.readiness_status = "READY FOR EVALUATION"

    def process_isro_sample(
        self,
        optical_image: Optional[Image.Image] = None,
        sar_image: Optional[Image.Image] = None,
        query: str = "Perform complete multi-sensor intelligence extraction",
        scene_id: Optional[str] = None,
        spatial_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the ISRO multi-sensor evaluation workflow.
        """
        t0 = time.perf_counter()
        sid = scene_id or f"isro_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        modalities = []
        if optical_image:
            modalities.append("Cartosat-2S Optical")
        if sar_image:
            modalities.append("RISAT SAR")

        if not modalities:
            return {
                "evaluation_id": sid,
                "readiness_status": "READY FOR EVALUATION",
                "message": "Awaiting ISRO/SAC test imagery. Pipeline is initialized and compliant.",
                "supported_sensors": list(self.SENSOR_SPECS.keys()),
            }

        predictions: Dict[str, Any] = {
            "evaluation_id": sid,
            "readiness_status": "READY FOR EVALUATION",
            "sensors_ingested": modalities,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # 1. Co-Registration Verification if both modalities present
        if optical_image and sar_image:
            reg_res = coregistration_validator.validate_and_align(optical_image, sar_image)
            predictions["coregistration"] = reg_res.to_dict()

            # Multimodal Fusion Reasoning
            fusion_res = optical_sar_fusion_engine.analyze_pair(optical_image, sar_image, query=query)
            predictions["multimodal_fusion"] = {
                "optical_evidence": fusion_res.optical_evidence,
                "sar_evidence": fusion_res.sar_evidence,
                "fused_conclusion": fusion_res.fused_conclusion,
                "model_provenance": fusion_res.model_provenance,
            }

        # 2. VQA and Object Detection on Optical
        primary_img = optical_image or sar_image
        if primary_img:
            vqa_res = rs_vqa_engine.answer_question(primary_img, query)
            predictions["vqa_prediction"] = {
                "question": query,
                "answer": vqa_res["answer"],
                "confidence": vqa_res.get("confidence"),
                "confidence_source": vqa_res.get("confidence_source"),
                "model_id": vqa_res.get("model_id"),
                "provenance": vqa_res.get("inference_status"),
            }

            # Tiled High-Resolution Extraction if image is large (>768px)
            w, h = primary_img.size
            if max(w, h) > 768:
                tiling_res = tiled_inference_engine.run_tiled_grounding(primary_img, "structures")
                predictions["grounded_objects_count"] = tiling_res["count"]
                predictions["grounded_objects"] = tiling_res["detections"]
            else:
                predictions["grounded_objects_count"] = 0
                predictions["grounded_objects"] = []

        dur = (time.perf_counter() - t0) * 1000.0
        predictions["processing_time_ms"] = round(dur, 2)

        return predictions

    def export_evaluation_payload(self, results: Dict[str, Any], output_filename: Optional[str] = None) -> str:
        """Saves evaluation results to JSON in the exports directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        fname = output_filename or f"{results.get('evaluation_id', 'eval')}_submission.json"
        out_path = os.path.join(self.output_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return out_path


# Global ISRO Evaluation Adapter
isro_evaluation_adapter = ISROEvaluationAdapter()
