"""
confidence_system.py
---------------------
Scientific Confidence Provenance & Calibration Engine.

Enforces strict epistemological integrity:
  - Eliminates fabricated / hardcoded static confidence numbers
  - Distinguishes "Uncalibrated model confidence", "Heuristic estimate", and "Calibrated probability"
  - Standardizes confidence metadata schema:
      * raw_score
      * normalized_confidence
      * confidence_type
      * confidence_source
      * confidence_label
      * model_id
      * evaluation_status
      * fallback_status
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

_dir = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ConfidenceRecord:
    raw_score: Optional[float]
    normalized_confidence: Optional[float]
    confidence_type: str  # "uncalibrated_model" | "calibrated_model" | "heuristic" | "unavailable"
    confidence_source: str
    confidence_label: str
    model_id: str
    evaluation_status: str  # "UNVERIFIED" | "BENCHMARK_VALIDATED" | "NOT_CALIBRATED"
    fallback_status: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_score": round(self.raw_score, 4) if self.raw_score is not None else None,
            "normalized_confidence": round(self.normalized_confidence, 2) if self.normalized_confidence is not None else None,
            "confidence_type": self.confidence_type,
            "confidence_source": self.confidence_source,
            "confidence_label": self.confidence_label,
            "model_id": self.model_id,
            "evaluation_status": self.evaluation_status,
            "fallback_status": self.fallback_status,
        }


class ConfidenceEngine:
    """
    Computes and formats scientifically truthful confidence records.
    """

    @staticmethod
    def evaluate(
        raw_score: Optional[float] = None,
        model_id: str = "generic_vision_model",
        task_type: str = "VQA",
        is_fallback: bool = False,
        evidence_count: int = 1,
        is_calibrated: bool = False,
        calibration_reference: Optional[str] = None,
    ) -> ConfidenceRecord:
        """
        Creates a verified confidence record without fake precision.
        """
        if is_fallback:
            # Fallback heuristic: bounded between 0.20 and 0.45, explicitly marked
            norm_conf = min(0.45, max(0.20, (raw_score or 0.30)))
            return ConfidenceRecord(
                raw_score=raw_score,
                normalized_confidence=round(norm_conf, 2),
                confidence_type="heuristic",
                confidence_source=f"{model_id}_fallback_heuristic",
                confidence_label="Heuristic estimate (unverified fallback)",
                model_id=model_id,
                evaluation_status="NOT_CALIBRATED",
                fallback_status=True,
            )

        if raw_score is None:
            return ConfidenceRecord(
                raw_score=None,
                normalized_confidence=None,
                confidence_type="unavailable",
                confidence_source=model_id,
                confidence_label="Confidence metric unavailable",
                model_id=model_id,
                evaluation_status="NOT_CALIBRATED",
                fallback_status=False,
            )

        # Genuine Model Score
        clamped_score = max(0.0, min(1.0, float(raw_score)))

        if is_calibrated and calibration_reference:
            conf_type = "calibrated_model"
            conf_label = f"Calibrated probability (Reference: {calibration_reference})"
            eval_status = "BENCHMARK_VALIDATED"
        else:
            conf_type = "uncalibrated_model"
            conf_label = "Uncalibrated model confidence"
            eval_status = "NOT_CALIBRATED"

        return ConfidenceRecord(
            raw_score=clamped_score,
            normalized_confidence=round(clamped_score, 2),
            confidence_type=conf_type,
            confidence_source=f"{model_id}_logits",
            confidence_label=conf_label,
            model_id=model_id,
            evaluation_status=eval_status,
            fallback_status=False,
        )


# Global Confidence Engine instance
confidence_engine = ConfidenceEngine()
