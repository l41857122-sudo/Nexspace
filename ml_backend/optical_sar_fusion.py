"""
optical_sar_fusion.py
----------------------
Multimodal Optical + SAR Joint Reasoning & Multimodal Fusion Engine.

Separates and integrates complementary physical phenomenology:
  1. Optical Remote Sensing: Multi-spectral reflection, colorimetry, visible roof geometry, vegetation indices
  2. SAR Radar Remote Sensing: Microwave backscatter, double-bounce dihedral reflection, volume scattering, speckle statistics
  3. Joint Multimodal Fusion: Feature embedding alignment, spatial cross-correlation, physical cross-validation

Output Structure:
  - OPTICAL EVIDENCE : Visible spectral & geometric observations
  - SAR EVIDENCE     : Microwave backscatter & dielectric properties
  - FUSED CONCLUSION : Synergistic multi-sensor synthesis with honest provenance
"""

from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from rs_vision_core import rs_vision_runtime


@dataclass
class MultimodalFusionResult:
    optical_evidence: str
    sar_evidence: str
    fused_conclusion: str
    optical_metrics: Dict[str, Any]
    sar_metrics: Dict[str, Any]
    cross_modal_metrics: Dict[str, Any]
    is_trained_model: bool
    model_provenance: str
    confidence: Optional[float]
    confidence_type: str
    confidence_source: str
    evidence_nodes: List[Dict[str, Any]]
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optical_evidence": self.optical_evidence,
            "sar_evidence": self.sar_evidence,
            "fused_conclusion": self.fused_conclusion,
            "optical_metrics": self.optical_metrics,
            "sar_metrics": self.sar_metrics,
            "cross_modal_metrics": self.cross_modal_metrics,
            "is_trained_model": self.is_trained_model,
            "model_provenance": self.model_provenance,
            "confidence": self.confidence,
            "confidence_type": self.confidence_type,
            "confidence_source": self.confidence_source,
            "evidence_nodes": self.evidence_nodes,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class OpticalSARFusionEngine:
    """
    Multimodal Optical + SAR physical reasoning and feature-level fusion engine.
    """

    def __init__(self):
        self.rs_vision = rs_vision_runtime

    def analyze_pair(
        self,
        optical_image: Image.Image,
        sar_image: Image.Image,
        query: str = "Compare optical and SAR imagery",
    ) -> MultimodalFusionResult:
        t0 = time.perf_counter()

        # 1. Optical Physical Feature Extraction
        opt_rgb = np.array(optical_image.convert("RGB"), dtype=np.float32)
        opt_gray = np.array(optical_image.convert("L"), dtype=np.float32)
        h, w = opt_gray.shape

        mean_opt = float(np.mean(opt_rgb))
        std_opt = float(np.std(opt_rgb))

        r, g, b = opt_rgb[..., 0], opt_rgb[..., 1], opt_rgb[..., 2]
        exg = (2.0 * g - r - b) / (2.0 * g + r + b + 1e-6)
        veg_coverage = float(np.mean(exg > 0.05))
        water_mask = (b > r + 15) & (b > g) & (r < 110)
        water_coverage = float(np.mean(water_mask))

        gy, gx = np.gradient(opt_gray)
        opt_edge_density = float(np.mean(np.hypot(gx, gy) > 28.0))

        # Optical Evidence formulation
        opt_observations = []
        if opt_edge_density > 0.12:
            opt_observations.append("orthogonal building roof contours and structural infrastructure")
        if veg_coverage > 0.20:
            opt_observations.append(f"dense photosynthetic vegetation canopy ({veg_coverage:.1%} coverage)")
        if water_coverage > 0.10:
            opt_observations.append(f"visible water body ({water_coverage:.1%} coverage)")

        if not opt_observations:
            opt_observations.append("heterogeneous open terrain with low structural contrast")

        optical_evidence = (
            f"Optical Multi-Spectral Signal: Detected {', '.join(opt_observations)}. "
            f"Mean brightness intensity: {mean_opt:.1f}/255, edge structural density: {opt_edge_density:.1%}."
        )

        # 2. SAR Radar Microwave Feature Extraction
        # Ensure dimensions match for array operations
        if sar_image.size != optical_image.size:
            sar_rescaled = sar_image.resize(optical_image.size, Image.Resampling.BILINEAR)
        else:
            sar_rescaled = sar_image

        sar_gray = np.array(sar_rescaled.convert("L"), dtype=np.float32)
        mean_sar = float(np.mean(sar_gray))
        std_sar = float(np.std(sar_gray))

        # Radar Speckle & Backscatter Metrics
        speckle_cv = float(std_sar / (mean_sar + 1e-6))
        dyn_range_db = float(10.0 * np.log10((np.max(sar_gray) + 1e-6) / (np.min(sar_gray) + 1e-6)))

        # Strong double-bounce / corner reflector pixels (built-up structures, metal assets)
        high_backscatter_mask = sar_gray > (mean_sar + 1.6 * std_sar)
        high_backscatter_ratio = float(np.mean(high_backscatter_mask))

        # Low backscatter specular reflection (calm water, flat paved tarmac)
        specular_mask = sar_gray < max(15.0, mean_sar - 1.2 * std_sar)
        specular_ratio = float(np.mean(specular_mask))

        # SAR Evidence formulation
        sar_observations = []
        if high_backscatter_ratio > 0.05:
            sar_observations.append(
                f"intense metallic/dihedral double-bounce backscatter ({high_backscatter_ratio:.1%} area), indicating rigid vertical structural facets"
            )
        if specular_ratio > 0.15:
            sar_observations.append(
                f"specular radar reflection zones ({specular_ratio:.1%} area) corresponding to smooth surfaces or calm water"
            )
        if speckle_cv > 0.40:
            sar_observations.append(f"volume scattering texture (coefficient of variation: {speckle_cv:.2f})")

        if not sar_observations:
            sar_observations.append("moderate diffuse backscatter return across bare terrain")

        sar_evidence = (
            f"SAR Microwave Radar Signal: Detected {', '.join(sar_observations)}. "
            f"Mean backscatter intensity: {mean_sar:.1f}/255, dynamic range: {dyn_range_db:.1f} dB."
        )

        # 3. Cross-Modal Spatial & Feature Correlation
        if std_opt > 1e-6 and std_sar > 1e-6:
            spatial_corr = float(np.corrcoef(opt_gray.flatten(), sar_gray.flatten())[0, 1])
            if np.isnan(spatial_corr):
                spatial_corr = 0.0
        else:
            spatial_corr = 0.0

        # High-level vision feature extraction
        f_opt = self.rs_vision.extract_features(optical_image)
        f_sar = self.rs_vision.extract_features(sar_rescaled)

        if f_opt is not None and f_sar is not None:
            cosine_sim = float(np.dot(f_opt, f_sar))
        else:
            cosine_sim = float(0.5 + 0.5 * spatial_corr)

        # 4. Joint Multimodal Reasoning & Fused Conclusion
        conclusions = []
        if opt_edge_density > 0.10 and high_backscatter_ratio > 0.04:
            conclusions.append(
                "Both optical multi-spectral shapes and SAR double-bounce radar returns strongly cross-validate the presence of built-up physical structures."
            )
        elif opt_edge_density > 0.10 and high_backscatter_ratio <= 0.04:
            conclusions.append(
                "Optical channels detect structural outlines, while SAR shows diffuse scattering, suggesting low-profile or non-metallic construction materials."
            )

        if water_coverage > 0.08 and specular_ratio > 0.10:
            conclusions.append(
                "Water presence identified in optical imagery is corroborated by specular microwave absorption in the SAR radar channel."
            )

        if veg_coverage > 0.20 and speckle_cv > 0.35:
            conclusions.append(
                "Vegetation detected via optical Excess Green index aligns with microwave volume scattering characteristics in the radar channel."
            )

        if not conclusions:
            conclusions.append(
                f"Optical and SAR data show complementary coverage across the target scene (cross-modal feature correlation: {cosine_sim:.2f})."
            )

        fused_conclusion = " ".join(conclusions)

        dur = (time.perf_counter() - t0) * 1000.0

        evidence_nodes = [
            {
                "type": "optical_evidence",
                "evidence_text": optical_evidence,
                "metrics": {
                    "mean_intensity": round(mean_opt, 1),
                    "edge_density": round(opt_edge_density, 3),
                    "vegetation_coverage": round(veg_coverage, 3),
                    "water_coverage": round(water_coverage, 3),
                },
            },
            {
                "type": "sar_evidence",
                "evidence_text": sar_evidence,
                "metrics": {
                    "mean_backscatter": round(mean_sar, 1),
                    "high_backscatter_ratio": round(high_backscatter_ratio, 3),
                    "specular_ratio": round(specular_ratio, 3),
                    "speckle_cv": round(speckle_cv, 3),
                    "dynamic_range_db": round(dyn_range_db, 1),
                },
            },
            {
                "type": "fused_conclusion",
                "conclusion_text": fused_conclusion,
                "cross_modal_metrics": {
                    "spatial_pearson_correlation": round(spatial_corr, 3),
                    "feature_cosine_similarity": round(cosine_sim, 3),
                },
            },
        ]

        return MultimodalFusionResult(
            optical_evidence=optical_evidence,
            sar_evidence=sar_evidence,
            fused_conclusion=fused_conclusion,
            optical_metrics={
                "mean_intensity": round(mean_opt, 1),
                "std_intensity": round(std_opt, 1),
                "edge_density": round(opt_edge_density, 3),
                "vegetation_fraction": round(veg_coverage, 3),
                "water_fraction": round(water_coverage, 3),
            },
            sar_metrics={
                "mean_backscatter": round(mean_sar, 1),
                "std_backscatter": round(std_sar, 1),
                "high_backscatter_ratio": round(high_backscatter_ratio, 3),
                "specular_ratio": round(specular_ratio, 3),
                "speckle_cv": round(speckle_cv, 3),
                "dynamic_range_db": round(dyn_range_db, 1),
            },
            cross_modal_metrics={
                "spatial_correlation": round(spatial_corr, 3),
                "feature_similarity": round(cosine_sim, 3),
            },
            is_trained_model=False,
            model_provenance="Research baseline — trained multimodal fusion checkpoint unavailable.",
            confidence=None,
            confidence_type="unavailable",
            confidence_source="optical_sar_physics_fusion_baseline",
            evidence_nodes=evidence_nodes,
            processing_time_ms=round(dur, 2),
        )


# Global Optical-SAR Fusion Engine instance
optical_sar_fusion_engine = OpticalSARFusionEngine()
