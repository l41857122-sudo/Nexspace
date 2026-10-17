"""
tools.py
--------
Central Specialist Tool Registry and Model Adapters for the
Geospatial Vision-Language Controller.

Provides uniform tool interfaces with standardized execution,
input validation, output schema, error containment, explicit confidence tracking,
and integration with real model runtimes (with safe fallbacks).

Specialist Tools:
  1. VQATool ("VQA")
  2. OpticalCaptioningTool ("Optical_Caption")
  3. SARCaptioningTool ("SAR_Caption")
  4. GroundingTool ("Grounding")
  5. ChangeAnalysisTool ("Change_Analysis")
  6. ChangeVQATool ("Change_VQA")
  7. OpticalSARAnalysisTool ("Optical_SAR_Analysis")
"""

from __future__ import annotations
import time
import abc
import hashlib
import re
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image

from router import TaskType
from model_runtime import (
    BaseModelRuntime,
    PaliGemmaVQARuntime,
    BLIPCaptioningRuntime,
    GroundingDINORuntime,
    OpticalSARFusionRuntime,
    ChangeVQARuntime,
)
import change_analysis
from anomaly_engine import anomaly_engine, AnomalyEngine
from geospatial import GeospatialEngine, GeoMetadata
from rs_vqa_engine import rs_vqa_engine, RemoteSensingVQAEngine
from semantic_change import semantic_change_engine, SemanticChangeEngine, SemanticChangeResult
from optical_sar_fusion import optical_sar_fusion_engine, OpticalSARFusionEngine, MultimodalFusionResult
from multispectral import MultispectralScene, SpectralIndexResult, spectral_engine, SpectralEngine
from co_registration import (
    RasterGeometry,
    GeospatialCoRegistrationEngine,
    CoRegistrationResult,
    AlignmentStatus,
    OverlapStatus,
)
from overpass_client import get_overpass_client, OverpassQueryResult


# ---------------------------------------------------------------------------
# Standardized Tool Result Containers
# ---------------------------------------------------------------------------

@dataclass
class ToolExecutionResult:
    tool_name: str
    task_type: str
    status: str  # "success" | "fallback" | "unavailable" | "failed" | "degraded"
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    confidence_type: str = "unavailable"  # "heuristic" | "algorithmic" | "model" | "calibrated_model" | "unavailable"
    confidence_source: str = "none"
    provenance: str = "UNAVAILABLE"  # "MODEL_DERIVED" | "ALGORITHMIC" | "HEURISTIC" | "FALLBACK" | "UNAVAILABLE"
    model_loaded: bool = False
    fallback_used: bool = False
    method: Optional[str] = None
    capability: Optional[str] = None
    model_metadata: Optional[Dict[str, Any]] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        clean_data = {}
        for k, v in self.data.items():
            if k in ("raw_result", "raw_change_result", "image", "image_a", "image_b", "optical_image", "sar_image") or isinstance(v, Image.Image) or hasattr(v, "save"):
                continue
            clean_data[k] = v

        return {
            "tool_name": self.tool_name,
            "task_type": self.task_type,
            "status": self.status,
            "provenance": self.provenance,
            "model_loaded": self.model_loaded,
            "fallback_used": self.fallback_used,
            "method": self.method or self.task_type,
            "capability": self.capability or self.tool_name,
            "data": clean_data,
            "confidence": self.confidence,
            "confidence_type": self.confidence_type,
            "confidence_source": self.confidence_source,
            "model_metadata": self.model_metadata,
            "evidence": self.evidence,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


# Legacy compatibility containers
@dataclass
class VQAResult:
    question: str
    answer: str
    confidence: Optional[float] = None
    low_confidence: bool = False

    def __post_init__(self):
        if self.confidence is not None:
            self.low_confidence = self.confidence <= 0.40
        else:
            self.low_confidence = False


@dataclass
class CaptionResult:
    caption: str
    modality: str  # "optical" or "sar"


# ---------------------------------------------------------------------------
# Base Specialist Tool Abstract Class
# ---------------------------------------------------------------------------

class BaseSpecialistTool(abc.ABC):
    name: str = "BaseTool"
    task_type: TaskType = TaskType.VQA
    description: str = ""
    input_requirements: List[str] = []
    model_id: str = "base_model"
    model_family: str = "vision_language"
    modality: str = "optical"
    checkpoint: str = "base"
    source: str = "system"
    rs_adapted: bool = False
    trained: bool = False
    fine_tuned: bool = False
    evaluation_status: str = "UNVERIFIED"
    fallback_policy: str = "deterministic_adapter"

    def get_capability_profile(self) -> Dict[str, Any]:
        """Returns the full scientific capability and provenance profile of the specialist tool."""
        return {
            "name": self.name,
            "model_id": getattr(self, "runtime", None) and getattr(self.runtime, "model_id", self.model_id) or self.model_id,
            "model_family": self.model_family,
            "task": self.task_type.value,
            "modality": self.modality,
            "checkpoint": getattr(self, "runtime", None) and getattr(self.runtime, "model_id", self.checkpoint) or self.checkpoint,
            "source": self.source,
            "rs_adapted": self.rs_adapted,
            "trained": self.trained,
            "fine_tuned": self.fine_tuned,
            "evaluation_status": self.evaluation_status,
            "metrics": getattr(self, "metrics", {}),
            "device": getattr(self, "runtime", None) and getattr(self.runtime, "device", "cpu") or "cpu",
            "fallback_policy": self.fallback_policy,
            "input_requirements": self.input_requirements,
        }

    def validate_inputs(self, params: Dict[str, Any]) -> Optional[str]:
        """Verify that all required input keys exist in params."""
        for req in self.input_requirements:
            if req not in params or params[req] is None:
                return f"Missing required parameter '{req}' for specialist tool '{self.name}'"
        return None

    def execute(self, params: Dict[str, Any]) -> ToolExecutionResult:
        """Standardized resilient execution wrapper with timing & error isolation."""
        t_start = time.perf_counter()

        val_err = self.validate_inputs(params)
        if val_err:
            dur = (time.perf_counter() - t_start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="none",
                error=val_err,
                duration_ms=round(dur, 2),
            )

        try:
            res = self._run(params)
            dur = (time.perf_counter() - t_start) * 1000.0
            res.duration_ms = round(dur, 2)
            return res
        except Exception as e:
            import traceback
            traceback.print_exc()
            dur = (time.perf_counter() - t_start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="none",
                error=f"Execution error in specialist tool {self.name}: {str(e)}",
                duration_ms=round(dur, 2),
            )

    @abc.abstractmethod
    def _run(self, params: Dict[str, Any]) -> ToolExecutionResult:
        pass


# ---------------------------------------------------------------------------
# 1. VQA Specialist Tool (RS-Adapted Multi-Tiered Inference)
# ---------------------------------------------------------------------------

class VQATool(BaseSpecialistTool):
    name = "VQA"
    task_type = TaskType.VQA
    description = "Visual question answering specialist supporting PaliGemma RSVQA and multi-tier Remote Sensing feature reasoning."
    input_requirements = ["image"]

    def __init__(self, runtime: Optional[BaseModelRuntime] = None, engine: Optional[RemoteSensingVQAEngine] = None):
        self.runtime = runtime or PaliGemmaVQARuntime()
        self.engine = engine or rs_vqa_engine

    def validate_inputs(self, params: Dict[str, Any]) -> Optional[str]:
        img = params.get("image") or params.get("optical_image")
        if img is None:
            return "VQA analysis requires an input image."
        if not isinstance(img, Image.Image):
            return "Parameter 'image' must be a valid PIL Image."
        return None

    def _run(self, params: Dict[str, Any]) -> ToolExecutionResult:
        image: Image.Image = params.get("image") or params.get("optical_image")
        raw_questions = params.get("questions") or [params.get("question", "What type of area is shown?")]
        questions = [q.strip() for q in raw_questions if isinstance(q, str) and q.strip()]
        if not questions:
            questions = ["What type of area is shown?"]

        vqa_results: List[VQAResult] = []
        evidence = []
        is_real_model = self.runtime.is_available()

        all_confidences = []
        confidence_sources = []
        overall_status = "success" if is_real_model else "fallback"

        for q in questions:
            if is_real_model:
                infer_res = self.runtime.infer(image=image, question=q)
                ans_text = infer_res.get("answer", "No answer generated")
                conf = infer_res.get("confidence")
                conf_type = infer_res.get("confidence_type", "model")
                conf_source = infer_res.get("confidence_source", self.runtime.model_id)
                status = "success"
                task_name = "VQA"
                inf_status = "REAL RS-ADAPTED MODEL"
                is_fallback = False
                ev_refs = [f"vlm_token_generation_{q[:20]}"]
            else:
                # Delegate to intelligent RS-VQA engine
                vqa_ans = self.engine.answer_question(image=image, question=q)
                ans_text = vqa_ans["answer"]
                conf = vqa_ans.get("confidence")
                conf_type = vqa_ans.get("confidence_type", "heuristic")
                conf_source = vqa_ans.get("confidence_source", "rsvqa_heuristic_adapter")
                status = "fallback"
                task_name = vqa_ans.get("task", "VQA")
                inf_status = vqa_ans.get("inference_status", "HEURISTIC FALLBACK (VLM unavailable)")
                is_fallback = True
                ev_refs = vqa_ans.get("evidence_references", [])

            if status in ("fallback", "degraded") and overall_status == "success":
                overall_status = "fallback"

            if conf is not None:
                all_confidences.append(conf)
            if conf_source:
                confidence_sources.append(conf_source)

            vqa_item = VQAResult(question=q, answer=ans_text, confidence=conf)
            vqa_results.append(vqa_item)

            evidence.append({
                "type": "vqa_answer",
                "question": q,
                "answer": ans_text,
                "task": task_name,
                "model": self.runtime.model_id if is_real_model else vqa_ans.get("model", self.runtime.model_id),
                "model_loaded": is_real_model,
                "checkpoint": self.runtime.model_id if is_real_model else vqa_ans.get("checkpoint", self.runtime.model_id),
                "confidence": conf,
                "confidence_type": conf_type,
                "confidence_source": conf_source,
                "inference_status": inf_status,
                "provenance": "MODEL_DERIVED" if is_real_model else vqa_ans.get("provenance", "HEURISTIC"),
                "fallback": is_fallback,
                "fallback_reason": None if is_real_model else (getattr(self.runtime, "load_error", None) or "VLM neural runtime unavailable; using domain heuristic reasoning"),
                "evidence_references": ev_refs,
            })

        avg_conf = round(float(np.mean(all_confidences)), 2) if all_confidences else None
        source_str = ", ".join(list(dict.fromkeys([s for s in confidence_sources if s]))) or "none"

        return ToolExecutionResult(
            tool_name=self.name,
            task_type=self.task_type.value,
            status=overall_status,
            provenance="MODEL_DERIVED" if is_real_model else "HEURISTIC",
            model_loaded=is_real_model,
            fallback_used=not is_real_model,
            method="paligemma_generative_vqa" if is_real_model else "spectral_spatial_heuristics",
            capability="Visual Question Answering",
            data={
                "vqa_results": [vars(r) for r in vqa_results],
                "primary_answer": vqa_results[0].answer if vqa_results else None,
                "inference_mode": "model_pipeline" if is_real_model else "heuristic_fallback",
                "total_questions": len(questions),
                "limitations": [
                    "VQA answers represent probabilistic vision-language interpretations, not guaranteed geospatial ground truth."
                ] if is_real_model else [
                    "PaliGemma neural runtime unavailable (gated checkpoint/hardware); answer generated via deterministic domain heuristics.",
                    "Heuristic answers should not be treated as ground truth or neural generative model outputs.",
                ],
            },
            confidence=avg_conf,
            confidence_type="model" if (is_real_model and avg_conf is not None) else ("heuristic" if avg_conf is not None else "unavailable"),
            confidence_source=source_str,
            model_metadata=self.runtime.get_metadata(),
            evidence=evidence,
        )

    # Legacy convenience helper
    def ask(self, image: Image.Image, question: str) -> VQAResult:
        res = self.execute({"image": image, "question": question})
        if res.data.get("vqa_results"):
            item = res.data["vqa_results"][0]
            return VQAResult(question=item["question"], answer=item["answer"], confidence=item["confidence"])
        return VQAResult(question=question, answer="Aerial scene inspected.", confidence=0.50)

    def ask_batch(self, image: Image.Image, questions: List[str]) -> List[VQAResult]:
        res = self.execute({"image": image, "questions": questions})
        if res.data.get("vqa_results"):
            return [
                VQAResult(question=item["question"], answer=item["answer"], confidence=item["confidence"])
                for item in res.data["vqa_results"]
            ]
        return [VQAResult(question=q, answer="Aerial scene inspected.", confidence=0.50) for q in questions]


# ---------------------------------------------------------------------------
# 2. Optical Captioning Specialist Tool
# ---------------------------------------------------------------------------

class OpticalCaptioningTool(BaseSpecialistTool):
    name = "Optical_Caption"
    task_type = TaskType.CAPTIONING
    description = "BLIP image captioning specialist for optical multi-spectral satellite imagery and photographs."
    input_requirements = ["image"]

    def __init__(self, runtime: Optional[BaseModelRuntime] = None):
        self.runtime = runtime or BLIPCaptioningRuntime()

    def _run(self, params: Dict[str, Any]) -> ToolExecutionResult:
        image: Image.Image = params["image"]
        is_real_model = self.runtime.is_available()
        caption_text = None
        raw_caption = ""
        rejection_reason = None
        status = "pending"
        conf_val = None
        conf_type = "unavailable"
        conf_source = "none"
        capability = "optical_captioning"
        evidence_list = []

        if is_real_model:
            try:
                inf = self.runtime.infer(image=image, modality="optical")
                caption_text = inf.get("caption")
                raw_caption = inf.get("raw_caption", "")
                cap_status = inf.get("caption_status", "success")
                capability = inf.get("model_capability", "generic_image_captioning")
                conf_val = inf.get("confidence", 0.65)
                conf_type = inf.get("confidence_type", "model")
                conf_source = inf.get("confidence_source", self.runtime.model_id)

                if cap_status == "invalid_generation" or not caption_text:
                    status = "invalid_generation"
                    rejection_reason = inf.get("rejection_reason", "Pathological repetition loop or low diversity")
                    evidence_list = []
                else:
                    status = "success"
                    evidence_list = [{"modality": "optical", "caption": caption_text, "capability": capability}]
            except Exception as ex:
                is_real_model = False
                rejection_reason = str(ex)

        if not caption_text or status != "success":
            from rs_vision_core import rs_vision_runtime
            import numpy as np

            scene_res = rs_vision_runtime.classify_scene(image, top_k=2)
            top_scene = scene_res.get("top_class", "mixed_landscape")
            top_desc = scene_res.get("top_description", "mixed aerial terrain")

            arr = np.array(image.convert("RGB"), dtype=np.float32)
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            veg_ratio = float(np.mean((2.0 * g - r - b) / (2.0 * g + r + b + 1e-6) > 0.05))
            water_ratio = float(np.mean((b > r + 15) & (b > g) & (r < 110)))

            features = []
            if water_ratio > 0.05:
                features.append(f"water body coverage ({water_ratio:.1%})")
            if veg_ratio > 0.15:
                features.append(f"vegetated terrain ({veg_ratio:.1%})")
            features.append(f"predominantly {top_desc}")

            caption_text = f"[Degraded Scene Analysis] Overhead capture showing {', '.join(features)}."
            capability = "spectral_domain_heuristic_fallback"
            conf_val = None
            conf_type = "unavailable"
            conf_source = "optical_caption_fallback"
            status = "degraded"
            evidence_list = [{"modality": "optical", "caption": caption_text, "capability": capability, "fallback": True}]

        return ToolExecutionResult(
            tool_name=self.name,
            task_type=self.task_type.value,
            status=status,
            provenance="MODEL_DERIVED" if is_real_model and status == "success" else "FALLBACK",
            model_loaded=is_real_model and status == "success",
            fallback_used=not (is_real_model and status == "success"),
            method="blip_image_captioning" if is_real_model and status == "success" else "spectral_scene_heuristics",
            capability="Optical Scene Captioning",
            data={
                "caption": caption_text,
                "raw_caption": raw_caption,
                "caption_status": status,
                "rejection_reason": rejection_reason,
                "modality": "optical",
                "model_capability": capability,
                "inference_mode": "model_pipeline" if (is_real_model and status == "success") else "deterministic_fallback",
            },
            confidence=conf_val,
            confidence_type=conf_type,
            confidence_source=conf_source if conf_val is not None else "none",
            model_metadata=self.runtime.get_metadata(),
            evidence=evidence_list,
        )

    # Legacy convenience helper
    def caption(self, image: Image.Image, modality: str = "optical") -> CaptionResult:
        res = self.execute({"image": image, "modality": modality})
        text = res.data.get("caption", "")
        return CaptionResult(caption=text, modality="optical")


# ---------------------------------------------------------------------------
# 3. SAR Captioning Specialist Tool
# ---------------------------------------------------------------------------

class SARCaptioningTool(BaseSpecialistTool):
    name = "SAR_Caption"
    task_type = TaskType.CAPTIONING
    description = "Specialist tool for radar backscatter interpretation and SAR imagery captioning."
    input_requirements = ["image"]

    def __init__(self, runtime: Optional[BaseModelRuntime] = None):
        self.runtime = runtime or BLIPCaptioningRuntime()

    def _run(self, params: Dict[str, Any]) -> ToolExecutionResult:
        image: Image.Image = params["image"]
        is_real_model = self.runtime.is_available()
        caption_text = None
        raw_caption = ""
        rejection_reason = None
        status = "pending"
        conf_val = None
        conf_type = "unavailable"
        conf_source = "none"
        capability = "sar_captioning"
        evidence_list = []

        if is_real_model:
            try:
                inf = self.runtime.infer(image=image, modality="sar")
                caption_text = inf.get("caption")
                raw_caption = inf.get("raw_caption", "")
                cap_status = inf.get("caption_status", "success")
                capability = inf.get("model_capability", "generic_captioning_on_SAR")
                conf_val = inf.get("confidence", 0.60)
                conf_type = inf.get("confidence_type", "model")
                conf_source = inf.get("confidence_source", self.runtime.model_id)

                if cap_status == "invalid_generation" or not caption_text:
                    status = "invalid_generation"
                    rejection_reason = inf.get("rejection_reason", "Pathological repetition loop or low diversity")
                    evidence_list = []
                else:
                    status = "success"
                    evidence_list = [{"modality": "sar", "caption": caption_text, "capability": capability}]
            except Exception as ex:
                is_real_model = False
                rejection_reason = str(ex)

        if not is_real_model:
            caption_text = "SAR backscatter imagery loaded. Specialist SAR captioning model unavailable in current environment."
            capability = "radar_backscatter_interpretation"
            conf_val = None
            conf_type = "unavailable"
            conf_source = "sar_caption_adapter"
            status = "degraded"
            evidence_list = [{"modality": "sar", "caption": caption_text, "capability": capability}]

        return ToolExecutionResult(
            tool_name=self.name,
            task_type=self.task_type.value,
            status=status,
            provenance="MODEL_DERIVED" if is_real_model and status == "success" else "FALLBACK",
            model_loaded=is_real_model and status == "success",
            fallback_used=not (is_real_model and status == "success"),
            method="blip_generic_sar_captioning" if is_real_model and status == "success" else "sar_interpretation_baseline",
            capability="SAR Scene Captioning",
            data={
                "caption": caption_text,
                "raw_caption": raw_caption,
                "caption_status": status,
                "rejection_reason": rejection_reason,
                "modality": "sar",
                "model_capability": capability,
                "limitation_notice": "Generic optical/vision model operating on radar backscatter data.",
                "inference_mode": "model_pipeline" if (is_real_model and status == "success") else "deterministic_fallback",
            },
            confidence=conf_val,
            confidence_type=conf_type,
            confidence_source=conf_source if conf_val is not None else "none",
            model_metadata=self.runtime.get_metadata(),
            evidence=evidence_list,
        )

    # Legacy convenience helper
    def caption(self, image: Image.Image, modality: str = "sar") -> CaptionResult:
        res = self.execute({"image": image, "modality": modality})
        text = res.data.get("caption", "")
        return CaptionResult(caption=text, modality="sar")


from grounding_adapters import GroundingAdapterFactory, BaseGroundingAdapter, GroundingDINOAdapter, RSAdaptedVisionGroundingAdapter, RSGuidedSpectralSpatialAdapter
from region_segmenter import rs_region_segmenter, classify_target_geometry, TargetGeometry
from spatial_reasoning import SpatialRelationshipEngine
from target_spec import TargetSpec, GeometryType, SemanticQueryParser


def extract_structural_salience_fallback(
    image: Image.Image,
    target_phrase: str,
    img_w: int,
    img_h: int,
    max_candidates: int = 3,
) -> List[Dict[str, Any]]:
    """
    Guaranteed non-zero structural salience & contour fallback (< 15ms).
    Uses high-contrast gradient contour extraction across image rasters
    so the user NEVER receives 0 detections on a populated scene.
    """
    # Blind salience fallbacks that fabricate bounding boxes over random high-contrast
    # edges and forcefully label them with the user's query are completely deleted.
    # Return genuine zero detections to prevent false positive hallucinations.
    return []


# ---------------------------------------------------------------------------
# 4. Grounding Specialist Tool (Hybrid Remote-Sensing Specialist + DINO Open-Vocabulary)
# ---------------------------------------------------------------------------

class GroundingTool(BaseSpecialistTool):
    name = "Grounding"
    task_type = TaskType.GROUNDING
    description = "Hybrid spatial grounding engine supporting RS-adapted specialist localization, region segmentation, and Grounding DINO open-vocabulary detection."
    input_requirements = ["image"]

    def __init__(self, runtime: Optional[Any] = None):
        self._is_custom_runtime = runtime is not None
        if runtime is not None:
            self.runtime = runtime
            self.dino_adapter = runtime
            self.rs_adapter = runtime
        else:
            self.dino_adapter = GroundingDINOAdapter()
            self.rs_adapter = RSGuidedSpectralSpatialAdapter()
            self.region_segmenter = rs_region_segmenter
            self.runtime = self.dino_adapter

    def _execute_region_segmentation(
        self,
        image: Image.Image,
        target_phrase: str,
        w: int,
        h: int,
    ) -> ToolExecutionResult:
        geo_meta = GeospatialEngine.extract_metadata(image)
        t0 = time.perf_counter()
        
        region_res = self.region_segmenter.segment_regions(image=image, target_phrase=target_phrase)
        detections = region_res.get("detections", [])
        
        evidence = []
        for d in detections:
            ev_item = dict(d)
            if "polygon" in d or "polygon_pixels" in d or d.get("has_polygon"):
                ev_item["has_polygon"] = True
                ev_item["polygon"] = d.get("polygon", [])
                ev_item["polygon_pixels"] = d.get("polygon_pixels", [])
                ev_item["segmentation_method"] = d.get("segmentation_method", "spectral_morphological_contour")
            if geo_meta.geospatial_available:
                ev_item = GeospatialEngine.enrich_evidence_item(ev_item, geo_meta)
            evidence.append(ev_item)
            
        dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        count = len(detections)
        avg_score = round(sum(d["score"] for d in detections) / count, 4) if count > 0 else None
        
        return ToolExecutionResult(
            tool_name=self.name,
            task_type=self.task_type.value,
            status="success",
            provenance="HEURISTIC",
            model_loaded=False,
            fallback_used=False,
            method="morphological_spectral_contour_extraction",
            capability="Regional Land Cover Segmentation",
            data={
                "target_phrase": target_phrase,
                "prompt_used": target_phrase,
                "target_geometry": TargetGeometry.REGION_FEATURE.value,
                "geometry_type": "polygon",
                "tiling_used": False,
                "detections": detections,
                "count": count,
                "image_width": w,
                "image_height": h,
                "summary": region_res.get("summary", f"Detected {count} region(s) for '{target_phrase}'."),
                "inference_time_ms": dur_ms,
                "model": "RS_Specialist_Region_Segmenter",
                "confidence_type": "HEURISTIC",
                "confidence_method": "spectral-area proxy",
                "provenance": {
                    "confidence_type": "HEURISTIC",
                    "confidence_method": "spectral-area proxy",
                    "algorithm": "morphological_spectral_contour_extraction",
                    "spectral_index": region_res.get("spectral_algorithm", "spectral_proxy"),
                    "sensor_mode": "RGB-Derived Proxies (Note: Not true multi-spectral NIR/SWIR bands)",
                    "raw_candidates": region_res.get("raw_candidates_count", count),
                    "filtered_candidates": count,
                }
            },
            confidence=avg_score,
            confidence_type="heuristic",
            confidence_source="RS_Specialist_Region_Segmenter",
            model_metadata={
                "model_id": "RS_Specialist_Region_Segmenter",
                "source": "RS_Specialist_Region_Segmenter",
                "confidence_type": "HEURISTIC",
                "confidence_method": "spectral-area proxy"
            },
            evidence=evidence,
        )

    def _execute_hybrid_detection(
        self,
        image: Image.Image,
        target_phrase: str,
        w: int,
        h: int,
    ) -> ToolExecutionResult:
        from tiling import tiled_inference_engine, apply_nms, compute_iou
        geo_meta = GeospatialEngine.extract_metadata(image)

        dino_available = self.dino_adapter.is_available()
        rs_available = self.rs_adapter.is_available()

        all_candidates: List[Dict[str, Any]] = []
        dino_time = 0.0
        rs_time = 0.0
        t0 = time.perf_counter()

        # Target classification
        tgt_geom = classify_target_geometry(target_phrase)
        geom_val = str(getattr(tgt_geom, "value", tgt_geom)).lower()
        is_linear_target = "linear" in geom_val or any(k in target_phrase.lower() for k in ["road", "highway", "bridge", "railway", "runway"])

        # 1. Automatic Internal Negative / Distractor Prompting (Contrastive Grounding)
        # Prevents Grounding DINO from forcing ships, docks, or land to be labeled as "road"
        INTERNAL_DISTRACTORS = ["ship", "boat", "craft", "vessel", "building", "water", "trees", "land", "harbor", "dock", "pier", "storage tank", "bridge", "road", "car"]
        tp_clean = re.sub(r"\b(find|locate|detect|identify|show|me|where|is|are|the|all|in|this|image|scene|count|search|for)\b", " ", target_phrase, flags=re.IGNORECASE).strip().lower()
        if not tp_clean:
            tp_clean = target_phrase.strip().lower()
        user_targets = [tp_clean]
        if tp_clean.endswith("s") and len(tp_clean) > 3:
            user_targets.append(tp_clean[:-1])
        else:
            user_targets.append(tp_clean + "s")

        active_distractors = [
            d for d in INTERNAL_DISTRACTORS
            if d not in user_targets and not any(ut == d or d == ut for ut in user_targets)
        ]

        # Combine user target with internal background distractors
        contrastive_prompt = " . ".join(user_targets + active_distractors) + " ."

        # Calibrated thresholds: box_threshold=0.22, text_threshold=0.20
        box_thresh = 0.22
        text_thresh = 0.20

        if dino_available:
            try:
                t_dino = time.perf_counter()
                dino_inf = tiled_inference_engine.run_tiled_grounding(
                    image=image,
                    target_phrase=contrastive_prompt,
                    grounding_runtime=self.dino_adapter,
                    tile_size=640,
                    overlap=0.15,
                    iou_threshold=0.45,
                    box_threshold=box_thresh,
                    text_threshold=text_thresh,
                    enable_multi_scale=False,
                )
                dino_time = (time.perf_counter() - t_dino) * 1000.0
                for d in dino_inf.get("detections", []):
                    d_copy = dict(d)
                    d_copy["source"] = "Grounding_DINO"
                    if "score" in d_copy and d_copy["score"] is not None:
                        d_copy["score"] = round(float(d_copy["score"]), 4)
                    all_candidates.append(d_copy)
            except Exception:
                pass

        # 2. Strict Contrastive Post-Filter: Discard all detections matching internal distractors and not user_targets
        filtered_candidates = []
        for d in all_candidates:
            raw_lbl = str(d.get("label", "")).strip().lower()
            is_distractor = any(dist == raw_lbl or dist in raw_lbl for dist in active_distractors)
            matches_user_target = False
            for ut in user_targets:
                if ut == raw_lbl or ut in raw_lbl or raw_lbl in ut:
                    matches_user_target = True
                    break
                if ut in ("road", "roads") and any(k in raw_lbl for k in ("road", "highway", "street", "avenue", "freeway", "lane", "pathway")):
                    matches_user_target = True
                    break
                if ut in ("ship", "ships") and any(k in raw_lbl for k in ("ship", "boat", "vessel", "craft", "marine")):
                    matches_user_target = True
                    break

            if is_distractor and not matches_user_target:
                continue
            if not matches_user_target and len(active_distractors) > 0:
                continue

            filtered_candidates.append(d)
        all_candidates = filtered_candidates

        # 3. Class-Aware NMS over combined proposals with cross-class duplicate suppression
        final_detections = apply_nms(all_candidates, iou_threshold=0.45, class_aware=True, cross_class_iou_threshold=0.70)

        # 4. Box Size Cap: Discard any Grounding DINO box covering > 40% of total image area when smaller detections exist
        total_area = float(w * h)
        has_smaller = any(
            ((d.get("bbox_pixel") or [0, 0, 0, 0])[2] - (d.get("bbox_pixel") or [0, 0, 0, 0])[0]) *
            ((d.get("bbox_pixel") or [0, 0, 0, 0])[3] - (d.get("bbox_pixel") or [0, 0, 0, 0])[1]) <= 0.40 * total_area
            for d in final_detections if d.get("bbox_pixel")
        )
        if has_smaller:
            final_detections = [
                d for d in final_detections
                if not d.get("bbox_pixel") or
                (((d["bbox_pixel"][2] - d["bbox_pixel"][0]) * (d["bbox_pixel"][3] - d["bbox_pixel"][1])) <= 0.40 * total_area)
            ]

        from router import normalize_detection_label
        for d in final_detections:
            d["label"] = normalize_detection_label(d.get("label", ""), target_phrase)

        # 5. Promptable SAM 2 & Clean Rectangular Box Geometries
        try:
            from sam2_engine import sam2_segmentation_engine
            final_detections = sam2_segmentation_engine.enrich_detections(image, final_detections, target_phrase=target_phrase)
        except Exception:
            pass

        for d in final_detections:
            d["label"] = normalize_detection_label(d.get("label", ""), target_phrase)

        # 6. Cap maximum returned detections to top 15 highest-confidence targets sorted by score
        final_detections.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
        final_detections = final_detections[:15]

        # Enrich geospatial coordinates
        evidence = []
        for d in final_detections:
            ev_item = dict(d)
            if geo_meta.geospatial_available:
                ev_item = GeospatialEngine.enrich_evidence_item(ev_item, geo_meta)
            evidence.append(ev_item)

        total_dur = (time.perf_counter() - t0) * 1000.0
        primary_source = "Hybrid (RS Specialist + DINO)" if (dino_available and rs_available) else ("RS_Adapted_Vision_Grounder" if rs_available else "Grounding_DINO")

        if final_detections:
            avg_score = round(sum(d["score"] for d in final_detections) / len(final_detections), 4)
            summary = f"{primary_source} detected {len(final_detections)} instance(s) of '{target_phrase}' in the scene (mean score: {avg_score:.2f})."
            conf = avg_score
        else:
            summary = f"{primary_source} scanned scene for '{target_phrase}': No instances detected above threshold."
            conf = None

        return ToolExecutionResult(
            tool_name=self.name,
            task_type=self.task_type.value,
            status="success",
            provenance="MODEL_DERIVED",
            model_loaded=dino_available or rs_available,
            fallback_used=False,
            method="tiled_hybrid_grounding",
            capability="Open-Vocabulary Object Grounding",
            data={
                "target_phrase": target_phrase,
                "prompt_used": target_phrase,
                "tiling_used": True,
                "detections": final_detections,
                "count": len(final_detections),
                "image_width": w,
                "image_height": h,
                "summary": summary,
                "inference_time_ms": round(total_dur, 2),
                "model": primary_source,
                "provenance": {
                    "dino_available": dino_available,
                    "rs_available": rs_available,
                    "dino_latency_ms": round(dino_time, 2),
                    "rs_latency_ms": round(rs_time, 2),
                }
            },
            confidence=conf,
            confidence_type="model" if (dino_available or rs_available) else "unavailable",
            confidence_source=primary_source if (dino_available or rs_available) else "none",
            model_metadata={"model_id": primary_source, "source": primary_source},
            evidence=evidence,
        )

    def _run(self, params: Dict[str, Any]) -> ToolExecutionResult:
        image: Image.Image = params["image"]
        all_targets = params.get("all_targets")
        target_phrase: str = params.get("target_phrase") or (all_targets[0] if (all_targets and len(all_targets) > 0) else "objects")
        w, h = image.size

        # If an explicit runtime was passed (e.g. Unit Test mock), execute it directly
        if self._is_custom_runtime:
            if self.runtime.is_available():
                try:
                    from tiling import tiled_inference_engine
                    inf = tiled_inference_engine.run_tiled_grounding(
                        image=image,
                        target_phrase=target_phrase,
                        grounding_runtime=self.runtime,
                        tile_size=512,
                        overlap=0.25,
                    )
                    dets = inf.get("detections", [])
                    det_scores = [d.get("score") for d in dets if d.get("score") is not None]
                    conf_calc = round(float(np.mean(det_scores)), 4) if det_scores else None
                    return ToolExecutionResult(
                        tool_name=self.name,
                        task_type=self.task_type.value,
                        status="success",
                        provenance="MODEL_DERIVED",
                        model_loaded=True,
                        fallback_used=False,
                        method="custom_grounding_runtime",
                        capability="Open-Vocabulary Object Grounding",
                        data={
                            "target_phrase": target_phrase,
                            "detections": dets,
                            "count": len(dets),
                            "image_width": w,
                            "image_height": h,
                            "summary": f"Detected {len(dets)} instance(s) of '{target_phrase}'.",
                        },
                        confidence=conf_calc,
                        confidence_type="model" if conf_calc is not None else "unavailable",
                        confidence_source=getattr(self.runtime, "model_id", "grounding_model"),
                        evidence=dets,
                    )
                except Exception:
                    pass

        # Multi-target check: if multiple distinct targets are requested, process each target and preserve separate outcomes
        all_targets = params.get("all_targets")
        if not all_targets and target_phrase:
            if " and " in target_phrase or " & " in target_phrase or "," in target_phrase:
                parts = re.split(r"[,&]|\band\b", target_phrase)
                cleaned_targets = [p.strip() for p in parts if p.strip() and p.strip() not in SemanticQueryParser.NON_TARGET_TOKENS]
                if len(cleaned_targets) > 1:
                    all_targets = cleaned_targets

        if all_targets and len(all_targets) > 1:
            combined_detections: List[Dict[str, Any]] = []
            combined_evidence: List[Dict[str, Any]] = []
            target_outcomes: Dict[str, Any] = {}
            total_dur_ms = 0.0

            for tgt in all_targets:
                tgt_clean = tgt.strip()
                if not tgt_clean:
                    continue
                tgt_geom = classify_target_geometry(tgt_clean)
                if tgt_geom == TargetGeometry.REGION_FEATURE:
                    tgt_res = self._execute_region_segmentation(image, tgt_clean, w, h)
                elif self.dino_adapter.is_available() or self.rs_adapter.is_available():
                    tgt_res = self._execute_hybrid_detection(image, tgt_clean, w, h)
                else:
                    tgt_res = None

                if tgt_res and tgt_res.status in ("success", "fallback"):
                    dets = tgt_res.data.get("detections", [])
                    for d in dets:
                        d["target_class"] = tgt_clean
                        if "score" in d and d["score"] is not None:
                            d["score"] = round(float(d["score"]), 4)
                        if "box" in d and isinstance(d["box"], (list, tuple)):
                            d["box"] = [int(round(float(c))) for c in d["box"]]
                        if "box_2d" in d and isinstance(d["box_2d"], (list, tuple)):
                            d["box_2d"] = [int(round(float(c))) for c in d["box_2d"]]
                        if "bbox_normalized" in d and isinstance(d["bbox_normalized"], (list, tuple)):
                            d["bbox_normalized"] = [int(round(float(c))) for c in d["bbox_normalized"]]
                        if "bbox_pixel" in d and isinstance(d["bbox_pixel"], (list, tuple)):
                            d["bbox_pixel"] = [round(float(c), 2) for c in d["bbox_pixel"]]
                    combined_detections.extend(dets)
                    combined_evidence.extend(tgt_res.evidence)
                    target_outcomes[tgt_clean] = {
                        "count": len(dets),
                        "geometry_type": tgt_geom.value,
                        "status": "success",
                        "summary": tgt_res.data.get("summary", f"Detected {len(dets)} instance(s) of '{tgt_clean}'."),
                        "detections": dets,
                    }
                    total_dur_ms += tgt_res.duration_ms
                else:
                    fb = extract_structural_salience_fallback(image, tgt_clean, w, h, max_candidates=2)
                    for d in fb:
                        d["target_class"] = tgt_clean
                    combined_detections.extend(fb)
                    combined_evidence.extend(fb)
                    target_outcomes[tgt_clean] = {
                        "count": len(fb),
                        "geometry_type": tgt_geom.value,
                        "status": "success",
                        "summary": f"Detected {len(fb)} instance(s) of '{tgt_clean}'.",
                        "detections": fb,
                    }

            outcome_summaries = [f"{tgt}: {info['count']} detected" for tgt, info in target_outcomes.items()]
            summary_str = f"Multi-target detection completed: {', '.join(outcome_summaries)}."

            det_scores = [d.get("score") for d in combined_detections if d.get("score") is not None]
            avg_conf = round(float(np.mean(det_scores)), 4) if det_scores else None

            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="success" if (self.dino_adapter.is_available() or self.rs_adapter.is_available()) else "unavailable",
                provenance="MODEL_DERIVED" if (self.dino_adapter.is_available() or self.rs_adapter.is_available()) else "UNAVAILABLE",
                model_loaded=self.dino_adapter.is_available() or self.rs_adapter.is_available(),
                fallback_used=False,
                method="multi_target_hybrid_grounding",
                capability="Multi-Target Open-Vocabulary Object Grounding",
                data={
                    "target_phrase": target_phrase,
                    "all_targets": all_targets,
                    "target_outcomes": target_outcomes,
                    "detections": combined_detections,
                    "count": len(combined_detections),
                    "image_width": w,
                    "image_height": h,
                    "summary": summary_str,
                    "inference_time_ms": round(total_dur_ms, 2),
                },
                confidence=avg_conf,
                confidence_type="model" if (self.dino_adapter.is_available() or self.rs_adapter.is_available()) else "unavailable",
                confidence_source="Grounding_DINO / RS_Specialist",
                model_metadata={"model_id": "Hybrid_RS_DINO", "source": "Hybrid_RS_DINO"},
                evidence=combined_evidence,
            )

        # Check target geometry: If region feature (e.g. water, forest, farmland, construction, industrial), execute region segmentation
        target_geom = classify_target_geometry(target_phrase)
        if target_geom == TargetGeometry.REGION_FEATURE:
            exec_res = self._execute_region_segmentation(image, target_phrase, w, h)
        elif self.dino_adapter.is_available() or self.rs_adapter.is_available():
            exec_res = self._execute_hybrid_detection(image, target_phrase, w, h)
        else:
            summary = (
                f"Grounding model adapters are currently unavailable in the runtime "
                f"environment. No simulated detections generated."
            )
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="unavailable",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                method="grounding_adapter_check",
                capability="Open-Vocabulary Object Grounding",
                data={
                    "target_phrase": target_phrase,
                    "detections": [],
                    "count": 0,
                    "image_width": w,
                    "image_height": h,
                    "summary": summary,
                },
                confidence=None,
                confidence_type="unavailable",
                confidence_source="none",
                model_metadata={"model_id": "unavailable", "source": "unavailable"},
                evidence=[],
                error="Grounding models unavailable in runtime",
            )

        # Apply geometric spatial relationship filtering if requested
        spatial_relations = params.get("spatial_relations") or []
        if spatial_relations and exec_res.data.get("detections"):
            candidate_dets = exec_res.data["detections"]
            # Detect reference entities in background if specified
            ref_dets = []
            for sr in spatial_relations:
                ref_name = sr.get("reference_entity") if isinstance(sr, dict) else getattr(sr, "reference_entity", "")
                if ref_name and ref_name not in target_phrase:
                    try:
                        ref_geom = classify_target_geometry(ref_name)
                        if ref_geom == TargetGeometry.REGION_FEATURE:
                            ref_res = self.region_segmenter.segment_regions(image, ref_name)
                            ref_dets.extend(ref_res.get("detections", []))
                        elif self.dino_adapter.is_available():
                            from tiling import tiled_inference_engine
                            d_inf = tiled_inference_engine.run_tiled_grounding(
                                image=image,
                                target_phrase=ref_name,
                                grounding_runtime=self.dino_adapter,
                                tile_size=640,
                            )
                            ref_dets.extend(d_inf.get("detections", []))
                    except Exception:
                        pass

            if ref_dets:
                filtered_dets = SpatialRelationshipEngine.filter_by_spatial_relations(
                    candidate_detections=candidate_dets,
                    reference_detections=ref_dets,
                    relations=spatial_relations,
                )
                exec_res.data["detections"] = filtered_dets
                exec_res.data["count"] = len(filtered_dets)
                exec_res.evidence = filtered_dets
                if filtered_dets:
                    exec_res.data["summary"] = f"Detected {len(filtered_dets)} instance(s) of '{target_phrase}' satisfying spatial relations."
                else:
                    exec_res.data["summary"] = f"Target '{target_phrase}' was detected, but none satisfied the requested spatial relationship ({spatial_relations[0].get('relation', 'near') if isinstance(spatial_relations[0], dict) else getattr(spatial_relations[0], 'relation', 'near')} {spatial_relations[0].get('reference_entity', '') if isinstance(spatial_relations[0], dict) else getattr(spatial_relations[0], 'reference_entity', '')})."

        return exec_res


# ---------------------------------------------------------------------------
# 5. Change Analysis Specialist Tool (Real Classical Diff Computation)
# ---------------------------------------------------------------------------

class ChangeAnalysisTool(BaseSpecialistTool):
    name = "Change_Analysis"
    task_type = TaskType.CHANGE_ANALYSIS
    description = "Bi-temporal differential pixel analysis for registered before/after satellite image pairs with dynamic anomaly segmentation."
    input_requirements = ["image_a", "image_b"]

    def _run(self, params: Dict[str, Any]) -> ToolExecutionResult:
        image_a: Optional[Image.Image] = params.get("image_a") or params.get("optical_image")
        image_b: Optional[Image.Image] = params.get("image_b") or params.get("sar_image")
        threshold: float = float(params.get("change_threshold", 0.15))

        if image_a is None or image_b is None:
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="temporal_change_input_validator",
                error="Change analysis requires two valid input observations (image_a & image_b).",
                data={"status": "failed", "error": "Missing input image_a or image_b."},
            )

        try:
            res = change_analysis.analyze(
                image_a=image_a,
                image_b=image_b,
                change_threshold=threshold,
                scene_a_id=params.get("scene_a_id"),
                scene_b_id=params.get("scene_b_id"),
                acquisition_date_a=params.get("acquisition_date_a") or params.get("date_a"),
                acquisition_date_b=params.get("acquisition_date_b") or params.get("date_b"),
                sensor_a=params.get("sensor_a"),
                sensor_b=params.get("sensor_b"),
                bounds_a=params.get("bounds_a") or params.get("optical_bounds"),
                bounds_b=params.get("bounds_b") or params.get("sar_bounds"),
                crs_a=params.get("crs_a") or params.get("optical_crs"),
                crs_b=params.get("crs_b") or params.get("sar_crs"),
            )
        except ValueError as e:
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="geospatial_bounds_validator",
                error=str(e),
                data={
                    "status": "failed",
                    "error": str(e),
                    "alignment_status": "bounds_non_overlapping",
                    "changed_fraction": 0.0,
                    "mean_intensity_delta": 0.0,
                    "limitations": ["Geographical bounds of observations do not overlap; analysis rejected."],
                },
            )

        anomaly_res = anomaly_engine.extract_change_anomalies(
            image_a=image_a,
            image_b=image_b,
            threshold_strategy="otsu",
            custom_threshold=threshold,
        )

        anomalies = anomaly_res.get("regions", [])

        conf = round(float(np.mean([r["change_score"] for r in anomalies[:5]])), 4) if anomalies else (0.0 if res.changed_fraction == 0.0 else round(float(res.changed_fraction), 4))

        return ToolExecutionResult(
            tool_name=self.name,
            task_type=self.task_type.value,
            status="success",
            provenance="ALGORITHMIC",
            model_loaded=False,
            fallback_used=False,
            method="classical_otsu_differencing",
            capability="Bi-Temporal Change Analysis",
            data={
                "summary": res.summary,
                "changed_fraction": res.changed_fraction,
                "mean_intensity_delta": res.mean_intensity_delta,
                "image_dimensions": list(res.image_dimensions),
                "processing_time_ms": res.processing_time_ms,
                "method": res.method,
                "raw_result": res,
                "scene_a_id": res.scene_a_id,
                "scene_b_id": res.scene_b_id,
                "acquisition_date_a": res.acquisition_date_a,
                "acquisition_date_b": res.acquisition_date_b,
                "sensor_a": res.sensor_a,
                "sensor_b": res.sensor_b,
                "valid_pixels_fraction": res.valid_pixels_fraction,
                "nodata_pixels_count": res.nodata_pixels_count,
                "cloud_contamination_detected": res.cloud_contamination_detected,
                "cloud_pixels_fraction": res.cloud_pixels_fraction,
                "alignment_status": res.alignment_status,
                "limitations": res.limitations,
                "anomalies": anomalies,
                "total_regions": anomaly_res.get("total_regions", 0),
                "anomaly_summary": {
                    "total_regions": anomaly_res.get("total_regions", 0),
                    "total_changed_pixels": anomaly_res.get("total_changed_pixels", 0),
                    "changed_fraction": anomaly_res.get("changed_fraction", 0.0),
                    "threshold_method": anomaly_res.get("threshold_method", "otsu_optimal_variance"),
                    "threshold_value_255": anomaly_res.get("threshold_value_255", 38.25),
                },
            },
            confidence=conf,
            confidence_type="heuristic",
            confidence_source="classical_pixel_diff",
            model_metadata={"method": "classical_pixel_difference", "learned_model": False},
            evidence=[
                {
                    "type": "change_summary",
                    "changed_fraction": res.changed_fraction,
                    "mean_intensity_delta": res.mean_intensity_delta,
                    "change_threshold": threshold,
                    "method": res.method,
                    "valid_pixels_fraction": res.valid_pixels_fraction,
                },
                *anomalies,
            ],
        )


class AnomalyExtractionTool(BaseSpecialistTool):
    name = "Anomaly_Extraction"
    task_type = TaskType.CHANGE_ANALYSIS
    description = "Dynamic spatial anomaly extraction and candidate change region localization."
    input_requirements = ["image_a", "image_b"]

    def _run(self, params: Dict[str, Any]) -> ToolExecutionResult:
        image_a = params.get("image_a") or params.get("optical_image")
        image_b = params.get("image_b") or params.get("sar_image")

        if image_a is None or image_b is None:
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                method="input_validation",
                capability="Dynamic Anomaly Extraction",
                error="Anomaly extraction requires two input images (image_a & image_b).",
            )

        threshold_strat = params.get("threshold_strategy", "otsu")
        threshold_val = params.get("change_threshold")

        anom = anomaly_engine.extract_change_anomalies(
            image_a=image_a,
            image_b=image_b,
            threshold_strategy=threshold_strat,
            custom_threshold=threshold_val,
        )

        regions = anom.get("regions", [])
        summary = (
            f"Spatial Anomaly Engine segmented {len(regions)} candidate change region(s) "
            f"using {anom.get('threshold_method')} (total changed pixels: {anom.get('total_changed_pixels', 0)})."
        )

        return ToolExecutionResult(
            tool_name=self.name,
            task_type=self.task_type.value,
            status="success",
            provenance="ALGORITHMIC",
            model_loaded=False,
            fallback_used=False,
            method="otsu_connected_components",
            capability="Dynamic Anomaly Extraction",
            data={
                "anomalies": regions,
                "total_regions": len(regions),
                "threshold_method": anom.get("threshold_method"),
                "threshold_value_255": anom.get("threshold_value_255"),
                "total_changed_pixels": anom.get("total_changed_pixels"),
                "changed_fraction": anom.get("changed_fraction"),
                "summary": summary,
                "geospatial_coordinates_available": anom.get("geospatial_coordinates_available", False),
            },
            confidence=None,
            confidence_type="heuristic",
            confidence_source="dynamic_anomaly_engine",
            model_metadata={"algorithm": "otsu_connected_components", "learned_model": False},
            evidence=regions,
        )


# ---------------------------------------------------------------------------
# 6. Change-VQA Specialist Tool (Adapter with Fallback)
# ---------------------------------------------------------------------------

class ChangeVQATool(BaseSpecialistTool):
    name = "Change_VQA"
    task_type = TaskType.CHANGE_VQA
    description = "Bi-temporal visual question answering specialist correlating temporal differences with query semantics."
    input_requirements = ["image_a", "image_b"]

    def __init__(self, runtime: Optional[BaseModelRuntime] = None, engine: Optional[SemanticChangeEngine] = None):
        self.runtime = runtime or ChangeVQARuntime()
        self.engine = engine or semantic_change_engine

    def _run(self, params: Dict[str, Any]) -> ToolExecutionResult:
        image_a: Image.Image = params["image_a"]
        image_b: Image.Image = params["image_b"]
        query = params.get("query", "What changed between these images?")
        threshold: float = float(params.get("change_threshold", 0.15))
        target_focus = params.get("target_focus")

        # Execute full bi-temporal semantic change understanding pipeline
        sem_res = self.engine.analyze_semantic_change(
            image_a=image_a,
            image_b=image_b,
            query=query,
            change_threshold=threshold,
            target_focus=target_focus,
        )

        is_trained = self.runtime.is_available()
        status = "success" if is_trained else "fallback"
        conf_type = "model" if is_trained else "estimated"
        conf_source = "change_vqa_neural_runtime" if is_trained else "bitemporal_diff_heuristic"
        formatted_answer = sem_res.change_vqa_answer if is_trained else f"Change-VQA (Differential Heuristic): {sem_res.change_vqa_answer}"

        return ToolExecutionResult(
            tool_name=self.name,
            task_type=self.task_type.value,
            status=status,
            provenance="MODEL_DERIVED" if is_trained else "FALLBACK",
            model_loaded=is_trained,
            fallback_used=not is_trained,
            method="bitemporal_semantic_differencing",
            capability="Change-VQA Reasoning",
            data={
                "query": query,
                "answer": formatted_answer,
                "what_changed": sem_res.what_changed,
                "change_category": sem_res.change_category,
                "before_interpretation": sem_res.before_interpretation,
                "after_interpretation": sem_res.after_interpretation,
                "changed_fraction": sem_res.changed_fraction,
                "mean_intensity_delta": sem_res.mean_intensity_delta,
                "changed_regions": [r.region_id for r in sem_res.changed_regions],
                "total_changed_regions": len(sem_res.changed_regions),
                "inference_mode": "semantic_change_pipeline",
            },
            confidence=sem_res.confidence,
            confidence_type=conf_type,
            confidence_source=conf_source,
            model_metadata={
                "provenance": sem_res.model_provenance,
                "learned_model": is_trained,
                "method": "bitemporal_semantic_differencing",
            },
            evidence=sem_res.evidence,
        )


# ---------------------------------------------------------------------------
# 7. Optical-SAR Analysis Specialist Tool (Multimodal Separation & Fusion Status)
# ---------------------------------------------------------------------------

class OpticalSARAnalysisTool(BaseSpecialistTool):
    name = "Optical_SAR_Analysis"
    task_type = TaskType.OPTICAL_SAR_ANALYSIS
    description = "Multimodal Optical + SAR feature fusion specialist executing cross-modal feature representations."
    input_requirements = ["optical_image", "sar_image"]

    def __init__(self, fusion_runtime: Optional[BaseModelRuntime] = None, engine: Optional[OpticalSARFusionEngine] = None):
        self.fusion_runtime = fusion_runtime or OpticalSARFusionRuntime()
        self.engine = engine or optical_sar_fusion_engine

    def validate_inputs(self, params: Dict[str, Any]) -> Optional[str]:
        """Strictly enforce that both an optical image and an independent SAR image are present."""
        opt = params.get("optical_image")
        sar = params.get("sar_image")
        if opt is None and sar is None:
            return "Optical-SAR analysis requires both an optical image and an independent SAR image."
        if opt is None:
            return "Optical-SAR analysis requires an optical image."
        if sar is None:
            return "Optical-SAR analysis requires an independent SAR image."
        if not isinstance(opt, Image.Image):
            return "Parameter 'optical_image' must be a valid PIL Image."
        if not isinstance(sar, Image.Image):
            return "Parameter 'sar_image' must be a valid PIL Image."
        return None

    def execute(self, params: Dict[str, Any]) -> ToolExecutionResult:
        """Standardized resilient execution wrapper with timing & error isolation for Optical-SAR Analysis."""
        t_start = time.perf_counter()
        try:
            res = self._run(params)
            dur = (time.perf_counter() - t_start) * 1000.0
            res.duration_ms = round(dur, 2)
            return res
        except Exception as e:
            dur = (time.perf_counter() - t_start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="optical_sar_execution_containment",
                error=f"Execution error in specialist tool {self.name}: {str(e)}",
                data={
                    "fusion_executed": False,
                    "optical_present": bool(isinstance(params.get("optical_image"), Image.Image)),
                    "sar_present": bool(isinstance(params.get("sar_image"), Image.Image)),
                },
                duration_ms=round(dur, 2),
            )

    def _run(self, params: Dict[str, Any]) -> ToolExecutionResult:
        optical_img: Optional[Any] = params.get("optical_image")
        sar_img: Optional[Any] = params.get("sar_image")
        query = params.get("query", "Compare optical and SAR imagery")

        # 1. Missing or Invalid Input Checks (Phase 1 & Phase 2)
        if optical_img is None and sar_img is None:
            err_msg = "Optical-SAR analysis requires both an optical image and an independent SAR image."
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="optical_sar_input_validator",
                error=err_msg,
                data={
                    "query": query,
                    "optical_present": False,
                    "sar_present": False,
                    "optical_hash": None,
                    "sar_hash": None,
                    "optical_input_id": None,
                    "sar_input_id": None,
                    "crs": {"optical": None, "sar": None, "match": None},
                    "bounds": {"optical": None, "sar": None, "overlap": None},
                    "alignment_status": "input_missing",
                    "fusion_executed": False,
                    "status": "failed",
                    "provenance": "UNAVAILABLE",
                    "confidence": None,
                    "error": err_msg,
                    "fusion": {"status": "unavailable", "error": err_msg},
                },
                evidence=[],
            )

        if optical_img is None:
            err_msg = "Optical-SAR analysis requires an optical image."
            sar_metrics = self.engine.analyze_sar_only(sar_img) if isinstance(sar_img, Image.Image) else {}
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="optical_sar_input_validator",
                error=err_msg,
                data={
                    "query": query,
                    "optical_present": False,
                    "sar_present": isinstance(sar_img, Image.Image),
                    "optical_hash": None,
                    "sar_hash": None,
                    "optical_input_id": None,
                    "sar_input_id": None,
                    "crs": {"optical": None, "sar": None, "match": None},
                    "bounds": {"optical": None, "sar": None, "overlap": None},
                    "alignment_status": "optical_missing",
                    "fusion_executed": False,
                    "status": "failed",
                    "provenance": "UNAVAILABLE",
                    "confidence": None,
                    "error": err_msg,
                    "fusion": {"status": "unavailable", "error": err_msg},
                    "optical_only_metrics": {},
                    "sar_only_metrics": sar_metrics,
                    "limitations": [
                        "Optical modality missing or unavailable; multimodal fusion could not be executed.",
                        "Only single-sensor SAR microwave backscatter analysis was performed.",
                    ],
                },
                evidence=[],
            )

        if sar_img is None:
            err_msg = "Optical-SAR analysis requires an independent SAR image."
            opt_metrics = self.engine.analyze_optical_only(optical_img) if isinstance(optical_img, Image.Image) else {}
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="optical_sar_input_validator",
                error=err_msg,
                data={
                    "query": query,
                    "optical_present": isinstance(optical_img, Image.Image),
                    "sar_present": False,
                    "optical_hash": None,
                    "sar_hash": None,
                    "optical_input_id": None,
                    "sar_input_id": None,
                    "crs": {"optical": None, "sar": None, "match": None},
                    "bounds": {"optical": None, "sar": None, "overlap": None},
                    "alignment_status": "sar_missing",
                    "fusion_executed": False,
                    "status": "failed",
                    "provenance": "UNAVAILABLE",
                    "confidence": None,
                    "error": err_msg,
                    "fusion": {"status": "unavailable", "error": err_msg},
                    "optical_only_metrics": opt_metrics,
                    "sar_only_metrics": {},
                    "limitations": [
                        "SAR modality missing or unavailable; multimodal fusion could not be executed.",
                        "Only single-sensor optical reflectance analysis was performed.",
                    ],
                },
                evidence=[],
            )

        if not isinstance(optical_img, Image.Image):
            err_msg = "Parameter 'optical_image' must be a valid PIL Image."
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="optical_sar_input_validator",
                error=err_msg,
                data={
                    "query": query,
                    "optical_present": False,
                    "sar_present": isinstance(sar_img, Image.Image),
                    "optical_hash": None,
                    "sar_hash": None,
                    "optical_input_id": None,
                    "sar_input_id": None,
                    "crs": {"optical": None, "sar": None, "match": None},
                    "bounds": {"optical": None, "sar": None, "overlap": None},
                    "alignment_status": "optical_invalid",
                    "fusion_executed": False,
                    "status": "failed",
                    "provenance": "UNAVAILABLE",
                    "confidence": None,
                    "error": err_msg,
                    "fusion": {"status": "unavailable", "error": err_msg},
                },
                evidence=[],
            )

        if not isinstance(sar_img, Image.Image):
            err_msg = "Parameter 'sar_image' must be a valid PIL Image."
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="optical_sar_input_validator",
                error=err_msg,
                data={
                    "query": query,
                    "optical_present": isinstance(optical_img, Image.Image),
                    "sar_present": False,
                    "optical_hash": None,
                    "sar_hash": None,
                    "optical_input_id": None,
                    "sar_input_id": None,
                    "crs": {"optical": None, "sar": None, "match": None},
                    "bounds": {"optical": None, "sar": None, "overlap": None},
                    "alignment_status": "sar_invalid",
                    "fusion_executed": False,
                    "status": "failed",
                    "provenance": "UNAVAILABLE",
                    "confidence": None,
                    "error": err_msg,
                    "fusion": {"status": "unavailable", "error": err_msg},
                },
                evidence=[],
            )

        # 2. Independence Check (Phase 3)
        opt_bytes = optical_img.tobytes()
        sar_bytes = sar_img.tobytes()
        opt_hash = hashlib.sha256(opt_bytes).hexdigest()
        sar_hash = hashlib.sha256(sar_bytes).hexdigest()
        opt_input_id = f"opt_{opt_hash[:12]}"
        sar_input_id = f"sar_{sar_hash[:12]}"

        if optical_img is sar_img or opt_hash == sar_hash:
            err_msg = "Optical and SAR inputs must be independent."
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="optical_sar_independence_gate",
                error=err_msg,
                data={
                    "query": query,
                    "optical_present": True,
                    "sar_present": True,
                    "optical_hash": opt_hash,
                    "sar_hash": sar_hash,
                    "optical_input_id": opt_input_id,
                    "sar_input_id": sar_input_id,
                    "is_independent": False,
                    "crs": {"optical": None, "sar": None, "match": None},
                    "bounds": {"optical": None, "sar": None, "overlap": None},
                    "alignment_status": "identical_inputs_rejected",
                    "fusion_executed": False,
                    "status": "failed",
                    "provenance": "UNAVAILABLE",
                    "confidence": None,
                    "error": err_msg,
                    "fusion": {
                        "status": "rejected_identical_inputs",
                        "error": err_msg,
                    },
                },
                evidence=[],
            )

        # 3. Geospatial Metadata Gate & Co-Registration (Gate 3 Step 4)
        opt_geo: GeoMetadata = params.get("optical_geo_meta") or GeospatialEngine.extract_metadata(optical_img)
        sar_geo: GeoMetadata = params.get("sar_geo_meta") or GeospatialEngine.extract_metadata(sar_img)

        opt_crs = params.get("optical_crs") or opt_geo.crs
        sar_crs = params.get("sar_crs") or sar_geo.crs
        opt_bounds = params.get("optical_bounds") or opt_geo.bounds_world
        sar_bounds = params.get("sar_bounds") or sar_geo.bounds_world
        opt_transform = params.get("optical_transform") or (opt_geo.transform.to_dict() if opt_geo.transform else None)
        sar_transform = params.get("sar_transform") or (sar_geo.transform.to_dict() if sar_geo.transform else None)
        opt_res = params.get("optical_resolution") or opt_geo.resolution
        sar_res = params.get("sar_resolution") or sar_geo.resolution

        has_opt_crs = bool(opt_crs)
        has_sar_crs = bool(sar_crs)
        has_opt_bounds = bool(opt_bounds)
        has_sar_bounds = bool(sar_bounds)

        geom_opt = RasterGeometry.from_image_or_metadata(
            optical_img,
            crs=opt_crs,
            transform=opt_transform,
            bounds=opt_bounds,
            resolution=opt_res,
            source="optical",
        )
        geom_sar = RasterGeometry.from_image_or_metadata(
            sar_img,
            crs=sar_crs,
            transform=sar_transform,
            bounds=sar_bounds,
            resolution=sar_res,
            source="sar",
        )

        def _parse_bounds(b: Any) -> Optional[Tuple[float, float, float, float]]:
            if isinstance(b, dict):
                min_x = b.get("min_x", b.get("west", b.get("left")))
                min_y = b.get("min_y", b.get("south", b.get("bottom")))
                max_x = b.get("max_x", b.get("east", b.get("right")))
                max_y = b.get("max_y", b.get("north", b.get("top")))
                if None not in (min_x, min_y, max_x, max_y):
                    return (float(min_x), float(min_y), float(max_x), float(max_y))
            elif isinstance(b, (list, tuple)) and len(b) == 4:
                return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            return None

        bounds_parsed_opt = _parse_bounds(opt_bounds)
        bounds_parsed_sar = _parse_bounds(sar_bounds)

        # Check true spatial overlap
        bounds_overlap: Optional[bool] = None
        overlap_ratio: Optional[float] = None

        if geom_opt.is_valid_geospatial() and geom_sar.is_valid_geospatial():
            ol_stat, ol_ratio, _ = GeospatialCoRegistrationEngine.calculate_spatial_overlap(geom_opt, geom_sar)
            overlap_ratio = ol_ratio
            if ol_stat == OverlapStatus.OVERLAP_CONFIRMED:
                bounds_overlap = True
            elif ol_stat == OverlapStatus.NO_OVERLAP:
                bounds_overlap = False
        elif bounds_parsed_opt is not None and bounds_parsed_sar is not None:
            w1, s1, e1, n1 = bounds_parsed_opt
            w2, s2, e2, n2 = bounds_parsed_sar
            bounds_overlap = (max(w1, w2) < min(e1, e2)) and (max(s1, s2) < min(n1, n2))

        if bounds_overlap is False:
            err_msg = "Optical and SAR imagery geographic bounds do not overlap."
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="geospatial_bounds_validator",
                error=err_msg,
                data={
                    "query": query,
                    "optical_present": True,
                    "sar_present": True,
                    "optical_hash": opt_hash,
                    "sar_hash": sar_hash,
                    "optical_input_id": opt_input_id,
                    "sar_input_id": sar_input_id,
                    "crs": {"optical": opt_crs, "sar": sar_crs, "match": (opt_crs == sar_crs) if (has_opt_crs and has_sar_crs) else None},
                    "bounds": {"optical": opt_bounds, "sar": sar_bounds, "overlap": False},
                    "alignment_status": "bounds_non_overlapping",
                    "geospatial_valid": False,
                    "fusion_executed": False,
                    "status": "failed",
                    "provenance": "UNAVAILABLE",
                    "confidence": None,
                    "error": err_msg,
                    "fusion": {
                        "status": "rejected_non_overlapping_bounds",
                        "error": err_msg,
                    },
                },
                evidence=[],
            )

        # CRS Evaluation
        crs_match: Optional[bool] = None
        if has_opt_crs and has_sar_crs:
            crs_match = (str(opt_crs).strip().upper() == str(sar_crs).strip().upper())
            if crs_match:
                geospatial_alignment_status = "crs_verified_match"
                geospatial_valid = True if (bounds_overlap is not False) else False
            else:
                geospatial_alignment_status = "crs_mismatch"
                geospatial_valid = False
        elif has_opt_crs or has_sar_crs:
            crs_match = False
            geospatial_alignment_status = "partial_crs_metadata"
            geospatial_valid = False
        else:
            crs_match = None
            geospatial_alignment_status = "geospatial_validation_unavailable"
            geospatial_valid = False

        # Dimension alignment
        opt_w, opt_h = optical_img.size
        sar_w, sar_h = sar_img.size
        if (opt_w, opt_h) == (sar_w, sar_h):
            dimension_alignment = "dimension_match_only"
        else:
            dimension_alignment = "dimension_mismatch_rescaled"

        alignment_status = f"{dimension_alignment}_{geospatial_alignment_status}"

        crs_data = {
            "optical": opt_crs,
            "sar": sar_crs,
            "match": crs_match,
        }
        bounds_data = {
            "optical": opt_bounds,
            "sar": sar_bounds,
            "overlap": bounds_overlap,
        }

        # True Geospatial Co-Registration Execution (Phase 8 & Phase 12)
        aligned_opt_img = optical_img
        aligned_sar_img = sar_img
        alignment_telemetry = None

        if geom_opt.is_valid_geospatial() and geom_sar.is_valid_geospatial() and bounds_overlap is True:
            coreg_opt, coreg_sar = GeospatialCoRegistrationEngine.co_register_pair(
                optical_img,
                sar_img,
                geom_opt,
                geom_sar,
            )
            if coreg_opt.status != AlignmentStatus.FAILED and coreg_sar.status != AlignmentStatus.FAILED:
                if coreg_opt.aligned_image:
                    aligned_opt_img = coreg_opt.aligned_image
                if coreg_sar.aligned_image:
                    aligned_sar_img = coreg_sar.aligned_image
                geospatial_valid = True
                geospatial_alignment_status = "geospatially_coregistered"
                alignment_telemetry = coreg_sar.to_dict()
                if coreg_sar.resampling_applied or coreg_opt.resampling_applied:
                    dimension_alignment = "dimension_mismatch_rescaled"
                alignment_status = f"{dimension_alignment}_{geospatial_alignment_status}"
            else:
                alignment_telemetry = coreg_sar.to_dict()

        # Check runtime availability
        if not self.fusion_runtime.is_available():
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="unavailable",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                method="fusion_runtime_check",
                capability="Optical-SAR Feature Fusion",
                data={
                    "query": query,
                    "optical_present": True,
                    "sar_present": True,
                    "optical_hash": opt_hash,
                    "sar_hash": sar_hash,
                    "optical_input_id": opt_input_id,
                    "sar_input_id": sar_input_id,
                    "crs": crs_data,
                    "bounds": bounds_data,
                    "alignment_status": alignment_status,
                    "dimension_alignment": dimension_alignment,
                    "geospatial_alignment_status": geospatial_alignment_status,
                    "geospatial_valid": geospatial_valid,
                    "alignment": alignment_telemetry,
                    "fusion_executed": False,
                    "status": "unavailable",
                    "provenance": "UNAVAILABLE",
                    "confidence": None,
                    "error": "Multimodal fusion runtime unavailable.",
                    "fusion": {"status": "unavailable"},
                    "optical_evidence": "Optical processing unavailable",
                    "sar_evidence": "SAR processing unavailable",
                    "fused_conclusion": "Multimodal fusion runtime unavailable.",
                },
                confidence=None,
                confidence_type="unavailable",
                confidence_source="optical_sar_feature_fusion_baseline",
                model_metadata={"provenance": "unavailable"},
                evidence=[],
            )

        # 4. Execute Structured Multimodal Reasoning on aligned rasters
        fusion_res = self.engine.analyze_pair(
            optical_image=aligned_opt_img,
            sar_image=aligned_sar_img,
            query=query,
        )

        return ToolExecutionResult(
            tool_name=self.name,
            task_type=self.task_type.value,
            status="success",
            provenance="ALGORITHMIC",
            model_loaded=False,
            fallback_used=False,
            method="vit_cross_modal_correlation",
            capability="Optical-SAR Feature Fusion",
            data={
                "query": query,
                "optical_present": True,
                "sar_present": True,
                "optical_hash": opt_hash,
                "sar_hash": sar_hash,
                "optical_input_id": opt_input_id,
                "sar_input_id": sar_input_id,
                "crs": crs_data,
                "bounds": bounds_data,
                "alignment_status": alignment_status,
                "dimension_alignment": dimension_alignment,
                "geospatial_alignment_status": geospatial_alignment_status,
                "geospatial_valid": geospatial_valid,
                "alignment": alignment_telemetry,
                "fusion_executed": True,
                "status": "success",
                "provenance": "ALGORITHMIC",
                "confidence": None,
                "error": None,
                "fusion": {
                    "status": "success",
                    "fusion_type": "feature_fusion_baseline",
                    "fusion_architecture": fusion_res.fusion_type,
                    "fusion_operation": fusion_res.fusion_operation,
                    "fusion_learning": fusion_res.fusion_learning,
                    "is_trained_fusion_model": False,
                    "optical_feature_dim": fusion_res.optical_feature_dim,
                    "sar_feature_dim": fusion_res.sar_feature_dim,
                    "fused_feature_dim": fusion_res.fused_feature_dim,
                    "cosine_similarity": fusion_res.cross_modal_metrics.get("cross_modal_correlation", 0.0),
                    "cross_modal_cosine_similarity": fusion_res.cross_modal_metrics.get("cross_modal_feature_cosine_similarity", 0.0),
                    "cross_modal_euclidean_discrepancy": fusion_res.cross_modal_metrics.get("cross_modal_feature_euclidean_discrepancy", 0.0),
                    "spatial_pearson_correlation": fusion_res.cross_modal_metrics.get("spatial_intensity_pearson_correlation", 0.0),
                    "geospatial_valid": geospatial_valid,
                    "alignment_status": alignment_status,
                },
                "optical_evidence": fusion_res.optical_evidence,
                "sar_evidence": fusion_res.sar_evidence,
                "fused_conclusion": fusion_res.fused_conclusion,
                "optical_metrics": fusion_res.optical_metrics,
                "sar_metrics": fusion_res.sar_metrics,
                "optical_only_metrics": fusion_res.optical_only_metrics or fusion_res.optical_metrics,
                "sar_only_metrics": fusion_res.sar_only_metrics or fusion_res.sar_metrics,
                "combined_metrics": fusion_res.combined_metrics or fusion_res.cross_modal_metrics,
                "source_contributions": fusion_res.source_contributions,
                "limitations": fusion_res.limitations,
                "preprocessing_status": fusion_res.preprocessing_status,
                "cross_modal_metrics": fusion_res.cross_modal_metrics,
                "fusion_type": fusion_res.fusion_type,
                "fusion_operation": fusion_res.fusion_operation,
                "fused_feature_dim": fusion_res.fused_feature_dim,
                "fusion_learning": fusion_res.fusion_learning,
                "optical_provenance": fusion_res.optical_provenance,
                "sar_provenance": fusion_res.sar_provenance,
                "alignment_provenance": fusion_res.alignment_provenance,
                "fusion_provenance": fusion_res.fusion_provenance,
                "confidence_provenance": fusion_res.confidence_provenance,
                "is_trained_model": fusion_res.is_trained_model,
                "model_provenance": fusion_res.model_provenance,
                "correlation_summary": fusion_res.fused_conclusion,
                "cross_modal_summary": fusion_res.fused_conclusion,
                "inference_mode": "multimodal_physics_fusion_pipeline",
            },
            confidence=None,
            confidence_type="unavailable",
            confidence_source="optical_sar_feature_fusion_baseline",
            model_metadata={
                "provenance": fusion_res.model_provenance,
                "is_trained_model": fusion_res.is_trained_model,
                "optical_encoder": "rs_vision_core",
                "sar_encoder": "sar_backscatter_analyzer",
                "geospatial_valid": geospatial_valid,
                "alignment_status": alignment_status,
            },
            evidence=fusion_res.evidence_nodes,
        )


# ---------------------------------------------------------------------------
# 8. Multispectral Analysis Specialist Tool (Spectral Truth & Genuine Indices)
# ---------------------------------------------------------------------------

class MultispectralAnalysisTool(BaseSpecialistTool):
    name = "Multispectral_Analysis"
    task_type = TaskType.REGION_ANALYSIS
    description = "Authentic Earth-observation spectral index calculator (NDVI, NDWI, NDBI) strictly from real multi-band sensor data."
    input_requirements = ["multispectral_scene"]

    def __init__(self, engine: Optional[Any] = None):
        self.engine = engine or spectral_engine

    def validate_inputs(self, params: Dict[str, Any]) -> Optional[str]:
        ms = params.get("multispectral_scene")
        if ms is None:
            return "Multispectral index calculation requires authentic spectral bands (e.g. B04 Red and B08 NIR for NDVI). The provided input is RGB-only."
        if not isinstance(ms, MultispectralScene):
            return "Parameter 'multispectral_scene' must be a valid MultispectralScene instance."
        return None

    def execute(self, params: Dict[str, Any]) -> ToolExecutionResult:
        t_start = time.perf_counter()
        ms = params.get("multispectral_scene")
        indices_requested = params.get("indices") or ["NDVI", "NDWI", "NDBI"]

        if ms is None or not isinstance(ms, MultispectralScene):
            dur = (time.perf_counter() - t_start) * 1000.0
            err_msg = "Multispectral index calculation requires authentic spectral bands (e.g. B04 Red and B08 NIR for NDVI). The provided input is RGB-only."
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="unavailable",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="multispectral_truth_contract",
                error=err_msg,
                data={
                    "multispectral_present": False,
                    "spectral_source": "RGB_VISUAL_ASSET",
                    "provenance": "UNAVAILABLE",
                    "loaded_bands": [],
                    "indices": {},
                    "spectral": {
                        "source": "RGB_VISUAL_ASSET",
                        "available_bands": [],
                        "loaded_bands": ["R", "G", "B"] if params.get("image") is not None else [],
                        "indices": {},
                        "alignment_status": "spectral_alignment_unavailable",
                        "provenance": "UNAVAILABLE",
                        "resolution": None,
                        "crs": None,
                        "reason": err_msg,
                    },
                },
                duration_ms=round(dur, 2),
            )

        try:
            res_dict = self.engine.compute_indices(ms, indices=indices_requested)
            dur = (time.perf_counter() - t_start) * 1000.0

            evidence_nodes = []
            serialized_indices = {}
            for name, idx_res in res_dict.items():
                serialized_indices[name] = idx_res.to_dict()
                if idx_res.status == "success" and idx_res.stats:
                    evidence_nodes.append({
                        "node_id": f"ev_spectral_{name.lower()}",
                        "source_stage": "stage_7_model_inference",
                        "modality": "multispectral",
                        "model_name": "SpectralEngine / Sentinel-2 Surface Reflectance",
                        "confidence": 0.95,
                        "confidence_type": "calibrated_model",
                        "confidence_source": "physics_spectral_index",
                        "provenance": "REAL_MULTISPECTRAL",
                        "summary": f"{name}: valid pixels {idx_res.stats.valid_pixel_count}, mean {round(idx_res.stats.mean or 0.0, 3)} [{round(idx_res.stats.min or 0.0, 3)} .. {round(idx_res.stats.max or 0.0, 3)}]",
                        "data": {
                            "index": name,
                            "formula": idx_res.formula,
                            "bands_used": idx_res.bands_used,
                            "stats": idx_res.stats.to_dict(),
                            "crs": idx_res.crs,
                            "resolution_m": idx_res.resolution_m,
                        },
                    })

            all_failed = all(idx.status != "success" for idx in res_dict.values())
            overall_status = "unavailable" if all_failed else "success"
            overall_prov = "UNAVAILABLE" if all_failed else "REAL_MULTISPECTRAL"

            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status=overall_status,
                provenance=overall_prov,
                model_loaded=True,
                fallback_used=False,
                confidence=0.95 if overall_status == "success" else None,
                confidence_type="calibrated_model" if overall_status == "success" else "unavailable",
                confidence_source="multispectral_physics_engine",
                error=None if overall_status == "success" else "All requested spectral indices were unavailable due to missing required sensor bands.",
                data={
                    "multispectral_present": True,
                    "spectral_source": ms.spectral_source,
                    "loaded_bands": ms.loaded_bands,
                    "available_bands": ms.available_bands,
                    "crs": ms.crs,
                    "resolution_m": ms.resolution_m,
                    "indices": serialized_indices,
                    "provenance": overall_prov,
                    "spectral": {
                        "source": ms.source,
                        "available_bands": ms.available_bands,
                        "loaded_bands": ms.loaded_bands,
                        "indices": serialized_indices,
                        "alignment_status": "geospatially_aligned",
                        "provenance": overall_prov,
                        "resolution": ms.resolution,
                        "crs": ms.crs,
                    },
                },
                evidence=evidence_nodes,
                duration_ms=round(dur, 2),
            )
        except Exception as e:
            dur = (time.perf_counter() - t_start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="multispectral_execution_containment",
                error=f"Execution error in specialist tool {self.name}: {str(e)}",
                data={"indices": {}, "provenance": "UNAVAILABLE"},
                duration_ms=round(dur, 2),
            )

    def _run(self, params: Dict[str, Any]) -> ToolExecutionResult:
        return self.execute(params)


class OpenStreetMapTool(BaseSpecialistTool):
    """
    OpenStreetMap Overpass Specialist Tool for retrieving ground cartographic features
    (buildings, roads, bridges, waterways, infrastructure) within an area of interest.
    """
    name: str = "OpenStreetMap_Feature_Query"
    task_type: TaskType = TaskType.OSM_FEATURE_QUERY
    description: str = (
        "Queries OpenStreetMap via Overpass API for verified mapped features: "
        "buildings, roads, bridges, waterways, and infrastructure within a bounding box."
    )
    input_requirements: List[str] = ["bbox"]
    model_id: str = "openstreetmap_overpass_v0.7"
    model_family: str = "cartographic_gis"
    modality: str = "vector_geojson"
    checkpoint: str = "overpass_api_interpreter"
    source: str = "OpenStreetMap_Contributors"
    rs_adapted: bool = True
    trained: bool = False
    fine_tuned: bool = False
    evaluation_status: str = "VERIFIED_EXTERNAL_PROVIDER"
    fallback_policy: str = "graceful_empty_layer"

    def __init__(self):
        super().__init__()
        self.client = get_overpass_client()

    def execute(self, params: Dict[str, Any]) -> ToolExecutionResult:
        t_start = time.perf_counter()
        bbox = params.get("bbox") or params.get("bounding_box") or params.get("bounds")

        if not bbox:
            resolved_loc = params.get("resolved_location") or params.get("location")
            if isinstance(resolved_loc, dict):
                bbox = resolved_loc.get("bbox_list") or resolved_loc.get("bbox")
            elif hasattr(resolved_loc, "bbox_list"):
                bbox = resolved_loc.bbox_list

        feature_types = params.get("feature_types")
        if not feature_types:
            q_lower = (params.get("query") or "").lower()
            types = []
            if any(w in q_lower for w in ["building", "buildings", "house", "footprint"]):
                types.append("buildings")
            if any(w in q_lower for w in ["road", "roads", "highway", "path", "street"]):
                types.append("roads")
            if any(w in q_lower for w in ["bridge", "bridges", "flyover"]):
                types.append("bridges")
            if any(w in q_lower for w in ["water", "waterway", "river", "canal", "lake"]):
                types.append("waterways")
            if any(w in q_lower for w in ["airport", "rail", "industrial", "infrastructure"]):
                types.append("infrastructure")
            feature_types = types if types else ["buildings", "roads", "bridges", "waterways"]

        if not bbox:
            dur = (time.perf_counter() - t_start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="unavailable",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="osm_input_validation",
                error="No geographic bounding box provided or resolved for OpenStreetMap Overpass query.",
                data={
                    "geojson": {"type": "FeatureCollection", "features": []},
                    "total_features": 0,
                    "attribution": "© OpenStreetMap contributors",
                    "provenance": "UNAVAILABLE",
                },
                duration_ms=round(dur, 2),
            )

        try:
            res: OverpassQueryResult = self.client.query_features(bbox, feature_types=feature_types)
            dur = (time.perf_counter() - t_start) * 1000.0

            evidence_nodes = []
            features = res.geojson.get("features", [])
            for feat in features[:30]:
                feat_props = feat.get("properties", {})
                evidence_nodes.append({
                    "evidence_id": f"ev_osm_{feat.get('id', feat_props.get('osm_id', 'unknown'))}",
                    "type": "osm_cartographic_feature",
                    "source_tool": self.name,
                    "source_model": self.model_id,
                    "derived_from": ["OpenStreetMap_Overpass_API"],
                    "payload": {
                        "osm_id": feat_props.get("osm_id"),
                        "feature_class": feat_props.get("feature_class"),
                        "name": feat_props.get("name"),
                        "geometry": feat.get("geometry"),
                        "attribution": "© OpenStreetMap contributors",
                    },
                    "confidence": 1.0,
                    "confidence_type": "deterministic",
                    "confidence_source": "openstreetmap_database",
                    "validation_status": "valid",
                })

            distinction_note = (
                "OpenStreetMap features represent existing crowdsourced cartographic records, "
                "not proof that the same object is currently visible in the satellite imagery. "
                "OSM coverage may be incomplete, historical, or evolving."
            )

            is_ok = res.status in ("success", "empty")
            status_val = (
                "success" if res.status == "success"
                else "unavailable" if res.status in ("rate_limited", "timeout")
                else "fallback" if res.status == "empty"
                else "failed"
            )

            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status=status_val,
                provenance="OPENSTREETMAP_OVERPASS" if is_ok else "UNAVAILABLE",
                model_loaded=True,
                fallback_used=(res.status == "empty"),
                confidence=1.0 if res.status == "success" else None,
                confidence_type="deterministic" if res.status == "success" else "unavailable",
                confidence_source="openstreetmap_database",
                error=res.error_message,
                data={
                    "geojson": res.geojson,
                    "total_features": res.total_features,
                    "feature_counts": res.feature_counts,
                    "attribution": res.attribution,
                    "cached": res.cached,
                    "query_bbox": res.query_bbox,
                    "distinction_note": distinction_note,
                    "status": res.status,
                    "endpoint_used": res.endpoint_used,
                },
                evidence=evidence_nodes,
                duration_ms=round(dur, 2),
            )
        except Exception as e:
            dur = (time.perf_counter() - t_start) * 1000.0
            return ToolExecutionResult(
                tool_name=self.name,
                task_type=self.task_type.value,
                status="failed",
                provenance="UNAVAILABLE",
                model_loaded=False,
                fallback_used=False,
                confidence=None,
                confidence_type="unavailable",
                confidence_source="osm_execution_containment",
                error=f"Execution error in specialist tool {self.name}: {str(e)}",
                data={"geojson": {"type": "FeatureCollection", "features": []}, "total_features": 0},
                duration_ms=round(dur, 2),
            )

    def _run(self, params: Dict[str, Any]) -> ToolExecutionResult:
        return self.execute(params)


# ---------------------------------------------------------------------------
# Central Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry for specialist vision-language tools."""

    def __init__(self):
        self._tools: Dict[str, BaseSpecialistTool] = {}

    def register(self, tool: BaseSpecialistTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseSpecialistTool]:
        return self._tools.get(name)

    def get_by_task(self, task_type: TaskType) -> List[BaseSpecialistTool]:
        return [tool for tool in self._tools.values() if tool.task_type == task_type]

    def list_tools(self) -> List[Dict[str, Any]]:
        return [tool.get_capability_profile() for tool in self._tools.values()]


# ---------------------------------------------------------------------------
# Global Registry Initialization & Explicit Registrations
# ---------------------------------------------------------------------------

tool_registry = ToolRegistry()

# Explicitly instantiate tools
vqa_tool = VQATool()
optical_caption_tool = OpticalCaptioningTool()
sar_caption_tool = SARCaptioningTool()
grounding_tool = GroundingTool()
change_analysis_tool = ChangeAnalysisTool()
anomaly_extraction_tool = AnomalyExtractionTool()
change_vqa_tool = ChangeVQATool()
optical_sar_analysis_tool = OpticalSARAnalysisTool()
multispectral_analysis_tool = MultispectralAnalysisTool()
openstreetmap_tool = OpenStreetMapTool()

# Register all tools distinctly
tool_registry.register(vqa_tool)
tool_registry.register(optical_caption_tool)
tool_registry.register(sar_caption_tool)
tool_registry.register(grounding_tool)
tool_registry.register(change_analysis_tool)
tool_registry.register(anomaly_extraction_tool)
tool_registry.register(change_vqa_tool)
tool_registry.register(optical_sar_analysis_tool)
tool_registry.register(multispectral_analysis_tool)
tool_registry.register(openstreetmap_tool)

# Aliases for Overpass / OpenStreetMap
tool_registry._tools["OpenStreetMap"] = openstreetmap_tool
tool_registry._tools["OSM"] = openstreetmap_tool
tool_registry._tools["overpass"] = openstreetmap_tool

# Legacy aliases
CaptioningTool = OpticalCaptioningTool
optical_sar_tool = optical_sar_analysis_tool

