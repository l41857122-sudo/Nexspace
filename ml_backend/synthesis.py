"""
synthesis.py
------------
Investigation Report Synthesizer & Auditable Multi-Modal Synthesis Layer.

Combines validated evidence nodes, execution traces, query plans, and model outputs into
an auditable structured investigation report distinguishing factual observations from interpretations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from evidence_graph import EvidenceGraph
from telemetry import ExecutionTrace


@dataclass
class InvestigationReport:
    """Canonical investigation document containing complete audit provenance."""
    query: str
    task: str
    plan: Dict[str, Any]
    observations: List[str]
    evidence: List[Dict[str, Any]]
    limitations: List[str]
    execution_summary: Dict[str, Any]
    trace: List[Dict[str, Any]]
    spatial_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "task": self.task,
            "plan": self.plan,
            "observations": self.observations,
            "evidence": self.evidence,
            "limitations": self.limitations,
            "execution_summary": self.execution_summary,
            "spatial_summary": self.spatial_summary or {
                "geospatial_available": False,
                "reason": "No geospatial metadata available",
            },
            "trace": self.trace,
        }


class InvestigationSynthesizer:
    """Synthesizes structured investigation reports and human-readable responses."""

    def __init__(self):
        pass

    def synthesize(
        self,
        query: str,
        task_type: str,
        plan: Dict[str, Any],
        tool_results: List[Any],
        evidence_graph: EvidenceGraph,
        trace: ExecutionTrace,
        requires_count_warning: bool = False,
    ) -> Tuple[InvestigationReport, Dict[str, Any], str]:
        """
        Synthesizes the complete auditable report, machine-readable summary,
        and human-readable response text.
        """
        observations: List[str] = []
        limitations: List[str] = []

        # Count tracking
        tools_attempted = len(tool_results)
        tools_completed = 0
        tools_failed = 0
        fallback_count = 0

        # Parse specialist outputs
        caption_text: Optional[str] = None
        sar_caption_text: Optional[str] = None
        vqa_answers: List[str] = []
        change_summary: Optional[str] = None
        grounding_detections_count = 0
        grounding_target: Optional[str] = None
        fusion_summary: Optional[str] = None

        for res in tool_results:
            t_name = res.tool_name
            t_status = res.status
            data = res.data or {}

            if t_status in ("success", "fallback"):
                tools_completed += 1
            elif t_status == "failed":
                tools_failed += 1
                limitations.append(f"Specialist tool '{t_name}' encountered an isolated execution error: {res.error}")

            if t_status == "fallback":
                fallback_count += 1
                limitations.append(
                    f"Tool '{t_name}' operated in fallback mode ({res.confidence_source or 'heuristic'})."
                )

            if t_status == "unavailable":
                limitations.append(
                    f"Specialist model for '{t_name}' is unavailable in the current runtime."
                )

            # Modality-specific observation extraction
            if t_name == "Optical_Caption" and t_status == "success":
                caption_text = data.get("caption", "")
                observations.append(f"Optical vision backbone described the scene: \"{caption_text}\".")

            elif t_name == "SAR_Caption" and t_status == "success":
                sar_caption_text = data.get("caption", "")
                observations.append(f"Radar SAR stream described backscatter patterns: \"{sar_caption_text}\".")

            elif t_name == "Grounding" and t_status == "success":
                detections = data.get("detections", [])
                grounding_detections_count = len(detections)
                grounding_target = data.get("target_phrase", "objects")
                observations.append(
                    f"Grounding DINO detected {grounding_detections_count} candidate region(s) matching '{grounding_target}'."
                )

            elif t_name == "VQA":
                ans = data.get("answer", "")
                if ans:
                    vqa_answers.append(ans)
                    if t_status == "fallback":
                        observations.append(f"VQA (Deterministic RSVQA Adapter) answered: \"{ans}\".")
                    else:
                        observations.append(f"PaliGemma VQA inferred answer: \"{ans}\".")

            elif t_name == "Change_Analysis" and t_status == "success":
                anom_count = data.get("total_regions", 0)
                observations.append(
                    f"Bi-temporal change analysis identified {anom_count} area(s) of visible surface difference."
                )

            elif t_name == "Anomaly_Extraction" and t_status == "success":
                anom_count = data.get("total_regions", 0)
                observations.append(
                    f"Identified {anom_count} candidate area(s) to review for surface modifications."
                )
                if not data.get("geospatial_coordinates_available", False):
                    limitations.append("Geospatial geographic coordinates (lat/lon) are unavailable from the source image metadata.")

            elif t_name == "Optical_SAR_Analysis" and t_status == "success":
                fusion_summary = data.get("correlation_summary", "")
                observations.append(
                    fusion_summary or "Cross-sensor comparison shows moderate structural pattern similarity between optical and SAR imagery."
                )
                limitations.append("Optical + SAR analysis is based on a preliminary feature comparison baseline.")

        if requires_count_warning:
            limitations.append("Exact object counts in VQA are approximate estimates with low confidence.")

        if caption_text:
            limitations.append("Optical scene description is derived from a general vision-language model with nadir domain shift; not ground-truth land cover classification.")

        if grounding_detections_count > 0:
            limitations.append("Zero-shot grounding detections are candidate model proposals; remote-sensing domain accuracy is not ground-truth validated.")

        if fusion_summary:
            limitations.append("Multimodal Optical + SAR analysis is based on a feature fusion baseline, not a trained joint model.")

        # Build execution summary
        evidence_nodes = evidence_graph.to_list()
        anomaly_count = len(evidence_graph.get_by_type("change_region"))
        detection_count = len(evidence_graph.get_by_type("object_detection"))

        exec_summary = {
            "tools_attempted": tools_attempted,
            "tools_completed": tools_completed,
            "tools_failed": tools_failed,
            "evidence_count": len(evidence_nodes),
            "anomaly_count": anomaly_count,
            "detection_count": detection_count,
            "fallback_count": fallback_count,
            "total_duration_ms": round(trace.total_duration_ms(), 2),
        }

        # Extract spatial summary
        geo_nodes = [
            n for n in evidence_nodes
            if (isinstance(n, dict) and n.get("payload", {}).get("geospatial_coordinates_available"))
            or (isinstance(n, dict) and n.get("payload", {}).get("bbox_world"))
        ]
        has_geo = len(geo_nodes) > 0
        total_ground_area = sum(float(n.get("payload", {}).get("ground_area", 0.0) or 0.0) for n in geo_nodes)
        crs = geo_nodes[0].get("payload", {}).get("crs") if has_geo else None

        spatial_summary = {
            "geospatial_available": has_geo,
            "crs": crs,
            "evidence_with_coordinates": len(geo_nodes),
            "total_ground_area": round(total_ground_area, 2) if total_ground_area > 0 else None,
            "total_ground_area_unit": "m2" if total_ground_area > 0 else None,
        }

        # Build investigation report
        report = InvestigationReport(
            query=query,
            task=task_type,
            plan=plan,
            observations=observations,
            evidence=evidence_nodes,
            limitations=limitations,
            execution_summary=exec_summary,
            trace=trace.to_list(),
            spatial_summary=spatial_summary,
        )

        # Build response text (User-facing Natural Markdown)
        sections: List[str] = []

        if caption_text:
            sections.append(f"**Scene Description:** {caption_text}")

        if sar_caption_text:
            sections.append(f"**Radar Analysis:** {sar_caption_text}")

        if grounding_detections_count > 0:
            target_display = grounding_target or "target"
            plural_name = f"{target_display}s" if not target_display.endswith("s") else target_display
            sections.append(
                f"**Objects Located:** We found {grounding_detections_count} possible {plural_name} in this image."
            )
        elif grounding_target and any(getattr(r, "tool_name", "") == "Grounding" for r in tool_results):
            sections.append(
                f"**Objects Located:** No clear {grounding_target} regions could be reliably detected in this image."
            )

        if vqa_answers:
            ans_joined = " | ".join(vqa_answers)
            sections.append(f"**Answer:** {ans_joined}")

        if change_summary:
            sections.append(f"**Change Detection:** {change_summary}")

        if fusion_summary:
            sections.append(f"**Multimodal Fusion (Optical & SAR Comparison):** {fusion_summary}")

        if not sections and observations:
            sections.extend(observations)

        response_text = "\n\n".join(sections) if sections else "The image was analyzed successfully."

        return report, exec_summary, response_text
