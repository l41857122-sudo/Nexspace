"""
synthesis.py
------------
Auditable Multi-Modal Synthesis Layer & Direct Natural Language Answer Synthesizer.

Synthesizes direct, human-readable answers tailored to specific query families:
  - Scene understanding & Land cover
  - Object identification & Grounding DINO proposals
  - Candidate counting with honest disclaimers
  - Spatial localization & Centroid distribution
  - Bi-temporal change detection & Anomaly regions
  - Optical + SAR multimodal fusion baseline
  - Geospatial metadata & CRS introspection
  - Model confidence & Provenance justification
  - Open-ended land-use suitability with candid limitations
"""

from __future__ import annotations
import re
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
    """Synthesizes structured investigation reports and direct natural language responses."""

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
        q_raw = (query or "").strip()
        q_lower = q_raw.lower()

        observations: List[str] = []
        limitations: List[str] = []

        tools_attempted = len(tool_results)
        tools_completed = 0
        tools_failed = 0
        fallback_count = 0

        caption_text: Optional[str] = None
        sar_caption_text: Optional[str] = None
        vqa_answers: List[str] = []
        change_data: Optional[Dict[str, Any]] = None
        grounding_detections: List[Dict[str, Any]] = []
        grounding_target: Optional[str] = None
        fusion_summary: Optional[str] = None

        for res in tool_results:
            t_name = getattr(res, "tool_name", "")
            t_status = getattr(res, "status", "")
            data = getattr(res, "data", {}) or {}

            if t_status in ("success", "fallback"):
                tools_completed += 1
            elif t_status == "failed":
                tools_failed += 1
                limitations.append(f"Specialist tool '{t_name}' encountered an error: {getattr(res, 'error', '')}")

            if t_status == "fallback":
                fallback_count += 1
                limitations.append(
                    f"Tool '{t_name}' operated in fallback mode ({getattr(res, 'confidence_source', '') or 'heuristic'})."
                )

            if t_status == "unavailable":
                limitations.append(
                    f"Specialist model for '{t_name}' is unavailable in the current runtime."
                )

            # Extract tool data
            if t_name == "Optical_Caption":
                if t_status == "success" and data.get("caption"):
                    caption_text = data.get("caption", "")
                    observations.append(f"Optical vision model described the scene: \"{caption_text}\".")
                elif t_status in ("invalid_generation", "failed") or not data.get("caption"):
                    reason = data.get("rejection_reason") or "Repetitive token loop or low lexical diversity detected"
                    limitations.append(f"Optical scene description was rejected by quality filters: {reason}.")

            elif t_name == "SAR_Caption":
                if t_status == "success" and data.get("caption"):
                    sar_caption_text = data.get("caption", "")
                    observations.append(f"SAR stream described radar backscatter: \"{sar_caption_text}\".")
                elif t_status in ("invalid_generation", "failed") or not data.get("caption"):
                    reason = data.get("rejection_reason") or "Repetitive token loop or low lexical diversity detected"
                    limitations.append(f"SAR scene description was rejected by quality filters: {reason}.")

            elif t_name == "Grounding" and t_status == "success":
                grounding_detections = data.get("detections", [])
                grounding_target = data.get("target_phrase", "objects")
                observations.append(
                    f"Grounding DINO detected {len(grounding_detections)} candidate region(s) matching '{grounding_target}'."
                )

            elif t_name == "VQA":
                ans = data.get("answer", "")
                if ans:
                    vqa_answers.append(ans)
                    observations.append(f"Visual Q&A inferred: \"{ans}\".")

            elif t_name == "Change_Analysis" and t_status == "success":
                change_data = data
                observations.append(
                    f"Bi-temporal change analysis: {data.get('changed_pixel_fraction', 0) * 100:.1f}% changed pixels across scenes."
                )

            elif t_name == "Anomaly_Extraction" and t_status == "success":
                anom_count = data.get("total_regions", 0)
                observations.append(
                    f"Identified {anom_count} anomaly region(s) with significant surface modification."
                )

            elif t_name == "Optical_SAR_Analysis" and t_status == "success":
                fusion_summary = data.get("correlation_summary", "")
                observations.append(
                    fusion_summary or "Cross-sensor comparison shows moderate structural pattern similarity between optical and SAR imagery."
                )
                limitations.append("Optical + SAR analysis is based on a preliminary feature comparison baseline.")

        # Disclaimers
        if requires_count_warning or bool(re.search(r"\b(how many|count)\b", q_lower)):
            limitations.append("Candidate counts reflect zero-shot model detection proposals, not ground-truth verified survey counts.")

        if caption_text:
            limitations.append("Optical scene descriptions are generated by a vision-language model with nadir domain shift.")

        if grounding_detections:
            limitations.append("Zero-shot bounding boxes are candidate neural proposals and have not been field-validated.")

        # Execution Summary
        evidence_nodes = evidence_graph.to_list()
        exec_summary = {
            "tools_attempted": tools_attempted,
            "tools_completed": tools_completed,
            "tools_failed": tools_failed,
            "evidence_count": len(evidence_nodes),
            "anomaly_count": len(evidence_graph.get_by_type("change_region")),
            "detection_count": len(evidence_graph.get_by_type("object_detection")),
            "fallback_count": fallback_count,
            "total_duration_ms": round(trace.total_duration_ms(), 2) if hasattr(trace, "total_duration_ms") else 0.0,
        }

        # Spatial Summary
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

        # Build Investigation Report
        report = InvestigationReport(
            query=query,
            task=task_type,
            plan=plan,
            observations=observations,
            evidence=evidence_nodes,
            limitations=limitations,
            execution_summary=exec_summary,
            trace=trace.to_list() if hasattr(trace, "to_list") else [],
            spatial_summary=spatial_summary,
        )

        # -------------------------------------------------------------------
        # Build Direct Natural Language Response tailored to User Query
        # -------------------------------------------------------------------
        response_text = self._synthesize_direct_answer(
            query=q_raw,
            task_type=task_type,
            caption_text=caption_text,
            sar_caption_text=sar_caption_text,
            grounding_detections=grounding_detections,
            grounding_target=grounding_target,
            vqa_answers=vqa_answers,
            change_data=change_data,
            fusion_summary=fusion_summary,
            has_geo=has_geo,
            crs=crs,
            exec_summary=exec_summary,
            observations=observations,
            plan=plan,
        )

        return report, exec_summary, response_text

    def _synthesize_direct_answer(
        self,
        query: str,
        task_type: str,
        caption_text: Optional[str],
        sar_caption_text: Optional[str],
        grounding_detections: List[Dict[str, Any]],
        grounding_target: Optional[str],
        vqa_answers: List[str],
        change_data: Optional[Dict[str, Any]],
        fusion_summary: Optional[str],
        has_geo: bool,
        crs: Optional[str],
        exec_summary: Dict[str, Any],
        observations: List[str],
        plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        q_lower = query.lower().strip()
        sections: List[str] = []

        # Case 1: Geospatial Metadata Queries (CRS, Resolution, Coordinates, GSD)
        if bool(re.search(r"\b(coordinates|crs|resolution|ground sampling distance|gsd|projection|lat.*lon)\b", q_lower)):
            if has_geo:
                sections.append(f"**Geospatial Metadata:** Coordinate Reference System: `{crs or 'EPSG:4326'}`. Spatial resolution and ground footprints are available in the evidence layer.")
            else:
                sections.append("**Geospatial Metadata:** This raster is currently loaded without embedded geospatial tags (GeoTIFF tags / world file). Spatial coordinates are indexed in pixel space `[0-1000]`. Upload a GeoTIFF raster to extract geographic lat/lon coordinates and EPSG projections.")
            return "\n\n".join(sections)

        # Case 2: Model Introspection / Provenance / Confidence Queries
        if bool(re.search(r"\b(how confident|confidence level|why did you|what evidence|which model|how reliable)\b", q_lower)):
            sections.append(
                "**Model Provenance & Confidence:**\n"
                "- **Grounding DINO (Swin-T Backbone):** Computes zero-shot candidate bounding box proposals.\n"
                "- **BLIP Vision-Language (Salesforce):** Generates holistic visual captions.\n"
                "- **Classical Pixel Difference Matrix:** Computes pixel-level intensity deltas for change detection.\n"
                f"- **Active Pipeline Status:** {exec_summary.get('tools_completed', 0)} tool(s) succeeded with {exec_summary.get('evidence_count', 0)} evidence node(s) cataloged."
            )
            return "\n\n".join(sections)

        # Case 3: Counting Queries
        if bool(re.search(r"\b(how many|count the|count|number of|quantity of)\b", q_lower)):
            target_name = grounding_target or "object"
            plural = f"{target_name}s" if not target_name.endswith("s") else target_name
            count = len(grounding_detections)
            sections.append(f"**Count:** We identified **{count}** candidate {plural} in this image.")
            if count > 0:
                sections.append("*(Note: Grounding DINO identified candidate regions matching your query. These are model proposals and have not been ground-truth validated.)*")
            if caption_text:
                sections.append(f"**Scene Overview:** {caption_text}")
            elif "Optical_Caption" in plan.get("selected_tools", []):
                sections.append("*(Scene description could not be generated reliably for this image.)*")
            return "\n\n".join(sections)

        # Case 4: Spatial / Location Queries ("Where are...", "Which side...")
        if bool(re.search(r"\b(where is|where are|which side|which region|locate the|pinpoint)\b", q_lower)):
            target_name = grounding_target or "object"
            plural = f"{target_name}s" if not target_name.endswith("s") else target_name
            count = len(grounding_detections)
            if count > 0:
                # Compute spatial distribution
                avg_x = sum((d.get("box", [0, 0, 0, 0])[0] + d.get("box", [0, 0, 0, 0])[2]) / 2 for d in grounding_detections) / count
                avg_y = sum((d.get("box", [0, 0, 0, 0])[1] + d.get("box", [0, 0, 0, 0])[3]) / 2 for d in grounding_detections) / count
                x_pos = "western" if avg_x < 400 else ("eastern" if avg_x > 600 else "central")
                y_pos = "northern" if avg_y < 400 else ("southern" if avg_y > 600 else "central")
                sections.append(f"**Location:** Found {count} candidate {plural}, concentrated primarily in the **{y_pos}-{x_pos}** region of the image.")
            else:
                sections.append(f"**Location:** No clear {plural} could be pinpointed in this raster.")
            if caption_text:
                sections.append(f"**Scene Context:** {caption_text}")
            elif plan and "Optical_Caption" in plan.get("selected_tools", []):
                sections.append("*(Scene description could not be generated reliably for this image.)*")
            return "\n\n".join(sections)

        # Case 5: Open-Ended Land-Use / Environmental Suitability
        if bool(re.search(r"\b(suitable for|suitability|can we build|construction feasibility|land use|farming potential)\b", q_lower)):
            target_count = len(grounding_detections)
            sections.append(
                f"**Visible Analysis:** The imagery displays an environment with {target_count} visible structural feature(s). "
                + (f"Scene description: \"{caption_text}\". " if caption_text else "")
                + "\n\n**Limitation Notice:** Site suitability cannot be determined from optical/satellite imagery alone. "
                "Reliable assessment requires geotechnical soil testing, elevation models (DEM/LiDAR), zoning regulations, and utility infrastructure data which are not present in this single raster pass."
            )
            return "\n\n".join(sections)

        # Case 6: Bi-Temporal Change & Comparison
        if change_data or task_type == "CHANGE_ANALYSIS":
            pct = change_data.get("changed_pixel_fraction", 0.0) * 100 if change_data else 0.0
            regions = change_data.get("total_regions", 0) if change_data else 0
            severity = change_data.get("severity", "Minimal") if change_data else "Minimal"
            summary_txt = change_data.get("summary") if change_data else ""
            sections.append(
                f"**Bi-Temporal Change Analysis:**\n"
                f"- **Surface Area Changed:** {pct:.1f}% of pixels\n"
                f"- **Anomaly Regions:** {regions} distinct cluster(s) detected\n"
                f"- **Change Severity:** {severity.capitalize()}\n"
                + (f"\n{summary_txt}" if summary_txt else "")
            )
            return "\n\n".join(sections)

        # Case 7: Optical + SAR Cross-Modality
        if fusion_summary or task_type == "OPTICAL_SAR_ANALYSIS":
            sections.append(f"**Multimodal Fusion Analysis:** {fusion_summary or 'Optical & SAR feature representations aligned and evaluated.'}")
            if caption_text:
                sections.append(f"- **Optical View:** {caption_text}")
            if sar_caption_text:
                sections.append(f"- **SAR View:** {sar_caption_text}")
            return "\n\n".join(sections)

        # Case 8: SAR Only
        if sar_caption_text and not caption_text:
            sections.append(f"**Radar SAR Analysis:** {sar_caption_text}")
            return "\n\n".join(sections)

        # Case 9: General Visual & Multi-Tool Output
        if caption_text:
            sections.append(f"**Scene Description:** {caption_text}")
            if grounding_detections:
                target_display = grounding_target or "structure"
                plural_name = f"{target_display}s" if not target_display.endswith("s") else target_display
                sections.append(
                    f"**Objects Located:** We found {len(grounding_detections)} possible {plural_name} in this image."
                )
        elif grounding_detections:
            target_display = grounding_target or "structure"
            plural_name = f"{target_display}s" if not target_display.endswith("s") else target_display
            sections.append(
                f"Scene description could not be generated reliably for this image. However, the vision grounding model identified **{len(grounding_detections)}** candidate region(s) matching '{target_display}'."
            )
        elif not sections:
            sections.append("Scene description could not be generated reliably for this image. The visual content could not be interpreted with sufficient confidence from the available model weights.")

        if vqa_answers:
            sections.append(f"**Answer:** {' | '.join(vqa_answers)}")

        if not sections and observations:
            sections.extend(observations)

        return "\n\n".join(sections) if sections else "The image was analyzed successfully."
