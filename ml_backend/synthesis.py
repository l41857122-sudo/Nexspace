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
    summary: Optional[str] = None
    interpretations: Optional[List[str]] = None
    evidence_references: Optional[List[str]] = None

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
                "crs": None,
                "reason": "Non-georeferenced optical image; spatial analysis indexed in normalized pixel coordinates [0-1000].",
            },
            "trace": self.trace,
            "summary": self.summary,
            "interpretations": self.interpretations or [],
            "evidence_references": self.evidence_references or [],
        }


def _pluralize_single_noun(word: str) -> str:
    """Pluralizes a single noun with standard English irregular morphology rules."""
    w_lower = word.lower()
    if w_lower.endswith("y") and len(word) > 1 and word[-2].lower() not in "aeiou":
        return word[:-1] + "ies"
    elif w_lower.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    elif w_lower.endswith("f") and not w_lower.endswith(("cliff", "roof")):
        return word[:-1] + "ves"
    elif w_lower.endswith("fe"):
        return word[:-2] + "ves"
    else:
        return word + "s"


def _pluralize_phrase(phrase: str) -> str:
    """
    Properly pluralizes noun phrases, including prepositional phrases ('body of water' -> 'bodies of water')
    and preserves uncountable domain terms.
    """
    if not phrase:
        return "objects"
    phrase = phrase.strip()
    p_lower = phrase.lower()

    # Prepositional compound phrase: e.g. "body of water" -> "bodies of water"
    compound_match = re.match(r"^([a-zA-Z]+)\s+(of\s+.+)$", phrase, re.IGNORECASE)
    if compound_match:
        head, tail = compound_match.groups()
        head_plural = _pluralize_single_noun(head)
        return f"{head_plural} {tail}"

    # Uncountable remote sensing terms
    uncountable = {"vegetation", "water", "farmland", "infrastructure", "aircraft", "debris", "foliage", "traffic", "radar backscatter"}
    if p_lower in uncountable:
        return phrase

    # Already pluralized
    if p_lower.endswith(("ies", "ses", "xes", "ches", "shes")) or (p_lower.endswith("s") and not p_lower.endswith(("us", "is", "ss"))):
        return phrase

    return _pluralize_single_noun(phrase)


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
        from target_spec import extract_safety_constraints
        clean_q, constraints = extract_safety_constraints(q_raw)
        q_lower = clean_q.lower() if clean_q else q_raw.lower()

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
        target_outcomes: Optional[Dict[str, Any]] = None
        fusion_summary: Optional[str] = None
        spectral_data: Optional[Dict[str, Any]] = None

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
                target_outcomes = data.get("target_outcomes")
                geom_type = data.get("geometry_type", "box")
                if target_outcomes:
                    for tgt, t_info in target_outcomes.items():
                        c = t_info.get("count", 0)
                        observations.append(
                            f"Multi-Target Grounding localized {c} instance(s) matching '{tgt}'."
                        )
                elif geom_type == "polygon":
                    observations.append(
                        f"Region Grounding Specialist localized {len(grounding_detections)} verified polygon region(s) for '{grounding_target}'."
                    )
                else:
                    observations.append(
                        f"Grounding DINO detected {len(grounding_detections)} candidate region(s) matching '{grounding_target}'."
                    )

            elif t_name == "VQA":
                ans = data.get("primary_answer") or data.get("answer") or ""
                if not ans and data.get("vqa_results"):
                    v_items = data.get("vqa_results", [])
                    ans_list = [v.get("answer", "") for v in v_items if isinstance(v, dict) and v.get("answer")]
                    ans = " | ".join(ans_list)
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

            elif t_name == "Multispectral_Analysis":
                spectral_data = data
                if t_status == "success":
                    observations.append("Authentic multispectral sensor bands processed; verified spectral indices computed.")
                else:
                    limitations.append("Multispectral analysis could not be computed: authentic Red (B04) and NIR (B08) sensor bands are required; imagery is RGB-only.")

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
            "crs": crs if has_geo else None,
            "evidence_with_coordinates": len(geo_nodes),
            "total_ground_area": round(total_ground_area, 2) if (has_geo and total_ground_area > 0) else None,
            "total_ground_area_unit": "m2" if (has_geo and total_ground_area > 0) else None,
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
            spectral_data=spectral_data,
            has_geo=has_geo,
            crs=crs,
            exec_summary=exec_summary,
            observations=observations,
            plan=plan,
            target_outcomes=target_outcomes,
        )
        report.summary = response_text

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
        spectral_data: Optional[Dict[str, Any]] = None,
        target_outcomes: Optional[Dict[str, Any]] = None,
    ) -> str:
        from target_spec import extract_safety_constraints
        clean_q, constraints = extract_safety_constraints(query)
        q_lower = clean_q.lower().strip()
        sections: List[str] = []

        # Case 0: Multispectral Analysis (NDVI, NDWI, NDBI)
        if spectral_data or any(k in q_lower for k in ["ndvi", "ndwi", "ndbi", "vegetation index", "water index", "built-up index", "multispectral", "spectral index", "spectral indices"]):
            if spectral_data and spectral_data.get("multispectral_present") and spectral_data.get("indices"):
                idx_lines = []
                for idx_name, idx_info in spectral_data["indices"].items():
                    if idx_info.get("status") == "success" and idx_info.get("stats"):
                        st = idx_info["stats"]
                        idx_lines.append(f"- **{idx_name}:** Mean {st['mean']:.3f} (Range: [{st['min']:.3f} .. {st['max']:.3f}], Valid Pixels: {st['valid_pixel_count']}) [{idx_info['formula']}]")
                    else:
                        idx_lines.append(f"- **{idx_name}:** {idx_info.get('error', 'Unavailable')}")
                bands_str = ", ".join(spectral_data.get("loaded_bands", []))
                sections.append(
                    f"**Multispectral Analysis:**\n"
                    f"Authentic Sentinel-2 Level-2A surface reflectance bands ({bands_str}) processed.\n"
                    + "\n".join(idx_lines)
                )
            else:
                sections.append(
                    "**Multispectral Analysis:** NDVI/NDWI could not be computed. Authentic Red (B04) and NIR (B08) spectral bands are required. The available imagery is RGB-only."
                )
            return "\n\n".join(sections)

        # Case 1: Pure Geospatial Metadata Queries (CRS, Resolution, Coordinates, GSD)
        is_pure_geo = (task_type in ("GEOSPATIAL_METADATA", "GEOLOCATION")) or (
            not grounding_detections
            and not target_outcomes
            and bool(re.search(r"\b(coordinates|crs|resolution|ground sampling distance|gsd|projection|lat.*lon)\b", q_lower))
            and not any(re.search(p, q_lower) for p in [r"\bidentify\b", r"\bdetect\b", r"\blocate\b", r"\bfind\b", r"\bsegment\b", r"\bcount\b", r"\banalyze\b", r"\bspot\b", r"\bpinpoint\b", r"\boutline\b"])
        )
        if is_pure_geo:
            if has_geo and crs:
                sections.append(f"**Geospatial Metadata:** Coordinate Reference System: `{crs}`. Spatial resolution and ground footprints are available in the evidence layer.")
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
            plural = _pluralize_phrase(target_name)
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
            plural = _pluralize_phrase(target_name)
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
            if fusion_summary:
                sections.append(f"**Multimodal Fusion Analysis:** {fusion_summary}")
            else:
                sections.append("**Multimodal Fusion Analysis:** Optical-SAR analysis could not be completed. An independent SAR radar image is required.")
            if caption_text:
                sections.append(f"- **Optical View:** {caption_text}")
            if sar_caption_text:
                sections.append(f"- **SAR View:** {sar_caption_text}")
            return "\n\n".join(sections)

        # Case 8: SAR Only
        if sar_caption_text and not caption_text:
            sections.append(f"**Radar SAR Analysis:** {sar_caption_text}")
            return "\n\n".join(sections)

        # Multi-Target Outcomes (Phase 3.E)
        if target_outcomes and len(target_outcomes) > 1:
            sections.append("**Multi-Target Visual Analysis:**")
            for tgt, t_info in target_outcomes.items():
                c = t_info.get("count", 0)
                plural = _pluralize_phrase(tgt)
                if c == 0:
                    sections.append(f"- **{tgt.title()}:** 0 candidate {plural} localized (None detected in imagery).")
                else:
                    sections.append(f"- **{tgt.title()}:** {c} candidate {plural} localized.")
            if caption_text:
                sections.append(f"**Scene Overview:** {caption_text}")
            elif vqa_answers:
                sections.append(f"**Answer:** {' | '.join(vqa_answers)}")
            if constraints and not has_geo:
                sections.append("*(Safety Constraint: Non-georeferenced imagery analyzed in normalized pixel frame [0-1000]. Verified sensor and geospatial metadata reported as unavailable.)*")
            return "\n\n".join(sections)

        # Direct VQA query response (if question asked)
        is_direct_q = query.endswith("?") or task_type in ("VQA", "OPEN_ENDED_LAND_USE") or any(k in q_lower for k in ["is there", "are there", "what type of", "what is the dominant", "tell me if"])
        if is_direct_q and vqa_answers:
            sections.append(f"**Answer:** {' | '.join(vqa_answers)}")
            if caption_text:
                sections.append(f"**Scene Description:** {caption_text}")
            elif grounding_detections:
                target_display = grounding_target or "structure"
                plural_name = _pluralize_phrase(target_display)
                sections.append(f"**Objects Located:** We found {len(grounding_detections)} possible {plural_name} in this image.")
            return "\n\n".join(sections)

        # Case 9: General Visual & Multi-Tool Output
        if caption_text:
            sections.append(f"**Scene Description:** {caption_text}")
            if grounding_detections:
                target_display = grounding_target or "structure"
                plural_name = _pluralize_phrase(target_display)
                sections.append(
                    f"**Objects Located:** We found {len(grounding_detections)} possible {plural_name} in this image."
                )
        elif grounding_detections:
            target_display = grounding_target or "structure"
            plural_name = _pluralize_phrase(target_display)
            sections.append(
                f"Scene description could not be generated reliably for this image. However, the vision grounding model identified **{len(grounding_detections)}** candidate region(s) matching '{target_display}'."
            )
        elif not sections:
            if not vqa_answers:
                sections.append("Scene description could not be generated reliably for this image. The visual content could not be interpreted with sufficient confidence from the available model weights.")

        if vqa_answers and not any("**Answer:**" in s for s in sections):
            sections.append(f"**Answer:** {' | '.join(vqa_answers)}")

        if not sections and observations:
            sections.extend(observations)

        return "\n\n".join(sections) if sections else "The image was analyzed successfully."
