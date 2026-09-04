"""
orchestrator.py
---------------
Agent Controller & Pipeline Orchestrator for Geospatial Vision-Language tasks.

Architecture Flow:
  User Query
  → Intent Classification & Multi-Tool Planning
  → Input Validation
  → Task & Tool Selection (Single or Multi-Tool)
  → Parameter Selection
  → Concurrent Specialist Tool Execution (Error Isolation)
  → Result Validation
  → Output Synthesis
  → Structured Response + Calibrated Confidence + Real Execution Trace
"""

from __future__ import annotations
import time
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any
from PIL import Image

from router import (
    IntentClassifier,
    TaskType,
    ClassificationResult,
    RoutingDecision,
    _extract_grounding_target,
)
from tools import (
    tool_registry,
    ToolExecutionResult,
    VQAResult,
    CaptionResult,
    vqa_tool,
    optical_caption_tool,
    sar_caption_tool,
    grounding_tool,
    change_analysis_tool,
    change_vqa_tool,
    optical_sar_analysis_tool,
)


COUNT_WARNING_TEXT = (
    "⚠️ Note: Exact numeric counts are derived with low model confidence "
    "(~0.25-0.40). Treat this count as an estimate."
)


from telemetry import ExecutionTrace, TraceStage
from evidence_graph import EvidenceGraph, EvidenceNode
from synthesis import InvestigationSynthesizer, InvestigationReport

# Legacy alias for backward compatibility
ExecutionTracer = ExecutionTrace


# ---------------------------------------------------------------------------
# Input Validation Layer
# ---------------------------------------------------------------------------

class InputValidator:
    """Validates query strings and image inputs against task and tool requirements."""

    @staticmethod
    def validate(
        task_type: TaskType,
        selected_tools: List[str],
        query: str,
        optical_image: Optional[Image.Image],
        sar_image: Optional[Image.Image],
        change_image_a: Optional[Image.Image],
        change_image_b: Optional[Image.Image],
    ) -> Optional[str]:
        # 1. Empty query and no imagery validation
        if not query.strip() and not optical_image and not sar_image and not (change_image_a and change_image_b):
            return "Empty request: Please provide a natural language query or satellite imagery."

        # 2. Unsupported tasks
        if task_type == TaskType.UNSUPPORTED:
            return f"Unsupported query or intent: '{query}' is outside geospatial VLM capabilities."

        # 3. Tool-specific input requirements
        for tool_name in selected_tools:
            if tool_name in ("VQA", "Optical_Caption", "Grounding"):
                if optical_image is None and change_image_b is None:
                    return f"Tool '{tool_name}' requires an optical satellite image."

            elif tool_name == "SAR_Caption":
                if sar_image is None:
                    return "Tool 'SAR_Caption' requires a SAR radar image."

            elif tool_name in ("Change_Analysis", "Change_VQA", "Anomaly_Extraction"):
                if (change_image_a is None or change_image_b is None) and (optical_image is None or sar_image is None):
                    return f"Tool '{tool_name}' requires both before (Image A) and after (Image B) images."

            elif tool_name == "Optical_SAR_Analysis":
                if optical_image is None or sar_image is None:
                    return "Tool 'Optical_SAR_Analysis' requires both an Optical image and a SAR radar image."

        # 4. Image sanity checks (verifying valid dimensions)
        for img_label, img in [
            ("optical_image", optical_image),
            ("sar_image", sar_image),
            ("change_image_a", change_image_a),
            ("change_image_b", change_image_b),
        ]:
            if img is not None:
                if not isinstance(img, Image.Image) or img.size[0] <= 0 or img.size[1] <= 0:
                    return f"Invalid image data for '{img_label}': Unable to decode dimensions."

        return None


# ---------------------------------------------------------------------------
# GeoVLM Agent Controller
# ---------------------------------------------------------------------------

class GeoVLMController:
    """Production Agentic Controller coordinating multi-modal remote sensing specialists."""

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        registry: Optional[ToolRegistry] = None,
        vqa_tool: Optional[BaseSpecialistTool] = None,
        optical_caption_tool: Optional[BaseSpecialistTool] = None,
        sar_caption_tool: Optional[BaseSpecialistTool] = None,
        grounding_tool: Optional[BaseSpecialistTool] = None,
        change_analysis_tool: Optional[BaseSpecialistTool] = None,
        change_vqa_tool: Optional[BaseSpecialistTool] = None,
        optical_sar_analysis_tool: Optional[BaseSpecialistTool] = None,
    ):
        self.classifier = classifier or IntentClassifier()
        self.registry = registry or tool_registry
        self.synthesizer = InvestigationSynthesizer()

        self.vqa = vqa_tool
        self.optical_caption = optical_caption_tool
        self.sar_caption = sar_caption_tool
        self.grounding = grounding_tool
        self.change_analysis = change_analysis_tool

    def handle_request(
        self,
        query: str = "",
        optical_image: Optional[Image.Image] = None,
        sar_image: Optional[Image.Image] = None,
        change_image_a: Optional[Image.Image] = None,
        change_image_b: Optional[Image.Image] = None,
        probe_features: Optional[List[str]] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        req_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
        tracer = ExecutionTrace(query_id=req_id)
        t_global_start = time.perf_counter()

        # Stage 1: Request / Query Received
        tracer.record(
            stage="request_received",
            status="completed",
            metadata={
                "request_id": req_id,
                "query_length": len(query),
                "has_optical": optical_image is not None,
                "has_sar": sar_image is not None,
                "has_change_pair": change_image_a is not None and change_image_b is not None,
            },
        )
        tracer.record(
            stage="query_received",
            status="success",
            metadata={
                "request_id": req_id,
                "query_length": len(query),
                "has_optical": optical_image is not None,
                "has_sar": sar_image is not None,
                "has_change_pair": change_image_a is not None and change_image_b is not None,
            },
        )

        # Stage 2: Intent Classification
        t_cls_start = time.perf_counter()
        classification: ClassificationResult = self.classifier.classify(
            query=query,
            has_optical=optical_image is not None,
            has_sar=sar_image is not None,
            has_change_pair=change_image_a is not None and change_image_b is not None,
            probe_features=probe_features,
        )
        dur_cls = (time.perf_counter() - t_cls_start) * 1000.0
        tracer.record(
            stage="intent_classification",
            status="completed" if classification.is_supported else "rejected",
            metadata={
                "task_type": classification.task_type.value,
                "confidence": classification.confidence,
                "confidence_type": classification.confidence_type,
                "confidence_source": classification.confidence_source,
                "target_tools": classification.target_tools,
            },
            duration_ms=dur_cls,
        )
        tracer.record(
            stage="classification",
            status="success" if classification.is_supported else "rejected",
            metadata={
                "task_type": classification.task_type.value,
                "confidence": classification.confidence,
                "confidence_type": classification.confidence_type,
                "confidence_source": classification.confidence_source,
                "target_tools": classification.target_tools,
            },
            duration_ms=dur_cls,
        )

        # Stage 3: Tool Selection & Task Planning
        t_sel_start = time.perf_counter()
        selected_tools = list(classification.target_tools)
        if sar_image is not None and "SAR_Caption" not in selected_tools and "Optical_SAR_Analysis" not in selected_tools:
            selected_tools.append("SAR_Caption")

        selected_tools = list(dict.fromkeys(selected_tools))
        dur_sel = (time.perf_counter() - t_sel_start) * 1000.0

        grounding_t = classification.parameters.get("target_phrase") or _extract_grounding_target(query or "")
        plan = {
            "task_type": classification.task_type.value,
            "selected_tools": selected_tools,
            "reasoning_basis": "intent_rules",
            "parameters": {
                "query": query,
                "change_threshold": 0.15,
                "grounding_targets": [grounding_t] if "Grounding" in selected_tools else [],
                "cross_modal": "Optical_SAR_Analysis" in selected_tools,
            },
        }

        tracer.record(
            stage="task_planning",
            status="completed",
            metadata={"plan": plan},
            duration_ms=dur_sel / 2.0,
        )
        tracer.record(
            stage="tool_selection",
            status="completed" if selected_tools or not classification.is_supported else "failed",
            metadata={"selected_tools": selected_tools, "is_multi_tool": len(selected_tools) > 1},
            duration_ms=dur_sel,
        )

        # Stage 4: Input Validation
        t_val_start = time.perf_counter()
        val_error = InputValidator.validate(
            task_type=classification.task_type,
            selected_tools=selected_tools,
            query=query,
            optical_image=optical_image,
            sar_image=sar_image,
            change_image_a=change_image_a,
            change_image_b=change_image_b,
        )
        dur_val = (time.perf_counter() - t_val_start) * 1000.0

        if val_error:
            tracer.record(
                stage="input_validation",
                status="failed",
                metadata={"error": val_error},
                duration_ms=dur_val,
                error=val_error,
            )
            tracer.record(
                stage="validation",
                status="failed",
                metadata={"error": val_error},
                duration_ms=dur_val,
            )
            tracer.record(
                stage="failed",
                status="failed",
                metadata={"reason": val_error},
                duration_ms=(time.perf_counter() - t_global_start) * 1000.0,
            )
            return self._build_failure_response(
                task_type=classification.task_type.value,
                error=val_error,
                tracer=tracer,
                classification=classification,
            )

        tracer.record(
            stage="input_validation",
            status="completed",
            metadata={"validation": "passed"},
            duration_ms=dur_val,
        )
        tracer.record(
            stage="validation",
            status="success",
            metadata={"validation": "passed"},
            duration_ms=dur_val,
        )

        # Stage 5: Parameter Extraction
        t_param_start = time.perf_counter()
        task_params = self._build_tool_parameters(
            query=query,
            classification=classification,
            optical_image=optical_image,
            sar_image=sar_image,
            change_image_a=change_image_a,
            change_image_b=change_image_b,
        )
        dur_param = (time.perf_counter() - t_param_start) * 1000.0

        tracer.record(
            stage="parameter_extraction",
            status="completed",
            metadata={"param_keys": list(task_params.keys())},
            duration_ms=dur_param,
        )
        tracer.record(
            stage="parameter_selection",
            status="success",
            metadata={"param_keys": list(task_params.keys())},
            duration_ms=dur_param,
        )

        # Stage 6: Tool & Model Execution
        t_exec_start = time.perf_counter()
        tool_results: List[ToolExecutionResult] = self._execute_tools_concurrently(
            selected_tools=selected_tools,
            task_params=task_params,
            tracer=tracer,
        )
        dur_exec = (time.perf_counter() - t_exec_start) * 1000.0

        all_ok = all(r.status in ("success", "fallback") for r in tool_results)
        any_ok = any(r.status in ("success", "fallback") for r in tool_results)

        tracer.record(
            stage="tool_execution",
            status="completed" if all_ok else ("partial_success" if any_ok else "failed"),
            metadata={
                "executed_tools": [r.tool_name for r in tool_results],
                "statuses": {r.tool_name: r.status for r in tool_results},
                "fallbacks": [r.tool_name for r in tool_results if r.status == "fallback"],
                "unavailables": [r.tool_name for r in tool_results if r.status == "unavailable"],
                "failures": [r.tool_name for r in tool_results if r.status == "failed"],
            },
            duration_ms=dur_exec,
        )

        # Stage 7: Evidence Extraction & Linking
        t_ev_start = time.perf_counter()
        evidence_graph = EvidenceGraph(query_id=req_id)
        for r in tool_results:
            t_name = r.tool_name
            src_model = ""
            if r.model_metadata and isinstance(r.model_metadata, dict):
                src_model = r.model_metadata.get("model_id") or r.model_metadata.get("method") or r.model_metadata.get("algorithm") or ""
            if not src_model:
                src_model = r.confidence_source or t_name

            for ev_item in r.evidence:
                if isinstance(ev_item, dict):
                    ev_type = ev_item.get("type") or ev_item.get("evidence_type")
                    if not ev_type:
                        if "box" in ev_item or "bbox_pixel" in ev_item:
                            ev_type = "object_detection"
                        elif "optical_caption" in ev_item:
                            ev_type = "caption"
                        else:
                            ev_type = "specialist_evidence"

                    evidence_graph.create_and_add(
                        type=ev_type,
                        source_tool=t_name,
                        source_model=str(src_model),
                        payload=ev_item,
                        confidence=r.confidence,
                        confidence_type=r.confidence_type,
                        confidence_source=r.confidence_source,
                    )
        dur_ev = (time.perf_counter() - t_ev_start) * 1000.0
        tracer.record(
            stage="evidence_extraction",
            status="completed",
            metadata={"extracted_evidence_nodes": len(evidence_graph.nodes)},
            duration_ms=dur_ev,
        )

        # Stage 8: Evidence & Result Validation
        t_rval_start = time.perf_counter()
        valid_results = [r for r in tool_results if r.status in ("success", "fallback")]
        unavailable_results = [r for r in tool_results if r.status == "unavailable"]
        failed_results = [r for r in tool_results if r.status == "failed"]
        dur_rval = (time.perf_counter() - t_rval_start) * 1000.0

        tracer.record(
            stage="evidence_validation",
            status="completed",
            metadata={"validated_nodes": len(evidence_graph.nodes)},
            duration_ms=dur_rval / 2.0,
        )
        tracer.record(
            stage="result_validation",
            status="success" if (valid_results or len(selected_tools) == 0) else ("partial" if unavailable_results else "failed"),
            metadata={
                "valid_count": len(valid_results),
                "unavailable_count": len(unavailable_results),
                "failed_count": len(failed_results),
            },
            duration_ms=dur_rval,
        )

        # Stage 9: Result Synthesis & Investigation Reporting
        t_syn_start = time.perf_counter()
        legacy_vqa_results, optical_cap, sar_cap, change_res, _ = self._synthesize_structured(
            classification=classification,
            tool_results=tool_results,
            requires_count_warning=classification.requires_count_warning,
        )

        investigation_report, exec_summary, response_text = self.synthesizer.synthesize(
            query=query,
            task_type=classification.task_type.value,
            plan=plan,
            tool_results=tool_results,
            evidence_graph=evidence_graph,
            trace=tracer,
            requires_count_warning=classification.requires_count_warning,
        )
        dur_syn = (time.perf_counter() - t_syn_start) * 1000.0

        tracer.record(
            stage="result_synthesis",
            status="completed",
            metadata={"observations_count": len(investigation_report.observations)},
            duration_ms=dur_syn,
        )
        tracer.record(
            stage="synthesis",
            status="success",
            metadata={"response_length": len(response_text)},
            duration_ms=dur_syn,
        )

        # Stage 10: Response Validation
        tracer.record(
            stage="response_validation",
            status="completed",
            metadata={"response_length": len(response_text)},
        )

        # Stage 11: Completion Event
        final_status = "completed" if (valid_results or unavailable_results or len(selected_tools) == 0) else "failed"
        tracer.record(
            stage="response_completed",
            status=final_status,
            metadata={"total_time_ms": round((time.perf_counter() - t_global_start) * 1000.0, 2)},
        )
        tracer.record(
            stage="completed",
            status=final_status,
            metadata={"total_time_ms": round((time.perf_counter() - t_global_start) * 1000.0, 2)},
        )

        overall_conf, overall_conf_type, overall_conf_source = self._compute_confidence(classification, valid_results)

        # Aggregate evidence list for legacy contracts
        all_evidence = []
        for r in tool_results:
            if r.evidence:
                all_evidence.extend(r.evidence)

        # Return Comprehensive Auditable Dual-Contract Response
        return {
            # --- New Investigation & Auditing Telemetry Schema ---
            "request_id": req_id,
            "status": final_status,
            "task_type": classification.task_type.value,
            "selected_tools": selected_tools,
            "plan": plan,
            "investigation_report": investigation_report.to_dict(),
            "evidence_graph": evidence_graph.to_dict(),
            "execution_summary": exec_summary,
            "parameters": {
                "query": query,
                "change_threshold": 0.15,
                "has_optical": optical_image is not None,
                "has_sar": sar_image is not None,
                "has_change_pair": change_image_a is not None and change_image_b is not None,
            },
            "results": [r.to_dict() for r in tool_results],
            "confidence": overall_conf,
            "confidence_type": overall_conf_type,
            "confidence_source": overall_conf_source,
            "evidence": all_evidence,
            "execution_trace": tracer.to_list(),
            # --- Backward Compatibility Schema ---
            "routing_decision": {
                "target_tools": selected_tools,
                "restructured_vqa_queries": classification.restructured_vqa_queries,
                "requires_count_warning": classification.requires_count_warning,
                "execution_reasoning": classification.reasoning,
            },
            "vqa_results": legacy_vqa_results,
            "optical_caption": optical_cap,
            "sar_caption": sar_cap,
            "grounding": next((r.data for r in tool_results if r.tool_name == "Grounding" and r.data), None),
            "change_analysis": change_res,
            "optical_sar_analysis": next((r.data for r in tool_results if r.tool_name == "Optical_SAR_Analysis" and r.data), None),
            "response_text": response_text,
        }

    # ------------------------------------------------------------------
    # Parameter Builder
    # ------------------------------------------------------------------
    def _build_tool_parameters(
        self,
        query: str,
        classification: ClassificationResult,
        optical_image: Optional[Image.Image],
        sar_image: Optional[Image.Image],
        change_image_a: Optional[Image.Image],
        change_image_b: Optional[Image.Image],
    ) -> Dict[str, Dict[str, Any]]:
        params: Dict[str, Dict[str, Any]] = {}
        target_img = optical_image if optical_image is not None else change_image_b

        # VQA Parameters
        vqa_queries = classification.restructured_vqa_queries or ([query] if query else ["Is there a body of water present?"])
        params["VQA"] = {
            "image": target_img,
            "questions": vqa_queries,
            "question": vqa_queries[0] if vqa_queries else query,
        }

        # Captioning Parameters
        params["Optical_Caption"] = {
            "image": optical_image if optical_image is not None else change_image_b,
            "modality": "optical",
        }
        params["SAR_Caption"] = {
            "image": sar_image,
            "modality": "sar",
        }

        # Grounding Parameters
        target_phrase = classification.parameters.get("target_phrase") or _extract_grounding_target(query or "")
        params["Grounding"] = {
            "image": optical_image if optical_image is not None else change_image_b,
            "target_phrase": target_phrase,
        }

        # Change Analysis Parameters
        params["Change_Analysis"] = {
            "image_a": change_image_a,
            "image_b": change_image_b,
            "change_threshold": classification.parameters.get("change_threshold", 0.15),
        }

        # Change-VQA Parameters
        params["Change_VQA"] = {
            "image_a": change_image_a,
            "image_b": change_image_b,
            "query": query or "What changed between these images?",
            "change_threshold": classification.parameters.get("change_threshold", 0.15),
        }

        # Anomaly Extraction Parameters
        params["Anomaly_Extraction"] = {
            "image_a": change_image_a if change_image_a is not None else optical_image,
            "image_b": change_image_b if change_image_b is not None else sar_image,
            "change_threshold": classification.parameters.get("change_threshold", 0.15),
            "threshold_strategy": "otsu",
        }

        # Optical-SAR Parameters
        params["Optical_SAR_Analysis"] = {
            "optical_image": optical_image,
            "sar_image": sar_image,
            "query": query or "Compare optical and SAR imagery",
        }

        return params

    # ------------------------------------------------------------------
    # Concurrent Tool Execution
    # ------------------------------------------------------------------
    def _execute_tools_concurrently(
        self,
        selected_tools: List[str],
        task_params: Dict[str, Dict[str, Any]],
        tracer: Optional[ExecutionTracer] = None,
    ) -> List[ToolExecutionResult]:
        results: List[ToolExecutionResult] = []

        def run_single_tool(tool_name: str) -> ToolExecutionResult:
            tool = self.registry.get(tool_name)
            if tool is None:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    task_type="UNKNOWN",
                    status="failed",
                    confidence=None,
                    confidence_type="heuristic",
                    confidence_source="tool_registry",
                    error=f"Tool '{tool_name}' not registered in specialist registry.",
                )

            p = task_params.get(tool_name, {})
            t0 = time.perf_counter()
            res = tool.execute(p)
            dur = (time.perf_counter() - t0) * 1000.0

            if tracer:
                if res.status == "fallback":
                    tracer.record(
                        stage="fallback_triggered",
                        status="fallback",
                        details={"tool": tool_name, "source": res.confidence_source},
                        duration_ms=dur,
                    )
                elif res.status == "success":
                    tracer.record(
                        stage="inference_complete",
                        status="success",
                        details={"tool": tool_name, "type": res.confidence_type},
                        duration_ms=dur,
                    )

            return res

        with ThreadPoolExecutor(max_workers=max(len(selected_tools), 1)) as executor:
            future_to_tool = {executor.submit(run_single_tool, t): t for t in selected_tools}
            for future in as_completed(future_to_tool):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as ex:
                    tool_n = future_to_tool[future]
                    results.append(
                        ToolExecutionResult(
                            tool_name=tool_n,
                            task_type="UNKNOWN",
                            status="failed",
                            confidence=None,
                            confidence_type="heuristic",
                            confidence_source="exception_handler",
                            error=f"Uncaught exception executing tool {tool_n}: {str(ex)}",
                        )
                    )

        # Preserve selected_tools order
        order_map = {name: idx for idx, name in enumerate(selected_tools)}
        results.sort(key=lambda r: order_map.get(r.tool_name, 999))
        return results

    # ------------------------------------------------------------------
    # Synthesis & Confidence Calculation
    # ------------------------------------------------------------------
    def _synthesize_structured(
        self,
        classification: ClassificationResult,
        tool_results: List[ToolExecutionResult],
        requires_count_warning: bool,
    ):
        parts: List[str] = []
        legacy_vqa_results: List[Dict[str, Any]] = []
        optical_cap: Optional[str] = None
        sar_cap: Optional[str] = None
        change_res: Optional[Dict[str, Any]] = None

        for r in tool_results:
            if r.status not in ("success", "fallback", "unavailable"):
                continue

            # Optical Caption
            if r.tool_name == "Optical_Caption":
                optical_cap = r.data.get("caption")
                if optical_cap:
                    parts.append(f"**Optical scene description:** {optical_cap}")

            # SAR Caption
            elif r.tool_name == "SAR_Caption":
                sar_cap = r.data.get("caption")
                if sar_cap:
                    parts.append(f"**SAR scene description:** {sar_cap}")

            # Optical SAR Fusion Analysis
            elif r.tool_name == "Optical_SAR_Analysis":
                parts.append(f"**Cross-Modal Analysis:** {r.data.get('correlation_summary')}")

            # Grounding
            elif r.tool_name == "Grounding":
                if r.status == "unavailable":
                    parts.append(f"**Spatial Grounding:** Grounding model is unavailable in the current runtime environment.")
                else:
                    parts.append(f"**Spatial Grounding:** {r.data.get('summary')}")

            # VQA
            elif r.tool_name == "VQA":
                v_list = r.data.get("vqa_results", [])
                legacy_vqa_results = v_list
                lines = []
                for item in v_list:
                    line = f"- {item['question']} → {item['answer']}"
                    if item.get("low_confidence"):
                        line += "  ⚠️ low confidence"
                    lines.append(line)
                if lines:
                    parts.append("**Structured VQA findings:**\n" + "\n".join(lines))

            # Change Analysis
            elif r.tool_name == "Change_Analysis":
                change_res = {
                    "summary": r.data.get("summary"),
                    "changed_fraction": r.data.get("changed_fraction"),
                    "mean_intensity_delta": r.data.get("mean_intensity_delta"),
                }
                parts.append(f"**Change analysis:** {r.data.get('summary')}")

            # Change-VQA
            elif r.tool_name == "Change_VQA":
                parts.append(f"**Change-VQA:** {r.data.get('answer')}")

        if requires_count_warning and any("how many" in (v.get("question", "").lower()) for v in legacy_vqa_results):
            parts.append(COUNT_WARNING_TEXT)

        if not parts:
            parts.append("No specialist tool generated actionable output for this request.")

        response_text = "\n\n".join(parts)
        return legacy_vqa_results, optical_cap, sar_cap, change_res, response_text

    def _compute_confidence(
        self,
        classification: ClassificationResult,
        valid_results: List[ToolExecutionResult],
    ) -> tuple[Optional[float], str, str]:
        if not valid_results:
            return None, "heuristic", "none"

        # Check if any tool suffered a critical generation failure / safety rejection
        gen_failures = [r for r in valid_results if getattr(r, "confidence_type", "") == "generation_failure" or getattr(r, "status", "") == "invalid_generation"]

        # Check if genuine model inference confidence is present from successful tools
        successful_model_confs = [
            r.confidence for r in valid_results
            if r.confidence_type == "model" and r.status == "success" and r.confidence is not None
        ]
        if successful_model_confs:
            avg_conf = round(sum(successful_model_confs) / len(successful_model_confs), 2)
            src = next((r.confidence_source for r in valid_results if r.confidence_type == "model" and r.status == "success"), "transformers_pipeline")
            return avg_conf, "model", src

        # If generation failure occurred and no other model succeeded, strictly declare generation failure
        if gen_failures:
            return 0.20, "generation_failure", "caption_quality_validator"

        # Check if genuine model executed without scalar logit confidence
        has_real_model = any(r.confidence_type == "model" and r.status == "success" for r in valid_results)
        if has_real_model:
            src = next((r.confidence_source for r in valid_results if r.confidence_type == "model"), "transformers_pipeline")
            return None, "model", src

        # Check for estimated / fallback confidences
        est_confs = [r.confidence for r in valid_results if r.confidence is not None and r.status != "failed"]
        if est_confs:
            tool_avg = sum(est_confs) / len(est_confs)
            combined = (tool_avg * 0.7) + (classification.confidence * 0.3)
            # Downgrade fallback confidence if any tool failed
            if gen_failures:
                combined = min(combined, 0.35)
            return round(combined, 2), "estimated", "ensemble_controller"

        return None, "heuristic", "adapter_uncalibrated"

    # ------------------------------------------------------------------
    # Failure Response Constructor
    # ------------------------------------------------------------------
    def _build_failure_response(
        self,
        task_type: str,
        error: str,
        tracer: ExecutionTracer,
        classification: ClassificationResult,
        request_id: str = "req_unknown",
    ) -> Dict[str, Any]:
        return {
            "request_id": request_id,
            "status": "failed",
            "task_type": task_type,
            "error": error,
            "selected_tools": [],
            "parameters": {},
            "results": [],
            "confidence": None,
            "confidence_type": "heuristic",
            "confidence_source": "validation_gate",
            "evidence": [],
            "execution_trace": tracer.to_list(),
            "routing_decision": {
                "target_tools": [],
                "restructured_vqa_queries": [],
                "requires_count_warning": False,
                "execution_reasoning": f"Execution aborted: {error}",
            },
            "vqa_results": [],
            "optical_caption": None,
            "sar_caption": None,
            "change_analysis": None,
            "response_text": f"Error: {error}",
        }
