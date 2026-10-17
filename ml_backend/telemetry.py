"""
telemetry.py
------------
Real-Time Execution Telemetry, Structured Logging & Lifecycle Tracing
for Geospatial Vision-Language Pipelines.

Features:
- Structured logging with timestamps, severity levels, correlation IDs, and secret redaction
- Millisecond-precision monotonic duration tracking
- Discrete sequential specialist tool execution tracking (pending, running, succeeded, failed, skipped, unavailable)
- Explicit error taxonomy: INPUT_VALIDATION_ERROR, RESOURCE_UNAVAILABLE, MODEL_EXECUTION_ERROR,
  GEOSPATIAL_ALIGNMENT_ERROR, TIMEOUT_ERROR, NETWORK_ERROR, PROVIDER_POLICY_ERROR, SYSTEM_ERROR
- Provenance linking final claims to actual processing steps
- Safe non-intrusive error containment (logging never crashes core pipeline)
- 100% backward compatible with existing test suites and caller contracts
"""

from __future__ import annotations
import os
import sys
import time
import logging
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Union


# Standard Discovery & Investigation Lifecycle Stages
STAGE_QUERY_RECEIVED = "QUERY_RECEIVED"
STAGE_LOCATION_PARSED = "LOCATION_PARSED"
STAGE_LOCATION_EXTRACTED = "LOCATION_EXTRACTED"
STAGE_LOCATION_RESOLVED = "LOCATION_RESOLVED"
STAGE_AOI_RESOLVED = "AOI_RESOLVED"
STAGE_AOI_GENERATED = "AOI_GENERATED"
STAGE_IMAGERY_REQUIREMENTS_RESOLVED = "IMAGERY_REQUIREMENTS_RESOLVED"
STAGE_STAC_SEARCH_STARTED = "STAC_SEARCH_STARTED"
STAGE_PROVIDER_QUERIED = "PROVIDER_QUERIED"
STAGE_STAC_SEARCH_COMPLETED = "STAC_SEARCH_COMPLETED"
STAGE_SCENES_RETURNED = "SCENES_RETURNED"
STAGE_SCENES_RANKED = "SCENES_RANKED"
STAGE_SCENE_SELECTED = "SCENE_SELECTED"
STAGE_RASTER_RETRIEVAL_STARTED = "RASTER_RETRIEVAL_STARTED"
STAGE_IMAGERY_RETRIEVED = "IMAGERY_RETRIEVED"
STAGE_RASTER_RETRIEVAL_COMPLETED = "RASTER_RETRIEVAL_COMPLETED"
STAGE_RGB_PROCESSING = "RGB_PROCESSING"
STAGE_GEOSPATIAL_METADATA_VALIDATED = "GEOSPATIAL_METADATA_VALIDATED"
STAGE_GEOREFERENCING = "GEOREFERENCING"
STAGE_SATELLITE_RESULT_READY = "SATELLITE_RESULT_READY"
STAGE_NEXSPACE_ANALYSIS_STARTED = "NEXSPACE_ANALYSIS_STARTED"
STAGE_AI_INFERENCE = "AI_INFERENCE"
STAGE_GEOGRAPHIC_PROJECTION = "GEOGRAPHIC_PROJECTION"
STAGE_EVIDENCE_GENERATED = "EVIDENCE_GENERATED"
STAGE_MAP_VISUALIZATION_READY = "MAP_VISUALIZATION_READY"

# Phase 2 Photo Investigation Stages
STAGE_IMAGE_RECEIVED = "IMAGE_RECEIVED"
STAGE_FILE_VALIDATED = "FILE_VALIDATED"
STAGE_METADATA_EXTRACTED = "METADATA_EXTRACTED"
STAGE_GEOREFERENCING_CHECKED = "GEOREFERENCING_CHECKED"
STAGE_EXIF_CHECKED = "EXIF_CHECKED"
STAGE_LANDMARK_RETRIEVAL = "LANDMARK_RETRIEVAL"
STAGE_VPR_INFERENCE = "VPR_INFERENCE"
STAGE_GLOBAL_GEOLOCATION = "GLOBAL_GEOLOCATION"
STAGE_CANDIDATE_FUSION = "CANDIDATE_FUSION"
STAGE_CONFIDENCE_CALIBRATION = "CONFIDENCE_CALIBRATION"
STAGE_CROSS_VIEW_VERIFICATION = "CROSS_VIEW_VERIFICATION"
STAGE_FINAL_RESULT = "FINAL_RESULT"

DISCOVERY_STAGES = [
    STAGE_QUERY_RECEIVED,
    STAGE_LOCATION_PARSED,
    STAGE_LOCATION_RESOLVED,
    STAGE_AOI_GENERATED,
    STAGE_STAC_SEARCH_STARTED,
    STAGE_STAC_SEARCH_COMPLETED,
    STAGE_SCENES_RANKED,
    STAGE_SCENE_SELECTED,
    STAGE_RASTER_RETRIEVAL_STARTED,
    STAGE_RASTER_RETRIEVAL_COMPLETED,
    STAGE_RGB_PROCESSING,
    STAGE_GEOREFERENCING,
    STAGE_SATELLITE_RESULT_READY,
]

PHASE2_INVESTIGATION_STAGES = [
    STAGE_IMAGE_RECEIVED,
    STAGE_FILE_VALIDATED,
    STAGE_METADATA_EXTRACTED,
    STAGE_GEOREFERENCING_CHECKED,
    STAGE_EXIF_CHECKED,
    STAGE_LANDMARK_RETRIEVAL,
    STAGE_VPR_INFERENCE,
    STAGE_GLOBAL_GEOLOCATION,
    STAGE_CANDIDATE_FUSION,
    STAGE_CONFIDENCE_CALIBRATION,
    STAGE_LOCATION_RESOLVED,
    STAGE_STAC_SEARCH_STARTED,
    STAGE_RASTER_RETRIEVAL_COMPLETED,
    STAGE_CROSS_VIEW_VERIFICATION,
    STAGE_FINAL_RESULT,
]


# ---------------------------------------------------------------------------
# Error Categorization Taxonomy
# ---------------------------------------------------------------------------

class ErrorCategory(str, Enum):
    INPUT_VALIDATION_ERROR = "INPUT_VALIDATION_ERROR"
    RESOURCE_UNAVAILABLE = "RESOURCE_UNAVAILABLE"
    MODEL_EXECUTION_ERROR = "MODEL_EXECUTION_ERROR"
    GEOSPATIAL_ALIGNMENT_ERROR = "GEOSPATIAL_ALIGNMENT_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    PROVIDER_POLICY_ERROR = "PROVIDER_POLICY_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class ToolExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Redaction & Sanitization Utilities
# ---------------------------------------------------------------------------

_SENSITIVE_KEY_PATTERNS = [
    r"token",
    r"password",
    r"secret",
    r"api[_-]?key",
    r"auth",
    r"authorization",
    r"credential",
    r"private[_-]?key",
    r"access[_-]?key",
]

_SENSITIVE_RE = re.compile("|".join(_SENSITIVE_KEY_PATTERNS), re.IGNORECASE)


def sanitize_value(val: Any, max_string_len: int = 500) -> Any:
    """Recursively sanitizes values, redacting giant payloads, base64 strings, and secrets."""
    if isinstance(val, str):
        if val.startswith("data:image/") or val.startswith("data:application/"):
            return f"[BASE64_IMAGE_DATA_REDACTED len={len(val)}]"
        if len(val) > max_string_len:
            return val[:max_string_len] + f"... [TRUNCATED len={len(val)}]"
        return val
    elif isinstance(val, (int, float, bool)) or val is None:
        return val
    elif isinstance(val, (list, tuple)):
        if len(val) > 50:
            return [sanitize_value(x, max_string_len) for x in val[:50]] + [f"... [{len(val)-50} items omitted]"]
        return [sanitize_value(x, max_string_len) for x in val]
    elif isinstance(val, dict):
        sanitized = {}
        for k, v in val.items():
            k_str = str(k)
            if _SENSITIVE_RE.search(k_str):
                sanitized[k_str] = "[REDACTED_CREDENTIAL]"
            else:
                sanitized[k_str] = sanitize_value(v, max_string_len)
        return sanitized
    elif hasattr(val, "shape"):  # NumPy array
        return f"[NUMPY_ARRAY shape={getattr(val, 'shape')} dtype={getattr(val, 'dtype')}]"
    else:
        return str(val)


# ---------------------------------------------------------------------------
# Structured Log Event Model
# ---------------------------------------------------------------------------

@dataclass
class StructuredLogEvent:
    """Represents a single structured execution log event."""
    timestamp: str
    level: str  # "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"
    logger: str
    event: str
    correlation_id: str
    stage: Optional[str] = None
    error_category: Optional[str] = None
    actionable_reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "level": self.level,
            "logger": self.logger,
            "event": self.event,
            "correlation_id": self.correlation_id,
        }
        if self.stage:
            d["stage"] = self.stage
        if self.error_category:
            d["error_category"] = self.error_category
        if self.actionable_reason:
            d["actionable_reason"] = self.actionable_reason
        if self.details:
            d["details"] = sanitize_value(self.details)
        return d


# ---------------------------------------------------------------------------
# Discrete Tool Execution Record Model
# ---------------------------------------------------------------------------

@dataclass
class ToolExecutionRecord:
    """Represents an individual specialist tool invocation lifecycle."""
    tool_name: str
    status: str = ToolExecutionStatus.PENDING.value
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    duration_ms: float = 0.0
    input_ref: Dict[str, Any] = field(default_factory=dict)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_category: Optional[str] = None
    actionable_reason: Optional[str] = None

    def complete(
        self,
        status: str = ToolExecutionStatus.SUCCEEDED.value,
        duration_ms: Optional[float] = None,
        output_summary: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        error_category: Optional[str] = None,
        actionable_reason: Optional[str] = None,
    ):
        self.status = status
        self.completed_at = datetime.now(timezone.utc).isoformat()
        if duration_ms is not None:
            self.duration_ms = duration_ms
        if output_summary:
            self.output_summary.update(sanitize_value(output_summary))
        if provenance:
            self.provenance.update(provenance)
        if error:
            self.error = error
        if error_category:
            self.error_category = error_category
        if actionable_reason:
            self.actionable_reason = actionable_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": round(self.duration_ms, 2),
            "input_ref": sanitize_value(self.input_ref),
            "output_summary": sanitize_value(self.output_summary),
            "provenance": self.provenance,
            "error": self.error,
            "error_category": self.error_category,
            "actionable_reason": self.actionable_reason,
        }


# ---------------------------------------------------------------------------
# Pipeline Lifecycle Stage Model
# ---------------------------------------------------------------------------

@dataclass
class TraceStage:
    """Represents a discrete lifecycle stage in the investigation pipeline."""
    stage: str
    status: str = "started"  # "started" | "completed" | "failed" | "skipped" | "partial_success"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_category: Optional[str] = None

    def complete(
        self,
        status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        error_category: Optional[str] = None,
    ):
        self.status = status
        self.completed_at = datetime.now(timezone.utc).isoformat()
        if metadata:
            self.metadata.update(sanitize_value(metadata))
        if provenance:
            self.provenance.update(provenance)
        if error:
            self.error = error
        if error_category:
            self.error_category = error_category

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_id": self.stage.lower(),
            "stage_name": self.stage,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "timestamp": self.started_at,
            "duration_ms": round(self.duration_ms, 2),
            "provenance": self.provenance,
            "metadata": sanitize_value(self.metadata),
            "details": sanitize_value(self.metadata),
            "error": self.error,
            "error_category": self.error_category,
        }


# ---------------------------------------------------------------------------
# Structured Execution Trace Class
# ---------------------------------------------------------------------------

class ExecutionTrace:
    """
    Maintains an ordered chronological timeline of pipeline execution stages,
    discrete specialist tool executions, structured diagnostic logs, and provenance.
    """

    def __init__(self, query_id: Optional[str] = None, request_id: Optional[str] = None, query: str = ""):
        self.investigation_id = query_id or request_id or f"inv_{time.strftime('%Y%m%d_%H%M%S')}"
        self.request_id = self.investigation_id
        self.query_id = self.investigation_id
        self.query = query
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self._start_perf = time.perf_counter()

        self.stages: List[TraceStage] = []
        self.tool_executions: List[ToolExecutionRecord] = []
        self.logs: List[StructuredLogEvent] = []

        self.query_metadata: Dict[str, Any] = {}
        self.validation_outcome: Dict[str, Any] = {"passed": True, "reason": None}
        self.data_sources: List[Dict[str, Any]] = []
        self.overall_outcome: str = "pending"
        self._sys_logger = logging.getLogger("nexspace.telemetry")

    def record(
        self,
        stage: str,
        status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        error_category: Optional[str] = None,
    ) -> TraceStage:
        """Records an instantaneous or pre-measured lifecycle stage."""
        now_utc = datetime.now(timezone.utc).isoformat()
        dur = duration_ms if duration_ms is not None else 0.0
        meta = dict(metadata or {})
        if details:
            meta.update(details)
        st = TraceStage(
            stage=stage,
            status=status,
            started_at=now_utc,
            completed_at=now_utc,
            duration_ms=dur,
            metadata=sanitize_value(meta),
            provenance=provenance or {},
            error=error,
            error_category=error_category,
        )
        self.stages.append(st)

        # Mirror as structured log event
        lvl = "ERROR" if (status in ("failed", "error") or error) else "INFO"
        self.log(
            level=lvl,
            event=f"Stage '{stage}' finished with status '{status}'",
            stage=stage,
            error_category=error_category,
            details={"duration_ms": round(dur, 2), "error": error},
        )
        return st

    def stage_context(self, stage_name: str, initial_metadata: Optional[Dict[str, Any]] = None):
        """Context manager for measuring stage execution duration automatically."""
        return _StageContextManager(self, stage_name, initial_metadata)

    def total_duration_ms(self) -> float:
        return (time.perf_counter() - self._start_perf) * 1000.0

    def start_stage(self, stage: str, metadata: Optional[Dict[str, Any]] = None) -> TraceStage:
        st = TraceStage(
            stage=stage,
            status="started",
            started_at=datetime.now(timezone.utc).isoformat(),
            metadata=sanitize_value(metadata or {}),
        )
        self.stages.append(st)
        self.log(level="INFO", event=f"Stage '{stage}' started", stage=stage)
        return st

    def complete_stage(
        self,
        stage: str,
        metadata: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        error_category: Optional[str] = None,
    ):
        for st in reversed(self.stages):
            if st.stage == stage:
                st.complete(
                    status="completed" if not error else "failed",
                    metadata=metadata,
                    provenance=provenance,
                    error=error,
                    error_category=error_category,
                )
                lvl = "ERROR" if error else "INFO"
                self.log(
                    level=lvl,
                    event=f"Stage '{stage}' completed ({st.status})",
                    stage=stage,
                    error_category=error_category,
                    details={"error": error},
                )
                break

    # -----------------------------------------------------------------------
    # Tool Execution Lifecycle Tracking
    # -----------------------------------------------------------------------

    def record_tool_start(self, tool_name: str, input_ref: Optional[Dict[str, Any]] = None) -> ToolExecutionRecord:
        """Records that a specialist tool invocation has started."""
        rec = ToolExecutionRecord(
            tool_name=tool_name,
            status=ToolExecutionStatus.RUNNING.value,
            started_at=datetime.now(timezone.utc).isoformat(),
            input_ref=sanitize_value(input_ref or {}),
        )
        self.tool_executions.append(rec)
        self.log(
            level="INFO",
            event=f"Tool '{tool_name}' execution initiated",
            stage="TOOL_EXECUTION",
            details={"tool": tool_name},
        )
        return rec

    def record_tool_completion(
        self,
        tool_name: str,
        status: str,
        duration_ms: float,
        output_summary: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        error_category: Optional[str] = None,
        actionable_reason: Optional[str] = None,
    ):
        """Records that a specialist tool invocation has finished."""
        # Find active tool record or create one
        rec = next((r for r in reversed(self.tool_executions) if r.tool_name == tool_name and r.status == ToolExecutionStatus.RUNNING.value), None)
        if not rec:
            rec = ToolExecutionRecord(tool_name=tool_name)
            self.tool_executions.append(rec)

        rec.complete(
            status=status,
            duration_ms=duration_ms,
            output_summary=output_summary,
            provenance=provenance,
            error=error,
            error_category=error_category,
            actionable_reason=actionable_reason,
        )

        lvl = "ERROR" if status in ("failed", "error") else ("WARNING" if status in ("unavailable", "skipped", "degraded") else "INFO")
        self.log(
            level=lvl,
            event=f"Tool '{tool_name}' finished with status '{status}' in {duration_ms:.1f}ms",
            stage="TOOL_EXECUTION",
            error_category=error_category,
            actionable_reason=actionable_reason,
            details={"tool": tool_name, "duration_ms": duration_ms, "error": error},
        )

    # -----------------------------------------------------------------------
    # Structured Logging Integration
    # -----------------------------------------------------------------------

    def log(
        self,
        level: str,
        event: str,
        logger_name: str = "nexspace.investigation",
        stage: Optional[str] = None,
        error_category: Optional[str] = None,
        actionable_reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Emits a structured log event, recording it in the trace and the python logger."""
        try:
            ev = StructuredLogEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                level=level.upper(),
                logger=logger_name,
                event=event,
                correlation_id=self.investigation_id,
                stage=stage,
                error_category=error_category,
                actionable_reason=actionable_reason,
                details=sanitize_value(details or {}),
            )
            self.logs.append(ev)

            # Emit to standard logging
            py_level = getattr(logging, level.upper(), logging.INFO)
            self._sys_logger.log(py_level, f"[{self.investigation_id}] [{level.upper()}] {event}")
        except Exception:
            # Defensive logging: logging failures must never crash execution
            pass

    def debug(self, event: str, **kwargs):
        self.log("DEBUG", event, **kwargs)

    def info(self, event: str, **kwargs):
        self.log("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs):
        self.log("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs):
        self.log("ERROR", event, **kwargs)

    # -----------------------------------------------------------------------
    # Investigation Context Setters
    # -----------------------------------------------------------------------

    def set_validation_outcome(self, passed: bool, reason: Optional[str] = None):
        self.validation_outcome = {"passed": passed, "reason": reason}
        if not passed:
            self.log(
                level="WARNING",
                event=f"Input validation rejected query: {reason}",
                stage="INPUT_VALIDATION",
                error_category=ErrorCategory.INPUT_VALIDATION_ERROR.value,
            )

    def set_query_metadata(
        self,
        parsed_intent: Optional[str] = None,
        geographical_target: Optional[str] = None,
        temporal_constraints: Optional[str] = None,
        selected_task: Optional[str] = None,
    ):
        if parsed_intent:
            self.query_metadata["intent"] = parsed_intent
        if geographical_target:
            self.query_metadata["geographical_target"] = geographical_target
        if temporal_constraints:
            self.query_metadata["temporal_constraints"] = temporal_constraints
        if selected_task:
            self.query_metadata["selected_task"] = selected_task

    def add_data_source(self, source_id: str, sensor_type: str, metadata: Optional[Dict[str, Any]] = None):
        self.data_sources.append({
            "source_id": source_id,
            "sensor_type": sensor_type,
            "metadata": sanitize_value(metadata or {}),
        })

    def complete(self, overall_outcome: str = "completed"):
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.overall_outcome = overall_outcome
        self.log(level="INFO", event=f"Investigation completed with outcome: {overall_outcome}")

    # -----------------------------------------------------------------------
    # Export and Serialization
    # -----------------------------------------------------------------------

    def to_list(self) -> List[Dict[str, Any]]:
        return [st.to_dict() for st in self.stages]

    def export_trace(self) -> List[Dict[str, Any]]:
        return self.to_list()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes complete structured trace, sequential tool executions, and logs."""
        return {
            "investigation_id": self.investigation_id,
            "request_id": self.request_id,
            "query_id": self.query_id,
            "query": self.query,
            "started_at": self.started_at,
            "completed_at": self.completed_at or datetime.now(timezone.utc).isoformat(),
            "total_duration_ms": round(self.total_duration_ms(), 2),
            "overall_outcome": self.overall_outcome,
            "validation_outcome": self.validation_outcome,
            "query_metadata": self.query_metadata,
            "data_sources": self.data_sources,
            "stages": [st.to_dict() for st in self.stages],
            "tool_executions": [t.to_dict() for t in self.tool_executions],
            "logs": [log.to_dict() for log in self.logs],
        }


# Aliases for Phase 1 & Backward Compatibility
InvestigationTelemetry = ExecutionTrace
STAGES_ORDERED = DISCOVERY_STAGES


class _StageContextManager:
    """Helper context manager to record stage start, duration, and error."""

    def __init__(self, trace: ExecutionTrace, stage_name: str, metadata: Optional[Dict[str, Any]] = None):
        self.trace = trace
        self.stage_name = stage_name
        self.metadata = metadata or {}
        self.t0 = 0.0
        self.stage_obj: Optional[TraceStage] = None

    def __enter__(self) -> TraceStage:
        self.t0 = time.perf_counter()
        self.stage_obj = TraceStage(
            stage=self.stage_name,
            status="started",
            started_at=datetime.now(timezone.utc).isoformat(),
            metadata=sanitize_value(self.metadata),
        )
        self.trace.stages.append(self.stage_obj)
        self.trace.log(level="INFO", event=f"Stage '{self.stage_name}' started", stage=self.stage_name)
        return self.stage_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        dur = (time.perf_counter() - self.t0) * 1000.0
        if self.stage_obj:
            self.stage_obj.duration_ms = dur
            if exc_type is not None:
                self.stage_obj.complete(status="failed", error=str(exc_val), error_category=ErrorCategory.SYSTEM_ERROR.value)
                self.trace.log(
                    level="ERROR",
                    event=f"Stage '{self.stage_name}' failed with exception: {str(exc_val)}",
                    stage=self.stage_name,
                    error_category=ErrorCategory.SYSTEM_ERROR.value,
                    details={"exception": str(exc_val)},
                )
            else:
                if self.stage_obj.status == "started":
                    self.stage_obj.complete(status="completed")
                    self.trace.log(
                        level="INFO",
                        event=f"Stage '{self.stage_name}' completed successfully in {dur:.1f}ms",
                        stage=self.stage_name,
                        details={"duration_ms": round(dur, 2)},
                    )
        return False  # Do not suppress exceptions
