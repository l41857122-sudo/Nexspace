"""
telemetry.py
------------
Real-Time Execution Telemetry & Lifecycle Tracing for Geospatial Vision-Language Pipelines.

Provides millisecond-precision monotonic duration tracking, real chronological ISO-8601 UTC timestamps,
hierarchical tool/model sub-stages, error capture, and execution metadata.
"""

from __future__ import annotations
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class TraceStage:
    """Represents a single discrete execution stage in the investigation pipeline."""
    stage: str
    status: str = "started"  # "started" | "completed" | "failed" | "skipped" | "partial_success"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def complete(self, status: str = "completed", metadata: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.status = status
        self.completed_at = datetime.now(timezone.utc).isoformat()
        if metadata:
            self.metadata.update(metadata)
        if error:
            self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "timestamp": self.started_at,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
            "details": self.metadata,
            "error": self.error,
        }


class ExecutionTrace:
    """Maintains an ordered chronological timeline of pipeline execution stages."""

    def __init__(self, query_id: Optional[str] = None, request_id: Optional[str] = None):
        self.query_id = query_id or request_id or "query_001"
        self.request_id = self.query_id
        self.stages: List[TraceStage] = []
        self._start_perf = time.perf_counter()

    def record(
        self,
        stage: str,
        status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
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
            metadata=meta,
            error=error,
        )
        self.stages.append(st)
        return st

    def stage_context(self, stage_name: str, initial_metadata: Optional[Dict[str, Any]] = None):
        """Context manager for measuring stage execution duration automatically."""
        return _StageContextManager(self, stage_name, initial_metadata)

    def total_duration_ms(self) -> float:
        return (time.perf_counter() - self._start_perf) * 1000.0

    def to_list(self) -> List[Dict[str, Any]]:
        return [st.to_dict() for st in self.stages]


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
            metadata=self.metadata,
        )
        self.trace.stages.append(self.stage_obj)
        return self.stage_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        dur = (time.perf_counter() - self.t0) * 1000.0
        if self.stage_obj:
            self.stage_obj.duration_ms = dur
            if exc_type is not None:
                self.stage_obj.complete(status="failed", error=str(exc_val))
            else:
                if self.stage_obj.status == "started":
                    self.stage_obj.complete(status="completed")
        return False  # Do not suppress exceptions
