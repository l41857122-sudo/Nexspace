"""
tests/test_phase1_execution_observability.py
--------------------------------------------
Comprehensive Test Suite for Phase 1:
Execution Trace & Structured Execution Logs

Verifies:
1. Unique investigation ID and request ID propagation.
2. Monotonic duration tracking and chronological ISO-8601 UTC timestamps.
3. Secret and payload redaction (tokens, API keys, credentials, large base64 strings).
4. Error categorization taxonomy (INPUT_VALIDATION_ERROR, RESOURCE_UNAVAILABLE, MODEL_EXECUTION_ERROR, etc.).
5. Discrete sequential tool execution lifecycle (pending -> running -> succeeded / failed / unavailable / skipped).
6. Downstream failure containment (failure reflects in overall trace, actionable reason captured).
7. Structured log event generation, correlation ID linking, and log level filtering.
8. Trace and log JSON serializability.
9. FastAPI middleware header propagation (X-Request-ID and X-Investigation-ID).
10. FastAPI endpoints: /api/investigation/{id}/trace and /api/investigation/{id}/logs.
"""

import os
import sys
import time
import unittest
from datetime import datetime, timezone

# Ensure ml_backend is on sys.path
_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(os.path.dirname(_dir), "ml_backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from telemetry import (
    ExecutionTrace,
    TraceStage,
    ToolExecutionRecord,
    ToolExecutionStatus,
    StructuredLogEvent,
    ErrorCategory,
    sanitize_value,
)
from server import app
from fastapi.testclient import TestClient
from investigation_pipeline import global_investigation_store


class TestPhase1ExecutionObservability(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    # 1. Unique Investigation & Request ID Propagation
    def test_01_unique_ids_and_correlation(self):
        inv_id = "inv_test_corr_001"
        trace = ExecutionTrace(query_id=inv_id, query="Locate vessels in Mumbai port")
        self.assertEqual(trace.investigation_id, inv_id)
        self.assertEqual(trace.request_id, inv_id)
        self.assertEqual(trace.query_id, inv_id)
        self.assertEqual(trace.query, "Locate vessels in Mumbai port")

        # Log event inherits correlation ID
        trace.log(level="INFO", event="Pipeline test event")
        self.assertEqual(len(trace.logs), 1)
        self.assertEqual(trace.logs[0].correlation_id, inv_id)

    # 2. Monotonic Duration & ISO-8601 UTC Timestamps
    def test_02_timing_and_iso_timestamps(self):
        trace = ExecutionTrace(query_id="inv_test_timing_002")
        time.sleep(0.01)
        stage = trace.record("preprocessing", status="completed", duration_ms=10.5)

        # Validate ISO-8601 UTC format
        dt_start = datetime.fromisoformat(stage.started_at)
        dt_comp = datetime.fromisoformat(stage.completed_at)
        self.assertIsNotNone(dt_start.tzinfo)
        self.assertIsNotNone(dt_comp.tzinfo)
        self.assertEqual(stage.duration_ms, 10.5)

        # Monotonic total duration is non-negative and grows
        total_dur = trace.total_duration_ms()
        self.assertGreaterEqual(total_dur, 10.0)

    # 3. Secret and Payload Redaction
    def test_03_secret_redaction(self):
        sensitive_dict = {
            "api_key": "sk-live-secret-key-12345",
            "access_token": "bearer-token-abcde",
            "password": "super-secret-password",
            "user_auth": "token-value",
            "safe_param": "mumbai_port",
            "b64_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
            "large_string": "x" * 1000,
        }
        sanitized = sanitize_value(sensitive_dict)

        self.assertEqual(sanitized["api_key"], "[REDACTED_CREDENTIAL]")
        self.assertEqual(sanitized["access_token"], "[REDACTED_CREDENTIAL]")
        self.assertEqual(sanitized["password"], "[REDACTED_CREDENTIAL]")
        self.assertEqual(sanitized["user_auth"], "[REDACTED_CREDENTIAL]")
        self.assertEqual(sanitized["safe_param"], "mumbai_port")
        self.assertTrue(sanitized["b64_image"].startswith("[BASE64_IMAGE_DATA_REDACTED"))
        self.assertTrue(sanitized["large_string"].endswith("[TRUNCATED len=1000]"))

    # 4. Error Categorization Taxonomy
    def test_04_error_categories(self):
        trace = ExecutionTrace(query_id="inv_test_err_004")
        trace.record(
            stage="input_validation",
            status="failed",
            error="Negative coordinate bounds provided",
            error_category=ErrorCategory.INPUT_VALIDATION_ERROR.value,
        )

        trace.record_tool_start("Sentinel1_SAR_Tool")
        trace.record_tool_completion(
            tool_name="Sentinel1_SAR_Tool",
            status=ToolExecutionStatus.UNAVAILABLE.value,
            duration_ms=5.0,
            error="Sentinel-1 GRD asset unavailable for given date",
            error_category=ErrorCategory.RESOURCE_UNAVAILABLE.value,
            actionable_reason="Select an alternative acquisition date or fall back to optical-only.",
        )

        d = trace.to_dict()
        val_stage = d["stages"][0]
        self.assertEqual(val_stage["error_category"], "INPUT_VALIDATION_ERROR")

        tool_exec = d["tool_executions"][0]
        self.assertEqual(tool_exec["error_category"], "RESOURCE_UNAVAILABLE")
        self.assertIn("Select an alternative", tool_exec["actionable_reason"])

    # 5. Discrete Sequential Tool Execution Tracking
    def test_05_tool_execution_lifecycle(self):
        trace = ExecutionTrace(query_id="inv_test_tools_005")

        # Start tool 1
        trace.record_tool_start("Optical_Caption", input_ref={"modality": "optical"})
        self.assertEqual(len(trace.tool_executions), 1)
        self.assertEqual(trace.tool_executions[0].status, ToolExecutionStatus.RUNNING.value)

        # Complete tool 1
        trace.record_tool_completion(
            tool_name="Optical_Caption",
            status=ToolExecutionStatus.SUCCEEDED.value,
            duration_ms=45.2,
            output_summary={"caption": "Harbor with docked vessels"},
            provenance={"model": "Salesforce/blip-base"},
        )
        self.assertEqual(trace.tool_executions[0].status, "succeeded")
        self.assertEqual(trace.tool_executions[0].duration_ms, 45.2)

        # Start tool 2 that skips
        trace.record_tool_start("SAR_Caption")
        trace.record_tool_completion(
            tool_name="SAR_Caption",
            status=ToolExecutionStatus.SKIPPED.value,
            duration_ms=0.5,
            actionable_reason="No SAR input raster supplied in query.",
        )
        self.assertEqual(trace.tool_executions[1].status, "skipped")

    # 6. Downstream Failure Does Not Report Success
    def test_06_failure_containment(self):
        trace = ExecutionTrace(query_id="inv_test_fail_006")
        trace.record_tool_start("Grounding")
        trace.record_tool_completion(
            tool_name="Grounding",
            status=ToolExecutionStatus.FAILED.value,
            duration_ms=30.0,
            error="CUDA out of memory",
            error_category=ErrorCategory.MODEL_EXECUTION_ERROR.value,
            actionable_reason="Retry with CPU execution profile or smaller tile.",
        )
        trace.complete(overall_outcome="failed")

        d = trace.to_dict()
        self.assertEqual(d["overall_outcome"], "failed")
        self.assertEqual(d["tool_executions"][0]["status"], "failed")
        self.assertIsNotNone(d["tool_executions"][0]["error"])

    # 7. Structured Log Event Filtering
    def test_07_structured_logging(self):
        trace = ExecutionTrace(query_id="inv_test_logs_007")
        trace.debug("Debug event 1")
        trace.info("Info event 2")
        trace.warning("Warning event 3", error_category="RESOURCE_UNAVAILABLE")
        trace.error("Error event 4", error_category="SYSTEM_ERROR")

        self.assertEqual(len(trace.logs), 4)
        levels = [l.level for l in trace.logs]
        self.assertEqual(levels, ["DEBUG", "INFO", "WARNING", "ERROR"])

        # Filter logs
        error_logs = [l for l in trace.logs if l.level == "ERROR"]
        self.assertEqual(len(error_logs), 1)
        self.assertEqual(error_logs[0].error_category, "SYSTEM_ERROR")

    # 8. Full Trace JSON Serializability
    def test_08_json_serializability(self):
        import json
        trace = ExecutionTrace(query_id="inv_test_json_008", query="Scan port")
        trace.set_query_metadata(
            parsed_intent="VESSEL_DETECTION",
            geographical_target="Mumbai",
            temporal_constraints="2026-Q3",
            selected_task="GROUNDING",
        )
        trace.set_validation_outcome(True)
        trace.add_data_source("S2B_20260926", "Sentinel-2 L2A", {"cloud_cover": 0.05})
        trace.record("query_understanding", status="completed", duration_ms=12.0)
        trace.record_tool_start("Grounding")
        trace.record_tool_completion("Grounding", status="succeeded", duration_ms=120.0)
        trace.complete("completed")

        trace_dict = trace.to_dict()
        # Must serialize cleanly to JSON string without errors
        serialized = json.dumps(trace_dict)
        self.assertIsInstance(serialized, str)
        self.assertIn("inv_test_json_008", serialized)
        self.assertIn("Mumbai", serialized)

    # 9. FastAPI Middleware Correlation Header Propagation
    def test_09_fastapi_header_propagation(self):
        custom_req_id = "test-client-req-9999"
        res = self.client.get("/api/health", headers={"X-Request-ID": custom_req_id})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("X-Request-ID"), custom_req_id)
        self.assertEqual(res.headers.get("X-Investigation-ID"), custom_req_id)

    # 10. FastAPI Trace & Logs Endpoints
    def test_10_fastapi_trace_and_logs_endpoints(self):
        inv_id = "inv_mock_store_010"
        mock_result = {
            "execution_trace": [{"stage": "test_stage", "status": "completed"}],
            "structured_trace": {
                "investigation_id": inv_id,
                "tool_executions": [{"tool_name": "Optical_Caption", "status": "succeeded"}],
                "total_duration_ms": 150.0,
            },
            "execution_logs": [
                {"level": "INFO", "event": "Stage completed", "timestamp": "2026-10-15T12:00:00Z"},
                {"level": "ERROR", "event": "Anomaly skipped", "timestamp": "2026-10-15T12:00:01Z"},
            ],
        }
        global_investigation_store.init_investigation(inv_id, "Test Query", ["test_stage"])
        global_investigation_store.set_completed(inv_id, mock_result)

        # GET /api/investigation/{id}/trace
        trace_res = self.client.get(f"/api/investigation/{inv_id}/trace")
        self.assertEqual(trace_res.status_code, 200)
        td = trace_res.json()
        self.assertEqual(td["investigation_id"], inv_id)
        self.assertEqual(len(td["tool_executions"]), 1)
        self.assertEqual(td["tool_executions"][0]["tool_name"], "Optical_Caption")

        # GET /api/investigation/{id}/logs
        logs_res = self.client.get(f"/api/investigation/{inv_id}/logs")
        self.assertEqual(logs_res.status_code, 200)
        ld = logs_res.json()
        self.assertEqual(ld["count"], 2)

        # GET /api/investigation/{id}/logs?level=ERROR
        error_logs_res = self.client.get(f"/api/investigation/{inv_id}/logs?level=ERROR")
        self.assertEqual(error_logs_res.status_code, 200)
        eld = error_logs_res.json()
        self.assertEqual(eld["count"], 1)
        self.assertEqual(eld["logs"][0]["level"], "ERROR")


if __name__ == "__main__":
    unittest.main()
