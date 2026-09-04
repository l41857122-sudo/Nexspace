"""
test_step8_telemetry.py
-----------------------
Comprehensive test suite for STEP 8:
Real Execution Telemetry + Evidence Graph + Investigation Reporting.
"""

import sys
import os
import unittest
import time
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from telemetry import ExecutionTrace, TraceStage
from evidence_graph import EvidenceGraph, EvidenceNode
from synthesis import InvestigationSynthesizer, InvestigationReport
from orchestrator import GeoVLMController
from router import IntentClassifier, TaskType
from tools import ToolExecutionResult
from server import app, encode_image_b64
from fastapi.testclient import TestClient


class TestStep8TelemetryAndEvidence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.controller = GeoVLMController()
        cls.classifier = IntentClassifier()
        cls.synthesizer = InvestigationSynthesizer()
        cls.client = TestClient(app)

        # Base 512x512 image (green landscape)
        cls.img_before = Image.new("RGB", (512, 512), (34, 139, 34))

        # Changed image: adds white rectangular structure
        cls.img_after = cls.img_before.copy()
        draw = ImageDraw.Draw(cls.img_after)
        draw.rectangle([120, 100, 240, 220], fill=(240, 240, 240))

        # Satellite samples
        cls.img_sat = Image.open(os.path.join(_dir, "sample_satellite.png"))
        cls.img_sar = Image.open(os.path.join(_dir, "sample_sar.png"))

    # 1. Monotonic Timing & Chronological Trace Integrity
    def test_trace_lifecycle_and_timing(self):
        print("\n--- [TEST 1] Monotonic Timing & Chronological Trace ---")
        trace = ExecutionTrace()
        trace.record("request_received", "completed", {"size": 100})
        time.sleep(0.01)
        with trace.stage_context("custom_computation", {"step": 1}) as st:
            time.sleep(0.02)
            st.complete(metadata={"computed": True})

        stages = trace.to_list()
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0]["stage"], "request_received")
        self.assertEqual(stages[1]["stage"], "custom_computation")
        self.assertGreater(stages[1]["duration_ms"], 10.0)
        self.assertIsNotNone(stages[1]["completed_at"])
        self.assertGreaterEqual(stages[1]["started_at"], stages[0]["started_at"])
        print(f"Recorded stages with real duration: {stages[1]['duration_ms']} ms")

    # 2. Evidence Graph Node Creation & Uniqueness
    def test_evidence_graph_uniqueness_and_provenance(self):
        print("\n--- [TEST 2] Evidence Graph Uniqueness & Provenance ---")
        graph = EvidenceGraph(query_id="query_test_001")
        n1 = graph.create_and_add(
            type="object_detection",
            source_tool="Grounding",
            source_model="IDEA-Research/grounding-dino-tiny",
            payload={"bbox_pixel": [10.0, 20.0, 100.0, 120.0], "label": "building"},
            confidence=0.85,
            confidence_type="model",
            confidence_source="grounding_dino_tiny",
        )
        n2 = graph.create_and_add(
            type="change_region",
            source_tool="Change_Analysis",
            source_model="classical_pixel_diff",
            payload={"area_pixels": 500, "area_fraction": 0.01},
            confidence=None,
            confidence_type="heuristic",
            confidence_source="classical_pixel_diff",
        )

        self.assertIsNotNone(n1)
        self.assertIsNotNone(n2)
        self.assertNotEqual(n1.evidence_id, n2.evidence_id)
        self.assertEqual(len(graph.nodes), 2)

        # Duplicate ID rejection test
        duplicate_node = EvidenceNode(
            evidence_id=n1.evidence_id,
            type="object_detection",
            source_tool="Grounding",
            source_model="test",
        )
        self.assertFalse(graph.add_node(duplicate_node))
        print(f"Evidence nodes unique: {list(graph.nodes.keys())}")

    # 3. Evidence Validation Rules (Reject NaN, Inf, Inverted BBox)
    def test_evidence_validation_rejection(self):
        print("\n--- [TEST 3] Evidence Validation (Sanity & Rejections) ---")
        graph = EvidenceGraph(query_id="query_val")

        # Inverted box [200, 200, 50, 50]
        bad_box_node = EvidenceNode(
            evidence_id="bad_001",
            type="object_detection",
            source_tool="Grounding",
            source_model="test",
            payload={"bbox_pixel": [200.0, 200.0, 50.0, 50.0]},
        )
        self.assertFalse(graph.add_node(bad_box_node))

        # NaN in payload float
        nan_node = EvidenceNode(
            evidence_id="nan_002",
            type="change_region",
            source_tool="Change_Analysis",
            source_model="test",
            payload={"mean_intensity_delta": float("nan")},
        )
        self.assertFalse(graph.add_node(nan_node))
        print("Invalid evidence nodes rejected correctly.")

    # 4. Complex Multi-Tool Query: "Compare these images, locate buildings, and tell me if there is water"
    def test_complex_multitool_investigation(self):
        print("\n--- [TEST 4] Complex Multi-Tool Investigation End-to-End ---")
        q = "Compare these images, locate buildings, and tell me if there is water"
        res = self.controller.handle_request(
            query=q,
            change_image_a=self.img_before,
            change_image_b=self.img_after,
        )

        self.assertEqual(res["status"], "completed")
        self.assertIn("Change_Analysis", res["selected_tools"])
        self.assertIn("Anomaly_Extraction", res["selected_tools"])
        self.assertIn("Grounding", res["selected_tools"])
        self.assertIn("VQA", res["selected_tools"])

        # Check Plan structure
        plan = res["plan"]
        self.assertEqual(plan["task_type"], "MULTI_TASK")
        self.assertEqual(plan["reasoning_basis"], "intent_rules")
        self.assertIn("grounding_targets", plan["parameters"])

        # Check Investigation Report
        report = res["investigation_report"]
        self.assertEqual(report["query"], q)
        self.assertGreater(len(report["observations"]), 0)
        self.assertIn("evidence", report)
        self.assertIn("execution_summary", report)

        # Check Execution Summary
        summary = res["execution_summary"]
        self.assertEqual(summary["tools_attempted"], len(res["selected_tools"]))
        self.assertGreater(summary["evidence_count"], 0)
        self.assertGreater(summary["total_duration_ms"], 0.0)

        # Check Execution Trace stages
        trace_stages = [s["stage"] for s in res["execution_trace"]]
        self.assertIn("request_received", trace_stages)
        self.assertIn("intent_classification", trace_stages)
        self.assertIn("task_planning", trace_stages)
        self.assertIn("tool_selection", trace_stages)
        self.assertIn("parameter_extraction", trace_stages)
        self.assertIn("tool_execution", trace_stages)
        self.assertIn("evidence_extraction", trace_stages)
        self.assertIn("evidence_validation", trace_stages)
        self.assertIn("result_synthesis", trace_stages)
        self.assertIn("response_completed", trace_stages)

        print(f"Complex Query Plan: {plan['selected_tools']}")
        print(f"Execution Summary: {summary}")
        print(f"Observations count: {len(report['observations'])}, Limitations count: {len(report['limitations'])}")

    # 5. Fallback Transparency in Synthesis
    def test_fallback_transparency(self):
        print("\n--- [TEST 5] Fallback Transparency ---")
        q = "Is there water in this image?"
        res = self.controller.handle_request(query=q, optical_image=self.img_sat)

        report = res["investigation_report"]
        # VQA runs in fallback mode when HF_TOKEN is unconfigured
        vqa_res = [r for r in res["results"] if r["tool_name"] == "VQA"][0]
        if vqa_res["status"] == "fallback":
            self.assertTrue(any("fallback" in lim.lower() for lim in report["limitations"]))
            self.assertEqual(res["execution_summary"]["fallback_count"], 1)
            print("Fallback usage transparently declared in report limitations.")

    # 6. Tool Failure Containment
    def test_tool_failure_containment(self):
        print("\n--- [TEST 6] Tool Failure Containment ---")
        class SimulatedFailingTool:
            name = "Simulated_Fail"
            def execute(self, params):
                raise RuntimeError("Driver crash simulated")

        tool_results = [
            ToolExecutionResult(tool_name="Optical_Caption", task_type="CAPTIONING", status="success", data={"caption": "green landscape"}),
            ToolExecutionResult(tool_name="Simulated_Fail", task_type="UNKNOWN", status="failed", error="Driver crash simulated"),
        ]

        graph = EvidenceGraph()
        graph.create_and_add("caption", "Optical_Caption", "BLIP", {"text": "green landscape"})

        trace = ExecutionTrace()
        trace.record("tool_execution", "partial_success")

        report, summary, resp_text = self.synthesizer.synthesize(
            query="Describe image",
            task_type="CAPTIONING",
            plan={"selected_tools": ["Optical_Caption", "Simulated_Fail"]},
            tool_results=tool_results,
            evidence_graph=graph,
            trace=trace,
        )

        self.assertEqual(summary["tools_completed"], 1)
        self.assertEqual(summary["tools_failed"], 1)
        self.assertTrue(any("Simulated_Fail" in lim for lim in report.limitations))
        self.assertIn("green landscape", resp_text)
        print("Failure isolated: successful tool output preserved alongside failure limitation.")

    # 7. Health Endpoint Capability Reporting
    def test_health_endpoint_capabilities(self):
        print("\n--- [TEST 7] GET /api/health Capability Statuses ---")
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("capabilities", data)
        caps = data["capabilities"]
        self.assertIn("captioning", caps)
        self.assertIn("grounding", caps)
        self.assertIn("vqa", caps)
        self.assertIn("change_analysis", caps)
        self.assertIn("optical_sar_fusion", caps)
        print(f"Reported Capabilities: {caps}")

    # 8. POST /api/query Schema Preservation & Extension
    def test_api_query_schema_extension(self):
        print("\n--- [TEST 8] POST /api/query Dual-Contract Schema ---")
        b64_opt = encode_image_b64(self.img_sat)
        res = self.client.post("/api/query", json={
            "query": "Describe this image and locate the buildings",
            "optical_image": b64_opt,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Check new investigation extensions
        self.assertIn("plan", data)
        self.assertIn("investigation_report", data)
        self.assertIn("evidence_graph", data)
        self.assertIn("execution_summary", data)
        self.assertIn("execution_trace", data)

        # Check existing fields preserved
        self.assertIn("task_type", data)
        self.assertIn("selected_tools", data)
        self.assertIn("evidence", data)
        self.assertIn("results", data)
        self.assertIn("response_text", data)
        self.assertIn("vqa_results", data)
        self.assertIn("optical_caption", data)
        print("API response preserves all legacy fields and adds full investigation telemetry.")

    # 9. Confidence Normalization Semantics
    def test_confidence_normalization_semantics(self):
        print("\n--- [TEST 9] Confidence Normalization Semantics ---")
        allowed_types = {"model", "heuristic", "unavailable"}
        res = self.controller.handle_request(
            query="Describe this image and locate the buildings",
            optical_image=self.img_sat,
        )
        self.assertIn(res["confidence_type"], allowed_types)
        for r in res["results"]:
            self.assertIn(r["confidence_type"], allowed_types)
            if r["tool_name"] == "Anomaly_Extraction":
                self.assertNotEqual(r["confidence_type"], "model")  # Anomaly severity must NOT be model confidence
        print(f"Overall confidence: {res['confidence']} (Type: {res['confidence_type']}, Source: {res['confidence_source']})")

    # 10. Query Plan Schema Integrity
    def test_query_plan_schema_integrity(self):
        print("\n--- [TEST 10] Query Plan Schema Integrity ---")
        res = self.controller.handle_request(
            query="Compare optical and SAR imagery",
            optical_image=self.img_sat,
            sar_image=self.img_sar,
        )
        plan = res["plan"]
        self.assertIn("task_type", plan)
        self.assertIn("selected_tools", plan)
        self.assertEqual(plan["reasoning_basis"], "intent_rules")
        self.assertIn("parameters", plan)
        self.assertTrue(plan["parameters"]["cross_modal"])
        print(f"Query Plan verified: {plan}")

    # 11. Investigation Report JSON Serializability
    def test_investigation_report_json_serializability(self):
        print("\n--- [TEST 11] Investigation Report JSON Serializability ---")
        import json
        res = self.controller.handle_request(
            query="Describe this image",
            optical_image=self.img_sat,
        )
        report = res["investigation_report"]
        # Ensure json.dumps succeeds without TypeError
        dumped = json.dumps(report, default=str)
        self.assertGreater(len(dumped), 50)
        self.assertIn("observations", dumped)
        self.assertIn("execution_summary", dumped)
        print(f"Investigation report serialized cleanly ({len(dumped)} chars).")

    # 12. Stage Error Capture in Telemetry
    def test_stage_error_capture(self):
        print("\n--- [TEST 12] Stage Error Capture in Telemetry ---")
        trace = ExecutionTrace()
        try:
            with trace.stage_context("failing_stage", {"input": "test"}):
                raise ValueError("Intentional computational failure")
        except ValueError:
            pass

        stages = trace.to_list()
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]["status"], "failed")
        self.assertIn("Intentional computational failure", stages[0]["error"])
        self.assertGreaterEqual(stages[0]["duration_ms"], 0.0)
        print(f"Failed stage telemetry captured: {stages[0]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

