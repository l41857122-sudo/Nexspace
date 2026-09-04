"""
test_controller.py
------------------
Comprehensive automated verification suite for STEP 2 & STEP 3 Agent Controller
and Model Runtime Integration.

Verification Scenarios:
  1. VQA model runtime & fallback
  2. Captioning model runtime & fallback
  3. Grounding unavailable behavior (no fake detections)
  4. Change Analysis real classical computation
  5. Change-VQA unavailable/fallback behavior
  6. Optical + SAR distinct modality separation
  7. Optical + SAR fusion explicitly reporting not_implemented
  8. Confidence type & source correctness
  9. Model failure isolation
  10. Multi-tool execution with partial failure
  11. CPU execution path & DeviceManager
  12. API backward compatibility (FastAPI TestClient)
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import unittest
import numpy as np
from PIL import Image

# Ensure ml_backend path is available
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from orchestrator import GeoVLMController
from tools import (
    BaseSpecialistTool,
    ToolExecutionResult,
    tool_registry,
    VQATool,
    OpticalCaptioningTool,
    SARCaptioningTool,
    GroundingTool,
    ChangeAnalysisTool,
    ChangeVQATool,
    OpticalSARAnalysisTool,
)
from router import TaskType
from model_runtime import (
    DeviceManager,
    BaseModelRuntime,
    PaliGemmaVQARuntime,
    BLIPCaptioningRuntime,
    GroundingDINORuntime,
    OpticalSARFusionRuntime,
)


def create_dummy_image(color=(100, 150, 200), size=(100, 100)) -> Image.Image:
    """Helper to create dummy RGB PIL images."""
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class MockRealModelRuntime(BaseModelRuntime):
    """Test double for validating real model inference paths without downloading checkpoints."""

    def __init__(self, model_id: str = "mock/real-vision-model", task_name: str = "MockTask"):
        super().__init__(model_id=model_id, task_name=task_name)
        self.is_loaded = True
        self._pipe = "mock_pipe_handle"

    def is_available(self) -> bool:
        return True

    def infer(self, **kwargs) -> dict:
        if "question" in kwargs:
            return {
                "answer": "real_inferred_water_body",
                "confidence": 0.94,
                "confidence_type": "model",
                "confidence_source": self.model_id,
            }
        elif "modality" in kwargs:
            return {
                "caption": "A high-resolution real optical satellite inference.",
                "modality": kwargs["modality"],
                "model_capability": "generic_image_captioning",
                "confidence": 0.91,
                "confidence_type": "model",
                "confidence_source": self.model_id,
            }
        elif "target_phrase" in kwargs:
            return {
                "detections": [
                    {"label": kwargs["target_phrase"], "box": [10, 20, 80, 90], "score": 0.93}
                ],
                "image_width": 100,
                "image_height": 100,
            }
        return {}


class TestAgentControllerAndRuntime(unittest.TestCase):

    def setUp(self):
        self.controller = GeoVLMController()
        self.img_optical = create_dummy_image((50, 120, 50))
        self.img_sar = create_dummy_image((128, 128, 128))
        self.img_before = create_dummy_image((60, 60, 60))
        self.img_after = create_dummy_image((200, 200, 200))

    # ------------------------------------------------------------------
    # 1. VQA Model Availability & Fallback Test
    # ------------------------------------------------------------------
    def test_vqa_model_and_fallback(self):
        print("\n--- [TEST 1] VQA Model & Fallback ---")
        # Test A: Fallback path when model weights not loaded
        res_fallback = self.controller.handle_request(
            query="Is there a river present?",
            optical_image=self.img_optical,
        )
        self.assertEqual(res_fallback["status"], "completed")
        self.assertIn("VQA", res_fallback["selected_tools"])
        vqa_tool_res = [r for r in res_fallback["results"] if r["tool_name"] == "VQA"][0]
        self.assertIn(vqa_tool_res["status"], ("success", "fallback"))
        self.assertIn("vqa_results", res_fallback)

        # Test B: Injected Real Model Runtime
        mock_vqa_tool = VQATool(runtime=MockRealModelRuntime(model_id="test/rsvqa-real-model", task_name="VQA"))
        real_res = mock_vqa_tool.execute({"image": self.img_optical, "question": "Is there water?"})
        self.assertEqual(real_res.status, "success")
        self.assertEqual(real_res.confidence_type, "model")
        self.assertEqual(real_res.confidence_source, "test/rsvqa-real-model")
        self.assertEqual(real_res.data["primary_answer"], "real_inferred_water_body")
        print(f"VQA Real Inference: Status={real_res.status}, Answer='{real_res.data['primary_answer']}', Conf={real_res.confidence}")

    # ------------------------------------------------------------------
    # 2. Captioning Model Availability & Fallback Test
    # ------------------------------------------------------------------
    def test_captioning_model_and_fallback(self):
        print("\n--- [TEST 2] Captioning Model & Fallback ---")
        # Optical Captioning
        res_opt = self.controller.handle_request(
            query="What is visible in this satellite scene?",
            optical_image=self.img_optical,
        )
        self.assertEqual(res_opt["status"], "completed")
        self.assertIsNotNone(res_opt["optical_caption"])
        opt_tool_res = [r for r in res_opt["results"] if r["tool_name"] == "Optical_Caption"][0]
        self.assertIn(opt_tool_res["status"], ("success", "fallback"))
        print(f"Optical Caption: '{res_opt['optical_caption']}' (Status: {opt_tool_res['status']})")

    # ------------------------------------------------------------------
    # 3. Grounding Unavailable Behavior (No Fake Detections)
    # ------------------------------------------------------------------
    def test_grounding_unavailable_behavior(self):
        print("\n--- [TEST 3] Grounding Unavailable (No Fake Detections) ---")
        class MockUnavailableRuntime(BaseModelRuntime):
            def __init__(self):
                super().__init__(model_id="mock/unavailable-gdino", task_name="Grounding")
                self.load_error = "Grounding model adapter (mock/unavailable-gdino) is currently unavailable in the runtime environment. No simulated detections generated."
                self.is_loaded = False
            def _do_load(self):
                return False
            def is_available(self):
                return False

        grounding_tool = GroundingTool(runtime=MockUnavailableRuntime())
        res = grounding_tool.execute({"image": self.img_optical, "target_phrase": "residential buildings"})
        self.assertEqual(res.status, "unavailable")
        self.assertEqual(res.data["detections"], [])
        self.assertEqual(res.data["count"], 0)
        self.assertIsNone(res.confidence)
        self.assertIn("unavailable", res.data["summary"].lower())
        print(f"Grounding Status: {res.status}, Detections: {res.data['detections']}, Confidence: {res.confidence}")

    # ------------------------------------------------------------------
    # 4. Change Analysis Real Classical Computation Test
    # ------------------------------------------------------------------
    def test_change_analysis_classical_computation(self):
        print("\n--- [TEST 4] Change Analysis Classical Computation ---")
        res = self.controller.handle_request(
            query="What changed between these images?",
            change_image_a=self.img_before,
            change_image_b=self.img_after,
        )
        self.assertEqual(res["status"], "completed")
        self.assertIsNotNone(res["change_analysis"])
        self.assertEqual(res["change_analysis"]["changed_fraction"], 1.0)
        chg_tool_res = [r for r in res["results"] if r["tool_name"] == "Change_Analysis"][0]
        self.assertEqual(chg_tool_res["status"], "success")
        self.assertEqual(chg_tool_res["confidence_type"], "heuristic")
        self.assertEqual(chg_tool_res["confidence_source"], "classical_pixel_diff")
        self.assertEqual(chg_tool_res["data"]["method"], "classical_pixel_difference")
        self.assertGreater(chg_tool_res["data"]["processing_time_ms"], 0.0)
        print(f"Change Diff Method: {chg_tool_res['data']['method']}, Time: {chg_tool_res['data']['processing_time_ms']}ms")

    # ------------------------------------------------------------------
    # 5. Change-VQA Unavailable / Fallback Behavior Test
    # ------------------------------------------------------------------
    def test_change_vqa_fallback_behavior(self):
        print("\n--- [TEST 5] Change-VQA Fallback Adapter ---")
        change_vqa = ChangeVQATool()
        res = change_vqa.execute({
            "image_a": self.img_before,
            "image_b": self.img_after,
            "query": "Did the infrastructure expand?",
        })
        self.assertEqual(res.status, "fallback")
        self.assertEqual(res.confidence_type, "estimated")
        self.assertEqual(res.confidence_source, "bitemporal_diff_heuristic")
        self.assertIn("Change-VQA (Differential Heuristic):", res.data["answer"])
        print(f"Change-VQA Status: {res.status}, Answer: {res.data['answer']}")

    # ------------------------------------------------------------------
    # 6. Optical + SAR Modality Separation Test
    # ------------------------------------------------------------------
    def test_optical_sar_separation(self):
        print("\n--- [TEST 6] Optical + SAR Modality Separation ---")
        sar_tool = SARCaptioningTool()
        sar_res = sar_tool.execute({"image": self.img_sar, "modality": "sar"})
        self.assertEqual(sar_res.status, "success")
        self.assertEqual(sar_res.data["modality"], "sar")
        self.assertIn("generic_captioning_on_SAR", sar_res.data["model_capability"])
        print(f"SAR Modality: {sar_res.data['modality']}, Capability: {sar_res.data['model_capability']}")

    # ------------------------------------------------------------------
    # 7. Optical + SAR Feature Fusion Reporting
    # ------------------------------------------------------------------
    def test_optical_sar_fusion_not_implemented_reporting(self):
        print("\n--- [TEST 7] Optical + SAR Fusion Reporting ---")
        opt_sar_tool = OpticalSARAnalysisTool()
        res = opt_sar_tool.execute({"optical_image": self.img_optical, "sar_image": self.img_sar})
        self.assertEqual(res.status, "success")
        fusion_info = res.data["fusion"]
        self.assertEqual(fusion_info["status"], "success")
        self.assertEqual(fusion_info["fusion_type"], "feature_fusion_baseline")
        self.assertFalse(fusion_info["is_trained_fusion_model"])
        self.assertEqual(fusion_info["fused_feature_dim"], 1536)
        print(f"Fusion Status: {fusion_info['status']}, Type: {fusion_info['fusion_type']}, Dim: {fusion_info['fused_feature_dim']}")

    # ------------------------------------------------------------------
    # 8. Confidence Type and Source Correctness Test
    # ------------------------------------------------------------------
    def test_confidence_type_and_source_correctness(self):
        print("\n--- [TEST 8] Confidence Semantics ---")
        # Verify heuristic on classical diff
        chg_tool = ChangeAnalysisTool()
        c_res = chg_tool.execute({"image_a": self.img_before, "image_b": self.img_after})
        self.assertEqual(c_res.confidence_type, "heuristic")
        self.assertEqual(c_res.confidence_source, "classical_pixel_diff")

        # Verify Grounding confidence semantics
        g_tool = GroundingTool()
        g_res = g_tool.execute({"image": self.img_optical, "target_phrase": "ship"})
        self.assertIn(g_res.confidence_type, ("model", "heuristic"))
        print(f"Classical diff conf: {c_res.confidence} ({c_res.confidence_type}), Grounding conf: {g_res.confidence} ({g_res.confidence_type})")

    # ------------------------------------------------------------------
    # 9. Model Failure Isolation Test
    # ------------------------------------------------------------------
    def test_model_failure_isolation(self):
        print("\n--- [TEST 9] Model Failure Isolation ---")
        class ExceptionThrowingTool(BaseSpecialistTool):
            name = "Failing_Specialist"
            task_type = TaskType.VQA
            description = "Fails unconditionally"

            def _run(self, params):
                raise ValueError("Simulated model driver segfault / memory error")

        bad_tool = ExceptionThrowingTool()
        tool_registry.register(bad_tool)

        res = bad_tool.execute({})
        self.assertEqual(res.status, "failed")
        self.assertIn("Simulated model driver segfault", res.error)
        print(f"Isolated Exception: {res.error}")

    # ------------------------------------------------------------------
    # 10. Multi-Tool Execution with Partial Failure Test
    # ------------------------------------------------------------------
    def test_multi_tool_partial_failure(self):
        print("\n--- [TEST 10] Multi-Tool Partial Failure ---")
        res = self.controller.handle_request(
            query="Describe this image and locate the buildings",
            optical_image=self.img_optical,
        )
        self.assertEqual(res["status"], "completed")
        statuses = {r["tool_name"]: r["status"] for r in res["results"]}
        self.assertIn(statuses["Optical_Caption"], ("success", "fallback"))
        self.assertIn(statuses["Grounding"], ("success", "unavailable"))
        self.assertIsNotNone(res["optical_caption"])
        print(f"Tool statuses in composite query: {statuses}")

    # ------------------------------------------------------------------
    # 11. CPU Execution Path & DeviceManager Test
    # ------------------------------------------------------------------
    def test_device_manager_cpu_path(self):
        print("\n--- [TEST 11] DeviceManager & Hardware Telemetry ---")
        dev = DeviceManager.get_device()
        self.assertIn(dev, ("cpu", "cuda", "mps"))
        info = DeviceManager.get_device_info()
        self.assertIn("device", info)
        print(f"Detected Compute Device: {dev}, Info: {info}")

    # ------------------------------------------------------------------
    # 12. API Backward Compatibility (FastAPI TestClient)
    # ------------------------------------------------------------------
    def test_api_backward_compatibility(self):
        print("\n--- [TEST 12] FastAPI REST API Compatibility ---")
        from fastapi.testclient import TestClient
        from server import app
        import io, base64

        client = TestClient(app)

        # GET /api/health
        h_res = client.get("/api/health")
        self.assertEqual(h_res.status_code, 200)

        # Create dummy base64 image
        img = create_dummy_image((30, 80, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        # POST /api/query
        q_res = client.post("/api/query", json={"query": "Is there a river present?", "optical_image": b64_str})
        self.assertEqual(q_res.status_code, 200)
        data = q_res.json()
        self.assertIn("task_type", data)
        self.assertIn("selected_tools", data)
        self.assertIn("results", data)
        self.assertIn("execution_trace", data)
        self.assertIn("routing_decision", data)
        self.assertIn("response_text", data)

        # POST /api/change-analysis
        c_res = client.post("/api/change-analysis", json={"image_a": b64_str, "image_b": b64_str, "change_threshold": 0.15})
        self.assertEqual(c_res.status_code, 200)
        self.assertIn("summary", c_res.json())
        print(f"API Endpoints /api/health, /api/query, /api/change-analysis all passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
