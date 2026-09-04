"""
verify_inference.py
-------------------
Verification script for STEP 3.1: Real Model Inference Verification on the host environment.
Executes live checks for:
  1. PaliGemma VQA
  2. BLIP Captioning
  3. Grounding DINO
  4. Change Analysis (classical diff)
  5. Optical + SAR Separation & Fusion status
  6. API endpoints (FastAPI TestClient)
"""

import sys
import os
import time
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image

# Ensure ml_backend is on path
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from model_runtime import (
    DeviceManager,
    PaliGemmaVQARuntime,
    BLIPCaptioningRuntime,
    GroundingDINORuntime,
    OpticalSARFusionRuntime,
)
from tools import (
    vqa_tool,
    optical_caption_tool,
    sar_caption_tool,
    grounding_tool,
    change_analysis_tool,
    optical_sar_analysis_tool,
)
from orchestrator import GeoVLMController
import change_analysis


def create_sample_image(color=(34, 139, 34), size=(224, 224), pattern=False) -> Image.Image:
    img = Image.new("RGB", size, color)
    if pattern:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill=(220, 20, 60))
        draw.ellipse([80, 80, 140, 140], fill=(30, 144, 255))
    return img


def run_checks():
    results = {}

    print("======================================================================")
    print("STEP 3.1: REAL MODEL INFERENCE SMOKE TEST & HARDWARE TELEMETRY")
    print("======================================================================")

    # ------------------------------------------------------------------
    # 0. Environment & Hardware Diagnostics
    # ------------------------------------------------------------------
    print("\n[0] ENVIRONMENT DIAGNOSTICS")
    torch_installed = False
    torch_importable = False
    torch_error = None
    transformers_installed = False
    transformers_error = None

    try:
        import torch
        torch_installed = True
        torch_importable = True
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
    except Exception as e:
        torch_error = str(e)
        torch_ver = "unimportable"
        cuda_avail = False

    try:
        import transformers
        transformers_installed = True
        trans_ver = transformers.__version__
    except Exception as e:
        transformers_error = str(e)
        trans_ver = "uninstalled/unimportable"

    device_info = DeviceManager.get_device_info()
    print(f"  * Python version       : {sys.version.split()[0]}")
    print(f"  * PyTorch importable   : {torch_importable} (Error: {torch_error})")
    print(f"  * Transformers version : {trans_ver} (Error: {transformers_error})")
    print(f"  * Device probed        : {device_info.get('device')} (CUDA: {device_info.get('cuda_available')})")

    results["env"] = {
        "torch_importable": torch_importable,
        "torch_error": torch_error,
        "transformers_installed": transformers_installed,
        "transformers_error": transformers_error,
        "device": device_info.get("device"),
    }

    sample_img = create_sample_image()
    sample_img_b = create_sample_image(pattern=True)

    # ------------------------------------------------------------------
    # 1. PaliGemma VQA Check
    # ------------------------------------------------------------------
    print("\n[1] PALIGEMMA VQA INFERENCE CHECK")
    vqa_runtime = PaliGemmaVQARuntime()
    t0 = time.perf_counter()
    vqa_load_ok = vqa_runtime.load()
    vqa_load_time = (time.perf_counter() - t0) * 1000.0

    print(f"  * Model Identifier     : {vqa_runtime.model_id}")
    print(f"  * Load attempted       : Success={vqa_load_ok}, Error={vqa_runtime.load_error}")
    print(f"  * Load duration        : {vqa_load_time:.2f}ms")

    # Run tool execution
    vqa_tool_res = vqa_tool.execute({"image": sample_img, "question": "Is there a river present?"})
    print(f"  * Tool Status          : {vqa_tool_res.status}")
    print(f"  * Primary Answer       : {vqa_tool_res.data.get('primary_answer')}")
    print(f"  * Confidence           : {vqa_tool_res.confidence} ({vqa_tool_res.confidence_type})")
    print(f"  * Confidence Source    : {vqa_tool_res.confidence_source}")

    results["vqa"] = {
        "model_id": vqa_runtime.model_id,
        "real_inference": vqa_load_ok,
        "load_error": vqa_runtime.load_error,
        "tool_status": vqa_tool_res.status,
        "primary_answer": vqa_tool_res.data.get("primary_answer"),
        "confidence": vqa_tool_res.confidence,
        "confidence_type": vqa_tool_res.confidence_type,
        "confidence_source": vqa_tool_res.confidence_source,
    }

    # ------------------------------------------------------------------
    # 2. BLIP Captioning Check
    # ------------------------------------------------------------------
    print("\n[2] BLIP CAPTIONING INFERENCE CHECK")
    blip_runtime = BLIPCaptioningRuntime()
    t0 = time.perf_counter()
    blip_load_ok = blip_runtime.load()
    blip_load_time = (time.perf_counter() - t0) * 1000.0

    print(f"  * Model Identifier     : {blip_runtime.model_id}")
    print(f"  * Load attempted       : Success={blip_load_ok}, Error={blip_runtime.load_error}")
    print(f"  * Load duration        : {blip_load_time:.2f}ms")

    # Run optical tool
    opt_tool_res = optical_caption_tool.execute({"image": sample_img, "modality": "optical"})
    print(f"  * Optical Tool Status  : {opt_tool_res.status}")
    print(f"  * Caption Text         : {opt_tool_res.data.get('caption')}")
    print(f"  * Capability Tag       : {opt_tool_res.data.get('model_capability')}")
    print(f"  * Confidence Source    : {opt_tool_res.confidence_source}")

    # Run SAR tool
    sar_tool_res = sar_caption_tool.execute({"image": sample_img, "modality": "sar"})
    print(f"  * SAR Tool Status      : {sar_tool_res.status}")
    print(f"  * SAR Caption Text     : {sar_tool_res.data.get('caption')}")
    print(f"  * SAR Capability Tag   : {sar_tool_res.data.get('model_capability')}")

    results["blip"] = {
        "model_id": blip_runtime.model_id,
        "real_inference": blip_load_ok,
        "load_error": blip_runtime.load_error,
        "optical_status": opt_tool_res.status,
        "optical_caption": opt_tool_res.data.get("caption"),
        "sar_status": sar_tool_res.status,
        "sar_caption": sar_tool_res.data.get("caption"),
    }

    # ------------------------------------------------------------------
    # 3. Grounding DINO Check
    # ------------------------------------------------------------------
    print("\n[3] GROUNDING DINO RUNTIME CHECK")
    grounding_runtime = GroundingDINORuntime()
    t0 = time.perf_counter()
    gd_load_ok = grounding_runtime.load()
    gd_load_time = (time.perf_counter() - t0) * 1000.0

    print(f"  * Model Identifier     : {grounding_runtime.model_id}")
    print(f"  * Load attempted       : Success={gd_load_ok}, Error={grounding_runtime.load_error}")
    print(f"  * Load duration        : {gd_load_time:.2f}ms")

    gd_tool_res = grounding_tool.execute({"image": sample_img, "target_phrase": "buildings"})
    print(f"  * Tool Status          : {gd_tool_res.status}")
    print(f"  * Detections list      : {gd_tool_res.data.get('detections')}")
    print(f"  * Confidence           : {gd_tool_res.confidence}")
    print(f"  * Summary              : {gd_tool_res.data.get('summary')}")

    results["grounding"] = {
        "model_id": grounding_runtime.model_id,
        "real_inference": gd_load_ok,
        "reason_unavailable": grounding_runtime.load_error,
        "tool_status": gd_tool_res.status,
        "detections": gd_tool_res.data.get("detections"),
        "confidence": gd_tool_res.confidence,
    }

    # ------------------------------------------------------------------
    # 4. Change Analysis Classical Computation Check
    # ------------------------------------------------------------------
    print("\n[4] CHANGE ANALYSIS (CLASSICAL DIFF) CHECK")
    chg_res = change_analysis.analyze(sample_img, sample_img_b, change_threshold=0.15)
    print(f"  * Method               : {chg_res.method}")
    print(f"  * Dimensions           : {chg_res.image_dimensions}")
    print(f"  * Processing time      : {chg_res.processing_time_ms:.2f}ms")
    print(f"  * Changed Fraction     : {chg_res.changed_fraction:.4f}")
    print(f"  * Mean Delta           : {chg_res.mean_intensity_delta:.2f}")
    print(f"  * Heatmap Mode         : {chg_res.heatmap.mode}, Overlay Mode: {chg_res.overlay.mode}")
    print(f"  * Summary Text         : {chg_res.summary}")

    results["change_analysis"] = {
        "method": chg_res.method,
        "processing_time_ms": chg_res.processing_time_ms,
        "changed_fraction": chg_res.changed_fraction,
        "mean_intensity_delta": chg_res.mean_intensity_delta,
        "summary": chg_res.summary,
    }

    # ------------------------------------------------------------------
    # 5. Optical + SAR Separation & Fusion Status Check
    # ------------------------------------------------------------------
    print("\n[5] OPTICAL + SAR SEPARATION & FUSION STATUS CHECK")
    opt_sar_res = optical_sar_analysis_tool.execute({"optical_image": sample_img, "sar_image": sample_img})
    print(f"  * Tool Status          : {opt_sar_res.status}")
    print(f"  * Optical stream status: {opt_sar_res.data.get('optical', {}).get('status')}")
    print(f"  * SAR stream status    : {opt_sar_res.data.get('sar', {}).get('status')}")
    print(f"  * Fusion status        : {opt_sar_res.data.get('fusion', {}).get('status')}")
    print(f"  * Fusion message       : {opt_sar_res.data.get('fusion', {}).get('message')}")

    results["optical_sar"] = {
        "tool_status": opt_sar_res.status,
        "optical_status": opt_sar_res.data.get("optical", {}).get("status"),
        "sar_status": opt_sar_res.data.get("sar", {}).get("status"),
        "fusion_status": opt_sar_res.data.get("fusion", {}).get("status"),
        "fusion_message": opt_sar_res.data.get("fusion", {}).get("message"),
    }

    # ------------------------------------------------------------------
    # 6. API Endpoints Live Execution Check
    # ------------------------------------------------------------------
    print("\n[6] API ENDPOINTS LIVE TEST VIA FASTAPI TESTCLIENT")
    from fastapi.testclient import TestClient
    from server import app, encode_image_b64

    client = TestClient(app)

    # 6A: GET /api/health
    h_res = client.get("/api/health")
    print(f"  * GET /api/health      : HTTP {h_res.status_code} -> {h_res.json()}")

    # 6B: POST /api/change-analysis
    b64_a = encode_image_b64(sample_img)
    b64_b = encode_image_b64(sample_img_b)
    c_res = client.post("/api/change-analysis", json={"image_a": b64_a, "image_b": b64_b, "change_threshold": 0.15})
    c_json = c_res.json()
    print(f"  * POST /api/change-analysis : HTTP {c_res.status_code} -> changed_fraction={c_json.get('changed_fraction')}")

    # 6C: Query 1: "Is there water in this image?"
    q1_res = client.post("/api/query", json={"query": "Is there water in this image?", "optical_image": b64_a})
    q1_data = q1_res.json()
    print(f"\n  --- Query 1: 'Is there water in this image?' ---")
    print(f"    * HTTP Status        : {q1_res.status_code}")
    print(f"    * task_type          : {q1_data.get('task_type')}")
    print(f"    * selected_tools     : {q1_data.get('selected_tools')}")
    print(f"    * confidence         : {q1_data.get('confidence')} ({q1_data.get('confidence_type')})")
    print(f"    * confidence_source  : {q1_data.get('confidence_source')}")
    print(f"    * tool statuses      : {[r.get('status') for r in q1_data.get('results', [])]}")
    print(f"    * execution trace    : {[e.get('stage') for e in q1_data.get('execution_trace', [])]}")
    print(f"    * response text      :\n      {q1_data.get('response_text')}")

    # 6D: Query 2: "Describe this image"
    q2_res = client.post("/api/query", json={"query": "Describe this image", "optical_image": b64_a})
    q2_data = q2_res.json()
    print(f"\n  --- Query 2: 'Describe this image' ---")
    print(f"    * HTTP Status        : {q2_res.status_code}")
    print(f"    * task_type          : {q2_data.get('task_type')}")
    print(f"    * selected_tools     : {q2_data.get('selected_tools')}")
    print(f"    * confidence         : {q2_data.get('confidence')} ({q2_data.get('confidence_type')})")
    print(f"    * confidence_source  : {q2_data.get('confidence_source')}")
    print(f"    * tool statuses      : {[r.get('status') for r in q2_data.get('results', [])]}")
    print(f"    * execution trace    : {[e.get('stage') for e in q2_data.get('execution_trace', [])]}")
    print(f"    * response text      :\n      {q2_data.get('response_text')}")

    # 6E: Query 3: "Describe this image and locate the buildings"
    q3_res = client.post("/api/query", json={"query": "Describe this image and locate the buildings", "optical_image": b64_a})
    q3_data = q3_res.json()
    print(f"\n  --- Query 3: 'Describe this image and locate the buildings' ---")
    print(f"    * HTTP Status        : {q3_res.status_code}")
    print(f"    * task_type          : {q3_data.get('task_type')}")
    print(f"    * selected_tools     : {q3_data.get('selected_tools')}")
    print(f"    * confidence         : {q3_data.get('confidence')} ({q3_data.get('confidence_type')})")
    print(f"    * confidence_source  : {q3_data.get('confidence_source')}")
    print(f"    * tool statuses      : {[(r.get('tool_name'), r.get('status')) for r in q3_data.get('results', [])]}")
    print(f"    * execution trace    : {[e.get('stage') for e in q3_data.get('execution_trace', [])]}")
    print(f"    * response text      :\n      {q3_data.get('response_text')}")

    results["api"] = {
        "health": h_res.status_code == 200,
        "change_analysis": c_res.status_code == 200,
        "query_1": q1_data,
        "query_2": q2_data,
        "query_3": q3_data,
    }

    return results


if __name__ == "__main__":
    run_checks()
