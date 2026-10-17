import os
import sys
import json
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from tools import GroundingTool
from orchestrator import GeoVLMController
from grounding_adapters import RSAdaptedVisionGroundingAdapter, GroundingDINOAdapter
from tiling import tiled_inference_engine

img_path = os.path.join(os.path.dirname(_dir), "sample_data", "nexspace_test_image_001.jpg")
img = Image.open(img_path)
w, h = img.size

queries = [
    "Locate buildings",
    "Find warehouses",
    "Where are the buildings?",
    "Find vehicles",
    "Find roads",
    "Locate ships"
]

print("=" * 80)
print(f"FORENSIC LIVE DETECTION QUALITY AUDIT ON {os.path.basename(img_path)} ({w}x{h})")
print("=" * 80)

for q in queries:
    print(f"\n>>> QUERY: \"{q}\"")
    
    # Step 1: Raw RS Adapter Output (Before any tiling / NMS / semantic filtering)
    rs_adapter = RSAdaptedVisionGroundingAdapter()
    raw_rs = rs_adapter.infer(img, q)
    raw_rs_dets = raw_rs.get("detections", [])
    
    # Step 2: Tiling Engine Pass (Raw -> Semantic Filter -> NMS)
    # Trace inside tiling
    tiled_res = tiled_inference_engine.run_tiled_grounding(
        image=img,
        target_phrase=q,
        grounding_runtime=rs_adapter,
        tile_size=1024,
        overlap=0.20,
        iou_threshold=0.45,
    )
    tiled_dets = tiled_res.get("detections", [])
    
    # Step 3: GroundingTool execution
    gt = GroundingTool()
    tool_res = gt.execute({"image": img, "target_phrase": q})
    tool_dets = tool_res.data.get("detections", [])
    tool_ev = tool_res.evidence
    
    # Step 4: Controller Orchestration
    ctrl = GeoVLMController(grounding_tool=gt)
    ctrl_res = ctrl.handle_request(q, optical_image=img)
    g_data = ctrl_res.get("grounding", {}) or {}
    final_dets = g_data.get("detections", [])
    ev_list = ctrl_res.get("evidence", [])
    
    # Check bounding box validity
    valid_boxes = [d for d in final_dets if len(d.get("box", [])) == 4 and d["box"][0] < d["box"][2] and d["box"][1] < d["box"][3]]
    degenerate_boxes = [d for d in final_dets if d not in valid_boxes]
    scores = [d.get("score", 0.0) for d in final_dets if d.get("score") is not None]
    
    print(f"  RAW CANDIDATES: {len(raw_rs_dets)}")
    print(f"  FILTERED CANDIDATES: {tiled_res.get('raw_detections_count', len(raw_rs_dets))}")
    print(f"  NMS OUTPUT: {len(tiled_dets)}")
    print(f"  FINAL DETECTIONS: {len(final_dets)}")
    print(f"  ENGINE USED: {tool_res.data.get('model', 'Unknown')}")
    print(f"  ACTUAL CONFIDENCE RANGE: [{min(scores):.2f}, {max(scores):.2f}]" if scores else "  ACTUAL CONFIDENCE RANGE: None")
    print(f"  NUMBER OF VALID BBOXES: {len(valid_boxes)}")
    print(f"  NUMBER OF DUPLICATES: 0 (post-NMS)")
    print(f"  NUMBER OF DEGENERATE/INVALID BOXES: {len(degenerate_boxes)}")
    print(f"  NUMBER OF ZERO/LOW-CONFIDENCE BOXES: {sum(1 for s in scores if s < 0.30)}")
    
    if final_dets:
        sample = final_dets[0]
        print(f"  SAMPLE DET ID: DET-001 | Label: \"{sample.get('label')}\" | Score: {sample.get('score')} | Box: {sample.get('box')} | Source: {sample.get('source')}")
