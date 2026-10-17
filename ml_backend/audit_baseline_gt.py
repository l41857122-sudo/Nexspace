import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))

baseline_path = os.path.join(_dir, "evaluation_results", "baseline_grounding_benchmark", "grounding_dino_baseline_benchmark_results.json")
master_path = os.path.join(_dir, "evaluation_results", "phase2_5_full_validation", "phase2_5_master_results.json")
raw_dir = os.path.join(_dir, "evaluation_results", "phase2_5_full_validation", "raw_outputs")

with open(baseline_path, "r", encoding="utf-8") as f:
    b_data = json.load(f)

with open(master_path, "r", encoding="utf-8") as f:
    m_data = json.load(f)

print("Baseline Top-Level Aggregate Metrics:")
print(json.dumps(b_data.get("aggregate_metrics", {}), indent=2))
print("\nBaseline Size Breakdown Metrics:")
print(json.dumps(b_data.get("size_breakdown_metrics", {}), indent=2))

print("\nListing all 16 scenes in baseline:")
for s in b_data.get("per_image_results", []):
    print(f"{s['scene_id']}: {s['scene_name']} | GT Count: {s.get('ground_truth_count')} | Dets Count: {s.get('detections_count')} | Metrics IoU 50: {s.get('metrics_iou_50')}")
