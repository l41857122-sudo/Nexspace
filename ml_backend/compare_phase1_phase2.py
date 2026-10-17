import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))

baseline_path = os.path.join(_dir, "evaluation_results", "baseline_grounding_benchmark", "grounding_dino_baseline_benchmark_results.json")
phase2_path = os.path.join(_dir, "evaluation_results", "phase2_grounding_benchmark", "grounding_dino_phase2_benchmark_results.json")

with open(baseline_path, "r", encoding="utf-8") as f:
    b_data = json.load(f)

with open(phase2_path, "r", encoding="utf-8") as f:
    p2_data = json.load(f)

# Scene-by-scene GT counts from benchmark dataset definition
GT_COUNTS = {
    "scene_01": {"count": 16, "size": "small", "cat": "buildings / residential areas"},
    "scene_02": {"count": 9, "size": "medium", "cat": "buildings / residential areas"},
    "scene_03": {"count": 6, "size": "large", "cat": "urban areas"},
    "scene_04": {"count": 2, "size": "large", "cat": "roads"},
    "scene_05": {"count": 24, "size": "small", "cat": "vehicles"},
    "scene_06": {"count": 8, "size": "small", "cat": "vehicles"},
    "scene_07": {"count": 4, "size": "large", "cat": "ships / ports"},
    "scene_08": {"count": 12, "size": "small", "cat": "ships / ports"},
    "scene_09": {"count": 4, "size": "medium", "cat": "airports"},
    "scene_10": {"count": 6, "size": "large", "cat": "rural/agricultural areas"},
    "scene_11": {"count": 5, "size": "medium", "cat": "rural/agricultural areas"},
    "scene_12": {"count": 2, "size": "large", "cat": "coastline/water"},
    "scene_13": {"count": 1, "size": "large", "cat": "roads"},
    "scene_14": {"count": 5, "size": "medium", "cat": "urban areas"},
    "scene_15": {"count": 2, "size": "large", "cat": "rural/agricultural areas"},
    "scene_16": {"count": 38, "size": "small", "cat": "buildings / residential areas"},
}

total_gt = sum(v["count"] for v in GT_COUNTS.values())

# Evaluate Phase 2 per scene matching
p2_scenes = p2_data["per_image_results"]
total_p2_dets = sum(s["detections_count"] for s in p2_scenes)

print(f"Total Benchmark GT: {total_gt} objects across 16 scenes")
print(f"Phase 1 Baseline Detections: {sum(s['detections_count'] for s in b_data['per_image_results'])}")
print(f"Phase 2 Tiled Detections:    {total_p2_dets}")

# Print scene comparison table
print("\n" + "="*110)
print(f"{'Scene ID':<10} | {'Scene Name':<45} | {'GT':<4} | {'P1 Dets':<8} | {'P2 Dets':<8} | {'P1 Giant':<8} | {'P2 Giant':<8}")
print("="*110)

p1_giants_total = 0
p2_giants_total = 0

for b_s, p2_s in zip(b_data["per_image_results"], p2_scenes):
    sid = b_s["scene_id"]
    sname = b_s["scene_name"]
    gt = GT_COUNTS[sid]["count"]
    p1_d = b_s["detections_count"]
    p2_d = p2_s["detections_count"]
    p1_g = len(b_s.get("giant_boxes_detected", []))
    p2_g = len(p2_s.get("giant_boxes_detected", []))
    p1_giants_total += p1_g
    p2_giants_total += p2_g
    print(f"{sid:<10} | {sname:<45} | {gt:<4} | {p1_d:<8} | {p2_d:<8} | {p1_g:<8} | {p2_g:<8}")

print("="*110)
print(f"TOTALS     | {'All 16 Benchmark Scenes':<45} | {total_gt:<4} | {sum(s['detections_count'] for s in b_data['per_image_results']):<8} | {total_p2_dets:<8} | {p1_giants_total:<8} | {p2_giants_total:<8}")
print("="*110)
