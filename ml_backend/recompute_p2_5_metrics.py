import os
import sys
import json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(_dir, "evaluation_results", "phase2_5_full_validation", "raw_outputs")
master_path = os.path.join(_dir, "evaluation_results", "phase2_5_full_validation", "phase2_5_master_results.json")
baseline_path = os.path.join(_dir, "evaluation_results", "baseline_grounding_benchmark", "grounding_dino_baseline_benchmark_results.json")

with open(baseline_path, "r", encoding="utf-8") as f:
    baseline_data = json.load(f)

raw_files = sorted(os.listdir(raw_dir))
print(f"Found {len(raw_files)} raw operation files.")

all_ops = []
all_detections = []
all_scores = []
all_latencies = []
scene_stats = []

for rf in raw_files:
    if not rf.endswith(".json"):
        continue
    fpath = os.path.join(raw_dir, rf)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    scene_id = rf.split("_")[0] + "_" + rf.split("_")[1]
    query = data["query_parameters"]["natural_language_query"]
    target_category = data["query_parameters"]["target_category"]
    dets = data["detections"]
    latency = data["performance"]["latency_ms"]
    
    all_latencies.append(latency)
    all_ops.append(data)
    
    for d in dets:
        all_detections.append(d)
        all_scores.append(d["score"])
        
    scene_stats.append({
        "scene_id": scene_id,
        "filename": data["image_parameters"]["filename"],
        "dimensions": data["image_parameters"]["dimensions"],
        "aspect_ratio": data["image_parameters"]["aspect_ratio"],
        "tiles_processed": data["image_parameters"]["tiles_processed"],
        "query": query,
        "target_category": target_category,
        "detections_count": len(dets),
        "scores": [d["score"] for d in dets],
        "latency_ms": latency
    })

print(f"Total Operations: {len(all_ops)}")
print(f"Total Raw Detections Extracted: {len(all_detections)}")
print(f"Latency: Mean = {np.mean(all_latencies):.2f} ms | Median = {np.median(all_latencies):.2f} ms | Min = {np.min(all_latencies):.2f} ms | Max = {np.max(all_latencies):.2f} ms")
print(f"Confidence: Mean = {np.mean(all_scores):.4f} | Median = {np.median(all_scores):.4f} | Min = {np.min(all_scores):.4f} | Max = {np.max(all_scores):.4f}")
print(f"Average Detections per Image: {len(all_detections) / len(all_ops):.2f}")
