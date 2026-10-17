import os
import sys
import json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(_dir, "evaluation_results", "phase2_5_full_validation", "raw_outputs")
baseline_path = os.path.join(_dir, "evaluation_results", "baseline_grounding_benchmark", "grounding_dino_baseline_benchmark_results.json")

with open(baseline_path, "r", encoding="utf-8") as f:
    baseline_data = json.load(f)

# Ground truth counts and categories for all 26 scenes
SCENE_GT_SPECS = {
    "scene_01": {"name": "High-Density Urban Residential", "cat": "buildings", "count": 16, "size": "small", "gt_status": "BENCHMARK_GT"},
    "scene_02": {"name": "Suburban Single-Family Homes", "cat": "houses", "count": 9, "size": "medium", "gt_status": "BENCHMARK_GT"},
    "scene_03": {"name": "Industrial Logistics Warehouses", "cat": "buildings", "count": 6, "size": "large", "gt_status": "BENCHMARK_GT"},
    "scene_04": {"name": "Highway Interchange & Arterial Network", "cat": "roads", "count": 2, "size": "large", "gt_status": "BENCHMARK_GT"},
    "scene_05": {"name": "Commercial Parking Lot Vehicles", "cat": "vehicles", "count": 24, "size": "small", "gt_status": "BENCHMARK_GT"},
    "scene_06": {"name": "Highway Vehicle Traffic", "cat": "vehicles", "count": 8, "size": "small", "gt_status": "BENCHMARK_GT"},
    "scene_07": {"name": "Commercial Seaport Cargo Ships", "cat": "ships", "count": 4, "size": "large", "gt_status": "BENCHMARK_GT"},
    "scene_08": {"name": "Marina Harbor Boats", "cat": "boats", "count": 12, "size": "small", "gt_status": "BENCHMARK_GT"},
    "scene_09": {"name": "Regional Airport Apron & Airplanes", "cat": "airplanes", "count": 4, "size": "medium", "gt_status": "BENCHMARK_GT"},
    "scene_10": {"name": "Segmented Agricultural Crop Plots", "cat": "agricultural structures", "count": 6, "size": "large", "gt_status": "BENCHMARK_GT"},
    "scene_11": {"name": "Rural Farmhouses & Silo Structures", "cat": "houses", "count": 4, "size": "medium", "gt_status": "BENCHMARK_GT"},
    "scene_12": {"name": "Coastal Shoreline & Water Estuary", "cat": "water/coastline", "count": 1, "size": "large", "gt_status": "BENCHMARK_GT"},
    "scene_13": {"name": "River Waterway & Highway Bridge Crossing", "cat": "bridges", "count": 2, "size": "large", "gt_status": "BENCHMARK_GT"},
    "scene_14": {"name": "Commercial Retail Center Buildings", "cat": "buildings", "count": 5, "size": "medium", "gt_status": "BENCHMARK_GT"},
    "scene_15": {"name": "Dense Forest Canopy Reserve", "cat": "vegetation", "count": 1, "size": "large", "gt_status": "BENCHMARK_GT"},
    "scene_16": {"name": "High-Res Dense Settlement Buildings", "cat": "buildings", "count": 36, "size": "small", "gt_status": "BENCHMARK_GT"},
    "scene_17": {"name": "Mountain Ridgeline Telecom Towers", "cat": "towers", "count": 4, "size": "medium", "gt_status": "VALIDATION_GT"},
    "scene_18": {"name": "Offshore Wind Turbine Farm", "cat": "arbitrary_targets", "count": 6, "size": "medium", "gt_status": "VALIDATION_GT"},
    "scene_19": {"name": "Desert Solar Photovoltaic Farm", "cat": "arbitrary_targets", "count": 8, "size": "large", "gt_status": "VALIDATION_GT"},
    "scene_20": {"name": "Petrochemical Refinery Storage Tanks", "cat": "arbitrary_targets", "count": 6, "size": "large", "gt_status": "VALIDATION_GT"},
    "scene_21": {"name": "Urban Construction Site Cranes", "cat": "arbitrary_targets", "count": 3, "size": "medium", "gt_status": "VALIDATION_GT"},
    "scene_22": {"name": "Railway Marshalling Yard Freight Cars", "cat": "arbitrary_targets", "count": 4, "size": "large", "gt_status": "VALIDATION_GT"},
    "scene_23": {"name": "Intermodal Shipping Container Terminal", "cat": "containers", "count": 20, "size": "medium", "gt_status": "VALIDATION_GT"},
    "scene_24": {"name": "Deepwater Port Multi-Target", "cat": "multi_target", "count": 8, "size": "large", "gt_status": "VALIDATION_GT"},
    "scene_25": {"name": "Urban Transit Hub Multi-Target", "cat": "multi_target", "count": 7, "size": "medium", "gt_status": "VALIDATION_GT"},
    "scene_26": {"name": "Agricultural Farm Silos & Barns", "cat": "agricultural structures", "count": 5, "size": "large", "gt_status": "VALIDATION_GT"},
}

raw_files = sorted(os.listdir(raw_dir))

scene_results = []
category_stats = {}
size_stats = {
    "small": {"gt": 0, "detections": 0, "scores": []},
    "medium": {"gt": 0, "detections": 0, "scores": []},
    "large": {"gt": 0, "detections": 0, "scores": []}
}

total_gt = 0
total_dets = 0
all_scores = []
all_latencies = []

for rf in raw_files:
    if not rf.endswith(".json"):
        continue
    fpath = os.path.join(raw_dir, rf)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    scene_id = rf.split("_")[0] + "_" + rf.split("_")[1]
    spec = SCENE_GT_SPECS[scene_id]
    gt_c = spec["count"]
    cat = spec["cat"]
    sz = spec["size"]
    
    dets = data["detections"]
    det_c = len(dets)
    lat = data["performance"]["latency_ms"]
    scores = [d["score"] for d in dets]
    
    total_gt += gt_c
    total_dets += det_c
    all_scores.extend(scores)
    all_latencies.append(lat)
    
    # Size breakdown based on pixel box sizes of detections
    for d in dets:
        b = d["box"]
        bw = (b[2] - b[0]) / 1000.0 * data["image_parameters"]["dimensions"][0]
        bh = (b[3] - b[1]) / 1000.0 * data["image_parameters"]["dimensions"][1]
        pixel_extent = max(bw, bh)
        if pixel_extent < 32:
            d_sz = "small"
        elif pixel_extent <= 96:
            d_sz = "medium"
        else:
            d_sz = "large"
        size_stats[d_sz]["detections"] += 1
        size_stats[d_sz]["scores"].append(d["score"])
        
    size_stats[sz]["gt"] += gt_c
    
    # Category stats
    c_rec = category_stats.setdefault(cat, {"gt": 0, "dets": 0, "scenes": 0, "scores": [], "latencies": []})
    c_rec["gt"] += gt_c
    c_rec["dets"] += det_c
    c_rec["scenes"] += 1
    c_rec["scores"].extend(scores)
    c_rec["latencies"].append(lat)
    
    scene_results.append({
        "scene_id": scene_id,
        "name": spec["name"],
        "category": cat,
        "gt_count": gt_c,
        "detections_count": det_c,
        "mean_score": np.mean(scores) if scores else 0.0,
        "median_score": np.median(scores) if scores else 0.0,
        "latency_ms": lat,
        "gt_status": spec["gt_status"]
    })

print("=== RECONCILED TOTALS ===")
print(f"Total Scenes: {len(scene_results)}")
print(f"Total Ground Truth: {total_gt}")
print(f"Total Detections Extracted: {total_dets}")

print("\n=== OBJECT SIZE BREAKDOWN (Pixel Extent of Detections) ===")
for sz_k, sz_v in size_stats.items():
    print(f"{sz_k.upper():8s} | GT: {sz_v['gt']:3d} | Detections: {sz_v['detections']:3d} | Mean Conf: {np.mean(sz_v['scores']) if sz_v['scores'] else 0.0:.4f}")

print("\n=== PER-CATEGORY BREAKDOWN ===")
for c_k, c_v in sorted(category_stats.items()):
    print(f"{c_k:24s} | Scenes: {c_v['scenes']:2d} | GT: {c_v['gt']:3d} | Dets: {c_v['dets']:3d} | Mean Conf: {np.mean(c_v['scores']) if c_v['scores'] else 0.0:.4f} | Mean Latency: {np.mean(c_v['latencies']):.0f}ms")
