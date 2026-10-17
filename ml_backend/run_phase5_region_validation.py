"""
run_phase5_region_validation.py
--------------------------------
Phase 5 Failure Investigation & Region-Feature Validation Suite.
Audits the root cause of the "is there a water body in the image if yes then locate it" failure,
evaluates generalized region detection (water bodies, vegetation, agriculture, construction, industrial, ports)
and hard-negative distractor pairs (water vs road, water vs rooftop, water vs shadow, water vs vegetation, etc.).
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image, ImageDraw

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from router import extract_grounding_target, get_canonical_target_label, IntentClassifier, get_target_geometry
from tools import GroundingTool
from tiling import compute_iou

RESULTS_DIR = os.path.join(_dir, "evaluation_results", "phase5_region_audit")
VISUALS_DIR = os.path.join(RESULTS_DIR, "phase5_region_visual_outputs")
TEST_IMAGES_DIR = os.path.join(RESULTS_DIR, "region_test_images")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

def build_region_validation_dataset():
    """Builds a diverse validation set for region features and hard-negative distractors."""
    scenes = []

    def add_scene(scene_id, name, img, query, target_cat, geom_type, gt_boxes, is_hard_neg=False, neg_target=None):
        filename = f"{scene_id}_{target_cat}.png"
        path = os.path.join(TEST_IMAGES_DIR, filename)
        img.save(path)
        scenes.append({
            "scene_id": scene_id,
            "name": name,
            "filename": filename,
            "path": path,
            "query": query,
            "target_category": target_cat,
            "geometry_type": geom_type,
            "gt_boxes": gt_boxes,
            "gt_count": len(gt_boxes),
            "is_hard_negative": is_hard_neg,
            "negative_target": neg_target,
            "width": img.width,
            "height": img.height
        })

    # 1. Primary Target Query: Unseen Coastal / Harbor Water Scene (Water body clearly present)
    img = Image.new("RGB", (768, 768), (110, 115, 110))
    draw = ImageDraw.Draw(img)
    # Coastal ocean / water body filling bottom-right half
    draw.polygon([(0, 300), (768, 150), (768, 768), (0, 768)], fill=(30, 70, 115), outline=(20, 50, 80))
    # Urban coastline strip on top-left
    draw.rectangle([50, 50, 180, 180], fill=(180, 80, 60))
    draw.rectangle([250, 40, 400, 160], fill=(160, 160, 170))
    gt_b = [[0, int(200/768*1000), 1000, 1000]]
    add_scene("reg_01_water", "Coastal Ocean & Harbor Water", img, "is there a water body in the image if yes then locate it", "water_body", "region", gt_b)

    # 2. Inland Lake
    img = Image.new("RGB", (512, 512), (110, 130, 90)) # grass
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 80, 430, 430], fill=(25, 65, 110), outline=(20, 50, 80), width=3)
    gt_b = [[int(80/512*1000), int(80/512*1000), int(430/512*1000), int(430/512*1000)]]
    add_scene("reg_02_lake", "Inland Freshwater Lake", img, "Locate water body in the landscape.", "water_body", "region", gt_b)

    # 3. River Channel & Estuary
    img = Image.new("RGB", (512, 512), (100, 110, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 180, 512, 330], fill=(35, 75, 120), outline=(20, 50, 80))
    gt_b = [[0, int(180/512*1000), 1000, int(330/512*1000)]]
    add_scene("reg_03_river", "Navigable River Channel", img, "Find the river and water bodies.", "water_body", "region", gt_b)

    # 4. Dense Forest Vegetation Canopy
    img = Image.new("RGB", (512, 512), (130, 120, 100))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 460, 460], fill=(30, 95, 40), outline=(20, 60, 30))
    gt_b = [[int(50/512*1000), int(50/512*1000), int(460/512*1000), int(460/512*1000)]]
    add_scene("reg_04_forest", "Dense Forest Canopy", img, "Locate vegetation in the scene.", "vegetation", "region", gt_b)

    # 5. Agricultural Crop Plots
    img = Image.new("RGB", (512, 512), (120, 120, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 60, 450, 450], fill=(160, 180, 70), outline=(80, 90, 30))
    gt_b = [[int(60/512*1000), int(60/512*1000), int(450/512*1000), int(450/512*1000)]]
    add_scene("reg_05_agri", "Cultivated Agricultural Farmland", img, "Find agricultural fields.", "agriculture", "region", gt_b)

    # 6. Construction Site Earthworks
    img = Image.new("RGB", (512, 512), (110, 110, 110))
    draw = ImageDraw.Draw(img)
    draw.rectangle([70, 70, 440, 440], fill=(185, 145, 85), outline=(130, 95, 40), width=3)
    gt_b = [[int(70/512*1000), int(70/512*1000), int(440/512*1000), int(440/512*1000)]]
    add_scene("reg_06_construction", "Excavation Construction Site", img, "Locate construction sites.", "construction_sites", "region", gt_b)

    # 7-12. Hard Negative Distractor Tests
    # 7. Water vs Road (Asphalt road intersection distractor)
    img = Image.new("RGB", (512, 512), (100, 120, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 220, 512, 290], fill=(50, 50, 55))
    draw.rectangle([220, 0, 290, 512], fill=(50, 50, 55))
    add_scene("reg_07_hn_road", "Asphalt Highway (Hard Neg for Water)", img, "is there a water body in the image if yes then locate it", "water_body", "region", [], is_hard_neg=True, neg_target="road")

    # 8. Water vs Dark Rooftop
    img = Image.new("RGB", (512, 512), (120, 120, 120))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 60, 450, 450], fill=(45, 45, 55), outline=(20, 20, 20), width=2)
    add_scene("reg_08_hn_rooftop", "Dark Commercial Rooftop (Hard Neg for Water)", img, "Locate water body.", "water_body", "region", [], is_hard_neg=True, neg_target="rooftop")

    # 9. Water vs Deep Cloud/Building Shadow
    img = Image.new("RGB", (512, 512), (140, 130, 110))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 400, 400], fill=(25, 25, 30)) # deep black shadow
    add_scene("reg_09_hn_shadow", "Deep Structural Shadow (Hard Neg for Water)", img, "Find water bodies.", "water_body", "region", [], is_hard_neg=True, neg_target="shadow")

    # 10. Water vs Dense Vegetation
    img = Image.new("RGB", (512, 512), (130, 110, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 460, 460], fill=(25, 90, 35))
    add_scene("reg_10_hn_veg", "Green Vegetation Canopy (Hard Neg for Water)", img, "Locate water body.", "water_body", "region", [], is_hard_neg=True, neg_target="vegetation")

    # 11. Water vs Dark Industrial Yard
    img = Image.new("RGB", (512, 512), (90, 90, 90))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 80, 430, 430], fill=(60, 60, 65))
    add_scene("reg_11_hn_industrial", "Dark Asphalt Yard (Hard Neg for Water)", img, "Find water body.", "water_body", "region", [], is_hard_neg=True, neg_target="industrial_surface")

    # 12. Water vs Parking Lot
    img = Image.new("RGB", (512, 512), (100, 100, 100))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 60, 450, 450], fill=(70, 70, 75))
    add_scene("reg_12_hn_parking", "Concrete Parking Lot (Hard Neg for Water)", img, "Locate water body.", "water_body", "region", [], is_hard_neg=True, neg_target="parking_lot")

    return scenes

def execute_region_audit():
    print("=" * 95)
    print("  [PHASE 5 FAILURE INVESTIGATION] REGION DETECTION & HARD-NEGATIVE AUDIT")
    print("=" * 95)

    scenes = build_region_validation_dataset()
    tool = GroundingTool()

    results = []
    total_gt = sum(s["gt_count"] for s in scenes)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    hard_neg_suppressed = 0
    hard_neg_count = 0

    print(f"Loaded {len(scenes)} region test scenes ({total_gt} GT regions).\n")

    for s in scenes:
        img = Image.open(s["path"]).convert("RGB")
        q = s["query"]
        extracted_t = extract_grounding_target(q)
        canonical_l = get_canonical_target_label(extracted_t)
        geom = get_target_geometry(extracted_t)

        t0 = time.perf_counter()
        tool_res = tool._run({"image": img, "target_phrase": extracted_t})
        lat = (time.perf_counter() - t0) * 1000.0
        dets = tool_res.data.get("detections", [])

        # Bipartite match with IoU >= 0.50
        matched_gt = set()
        matched_pred = set()
        matches = []
        for p_i, p in enumerate(dets):
            p_box = p["box"]
            best_iou = 0.0
            best_g = -1
            for g_i, g_box in enumerate(s["gt_boxes"]):
                if g_i in matched_gt:
                    continue
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_g = g_i
            if best_iou >= 0.50 and best_g != -1:
                matched_gt.add(best_g)
                matched_pred.add(p_i)
                matches.append((p_i, best_g, best_iou, p))

        tp = len(matches)
        fp = len(dets) - tp
        fn = len(s["gt_boxes"]) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if s["is_hard_negative"]:
            hard_neg_count += 1
            if fp == 0:
                hard_neg_suppressed += 1

        # Generate Visual Artifact
        vis_img = Image.new("RGB", (img.width * 2, img.height), (15, 20, 25))
        vis_img.paste(img, (0, 0))
        vis_img.paste(img, (img.width, 0))
        draw = ImageDraw.Draw(vis_img)

        # Left: GT
        for gb in s["gt_boxes"]:
            x0 = int(gb[0] / 1000.0 * img.width)
            y0 = int(gb[1] / 1000.0 * img.height)
            x1 = int(gb[2] / 1000.0 * img.width)
            y1 = int(gb[3] / 1000.0 * img.height)
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
        draw.text((10, 10), f"GROUND TRUTH ({s['gt_count']} regions)", fill=(0, 255, 0))

        # Right: Predictions
        for p in dets:
            pb = p["box"]
            x0 = int(pb[0] / 1000.0 * img.width) + img.width
            y0 = int(pb[1] / 1000.0 * img.height)
            x1 = int(pb[2] / 1000.0 * img.width) + img.width
            y1 = int(pb[3] / 1000.0 * img.height)
            draw.rectangle([x0, y0, x1, y1], outline=(0, 220, 255), width=3)
        draw.text((img.width + 10, 10), f"PREDICTIONS (TP:{tp}, FP:{fp})", fill=(0, 220, 255))

        vis_path = os.path.join(VISUALS_DIR, f"{s['scene_id']}_overlay.png")
        vis_img.save(vis_path)

        rec = {
            "scene_id": s["scene_id"],
            "scene_name": s["name"],
            "raw_query": q,
            "extracted_target": extracted_t,
            "canonical_label": canonical_l,
            "geometry_type": geom.value,
            "gt_count": s["gt_count"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "detections_count": len(dets),
            "latency_ms": round(lat, 2),
            "visual_artifact": vis_path
        }
        results.append(rec)
        print(f"  [{s['scene_id']}] {s['name'][:30]:30s} | Extracted: '{extracted_t}' ({geom.value}) | GT:{s['gt_count']} | TP:{tp} | FP:{fp} | Lat:{lat:6.1f}ms")

    prec = total_tp / max(1, total_tp + total_fp)
    rec = total_tp / max(1, total_gt)
    f1 = (2 * prec * rec) / max(1e-5, prec + rec)

    print("\n" + "=" * 95)
    print("  [REGION VALIDATION & AUDIT SUMMARY]")
    print("=" * 95)
    print(f"  Total Ground Truth Regions:       {total_gt}")
    print(f"  True Positives (IoU >= 0.50):     {total_tp} / {total_gt} ({rec*100:.2f}% Recall)")
    print(f"  False Positives:                  {total_fp}")
    print(f"  Precision:                        {prec*100:.2f}%")
    print(f"  F1-Score:                         {f1*100:.2f}%")
    print(f"  Hard-Negative Suppression Rate:   {hard_neg_suppressed} / {hard_neg_count} ({hard_neg_suppressed/max(1, hard_neg_count)*100:.1f}%)")
    print("=" * 95)

    # Save JSON Report
    out_json = os.path.join(RESULTS_DIR, "phase5_region_validation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": "2026-09-15T13:50:00Z",
            "benchmark_summary": {
                "total_gt": total_gt,
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "hard_negative_suppression_rate": hard_neg_suppressed / max(1, hard_neg_count)
            },
            "scenes": results
        }, f, indent=2)

    print(f"\n  [OK] Region Validation Report saved to: {out_json}\n")

if __name__ == "__main__":
    execute_region_audit()
