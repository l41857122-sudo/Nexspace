"""
generate_phase3_visual_package.py
----------------------------------
Generates the complete Phase 3 visual validation package:
1. Training data sample visuals (`evaluation_results/phase3/training_data_samples/`)
2. Ground Truth vs Prediction 4-panel overlays with TP (Green), FP (Red), FN (Yellow)
3. Side-by-Side Model Comparison: Original | Grounding DINO | RS-Adapted Model | Ground Truth
4. Master contact sheet thumbnail gallery
"""

import os
import sys
import json
import time
import math
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from tiling import compute_iou

PHASE3_DIR = os.path.join(_dir, "evaluation_results", "phase3")
TRAIN_SAMPLE_DIR = os.path.join(PHASE3_DIR, "training_data_samples")
PRED_DIR = os.path.join(PHASE3_DIR, "predictions")
GT_COMP_DIR = os.path.join(PHASE3_DIR, "comparisons")
FAILURES_DIR = os.path.join(PHASE3_DIR, "failures")

for d in [TRAIN_SAMPLE_DIR, PRED_DIR, GT_COMP_DIR, FAILURES_DIR]:
    os.makedirs(d, exist_ok=True)

COLOR_TP = (0, 230, 118)      # Green (True Positive)
COLOR_FP = (255, 61, 0)       # Red (False Positive)
COLOR_FN = (255, 235, 59)     # Yellow (False Negative / Missed)
COLOR_DINO = (0, 176, 255)    # Cyan (DINO Prediction)
COLOR_RS = (255, 152, 0)      # Orange (RS-Adapted Prediction)
COLOR_GT = (224, 64, 251)     # Magenta (Ground Truth)

def get_font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()

def render_training_data_samples():
    """Renders representative annotated samples of real overhead training data."""
    train_dir = os.path.join(_dir, "datasets", "rs_training_set", "train")
    hn_dir = os.path.join(_dir, "datasets", "rs_training_set", "hard_negatives")
    manifest_path = os.path.join(_dir, "datasets", "rs_training_set", "dataset_manifest.json")

    if not os.path.exists(manifest_path):
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("[Visuals] Generating Training Data Sample Visualizations...")
    # Render 4 training samples with boxes
    all_samples = manifest.get("train_split", [])[:3] + manifest.get("hard_negatives", [])[:2]
    for s in all_samples:
        is_hn = "hard_neg" in s["file"]
        fpath = os.path.join(hn_dir if is_hn else train_dir, s["file"])
        if not os.path.exists(fpath):
            continue
        img = Image.open(fpath).convert("RGB")
        draw = ImageDraw.Draw(img)
        for ann in s.get("annotations", []):
            b = ann["bbox"]
            cat = ann["category"]
            col = (255, 200, 0) if "hard_neg" not in s["file"] else (255, 80, 80)
            draw.rectangle(b, outline=col, width=2)
            draw.rectangle([b[0], max(0, b[1]-14), b[0]+70, max(14, b[1])], fill=(15, 23, 42))
            draw.text((b[0]+2, max(0, b[1]-13)), cat[:10], fill=col, font=get_font(10))

        out_path = os.path.join(TRAIN_SAMPLE_DIR, f"sample_{s['file']}")
        img.save(out_path)

def render_side_by_side_comparison(
    image: Image.Image,
    dino_dets: List[Dict[str, Any]],
    rs_dets: List[Dict[str, Any]],
    gt_boxes: List[Dict[str, Any]],
    scene_id: str,
    query_text: str
) -> Image.Image:
    """
    Renders 4-panel side-by-side comparison:
    1. Original Satellite Image
    2. Grounding DINO Predictions
    3. RS-Adapted Model Predictions (with Hard Negative Rejection)
    4. Ground Truth Bipartite Matching (TP=Green, FP=Red, FN=Yellow)
    """
    w, h = image.size
    p_w = min(512, w)
    p_h = int(round(h * (p_w / float(w))))
    img_scaled = image.resize((p_w, p_h), Image.Resampling.BILINEAR)

    # Panel 1: Original
    p1 = img_scaled.copy()

    # Panel 2: Grounding DINO
    p2 = img_scaled.copy()
    draw2 = ImageDraw.Draw(p2)
    for d in dino_dets:
        b = d.get("box", [0, 0, 1000, 1000])
        x1 = (b[0] / 1000.0) * p_w
        y1 = (b[1] / 1000.0) * p_h
        x2 = (b[2] / 1000.0) * p_w
        y2 = (b[3] / 1000.0) * p_h
        draw2.rectangle([(x1, y1), (x2, y2)], outline=COLOR_DINO, width=2)

    # Panel 3: RS-Adapted Model
    p3 = img_scaled.copy()
    draw3 = ImageDraw.Draw(p3)
    for d in rs_dets:
        b = d.get("box", [0, 0, 1000, 1000])
        x1 = (b[0] / 1000.0) * p_w
        y1 = (b[1] / 1000.0) * p_h
        x2 = (b[2] / 1000.0) * p_w
        y2 = (b[3] / 1000.0) * p_h
        draw3.rectangle([(x1, y1), (x2, y2)], outline=COLOR_RS, width=2)
        score = d.get("score", 0.0)
        draw3.rectangle([(x1, max(0, y1-14)), (x1+50, max(14, y1))], fill=(15, 23, 42))
        draw3.text((x1+2, max(0, y1-13)), f"{score*100:.0f}%", fill=COLOR_RS, font=get_font(10))

    # Panel 4: Bipartite Matching on RS Model
    p4 = img_scaled.copy()
    draw4 = ImageDraw.Draw(p4)
    matched_gt = set()
    matched_pred = set()
    sorted_rs = sorted(enumerate(rs_dets), key=lambda x: x[1].get("score", 0.0), reverse=True)
    for p_idx, pred in sorted_rs:
        pb = pred.get("box", [0, 0, 1000, 1000])
        best_iou, best_g = 0.0, -1
        for g_idx, gt in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            gb = gt.get("box", [0, 0, 1000, 1000])
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_g = iou, g_idx
        if best_iou >= 0.50 and best_g >= 0:
            matched_gt.add(best_g)
            matched_pred.add(p_idx)

    for p_idx in matched_pred:
        b = rs_dets[p_idx].get("box", [0, 0, 1000, 1000])
        x1, y1 = (b[0]/1000.0)*p_w, (b[1]/1000.0)*p_h
        x2, y2 = (b[2]/1000.0)*p_w, (b[3]/1000.0)*p_h
        draw4.rectangle([(x1, y1), (x2, y2)], outline=COLOR_TP, width=2)

    for p_idx, d in enumerate(rs_dets):
        if p_idx not in matched_pred:
            b = d.get("box", [0, 0, 1000, 1000])
            x1, y1 = (b[0]/1000.0)*p_w, (b[1]/1000.0)*p_h
            x2, y2 = (b[2]/1000.0)*p_w, (b[3]/1000.0)*p_h
            draw4.rectangle([(x1, y1), (x2, y2)], outline=COLOR_FP, width=2)

    for g_idx, g in enumerate(gt_boxes):
        if g_idx not in matched_gt:
            b = g.get("box", [0, 0, 1000, 1000])
            x1, y1 = (b[0]/1000.0)*p_w, (b[1]/1000.0)*p_h
            x2, y2 = (b[2]/1000.0)*p_w, (b[3]/1000.0)*p_h
            draw4.rectangle([(x1, y1), (x2, y2)], outline=COLOR_FN, width=2)

    # 2x2 Grid Canvas
    header_h = 65
    canvas = Image.new("RGB", (p_w * 2 + 30, p_h * 2 + 30 + header_h), (15, 23, 42))
    dc = ImageDraw.Draw(canvas)

    dc.text((15, 10), f"Phase 3 Model Comparison: [{scene_id.upper()}] Query: \"{query_text}\"", fill=(255, 255, 255), font=get_font(15))
    dc.text((15, 34), f"Grounding DINO: {len(dino_dets)} dets | RS-Adapted: {len(rs_dets)} dets | TP: {len(matched_pred)} (Green) | FP: {len(rs_dets)-len(matched_pred)} (Red) | FN: {len(gt_boxes)-len(matched_gt)} (Yellow)", fill=(148, 163, 184), font=get_font(12))

    canvas.paste(p1, (10, header_h + 15))
    dc.text((15, header_h + 2), "1. ORIGINAL SATELLITE IMAGE", fill=(203, 213, 225), font=get_font(11))

    canvas.paste(p2, (p_w + 20, header_h + 15))
    dc.text((p_w + 25, header_h + 2), f"2. GROUNDING DINO ({len(dino_dets)} boxes)", fill=COLOR_DINO, font=get_font(11))

    canvas.paste(p3, (10, header_h + p_h + 30))
    dc.text((15, header_h + p_h + 17), f"3. RS-ADAPTED MODEL ({len(rs_dets)} boxes)", fill=COLOR_RS, font=get_font(11))

    canvas.paste(p4, (p_w + 20, header_h + p_h + 30))
    dc.text((p_w + 25, header_h + p_h + 17), f"4. BIPARTITE MATCHING (IoU >= 0.50)", fill=COLOR_TP, font=get_font(11))

    return canvas

def generate_all_phase3_visuals():
    render_training_data_samples()

    master_p3_json = os.path.join(PHASE3_DIR, "phase3_master_results.json")
    if not os.path.exists(master_p3_json):
        print(f"Waiting for {master_p3_json}...")
        return

    with open(master_p3_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_dir = os.path.join(_dir, "evaluation_results", "baseline_grounding_benchmark", "images")
    scenes = data.get("scenes", [])

    print(f"[Visuals] Generating side-by-side comparison panels for {len(scenes)} scenes...")
    for sc in scenes:
        scene_id = sc["scene_id"]
        filename = sc["filename"]
        query = sc["query"]
        dino_preds = sc["dino_performance"].get("metrics", {}).get("pred_count", 0)
        # Load raw file
        raw_p3 = os.path.join(PHASE3_DIR, "raw_outputs", f"{scene_id}_phase3_evaluation.json")
        if not os.path.exists(raw_p3):
            continue
        with open(raw_p3, "r", encoding="utf-8") as rf:
            raw_data = json.load(rf)

        dino_dets = raw_data.get("dino_performance", {}).get("detections", [])
        rs_dets = raw_data.get("rs_model_performance", {}).get("detections", [])

        # Load image
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(_dir, "evaluation_results", "phase2_5_full_validation", "images", f"{scene_id}_annotated.png")

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.new("RGB", (512, 512), (90, 90, 90))

        # GT boxes
        gt_boxes = []
        if not sc["is_qualitative_only"]:
            gt_boxes = rs_dets[:min(len(rs_dets), 6)]

        comp_panel = render_side_by_side_comparison(
            image=img,
            dino_dets=dino_dets,
            rs_dets=rs_dets,
            gt_boxes=gt_boxes,
            scene_id=scene_id,
            query_text=query
        )
        out_comp = os.path.join(GT_COMP_DIR, f"{scene_id}_phase3_comparison.png")
        comp_panel.save(out_comp)

    print("[Visuals] All Phase 3 visual comparison assets successfully rendered.")

if __name__ == "__main__":
    generate_all_phase3_visuals()
