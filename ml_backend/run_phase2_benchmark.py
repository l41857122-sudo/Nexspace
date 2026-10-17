import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)
for p in [_dir, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_backend.tiling import tiled_inference_engine, compute_iou
from ml_backend.model_runtime import GroundingDINORuntime
from ml_backend.router import build_generalized_aerial_prompt, MACRO_EXTENDED_CATEGORIES


def match_boxes_iou(gt_boxes: List[Dict[str, Any]], pred_boxes: List[Dict[str, Any]], iou_threshold: float = 0.50) -> Tuple[int, int, int, float]:
    """Greedy bipartite matching between GT and predicted bounding boxes."""
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0, 1.0
    if not gt_boxes:
        return 0, len(pred_boxes), 0, 0.0
    if not pred_boxes:
        return 0, 0, len(gt_boxes), 0.0

    matched_gt = set()
    matched_pred = set()
    iou_sum = 0.0

    # Sort predictions by confidence score descending
    sorted_preds = sorted(enumerate(pred_boxes), key=lambda x: x[1].get("score", 0.0), reverse=True)

    for p_idx, pred in sorted_preds:
        p_box = pred.get("box") or pred.get("bbox_normalized")
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, gt in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            g_box = gt.get("box") or gt.get("bbox_normalized")
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matched_pred.add(p_idx)
            iou_sum += best_iou

    tp = len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    mean_iou = iou_sum / max(1, tp)

    return tp, fp, fn, mean_iou


def categorize_box_size(box: List[int], img_w: int, img_h: int) -> str:
    """Classifies box into small, medium, or large based on area in 1000x1000 space."""
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = (w / 1000.0) * (h / 1000.0)
    if area < 0.02:
        return "small"
    elif area < 0.10:
        return "medium"
    else:
        return "large"


def run_phase2_benchmark():
    baseline_json_path = os.path.join(
        _dir, "evaluation_results", "baseline_grounding_benchmark", "grounding_dino_baseline_benchmark_results.json"
    )
    images_dir = os.path.join(_dir, "evaluation_results", "baseline_grounding_benchmark", "images")
    output_dir = os.path.join(_dir, "evaluation_results", "phase2_grounding_benchmark")
    os.makedirs(output_dir, exist_ok=True)

    with open(baseline_json_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    runtime = GroundingDINORuntime()
    runtime.load()

    scenes = baseline_data["per_image_results"]
    print(f"Loaded {len(scenes)} benchmark scenes. Initializing Phase 2 Tiled Inference Engine...")

    total_gt = 0
    total_tp_50 = 0
    total_fp_50 = 0
    total_fn_50 = 0

    size_stats = {
        "small": {"gt": 0, "tp": 0, "fp": 0, "fn": 0},
        "medium": {"gt": 0, "tp": 0, "fp": 0, "fn": 0},
        "large": {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
    }

    category_stats: Dict[str, Dict[str, int]] = {}
    phase2_scene_results = []
    all_confidences = []
    giant_boxes_phase2 = []

    t_start = time.perf_counter()

    for idx, scene in enumerate(scenes):
        scene_id = scene["scene_id"]
        scene_name = scene["scene_name"]
        category = scene["category"]
        target_phrase = scene["target_phrase"]
        gt_objects = scene.get("ground_truth_objects", [])

        # Match image file
        img_name = [
            f for f in os.listdir(images_dir)
            if f.startswith(scene_id) or scene_name.lower().replace(" ", "_") in f.lower()
        ][0]
        img_path = os.path.join(images_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Run Phase 2 Tiled Grounding
        t0 = time.perf_counter()
        res = tiled_inference_engine.run_tiled_grounding(
            image=img,
            target_phrase=target_phrase,
            grounding_runtime=runtime,
            tile_size=512,
            overlap=0.20,
            iou_threshold=0.45,
            box_threshold=0.20,
            text_threshold=0.20,
            max_area_ratio=0.35,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections = res["detections"]
        scores = [d["score"] for d in detections]
        all_confidences.extend(scores)

        # Match at IoU 0.50
        tp, fp, fn, mean_iou = match_boxes_iou(gt_objects, detections, iou_threshold=0.50)
        tp30, fp30, fn30, mean_iou30 = match_boxes_iou(gt_objects, detections, iou_threshold=0.30)

        total_gt += len(gt_objects)
        total_tp_50 += tp
        total_fp_50 += fp
        total_fn_50 += fn

        # Size breakdown
        for gt in gt_objects:
            sz = categorize_box_size(gt["box"], w, h)
            size_stats[sz]["gt"] += 1

        # Check matched GTs for size assignment
        # For simplicity, assign TP by IoU matching per size
        for sz in ["small", "medium", "large"]:
            sz_gts = [g for g in gt_objects if categorize_box_size(g["box"], w, h) == sz]
            sz_preds = [d for d in detections if categorize_box_size(d["box"], w, h) == sz]
            sz_tp, sz_fp, sz_fn, _ = match_boxes_iou(sz_gts, detections, iou_threshold=0.50)
            size_stats[sz]["tp"] += sz_tp
            size_stats[sz]["fn"] += (len(sz_gts) - sz_tp)
            size_stats[sz]["fp"] += sz_fp

        # Category breakdown
        cat_data = category_stats.setdefault(category, {"gt": 0, "tp": 0, "fp": 0, "fn": 0})
        cat_data["gt"] += len(gt_objects)
        cat_data["tp"] += tp
        cat_data["fp"] += fp
        cat_data["fn"] += fn

        # Check for giant boxes (> 0.35 area)
        scene_giant = []
        is_macro = any(m in target_phrase.lower() for m in MACRO_EXTENDED_CATEGORIES)
        if not is_macro:
            for d in detections:
                bx = d["box"]
                bw = (bx[2] - bx[0]) / 1000.0
                bh = (bx[3] - bx[1]) / 1000.0
                area_frac = bw * bh
                if area_frac > 0.35:
                    scene_giant.append({"box": bx, "score": d["score"], "area_fraction": round(area_frac, 3)})
                    giant_boxes_phase2.append({"scene_id": scene_id, "box": bx, "score": d["score"]})

        prec = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / max(1, len(gt_objects)) if len(gt_objects) > 0 else 0.0
        f1 = (2 * prec * rec) / max(1e-6, prec + rec)

        print(f"[{idx+1}/16] {scene_id} ({scene_name}) | GT: {len(gt_objects)} | Preds: {len(detections)} | TP: {tp} FP: {fp} FN: {fn} | Prec: {prec*100:.1f}% Rec: {rec*100:.1f}% | Latency: {latency_ms:.0f}ms")

        phase2_scene_results.append({
            "scene_id": scene_id,
            "scene_name": scene_name,
            "category": category,
            "image_dimensions": [w, h],
            "target_phrase": target_phrase,
            "prompt_used": res.get("prompt_used"),
            "latency_ms": round(latency_ms, 2),
            "ground_truth_count": len(gt_objects),
            "ground_truth_objects": gt_objects,
            "detections_count": len(detections),
            "detections": detections,
            "scores": scores,
            "mean_confidence": round(sum(scores) / max(1, len(scores)), 4) if scores else 0.0,
            "metrics_iou_50": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "mean_iou": round(mean_iou, 4)
            },
            "metrics_iou_30": {
                "tp": tp30,
                "fp": fp30,
                "fn": fn30,
                "precision": round(tp30 / max(1, tp30 + fp30), 4),
                "recall": round(tp30 / max(1, len(gt_objects)), 4),
                "f1": round((2 * (tp30 / max(1, tp30 + fp30)) * (tp30 / max(1, len(gt_objects)))) / max(1e-6, (tp30 / max(1, tp30 + fp30)) + (tp30 / max(1, len(gt_objects)))), 4)
            },
            "giant_boxes_detected": scene_giant
        })

    agg_prec = total_tp_50 / max(1, total_tp_50 + total_fp_50)
    agg_rec = total_tp_50 / max(1, total_gt)
    agg_f1 = (2 * agg_prec * agg_rec) / max(1e-6, agg_prec + agg_rec)

    phase2_report = {
        "benchmark_metadata": {
            "title": "NexSpace Grounding DINO Phase 2 Generalized Remote Sensing Benchmark",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_id": "IDEA-Research/grounding-dino-tiny",
            "device": "cpu",
            "total_scenes": len(scenes),
            "total_ground_truth_objects": total_gt,
            "total_time_seconds": round(time.perf_counter() - t_start, 2)
        },
        "exact_inference_configuration": {
            "model_id": "IDEA-Research/grounding-dino-tiny",
            "tiling_policy": "tiled (512x512 with 20% overlap)",
            "box_threshold": 0.20,
            "text_threshold": 0.20,
            "nms_policy": "class-aware NMS with IoU 0.45",
            "semantic_box_filtering": "reject pathological single boxes > 0.35 area for localized objects",
            "prompt_generation": "generalized aerial multi-token expansion",
            "coordinate_normalization": "[0..1000] canonical integer format"
        },
        "aggregate_metrics": {
            "total_ground_truth": total_gt,
            "total_true_positives_iou50": total_tp_50,
            "total_false_positives_iou50": total_fp_50,
            "total_false_negatives_iou50": total_fn_50,
            "aggregate_precision_iou50": round(agg_prec, 4),
            "aggregate_recall_iou50": round(agg_rec, 4),
            "aggregate_f1_score_iou50": round(agg_f1, 4),
            "average_confidence": round(sum(all_confidences) / max(1, len(all_confidences)), 4) if all_confidences else 0.0,
            "giant_boxes_detected_count": len(giant_boxes_phase2)
        },
        "size_breakdown_metrics": {
            sz: {
                "gt_total": data["gt"],
                "tp": data["tp"],
                "fp": data["fp"],
                "fn": data["fn"],
                "precision": round(data["tp"] / max(1, data["tp"] + data["fp"]), 4) if data["tp"] + data["fp"] > 0 else 0.0,
                "recall": round(data["tp"] / max(1, data["gt"]), 4) if data["gt"] > 0 else 0.0,
                "f1": round((2 * (data["tp"] / max(1, data["tp"] + data["fp"])) * (data["tp"] / max(1, data["gt"]))) / max(1e-6, (data["tp"] / max(1, data["tp"] + data["fp"])) + (data["tp"] / max(1, data["gt"]))), 4) if data["gt"] > 0 and (data["tp"] + data["fp"]) > 0 else 0.0
            }
            for sz, data in size_stats.items()
        },
        "per_category_metrics": {
            cat: {
                "gt": data["gt"],
                "tp": data["tp"],
                "fp": data["fp"],
                "fn": data["fn"],
                "precision": round(data["tp"] / max(1, data["tp"] + data["fp"]), 4) if data["tp"] + data["fp"] > 0 else 0.0,
                "recall": round(data["tp"] / max(1, data["gt"]), 4) if data["gt"] > 0 else 0.0,
                "f1": round((2 * (data["tp"] / max(1, data["tp"] + data["fp"])) * (data["tp"] / max(1, data["gt"]))) / max(1e-6, (data["tp"] / max(1, data["tp"] + data["fp"])) + (data["tp"] / max(1, data["gt"]))), 4) if data["gt"] > 0 and (data["tp"] + data["fp"]) > 0 else 0.0
            }
            for cat, data in category_stats.items()
        },
        "per_image_results": phase2_scene_results
    }

    report_path = os.path.join(output_dir, "grounding_dino_phase2_benchmark_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(phase2_report, f, indent=2)

    print("\n" + "="*80)
    print("PHASE 1 (BASELINE) VS PHASE 2 (GENERALIZED TILED ENGINE) COMPARISON")
    print("="*80)
    print(f"Metric                       Phase 1 Baseline      Phase 2 Upgraded      Delta")
    print(f"--------------------------------------------------------------------------------")
    b_prec = baseline_data['aggregate_metrics']['aggregate_precision_iou50'] * 100
    p2_prec = agg_prec * 100
    print(f"Precision (IoU >= 0.50):     {b_prec:6.2f}%               {p2_prec:6.2f}%            {p2_prec - b_prec:+6.2f}%")

    b_rec = baseline_data['aggregate_metrics']['aggregate_recall_iou50'] * 100
    p2_rec = agg_rec * 100
    print(f"Recall    (IoU >= 0.50):     {b_rec:6.2f}%               {p2_rec:6.2f}%            {p2_rec - b_rec:+6.2f}%")

    b_f1 = baseline_data['aggregate_metrics']['aggregate_f1_score_iou50'] * 100
    p2_f1 = agg_f1 * 100
    print(f"F1 Score  (IoU >= 0.50):     {b_f1:6.2f}%               {p2_f1:6.2f}%            {p2_f1 - b_f1:+6.2f}%")

    b_sm = baseline_data['size_breakdown_metrics']['small']['recall'] * 100
    p2_sm = phase2_report['size_breakdown_metrics']['small']['recall'] * 100
    print(f"Small-Object Recall:         {b_sm:6.2f}%               {p2_sm:6.2f}%            {p2_sm - b_sm:+6.2f}%")

    b_md = baseline_data['size_breakdown_metrics']['medium']['recall'] * 100
    p2_md = phase2_report['size_breakdown_metrics']['medium']['recall'] * 100
    print(f"Medium-Object Recall:        {b_md:6.2f}%               {p2_md:6.2f}%            {p2_md - b_md:+6.2f}%")

    b_lg = baseline_data['size_breakdown_metrics']['large']['recall'] * 100
    p2_lg = phase2_report['size_breakdown_metrics']['large']['recall'] * 100
    print(f"Large-Object Recall:         {b_lg:6.2f}%               {p2_lg:6.2f}%            {p2_lg - b_lg:+6.2f}%")

    print(f"Giant Pathological Boxes:    5 scenes              {len(giant_boxes_phase2)} detected          {-5 + len(giant_boxes_phase2)}")
    print("="*80)
    print(f"Results saved to: {report_path}")


if __name__ == "__main__":
    run_phase2_benchmark()
