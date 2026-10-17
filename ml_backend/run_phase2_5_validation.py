import os
import sys
import json
import time
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_dir)
for p in [_dir, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml_backend.tiling import (
    tiled_inference_engine,
    compute_iou,
    apply_nms,
    filter_boxes_by_semantics,
    normalize_bbox_1000,
)
from ml_backend.model_runtime import GroundingDINORuntime
from ml_backend.router import (
    build_generalized_aerial_prompt,
    extract_detection_targets,
    extract_grounding_target,
    is_macro_target,
    MACRO_EXTENDED_CATEGORIES,
)


def get_font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


COLOR_PALETTE = [
    (0, 230, 118),    # Vibrant Mint
    (0, 176, 255),    # Electric Blue
    (255, 145, 0),    # Bright Amber
    (245, 0, 87),     # Vivid Crimson
    (156, 39, 176),   # Royal Violet
    (255, 235, 59),   # Golden Yellow
    (0, 229, 255),    # Cyan
    (255, 87, 34),    # Deep Orange
]


def draw_bounding_boxes_annotated(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    query_text: str,
    prompt_used: str,
    latency_ms: float,
    scene_id: str,
    scene_name: str,
) -> Image.Image:
    """Renders high-visibility bounding box overlays with labels, target IDs, coordinates, and query metadata."""
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    
    banner_height = 64
    canvas = Image.new("RGB", (w, h + banner_height), color=(18, 24, 38))
    canvas.paste(img_rgb, (0, banner_height))
    
    draw = ImageDraw.Draw(canvas)
    font_header = get_font(15)
    font_sub = get_font(12)
    font_box = get_font(13)
    
    draw.rectangle([(0, 0), (w, banner_height)], fill=(15, 23, 42))
    draw.line([(0, banner_height), (w, banner_height)], fill=(51, 65, 85), width=2)
    
    header_title = f"[{scene_id.upper()}] {scene_name} | Res: {w}x{h} | Detections: {len(detections)} | Time: {latency_ms:.0f}ms"
    draw.text((12, 10), header_title, fill=(255, 255, 255), font=font_header)
    
    query_info = f'Query: "{query_text}"  ➔  Prompt: "{prompt_used}"'
    draw.text((12, 34), query_info, fill=(148, 163, 184), font=font_sub)
    
    unique_labels = list(set(d.get("label", "target") for d in detections))
    label_to_color = {lbl: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, lbl in enumerate(unique_labels)}
    
    for idx, det in enumerate(detections):
        box_1000 = det.get("box") or [0, 0, 1000, 1000]
        x1 = (box_1000[0] / 1000.0) * w
        y1 = (box_1000[1] / 1000.0) * h + banner_height
        x2 = (box_1000[2] / 1000.0) * w
        y2 = (box_1000[3] / 1000.0) * h + banner_height
        
        lbl = det.get("label", "object")
        score = det.get("score", 0.0)
        target_id = f"T{idx+1:02d}"
        color = label_to_color.get(lbl, (0, 230, 118))
        
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        
        badge_text = f"#{target_id} {lbl} {score*100:.0f}%"
        try:
            tb = draw.textbbox((0, 0), badge_text, font=font_box)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = len(badge_text) * 8, 14
            
        badge_x1 = x1
        badge_y1 = max(banner_height + 2, y1 - th - 6)
        badge_x2 = x1 + tw + 8
        badge_y2 = badge_y1 + th + 6
        
        draw.rectangle([(badge_x1, badge_y1), (badge_x2, badge_y2)], fill=(15, 23, 42), outline=color, width=1)
        draw.text((badge_x1 + 4, badge_y1 + 2), badge_text, fill=color, font=font_box)
        
    return canvas


def create_before_after_comparison(
    image: Image.Image,
    p1_detections: List[Dict[str, Any]],
    p2_detections: List[Dict[str, Any]],
    scene_name: str,
    query_text: str,
) -> Image.Image:
    """Renders a side-by-side Before (Phase 1 Baseline) vs After (Phase 2 Upgraded) visual image."""
    w, h = image.size
    side_w = min(800, w)
    side_h = int(round(h * (side_w / float(w))))
    
    img_resized = image.resize((side_w, side_h), Image.Resampling.BILINEAR)
    
    # Left: Phase 1
    canvas_p1 = img_resized.copy()
    draw_p1 = ImageDraw.Draw(canvas_p1)
    for idx, d in enumerate(p1_detections):
        bx = d.get("box", [0, 0, 1000, 1000])
        x1 = (bx[0] / 1000.0) * side_w
        y1 = (bx[1] / 1000.0) * side_h
        x2 = (bx[2] / 1000.0) * side_w
        y2 = (bx[3] / 1000.0) * side_h
        draw_p1.rectangle([(x1, y1), (x2, y2)], outline=(245, 0, 87), width=3)
        lbl = f"P1: {d.get('label','target')} {d.get('score',0)*100:.0f}%"
        draw_p1.rectangle([(x1, y1), (x1 + 140, y1 + 18)], fill=(15, 23, 42))
        draw_p1.text((x1 + 4, y1 + 2), lbl, fill=(255, 82, 82), font=get_font(11))
        
    # Right: Phase 2
    canvas_p2 = img_resized.copy()
    draw_p2 = ImageDraw.Draw(canvas_p2)
    for idx, d in enumerate(p2_detections):
        bx = d.get("box", [0, 0, 1000, 1000])
        x1 = (bx[0] / 1000.0) * side_w
        y1 = (bx[1] / 1000.0) * side_h
        x2 = (bx[2] / 1000.0) * side_w
        y2 = (bx[3] / 1000.0) * side_h
        draw_p2.rectangle([(x1, y1), (x2, y2)], outline=(0, 230, 118), width=2)
        lbl = f"#{idx+1} {d.get('label','target')} {d.get('score',0)*100:.0f}%"
        draw_p2.rectangle([(x1, max(0, y1-18)), (x1 + 130, max(18, y1))], fill=(15, 23, 42))
        draw_p2.text((x1 + 3, max(2, y1-16)), lbl, fill=(0, 230, 118), font=get_font(11))
        
    header_h = 70
    combo = Image.new("RGB", (side_w * 2 + 20, side_h + header_h), color=(15, 23, 42))
    draw_c = ImageDraw.Draw(combo)
    
    font_h = get_font(16)
    font_sub = get_font(13)
    
    draw_c.text((20, 12), f"Comparative Analysis: {scene_name}", fill=(255, 255, 255), font=font_h)
    draw_c.text((20, 38), f'Query: "{query_text}"  |  Phase 1 Baseline (Left: {len(p1_detections)} boxes) vs Phase 2 Upgraded (Right: {len(p2_detections)} boxes)', fill=(148, 163, 184), font=font_sub)
    
    combo.paste(canvas_p1, (0, header_h))
    combo.paste(canvas_p2, (side_w + 20, header_h))
    draw_c.line([(side_w + 10, header_h), (side_w + 10, header_h + side_h)], fill=(51, 65, 85), width=2)
    
    return combo


def create_synthetic_scene_if_missing(img_path: str, scene_type: str, w: int, h: int) -> Image.Image:
    """Generates synthetic high-resolution remote-sensing imagery for specialized validation tests if not present."""
    if os.path.exists(img_path):
        return Image.open(img_path).convert("RGB")
        
    img = Image.new("RGB", (w, h), color=(40, 50, 45))
    draw = ImageDraw.Draw(img)
    
    if scene_type == "towers":
        # Green mountain ridge with 4 transmission / telecom towers
        draw.rectangle([(0, 0), (w, h)], fill=(34, 52, 38))
        # Draw 4 distinct lattice towers
        tower_locs = [(w * 0.2, h * 0.4), (w * 0.45, h * 0.35), (w * 0.7, h * 0.45), (w * 0.85, h * 0.38)]
        for tx, ty in tower_locs:
            # Tower base and antenna mast
            draw.polygon([(tx-15, ty+60), (tx+15, ty+60), (tx+4, ty-40), (tx-4, ty-40)], fill=(180, 185, 190), outline=(220, 220, 220))
            draw.line([(tx, ty-40), (tx, ty-80)], fill=(255, 50, 50), width=3)  # red mast top
            draw.ellipse([(tx-25, ty-10), (tx+25, ty+40)], outline=(160, 170, 180), width=2)
            
    elif scene_type == "wind_turbines":
        # Coastal wind farm with 6 wind turbines
        draw.rectangle([(0, 0), (w, h)], fill=(28, 65, 85))  # ocean water
        for wx in range(150, w - 100, 200):
            wy = int(h * 0.5 + 40 * math.sin(wx))
            # Turbine white hub and 3 blades
            draw.ellipse([(wx-12, wy-12), (wx+12, wy+12)], fill=(240, 245, 250))
            draw.line([(wx, wy), (wx, wy - 55)], fill=(255, 255, 255), width=3)
            draw.line([(wx, wy), (wx + 45, wy + 30)], fill=(255, 255, 255), width=3)
            draw.line([(wx, wy), (wx - 45, wy + 30)], fill=(255, 255, 255), width=3)
            
    elif scene_type == "solar_farm":
        # Semi-arid ground with 8 rectangular solar photovoltaic arrays
        draw.rectangle([(0, 0), (w, h)], fill=(120, 105, 80))
        for row in range(2):
            for col in range(4):
                sx = 80 + col * 220
                sy = 120 + row * 380
                draw.rectangle([(sx, sy), (sx + 180, sy + 300)], fill=(20, 35, 75), outline=(180, 190, 200), width=2)
                # Grid lines
                for gx in range(sx, sx + 180, 30):
                    draw.line([(gx, sy), (gx, sy + 300)], fill=(45, 65, 110), width=1)
                    
    elif scene_type == "fuel_tanks":
        # Industrial refinery with 6 large cylindrical white/silver storage tanks
        draw.rectangle([(0, 0), (w, h)], fill=(75, 75, 75))
        tank_centers = [(200, 250), (450, 250), (700, 250), (200, 650), (450, 650), (700, 650)]
        for cx, cy in tank_centers:
            draw.ellipse([(cx - 90, cy - 90), (cx + 90, cy + 90)], fill=(235, 240, 245), outline=(100, 105, 110), width=3)
            draw.ellipse([(cx - 75, cy - 75), (cx + 75, cy + 75)], outline=(180, 185, 190), width=1)
            
    elif scene_type == "construction_cranes":
        # Construction site with excavation pit and 3 yellow tower cranes
        draw.rectangle([(0, 0), (w, h)], fill=(90, 80, 65))
        draw.rectangle([(150, 150), (w - 150, h - 150)], fill=(55, 50, 40), outline=(110, 100, 80), width=2)
        crane_locs = [(w * 0.3, h * 0.35), (w * 0.65, h * 0.4), (w * 0.45, h * 0.7)]
        for kx, ky in crane_locs:
            draw.rectangle([(kx - 10, ky - 10), (kx + 10, ky + 10)], fill=(255, 190, 0))
            draw.line([(kx - 70, ky), (kx + 110, ky)], fill=(255, 210, 0), width=4)  # Jib
            draw.ellipse([(kx - 6, ky - 6), (kx + 6, ky + 6)], fill=(20, 20, 20))
            
    elif scene_type == "railway_yard":
        # Rail marshalling yard with parallel tracks and 4 locomotive train lines
        draw.rectangle([(0, 0), (w, h)], fill=(60, 60, 60))
        for y_track in range(120, h - 100, 90):
            draw.line([(0, y_track), (w, y_track)], fill=(120, 120, 130), width=3)
            draw.line([(0, y_track + 15), (w, y_track + 15)], fill=(120, 120, 130), width=3)
            # Train carriages
            tx = 100 + (y_track % 200)
            draw.rectangle([(tx, y_track - 5), (tx + 320, y_track + 20)], fill=(180, 40, 40), outline=(230, 230, 230), width=2)
            
    elif scene_type == "container_terminal":
        # Intermodal port terminal with colored container stacks
        draw.rectangle([(0, 0), (w, h)], fill=(80, 82, 85))
        colors = [(200, 40, 40), (40, 90, 200), (40, 160, 80), (220, 180, 30), (200, 100, 30)]
        for r in range(4):
            for c in range(5):
                cx = 80 + c * 170
                cy = 100 + r * 210
                fill_c = colors[(r * 5 + c) % len(colors)]
                draw.rectangle([(cx, cy), (cx + 140, cy + 170)], fill=fill_c, outline=(240, 240, 240), width=1)
                
    elif scene_type == "multi_target_harbor":
        # Port with both cargo ships in water and container stacks on dock
        draw.rectangle([(0, 0), (w * 0.55, h)], fill=(30, 65, 90))  # water
        draw.rectangle([(w * 0.55, 0), (w, h)], fill=(80, 80, 85))   # dock
        # 2 Ships in water
        draw.polygon([(80, 150), (240, 150), (280, 220), (240, 290), (80, 290)], fill=(160, 45, 45), outline=(220, 220, 220))
        draw.polygon([(100, 520), (280, 520), (320, 590), (280, 660), (100, 660)], fill=(40, 75, 140), outline=(220, 220, 220))
        # Containers on dock
        for r in range(3):
            for c in range(2):
                cx = int(w * 0.62) + c * 130
                cy = 120 + r * 240
                draw.rectangle([(cx, cy), (cx + 100, cy + 180)], fill=(220, 140, 30), outline=(255, 255, 255))
                
    elif scene_type == "multi_target_urban":
        # Transit hub with road vehicles and office buildings
        draw.rectangle([(0, 0), (w, h)], fill=(50, 50, 52))
        # 3 Office buildings
        draw.rectangle([(60, 60), (450, 280)], fill=(190, 195, 205), outline=(240, 240, 250), width=2)
        draw.rectangle([(550, 60), (w - 60, 280)], fill=(160, 175, 190), outline=(240, 240, 250), width=2)
        draw.rectangle([(200, 460), (w - 200, h - 60)], fill=(140, 155, 170), outline=(240, 240, 250), width=2)
        # Road corridor in middle with 4 cars
        draw.rectangle([(0, 320), (w, 420)], fill=(35, 35, 38))
        for vx in [150, 400, 700, 1000]:
            draw.rectangle([(vx, 350), (vx + 80, 390)], fill=(230, 40, 40), outline=(255, 255, 255))
            
    elif scene_type == "agricultural_silos":
        # Farm complex with round silo towers and rectangular red barns
        draw.rectangle([(0, 0), (w, h)], fill=(85, 115, 65))
        # 2 Barns
        draw.rectangle([(120, 150), (420, 380)], fill=(175, 35, 35), outline=(240, 240, 240), width=2)
        draw.rectangle([(550, 150), (850, 380)], fill=(175, 35, 35), outline=(240, 240, 240), width=2)
        # 3 Grain silos
        for sx in [200, 480, 760]:
            draw.ellipse([(sx - 45, 520), (sx + 45, 610)], fill=(210, 215, 220), outline=(80, 80, 80), width=2)
            
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    img.save(img_path)
    return img


def run_full_validation():
    print("================================================================================")
    print("PHASE 2.5: GENERALIZED DETECTION VALIDATION & VISUAL BENCHMARK RUNNER")
    print("================================================================================")
    
    output_base = os.path.join(_dir, "evaluation_results", "phase2_5_full_validation")
    images_out_dir = os.path.join(output_base, "images")
    before_after_dir = os.path.join(output_base, "before_after")
    failures_dir = os.path.join(output_base, "failures")
    raw_out_dir = os.path.join(output_base, "raw_outputs")
    report_dir = os.path.join(output_base, "report")
    
    for d in [images_out_dir, before_after_dir, failures_dir, raw_out_dir, report_dir]:
        os.makedirs(d, exist_ok=True)
        
    baseline_json_path = os.path.join(
        _dir, "evaluation_results", "baseline_grounding_benchmark", "grounding_dino_baseline_benchmark_results.json"
    )
    with open(baseline_json_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
        
    baseline_scenes_map = {s["scene_id"]: s for s in baseline_data["per_image_results"]}
    
    # Initialize Grounding DINO runtime
    runtime = GroundingDINORuntime()
    runtime.load()
    
    images_dir = os.path.join(_dir, "evaluation_results", "baseline_grounding_benchmark", "images")
    
    # 26 Comprehensive Validation Scenes
    VALIDATION_SCENES = [
        # 16 Core Phase 1 Benchmark Scenes
        ("scene_01", "High-Density Urban Residential", "scene_01_urban_residential.png", "Locate all buildings.", "buildings", 16, "buildings / residential areas", "small"),
        ("scene_02", "Suburban Single-Family Homes", "scene_02_suburban_houses.png", "Find houses and residential rooftops.", "houses", 9, "buildings / residential areas", "medium"),
        ("scene_03", "Industrial Logistics Warehouses", "scene_03_industrial_park.png", "Locate warehouses and industrial structures.", "warehouses", 6, "urban areas", "large"),
        ("scene_04", "Highway Interchange & Arterial Network", "scene_04_highway_interchange.png", "Where are the highways and roads?", "roads", 2, "roads", "large"),
        ("scene_05", "Commercial Parking Lot Vehicles", "scene_05_parking_vehicles.png", "Find cars in the parking area.", "cars", 24, "vehicles", "small"),
        ("scene_06", "Highway Vehicle Traffic", "scene_06_highway_traffic.png", "Detect vehicles on the road.", "vehicles", 8, "vehicles", "small"),
        ("scene_07", "Commercial Seaport Cargo Ships", "scene_07_seaport_cargo.png", "Detect ships near the port.", "ships", 4, "ships / ports", "large"),
        ("scene_08", "Marina Harbor Boats", "scene_08_harbor_marina.png", "Locate boats in the marina harbor.", "boats", 12, "ships / ports", "small"),
        ("scene_09", "Regional Airport Apron & Airplanes", "scene_09_airport_apron.png", "Identify airplanes on the airport apron.", "airplanes", 4, "airports", "medium"),
        ("scene_10", "Segmented Agricultural Crop Plots", "scene_10_agricultural_farmland.png", "Where are the agricultural crop fields?", "farmland", 6, "rural/agricultural areas", "large"),
        ("scene_11", "Rural Farmhouses & Silo Structures", "scene_11_rural_homesteads.png", "Find rural farm buildings.", "buildings", 5, "rural/agricultural areas", "medium"),
        ("scene_12", "Coastal Shoreline & Water Estuary", "scene_12_coastal_shoreline.png", "Detect the water body and shoreline.", "water", 2, "coastline/water", "large"),
        ("scene_13", "River Waterway & Highway Bridge Crossing", "scene_13_river_bridge.png", "Where is the bridge crossing the river?", "bridge", 1, "roads", "large"),
        ("scene_14", "Commercial Retail Center Buildings", "scene_14_commercial_center.png", "Locate commercial center buildings.", "buildings", 5, "urban areas", "medium"),
        ("scene_15", "Dense Forest Canopy Reserve", "scene_15_forest_reserve.png", "Identify trees and dense vegetation.", "vegetation", 2, "rural/agricultural areas", "large"),
        ("scene_16", "High-Res Dense Settlement Buildings", "scene_16_highres_dense_settlement.png", "Locate all small settlement buildings.", "buildings", 38, "buildings / residential areas", "small"),
        
        # 10 Additional Diverse Validation Scenes (Specialized Classes, Arbitrary Targets, Multi-Target)
        ("scene_17", "Mountain Ridgeline Telecom Towers", "scene_17_telecom_towers.png", "Locate communication towers and antenna masts.", "towers", 4, "towers", "medium"),
        ("scene_18", "Offshore Wind Turbine Farm", "scene_18_wind_turbine_farm.png", "Detect wind turbine generators.", "wind turbine generator", 6, "arbitrary targets", "medium"),
        ("scene_19", "Desert Solar Photovoltaic Farm", "scene_19_solar_photovoltaic_farm.png", "Identify solar panel arrays.", "solar panel array", 8, "arbitrary targets", "large"),
        ("scene_20", "Petrochemical Refinery Storage Tanks", "scene_20_petrochemical_fuel_tanks.png", "Locate fuel storage tanks.", "fuel storage tank", 6, "arbitrary targets", "large"),
        ("scene_21", "Urban Construction Site Cranes", "scene_21_construction_cranes_site.png", "Find construction cranes.", "construction crane", 3, "arbitrary targets", "medium"),
        ("scene_22", "Railway Marshalling Yard Freight Cars", "scene_22_railway_marshalling_yard.png", "Locate railway locomotive cars.", "railway locomotive", 4, "arbitrary targets", "large"),
        ("scene_23", "Intermodal Shipping Container Terminal", "scene_23_shipping_container_terminal.png", "Detect shipping containers in the terminal.", "containers", 20, "containers / industrial structures", "medium"),
        ("scene_24", "Deepwater Port Multi-Target", "scene_24_multi_target_harbor.png", "Locate ships and containers.", "ships and containers", 8, "multi-target queries", "large"),
        ("scene_25", "Urban Transit Hub Multi-Target", "scene_25_multi_target_urban_transit.png", "Find buildings and vehicles.", "buildings and vehicles", 7, "multi-target queries", "medium"),
        ("scene_26", "Agricultural Farm Silos & Barns", "scene_26_agricultural_silos_barns.png", "Locate barns and agricultural silos.", "agricultural structures", 5, "agricultural structures", "large"),
    ]
    
    # Synthetic generators for scenes 17 to 26
    synth_configs = {
        "scene_17": ("towers", 1024, 768),
        "scene_18": ("wind_turbines", 1280, 720),
        "scene_19": ("solar_farm", 1000, 1000),
        "scene_20": ("fuel_tanks", 900, 900),
        "scene_21": ("construction_cranes", 1200, 900),
        "scene_22": ("railway_yard", 1400, 700),
        "scene_23": ("container_terminal", 1024, 1024),
        "scene_24": ("multi_target_harbor", 1000, 1000),
        "scene_25": ("multi_target_urban", 1280, 720),
        "scene_26": ("agricultural_silos", 1024, 768),
    }
    
    master_results = {
        "metadata": {
            "title": "NexSpace Phase 2.5 Generalized Detection Master Validation Benchmark",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": "IDEA-Research/grounding-dino-tiny",
            "total_images_evaluated": len(VALIDATION_SCENES),
            "tile_size": 512,
            "overlap": 0.20,
            "nms_iou_threshold": 0.45,
            "box_threshold": 0.20,
            "text_threshold": 0.20,
        },
        "operations": [],
        "failures": [],
        "category_summary": {},
        "scale_summary": {
            "small": {"gt": 0, "detected": 0},
            "medium": {"gt": 0, "detected": 0},
            "large": {"gt": 0, "detected": 0},
        }
    }
    
    total_ops = 0
    total_detections_all = 0
    total_gt_all = 0
    all_latencies = []
    
    print(f"\nStarting evaluation of {len(VALIDATION_SCENES)} remote-sensing scenes...")
    
    for idx, (scene_id, scene_name, img_filename, query, target_phrase, gt_count, category, scale_type) in enumerate(VALIDATION_SCENES):
        img_path = os.path.join(images_dir, img_filename)
        
        # Load or generate scene
        if scene_id in synth_configs:
            stype, sw, sh = synth_configs[scene_id]
            img = create_synthetic_scene_if_missing(img_path, stype, sw, sh)
        else:
            img = Image.open(img_path).convert("RGB")
            
        w, h = img.size
        aspect_ratio = f"{w}:{h} ({w/max(1,h):.2f})"
        
        # 1. Multi-target check
        extracted_targets = extract_detection_targets(query)
        clean_prompt = build_generalized_aerial_prompt(target_phrase)
        
        # 2. Execute Tiled Grounding DINO
        t0 = time.perf_counter()
        t_res = tiled_inference_engine.run_tiled_grounding(
            image=img,
            target_phrase=clean_prompt,
            grounding_runtime=runtime,
            tile_size=512,
            overlap=0.20,
            iou_threshold=0.45,
            box_threshold=0.20,
            text_threshold=0.20,
            max_area_ratio=0.35,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        all_latencies.append(latency_ms)
        
        detections = t_res.get("detections", [])
        num_dets = len(detections)
        total_detections_all += num_dets
        total_gt_all += gt_count
        total_ops += 1
        
        # Update scale metrics
        master_results["scale_summary"][scale_type]["gt"] += gt_count
        master_results["scale_summary"][scale_type]["detected"] += num_dets
        
        # Update category metrics
        cat_rec = master_results["category_summary"].setdefault(category, {
            "image_count": 0, "gt_total": 0, "detections_total": 0, "latencies": [], "scores": []
        })
        cat_rec["image_count"] += 1
        cat_rec["gt_total"] += gt_count
        cat_rec["detections_total"] += num_dets
        cat_rec["latencies"].append(latency_ms)
        cat_rec["scores"].extend([d["score"] for d in detections])
        
        # 3. Generate Annotated Visualization Image
        annotated_img = draw_bounding_boxes_annotated(
            image=img,
            detections=detections,
            query_text=query,
            prompt_used=clean_prompt,
            latency_ms=latency_ms,
            scene_id=scene_id,
            scene_name=scene_name,
        )
        annotated_out_path = os.path.join(images_out_dir, f"{scene_id}_{target_phrase.replace(' ','_')}_detections.png")
        annotated_img.save(annotated_out_path)
        
        # 4. Generate Before vs After Image for representative scenes
        if scene_id in baseline_scenes_map:
            b_scene = baseline_scenes_map[scene_id]
            p1_dets = b_scene.get("detections", [])
            ba_img = create_before_after_comparison(
                image=img,
                p1_detections=p1_dets,
                p2_detections=detections,
                scene_name=scene_name,
                query_text=query,
            )
            ba_out_path = os.path.join(before_after_dir, f"{scene_id}_before_after.png")
            ba_img.save(ba_out_path)
            
        # 5. Check for failure case observations
        is_failure = False
        failure_reasons = []
        if gt_count > 0 and num_dets == 0:
            is_failure = True
            failure_reasons.append("Zero detections on small/difficult targets (<12px resolution limit)")
        elif any(d["score"] < 0.22 for d in detections):
            failure_reasons.append("Marginal model confidence near threshold")
            
        if is_failure:
            fail_record = {
                "scene_id": scene_id,
                "scene_name": scene_name,
                "query": query,
                "gt_count": gt_count,
                "detections_count": num_dets,
                "reasons": failure_reasons,
                "image_file": os.path.basename(annotated_out_path),
            }
            master_results["failures"].append(fail_record)
            fail_img_path = os.path.join(failures_dir, f"failure_{scene_id}.png")
            annotated_img.save(fail_img_path)
            
        # Save individual JSON
        op_json_record = {
            "image_parameters": {
                "filename": img_filename,
                "dimensions": [w, h],
                "aspect_ratio": aspect_ratio,
                "tiles_processed": t_res.get("tiles_processed", 1),
                "tile_size": 512,
                "overlap": 0.20,
            },
            "query_parameters": {
                "natural_language_query": query,
                "extracted_targets": extracted_targets,
                "generated_prompt": clean_prompt,
                "target_category": category,
            },
            "performance": {
                "latency_ms": round(latency_ms, 2),
                "raw_detections": t_res.get("raw_detections_count", num_dets),
                "final_detections": num_dets,
            },
            "detections": detections,
            "visualization_path": annotated_out_path,
        }
        
        single_json_path = os.path.join(raw_out_dir, f"{scene_id}_{target_phrase.replace(' ','_')}_results.json")
        with open(single_json_path, "w", encoding="utf-8") as f:
            json.dump(op_json_record, f, indent=2)
            
        master_results["operations"].append(op_json_record)
        
        print(f"[{idx+1:02d}/26] {scene_id} ({scene_name[:35]:35s}) | GT: {gt_count:2d} | Found: {num_dets:2d} | Latency: {latency_ms:6.0f}ms | Visual: {os.path.basename(annotated_out_path)}")
        
    # Write master JSON
    master_json_path = os.path.join(output_base, "phase2_5_master_results.json")
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)
        
    print("\n" + "="*80)
    print("PHASE 2.5 VALIDATION SUMMARY")
    print("="*80)
    print(f"Total Images Evaluated:       {len(VALIDATION_SCENES)}")
    print(f"Total Operations Executed:    {total_ops}")
    print(f"Total Detections Extracted:   {total_detections_all}")
    print(f"Total Ground-Truth Objects:   {total_gt_all}")
    print(f"Mean Latency:                 {np.mean(all_latencies):.2f} ms")
    print(f"Worst-case Latency:           {np.max(all_latencies):.2f} ms")
    print(f"Master Results JSON:          {master_json_path}")
    print("="*80)


if __name__ == "__main__":
    run_full_validation()
