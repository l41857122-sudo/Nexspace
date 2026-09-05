"""
profile_lifecycle.py
--------------------
Detailed lifecycle profiler for NexSpace ML Backend.
Measures:
- Image decoding
- Model loading vs warm inference
- Request parsing & dispatch
- Forward pass, text encoding, vision encoding
- Postprocessing, NMS, synthesis
- End-to-end query latency for queries A-H
"""

import os
import sys
import time
import json
import torch
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from orchestrator import GeoVLMController
from model_runtime import (
    DeviceManager,
    BLIPCaptioningRuntime,
    GroundingDINORuntime,
    PaliGemmaVQARuntime,
)
from rs_vqa_engine import rs_vqa_engine
from rs_vision_core import rs_vision_runtime

def run_profiling():
    test_image_path = os.path.join(os.path.dirname(_dir), "sample_data", "nexspace_test_image_001.jpg")
    if not os.path.exists(test_image_path):
        test_image_path = os.path.join(_dir, "sample_satellite.png")

    print(f"=== NEXSPACE REQUEST LIFECYCLE PROFILER ===")
    print(f"Target Image: {test_image_path}")
    print(f"Device: {DeviceManager.get_device_info()}")
    print(f"PyTorch Thread Count: {torch.get_num_threads()}, Inter-op: {torch.get_num_interop_threads()}")

    # 1. Measure Image Loading & Decoding
    t0 = time.perf_counter()
    with open(test_image_path, "rb") as f:
        img_bytes = f.read()
    t_read = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    import io
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    t_decode = (time.perf_counter() - t0) * 1000.0

    print(f"\n--- [1] Image I/O ---")
    print(f"File Read: {t_read:.2f} ms ({len(img_bytes)} bytes)")
    print(f"Image Decode to RGB: {t_decode:.2f} ms ({pil_img.size})")

    # 2. Measure Component Singletons & Loading vs Warm-up
    print(f"\n--- [2] Model Initialization & Inference Breakdown ---")

    # A. VQA
    print("\n[A] RSVQA Engine / PaliGemma:")
    t0 = time.perf_counter()
    vqa_res1 = rs_vqa_engine.answer_question(pil_img, "Is there a body of water?")
    t_vqa_warm1 = (time.perf_counter() - t0) * 1000.0
    print(f"  VQA Query 1: {t_vqa_warm1:.2f} ms -> Answer: {vqa_res1.get('answer')}")

    t0 = time.perf_counter()
    vqa_res2 = rs_vqa_engine.answer_question(pil_img, "Are there bridges?")
    t_vqa_warm2 = (time.perf_counter() - t0) * 1000.0
    print(f"  VQA Query 2: {t_vqa_warm2:.2f} ms -> Answer: {vqa_res2.get('answer')}")

    # B. BLIP Captioning
    print("\n[B] BLIP Captioning Runtime:")
    blip = BLIPCaptioningRuntime()
    t0 = time.perf_counter()
    blip_loaded = blip.is_available()
    t_blip_load = (time.perf_counter() - t0) * 1000.0
    print(f"  BLIP Load Check: {t_blip_load:.2f} ms (Loaded={blip_loaded})")

    # Profile sub-steps of BLIP
    if blip_loaded:
        t0 = time.perf_counter()
        rgb_img = pil_img.convert("RGB")
        w, h = rgb_img.size
        scale = 768.0 / max(w, h)
        inf_image = rgb_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
        t_blip_prep = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        inputs = blip._processor(inf_image, return_tensors="pt")
        t_blip_proc = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        with torch.no_grad():
            out = blip._model.generate(
                **inputs,
                num_beams=3,
                max_new_tokens=40,
                min_new_tokens=4,
                repetition_penalty=1.25,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=False,
            )
        t_blip_gen = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        raw_cap = blip._processor.decode(out[0], skip_special_tokens=True).strip()
        t_blip_decode = (time.perf_counter() - t0) * 1000.0

        print(f"  BLIP Preprocessing: {t_blip_prep:.2f} ms")
        print(f"  BLIP Tokenize/Processor: {t_blip_proc:.2f} ms")
        print(f"  BLIP Forward Generate (beams=3, max=40): {t_blip_gen:.2f} ms")
        print(f"  BLIP Decode Text: {t_blip_decode:.2f} ms")
        print(f"  BLIP Raw Output: '{raw_cap}'")

        # Measure greedy decoding (num_beams=1)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out_greedy = blip._model.generate(
                **inputs,
                num_beams=1,
                max_new_tokens=30,
                min_new_tokens=4,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                do_sample=False,
            )
        t_blip_greedy = (time.perf_counter() - t0) * 1000.0
        raw_greedy = blip._processor.decode(out_greedy[0], skip_special_tokens=True).strip()
        print(f"  BLIP Greedy Generate (beams=1, max=30, inference_mode): {t_blip_greedy:.2f} ms -> '{raw_greedy}'")

    # C. Grounding DINO
    print("\n[C] Grounding DINO Runtime:")
    gdino = GroundingDINORuntime()
    t0 = time.perf_counter()
    gdino_loaded = gdino.is_available()
    t_gdino_load = (time.perf_counter() - t0) * 1000.0
    print(f"  Grounding DINO Load Check: {t_gdino_load:.2f} ms (Loaded={gdino_loaded})")

    if gdino_loaded:
        # Step breakdown for target "bridge"
        target_phrase = "bridge"
        clean_phrase = target_phrase.strip().lower() + "."
        w, h = pil_img.size
        scale = 768.0 / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        t0 = time.perf_counter()
        inf_image = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        t_gdino_prep = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        inputs = gdino._processor(images=inf_image, text=clean_phrase, return_tensors="pt")
        t_gdino_proc = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = gdino._model(**inputs)
        t_gdino_forward = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        processed = gdino._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"] if "input_ids" in inputs else None,
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=[(h, w)],
        )
        t_gdino_post = (time.perf_counter() - t0) * 1000.0

        print(f"  GDINO Resize ({w}x{h} -> {new_w}x{new_h}): {t_gdino_prep:.2f} ms")
        print(f"  GDINO Processor/Tokenize: {t_gdino_proc:.2f} ms")
        print(f"  GDINO Forward Pass (no_grad): {t_gdino_forward:.2f} ms")
        print(f"  GDINO Post-process/NMS: {t_gdino_post:.2f} ms")
        print(f"  GDINO Found: {len(processed[0]['boxes']) if processed else 0} boxes")

        # Test torch.inference_mode()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs_inf = gdino._model(**inputs)
        t_gdino_infmode = (time.perf_counter() - t0) * 1000.0
        print(f"  GDINO Forward Pass (inference_mode): {t_gdino_infmode:.2f} ms")

    # 3. End-to-End Benchmark for Queries A through H
    print(f"\n--- [3] End-to-End Controller Baseline for Queries A through H ---")
    ctrl = GeoVLMController()

    queries = [
        ("A", "Is there a body of water?"),
        ("B", "Are there bridges?"),
        ("C", "Is vegetation present?"),
        ("D", "How many boats are visible?"),
        ("E", "Where is the bridge?"),
        ("F", "Describe the scene."),
        ("G", "What is near the bridge?"),
        ("H", "What objects are visible?"),
    ]

    benchmark_data = {}

    for case_id, q in queries:
        t0 = time.perf_counter()
        resp = ctrl.handle_request(query=q, optical_image=pil_img)
        dur = (time.perf_counter() - t0) * 1000.0

        tools = resp.get("selected_tools", [])
        ans = (resp.get("response_text", "") or "")[:120].replace("\n", " ")
        conf = resp.get("confidence")
        conf_prov = f"{conf} ({resp.get('confidence_type')}, {resp.get('confidence_source')})"
        dets = len(resp.get("grounding", {}).get("detections", [])) if resp.get("grounding") else 0
        ev_count = len(resp.get("evidence", []))

        print(f"\nCase {case_id}: \"{q}\"")
        print(f"  Tools: {tools}")
        print(f"  Latency: {dur:.2f} ms ({dur/1000.0:.2f} s)")
        print(f"  Detections: {dets}, Evidence Items: {ev_count}")
        print(f"  Conf Provenance: {conf_prov}")
        print(f"  Summary: {ans}...")

        benchmark_data[case_id] = {
            "query": q,
            "tools": tools,
            "latency_ms": round(dur, 2),
            "detections": dets,
            "evidence_count": ev_count,
            "confidence": conf,
            "confidence_type": resp.get("confidence_type"),
            "confidence_source": resp.get("confidence_source"),
            "answer_preview": ans,
        }

    with open(os.path.join(_dir, "baseline_group2_profile.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print("\nSaved baseline profile to ml_backend/baseline_group2_profile.json")

if __name__ == "__main__":
    run_profiling()
