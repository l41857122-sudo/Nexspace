"""
profile_dino_internals.py
-------------------------
Measures the internal timing breakdown of Grounding DINO:
1. Image conversion & resize
2. Tokenizer & Processor inputs preparation
3. Vision backbone (Swin-T)
4. Text backbone (BERT-base)
5. Multi-scale feature enhancer + Deformable cross-attention
6. Transformer decoder layers
7. Postprocessing, thresholding, and coordinate mapping
8. Multi-target joint prompt evaluation: "boat . bridge . buildings ."
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from model_runtime import GroundingDINORuntime, DeviceManager

def profile_dino():
    test_image_path = os.path.join(os.path.dirname(_dir), "sample_data", "nexspace_test_image_001.jpg")
    img = Image.open(test_image_path).convert("RGB")
    w, h = img.size

    print(f"=== GROUNDING DINO INTERNAL TIMING PROFILER ===")
    print(f"Target Image: {w}x{h}")
    print(f"Device: {DeviceManager.get_device_info()}")

    runtime = GroundingDINORuntime()
    runtime.load()

    # Test 1: Single Target "boats ."
    print("\n--- [TEST 1] Single Target: 'boats .' ---")
    t0 = time.perf_counter()
    scale = 768.0 / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    inf_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    t_resize = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    prompt = "boats ."
    inputs = runtime._processor(images=inf_img, text=prompt, return_tensors="pt")
    t_proc = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = runtime._model(**inputs)
    t_forward = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    processed = runtime._processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"] if "input_ids" in inputs else None,
        threshold=0.25,
        text_threshold=0.25,
        target_sizes=[(h, w)],
    )
    t_post = (time.perf_counter() - t0) * 1000.0

    dets = processed[0] if processed else {}
    n_boxes = len(dets.get("boxes", []))

    print(f"  Resize ({w}x{h} -> {new_w}x{new_h}): {t_resize:.2f} ms")
    print(f"  Processor / Tokenizer: {t_proc:.2f} ms")
    print(f"  Model Forward Pass: {t_forward:.2f} ms")
    print(f"  Postprocessing / NMS: {t_post:.2f} ms")
    print(f"  Found Detections: {n_boxes} boxes")

    # Test 2: Joint Multi-Target Prompt "boats . bridge . buildings ."
    print("\n--- [TEST 2] Joint Multi-Target Prompt: 'boats . bridge . buildings .' ---")
    multi_prompt = "boats . bridge . buildings ."
    t0 = time.perf_counter()
    multi_inputs = runtime._processor(images=inf_img, text=multi_prompt, return_tensors="pt")
    t_multi_proc = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    with torch.inference_mode():
        multi_outputs = runtime._model(**multi_inputs)
    t_multi_forward = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    multi_processed = runtime._processor.post_process_grounded_object_detection(
        multi_outputs,
        multi_inputs["input_ids"] if "input_ids" in multi_inputs else None,
        threshold=0.25,
        text_threshold=0.25,
        target_sizes=[(h, w)],
    )
    t_multi_post = (time.perf_counter() - t0) * 1000.0

    m_res = multi_processed[0] if multi_processed else {}
    m_boxes = m_res.get("boxes", [])
    m_labels = m_res.get("labels", [])
    m_scores = m_res.get("scores", [])

    print(f"  Multi-Prompt Tokenize: {t_multi_proc:.2f} ms")
    print(f"  Multi-Prompt Single Forward Pass: {t_multi_forward:.2f} ms")
    print(f"  Multi-Prompt Postprocess: {t_multi_post:.2f} ms")
    print(f"  Total Multi-Prompt Detections: {len(m_boxes)}")
    for b, l, s in zip(m_boxes[:8], m_labels[:8], m_scores[:8]):
        print(f"    Label: {l:15s} | Score: {float(s):.4f}")

    # Test 3: Resolution Scaling Comparison (768 vs 640 vs 512)
    print("\n--- [TEST 3] Resolution Scaling Comparison ---")
    for max_dim in [768, 640, 512]:
        s = max_dim / max(w, h)
        cur_w, cur_h = max(1, int(round(w * s))), max(1, int(round(h * s)))
        scaled_img = img.resize((cur_w, cur_h), Image.Resampling.BILINEAR)
        inp = runtime._processor(images=scaled_img, text="boats .", return_tensors="pt")
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = runtime._model(**inp)
        t_fwd = (time.perf_counter() - t0) * 1000.0
        post = runtime._processor.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=0.25, text_threshold=0.25, target_sizes=[(h, w)]
        )
        cnt = len(post[0]["boxes"]) if post else 0
        print(f"  Resolution {cur_w}x{cur_h} (max={max_dim}): Latency = {t_fwd:8.2f} ms | Boat Detections = {cnt}")

if __name__ == "__main__":
    profile_dino()
