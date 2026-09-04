# NexSpace Model Quality & Scientific Validation Audit (Step 13)

**Document Version:** 1.0.0  
**Evaluated Systems:** PaliGemma RSVQA, BLIP Captioning, Grounding DINO, Change Analysis Engine, Change-VQA, Optical + SAR Fusion Runtime  
**Hardware Profile:** CPU (PyTorch 2.6.0+cpu, Transformers 4.51.0, Pillow, OpenCV)

---

## 1. Executive Summary & Capability Matrix

| Capability / Model | Technical Inference | Remote-Sensing Domain Validation | Checkpoint ID / Method | Real CPU Latency | Operational Status |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **PaliGemma RSVQA** | 🟡 Fallback Active | Gated checkpoint | `google/paligemma-3b-ft-rsvqa-lr-224` | 0.8 ms (Adapter) | ⚫ **BLOCKED** (Auth Required) |
| **BLIP Captioning** | 🟢 **REAL** | General Domain (Nadir Shift) | `Salesforce/blip-image-captioning-base` | ~1.5s - 2.8s | 🟢 **LIVE** |
| **Grounding DINO** | 🟢 **REAL** | General Domain (Nadir Shift) | `IDEA-Research/grounding-dino-tiny` | ~12.2s - 17.1s | 🟢 **LIVE** |
| **Change Analysis** | 🟢 **REAL** | Classical Computer Vision | `cv2.absdiff` + Dynamic Otsu | ~0.9 ms - 1.5 ms | 🟢 **LIVE** |
| **Change-VQA** | 🟡 **REAL** | Differential Heuristic Adapter | Rule-based surface delta mapping | ~0.5 ms | 🟡 **PARTIAL** (Heuristic) |
| **Optical + SAR Fusion**| 🟡 **REAL** | Feature Extraction Baseline | Dual-backbone cosine similarity | ~1.8s - 3.2s | 🟡 **PARTIAL** (Baseline) |

---

## 2. Model-by-Model Quality & Diagnostic Breakdown

### A. PaliGemma RSVQA (`google/paligemma-3b-ft-rsvqa-lr-224`)
- **Model Checkpoint**: `google/paligemma-3b-ft-rsvqa-lr-224` (3 Billion parameters, fine-tuned on Remote Sensing Visual Question Answering).
- **Authentication State**: Gated repository requiring Hugging Face token with accepted user agreement. When `HF_TOKEN` is unconfigured, the system automatically and safely activates the deterministic RSVQA fallback adapter.
- **Confidence Semantics**:
  - Genuine PaliGemma output logits are uncalibrated generation likelihoods; hence `confidence = null` and `confidence_type = "unavailable"`.
  - Fallback responses report `confidence = 0.85` (binary) or `0.32` (counting), explicitly marked as `confidence_type = "heuristic"`.
- **Status**: ⚫ **BLOCKED (Gated Checkpoint)** / 🟡 **PARTIAL (Deterministic RSVQA Fallback)**.

---

### B. BLIP Optical & SAR Captioning (`Salesforce/blip-image-captioning-base`)
- **Technical Inference**: **100% Genuine Neural Inference** using PyTorch vision transformer and text decoder.
- **Domain Shift Limitation**: BLIP was pre-trained on generic oblique perspective imagery (COCO, Visual Genome, web images). When exposed to overhead nadir satellite imagery, it generates generic descriptions (e.g. "a green background with a white border", "the facebook logo" for port docks, "aerial view of a city").
- **Scientific Honesty Rule**: The UI must display BLIP outputs as general scene descriptions without claiming specialized remote-sensing land-use classification.
- **Status**: 🟢 **LIVE**.

---

### C. Grounding DINO Quality & Diagnostics (`IDEA-Research/grounding-dino-tiny`)
- **Technical Inference**: **100% Genuine Neural Inference** using Zero-Shot Object Detection transformer.
- **Empirical Diagnostics on Real 512x512 Satellite Image (`sample_satellite.png`)**:

| Target Prompt | Detections Count | Mean Score | Median Score | Mean Box/Image Area | Near-Full-Image Detections (>90% Area) | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **"rooftops"** | 10 | 0.4107 | 0.4050 | 5.29% | 0 (0.0%) | 12.4s |
| **"trees"** | 9 | 0.3510 | 0.3288 | 10.43% | 0 (0.0%) | 12.8s |
| **"ships"** | 8 | 0.3274 | 0.2894 | 11.61% | 0 (0.0%) | 12.2s |
| **"vehicles"** | 10 | 0.3357 | 0.3241 | 19.31% | 1 (10.0%) | 13.2s |
| **"buildings"** | 5 | 0.3516 | 0.2918 | 36.82% | 1 (20.0%) | 17.1s |
| **"roads"** | 3 | 0.3681 | 0.3236 | 41.27% | 0 (0.0%) | 12.5s |
| **"water"** | 3 | 0.3568 | 0.3806 | 60.47% | 1 (33.3%) | 13.6s |

- **Key Diagnostic Finding**:
  - For discrete, well-bounded objects ("rooftops", "trees", "ships"), Grounding DINO detects localized regions (5%–11% image area) with zero oversized artifacts.
  - For large continuous regions ("water", "buildings"), generic text grounding can produce near-full-image bounding boxes (up to 33.3% of detections for "water").
- **Scientific Honesty Rule**: Detections are labeled as "candidate model detections" rather than "verified structures" or "ground truth".
- **Status**: 🟢 **LIVE** (Technical) / **NOT SCIENTIFICALLY VALIDATED** (Domain Accuracy).

---

### D. Change Analysis Engine (Classical CV)
- **Methodology**: Co-registered grayscale differential analysis (`cv2.absdiff`), Gaussian smoothing, dynamic Otsu thresholding, morphological noise removal, and connected components contour extraction.
- **Area Computation**: Computes exact pixel count and maps to real surface area ($\text{m}^2$) when geospatial resolution (GSD) is present.
- **Status**: 🟢 **LIVE**.

---

### E. Multimodal Optical + SAR Fusion Runtime
- **Methodology**: Dual-backbone feature extraction (768-dim optical + 768-dim SAR) fused into a 1536-dimensional joint representation; spatial correlation and cosine similarity are computed across modal feature maps.
- **Classification**: Explicitly declared as `feature_fusion_baseline` (`is_trained_fusion_model = False`).
- **Scientific Honesty Rule**: Prohibits claiming "trained deep multimodal fusion" or "SAR-optical joint AI model".
- **Status**: 🟡 **PARTIAL (Baseline)**.

---

## 3. Scientific Terminology Standards

To preserve scientific honesty and regulatory compliance, NexSpace enforces the following vocabulary rules across frontend and backend:

| Prohibited / Unsubstantiated Term | Approved Scientific Term | Rationale |
| :--- | :--- | :--- |
| "Confirmed building / structure" | "Model detection candidate" | General VLM bounding boxes have not been validated against ground survey |
| "Confirmed damage / destroyed" | "Candidate anomaly cluster" | Pixel-differencing detects intensity delta, not structural failure |
| "Calibrated confidence (99%)" | "Model score / Heuristic confidence" | Raw scores and heuristics are not calibrated posterior probabilities |
| "Trained multimodal fusion AI" | "Feature fusion baseline" | Joint embedding is computed via feature concatenation, not learned fusion |
| "Autonomous agent reasoning" | "Deterministic rule orchestration" | Routing is governed by predictable pattern classifiers, not unconstrained LLMs |
| "Zero-error detection" | "Unverified bounding box" | False positives occur under domain shift |
