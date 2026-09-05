# Remote Sensing Datasets Reference

This document outlines the standard datasets supported by the NexSpace multimodal remote-sensing AI framework, including their canonical schemas, split structures, and roles in zero-shot evaluation, benchmark scoring, and PEFT adaptation.

---

## 1. Supported Benchmark & Adaptation Datasets

### 1.1 RSICD (Remote Sensing Image Captioning Dataset)
* **Domain:** Optical remote-sensing scene classification and natural language captioning.
* **Size:** ~10,921 high-resolution images (224x224 and 512x512) spanning 30+ land-use categories (airport, farmland, industrial, residential, port, etc.).
* **Role in NexSpace:** 
  - Backbone zero-shot alignment via `flax-community/clip-rsicd`.
  - Supervised contrastive adaptation for custom PEFT linear projection heads (`train_rs_adaptation.py`).
* **Expected Directory Structure:**
  ```text
  data/datasets/RSICD/
  ├── dataset_rsicd.json
  └── images/
      ├── airport_01.jpg
      ├── farmland_02.jpg
      └── ...
  ```

---

### 1.2 RSVQA (Remote Sensing Visual Question Answering)
* **Domain:** Natural-language question answering over high-resolution (HR) and low-resolution (LR) overhead imagery.
* **Question Types:**
  - `presence`: Presence or absence of features ("Is there a residential building?").
  - `count`: Discrete count queries ("How many storage tanks are present?").
  - `comparison`: Structural or area comparison ("Are there more trees than buildings?").
  - `area`: Land cover estimation ("Is the rural area larger than the urban area?").
* **Role in NexSpace:**
  - Evaluated in `ml_backend/benchmarks/evaluator.py` under `rsvqa_lr` and `rsvqa_hr`.
  - Ground truth comparison for VQA accuracy calculation.
* **Expected Directory Structure:**
  ```text
  data/datasets/RSVQA/
  ├── Questions_LR.json
  ├── Answers_LR.json
  └── Images_LR/
  ```

---

### 1.3 VRSBench (Visual Remote Sensing Benchmark)
* **Domain:** Unified visual grounding, dense captioning, and multi-turn visual dialogue for remote sensing.
* **Size:** Multi-scale aerial and satellite images with high-precision bounding box annotations.
* **Role in NexSpace:**
  - Object localization and grounding evaluation (IoU calculation).
  - Open-vocabulary remote-sensing detection validation.
* **Expected Directory Structure:**
  ```text
  data/datasets/VRSBench/
  ├── annotations/
  │   ├── test_grounding.json
  │   └── test_vqa.json
  └── images/
  ```

---

### 1.4 CDVQA (Change Detection Visual Question Answering)
* **Domain:** Bi-temporal change interpretation and reasoning over co-registered satellite pairs.
* **Question Types:**
  - Change existence ("Did any construction happen in this zone?").
  - Semantic nature ("What replaced the agricultural field?").
  - Change localization ("Where did building demolitions occur?").
* **Role in NexSpace:**
  - Benchmarks the `semantic_change.py` engine against human annotations.
* **Expected Directory Structure:**
  ```text
  data/datasets/CDVQA/
  ├── questions.json
  └── pairs/
      ├── 0001_before.tif
      ├── 0001_after.tif
      └── ...
  ```

---

### 1.5 BigEarthNet (Sentinel-1 / Sentinel-2)
* **Domain:** Multi-spectral optical (Sentinel-2) and SAR dual-polarization (Sentinel-1) multi-label scene understanding.
* **Size:** 590,326 pairs across 19 CORINE Land Cover classes.
* **Role in NexSpace:**
  - Multi-label classification benchmarking.
  - Optical-SAR cross-modal alignment evaluation.
* **Expected Directory Structure:**
  ```text
  data/datasets/BigEarthNet/
  ├── BigEarthNet-S2/
  └── BigEarthNet-S1/
  ```

---

## 2. Evaluation State

When any of the above datasets are not installed on local disk, the NexSpace benchmark runner (`ml_backend/benchmarks/evaluator.py`) reports:
```json
{
  "status": "NOT RUN — DATASET NOT AVAILABLE",
  "dataset": "RSVQA",
  "metrics": {}
}
```
In compliance with the **Scientific Honesty Rule**, no mock accuracy figures or synthetic benchmark scores are fabricated.
