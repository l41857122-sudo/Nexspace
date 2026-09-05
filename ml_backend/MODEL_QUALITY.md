# NexSpace Model Quality & Scientific Validation Audit

**Document Version:** 2.0.0  
**Evaluated Systems:** RSICD Domain Vision Core (`flax-community/clip-rsicd`), Remote Sensing VQA Engine (`rs_vqa_engine.py`), Zero-Shot Grounding DINO (`IDEA-Research/grounding-dino-tiny`), BLIP Scene Captioning (`Salesforce/blip-image-captioning-base`), Bi-Temporal Semantic Change Engine (`semantic_change.py`), Physical Optical + SAR Fusion Engine (`optical_sar_fusion.py`), Co-Registration Validator (`coregistration.py`), High-Resolution Tiling Engine (`tiling.py`), Benchmark Framework (`benchmarks/evaluator.py`), ISRO/SAC Evaluation Pipeline (`isro_evaluation.py`).  
**Hardware Profile:** CPU (PyTorch 2.6.0+cpu, Transformers 4.51.0, Pillow, OpenCV, NumPy)

---

## 1. Executive Summary & Capability Matrix

| Capability / Tool | Technical Inference | Remote-Sensing Domain Adaptation | Provenance / Checkpoint ID | Inference Latency | Operational Status |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Domain-Adapted Vision Core** | 🟢 **REAL** | **Domain-Specific (RSICD)** | `flax-community/clip-rsicd` | ~45 ms - 90 ms | 🟢 **LIVE (RS-Adapted)** |
| **PEFT / Adaptation Pipeline** | 🟢 **REAL** | Supervised Contrastive Training | `train_rs_adaptation.py` (Linear / LoRA) | N/A (Offline) | 🟢 **REPRODUCIBLE** |
| **Remote-Sensing VQA Engine** | 🟢 **REAL** | Domain Taxonomies + DINO + ExG | `rs_vqa_engine.py` (Multi-Component) | ~1.2s - 14.5s | 🟢 **LIVE (Domain Reasoning)** |
| **Zero-Shot Grounding** | 🟢 **REAL** | Open-Vocabulary (Nadir Shift) | `IDEA-Research/grounding-dino-tiny` | ~12.2s - 16.5s | 🟢 **LIVE** |
| **Scene Captioning** | 🟢 **REAL** | General Domain (Nadir Shift) | `Salesforce/blip-image-captioning-base` | ~1.5s - 2.8s | 🟢 **LIVE (Generic Model)** |
| **Semantic Change Understanding**| 🟢 **REAL** | 3-Tier (Pixel/Object/Semantic) | `semantic_change.py` (RSICD + DINO + CV) | ~14.0s - 18.0s | 🟢 **LIVE (Research Baseline)** |
| **Optical + SAR Microwave Fusion**| 🟢 **REAL** | Physical Backscatter + ExG | `optical_sar_fusion.py` | ~80 ms - 150 ms | 🟢 **LIVE (Research Baseline)** |
| **Rigorous Co-Registration** | 🟢 **REAL** | CRS, Affine, Extents, Overlap | `coregistration.py` | ~1.2 ms | 🟢 **LIVE (Geospatial Engine)**|
| **Multi-Scale Sliding Tiling** | 🟢 **REAL** | Canonical [0, 1000] Boxes + NMS | `tiling.py` | Dynamic | 🟢 **LIVE** |
| **Honest Confidence Scoring** | 🟢 **REAL** | Uncalibrated vs Heuristic | `confidence_system.py` | < 0.1 ms | 🟢 **LIVE** |
| **Benchmark Framework** | 🟢 **REAL** | RSVQA, VRSBench, CDVQA, BigEarthNet | `benchmarks/evaluator.py` | Dynamic | 🟢 **EVALUATION READY** |
| **ISRO/SAC Adapter** | 🟢 **REAL** | Cartosat-2S + RISAT Export | `isro_evaluation.py` | Dynamic | 🟢 **READY FOR EVALUATION** |

---

## 2. Model-by-Model Quality & Diagnostic Breakdown

### A. RS-Adapted Vision Core (`flax-community/clip-rsicd`)
- **Architecture**: Contrastive Vision-Language Transformer adapted to satellite and aerial imagery.
- **Taxonomy Coverage**: 14 standardized RS land-use and land-cover categories (airport, port, residential, industrial, farmland, forest, desert, mountain, water body, stadium, railway, parking lot, storage tank, solar farm).
- **Adaptation Weight Hook**: Supports optional local PEFT adapter checkpoint loading from `weights/rs_adapter/`.
- **Status**: 🟢 **REAL RS-ADAPTED MODEL (LIVE)**.

---

### B. Remote Sensing VQA Engine (`rs_vqa_engine.py`)
- **Question Coverage**:
  - `presence`: Zero-shot RSICD probability estimation against land-cover taxonomies.
  - `counting`: Zero-shot proposal extraction via Grounding DINO with non-hardcoded counts.
  - `vegetation`: Spectral Excess Green Index ($2G - R - B$) with spatial coverage percentage.
  - `roads & transportation`: Infrastructure detection and density analysis.
  - `ships & maritime`: Water surface anomaly detection.
  - `spatial localization`: Centroid and quadrant calculation.
  - `structural comparison`: Surrounding context vs built-up ratio comparison.
- **Status**: 🟢 **LIVE (Domain Reasoning)**.

---

### C. Grounding DINO Quality & Diagnostics (`IDEA-Research/grounding-dino-tiny`)
- **Technical Inference**: **100% Genuine Neural Inference** using Zero-Shot Object Detection transformer.
- **Coordinate Convention**: Strict canonical `[xmin, ymin, xmax, ymax]` normalized to `[0, 1000]` coordinate space.
- **Status**: 🟢 **LIVE (Zero-Shot Detection)**.

---

### D. Bi-Temporal Semantic Change Understanding (`semantic_change.py`)
- **Pipeline**: Temporal feature extraction, multi-scale differencing, anomaly clustering, localized bounding box proposal extraction, and Change-VQA natural language resolution.
- **Categorization**: Strict separation between:
  - `PIXEL CHANGE` (Photometric shift / seasonal variance)
  - `OBJECT CHANGE` (Structural appearance / disappearance)
  - `SEMANTIC CHANGE` (Functional land-use transition)
- **Status**: 🟢 **RESEARCH BASELINE (LIVE)**.

---

### E. Multimodal Optical + SAR Physical Fusion (`optical_sar_fusion.py`)
- **Principle**: Processes physical microwave radar signatures (double-bounce dihedral reflections, specular water scatter, volume canopy scatter, speckle variance, dynamic range) against optical multi-spectral reflectance and vegetation indices (ExG).
- **Output Integrity**: Explicitly separates `OPTICAL EVIDENCE`, `SAR EVIDENCE`, and `FUSED CONCLUSION`.
- **Status**: 🟢 **RESEARCH BASELINE (LIVE)**.

---

### F. Co-Registration & High-Resolution Tiling (`coregistration.py`, `tiling.py`)
- **Co-Registration**: Calculates exact geographical bounding intersection, resolution ratio, and spatial overlap. Blocks execution when overlap is under 10% on co-dependent tasks.
- **Tiling**: Sliding window with configurable stride, boundary padding, coordinate remapping, and vectorized Non-Maximum Suppression (NMS).
- **Status**: 🟢 **LIVE**.

---

## 3. Scientific Honesty & Terminology Standards

NexSpace strictly enforces scientific honesty across all system logs, API responses, and user interfaces:

| Prohibited / Unsubstantiated Term | Mandatory Approved Scientific Term | Rationale |
| :--- | :--- | :--- |
| "Confirmed building / structure" | "Model detection candidate" | General VLM bounding boxes have not been validated against ground survey |
| "Confirmed damage / destroyed" | "Candidate anomaly cluster" | Pixel-differencing detects intensity delta, not structural failure |
| "Calibrated confidence (99%)" | "Uncalibrated model confidence" | Raw scores and heuristics are not calibrated posterior probabilities |
| "Trained multimodal fusion AI" | "Research baseline — trained fusion unavailable"| Joint embedding is computed via physical rule fusion, not learned end-to-end model |
| "Evaluated on ISRO dataset" | "Ready for evaluation" | Evaluation scripts prepared; held-out evaluation dataset not yet ingested |
| "Benchmark accuracy: 98%" | "Not run — dataset not available" | Benchmark evaluators run only when official datasets exist locally |
