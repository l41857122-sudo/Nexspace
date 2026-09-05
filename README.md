# 🌌 NexSpace — Multi-Modal Geospatial Intelligence Platform

NexSpace is an auditable Agentic Multi-Modal Remote Sensing Intelligence Platform integrating Remote-Sensing Vision-Language Models (VLM), Zero-Shot Grounding, Bi-Temporal Semantic Change Reasoning, Physical Optical+SAR Microwave Fusion, Multi-Scale High-Resolution Tiling, and Geospatial Intelligence into an interactive investigation terminal.

---

## 🚀 Fresh Developer Machine Setup

### Prerequisites
- **Node.js**: `v18.x` or `v20.x+` ([Download Node.js](https://nodejs.org/))
- **Python**: `3.11.x` ([Download Python 3.11](https://www.python.org/downloads/)) — *Ensure "Add Python to PATH" is checked during installation.*
- **Git**: ([Download Git](https://git-scm.com/))

---

### Option A: Automated Setup on Windows (Recommended)

Run the safe, idempotent batch script from the repository root:

```cmd
setup.bat
```

The `setup.bat` script automatically:
1. Validates Node.js and Python 3.11.x availability in your system PATH.
2. Creates an isolated local Python virtual environment (`.venv`) if one does not already exist.
3. Upgrades `pip` and installs all backend machine learning dependencies from `requirements.txt`.
4. Installs all Node.js and Next.js frontend dependencies via `npm install`.
5. Initializes `.env.local` from `.env.example` if not already present.
6. Verifies system integrity, required file trees, and Python backend module imports.

---

### Option B: Manual Cross-Platform Setup (Windows / macOS / Linux)

```bash
# 1. Clone repository and navigate to root
cd Nexspace

# 2. Configure Python 3.11 Virtual Environment
python -m venv .venv

# Windows activation:
.venv\Scripts\activate
# Linux/macOS activation:
# source .venv/bin/activate

# 3. Upgrade pip and install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install Node.js frontend dependencies
npm install

# 5. Initialize environment variables
cp .env.example .env.local
```

---

## ⚡ Running the Application

### 1. One-Click Live Demo Launcher (FastAPI + Next.js)
```bash
npm run demo
```
The demo runner automatically:
1. Validates Python 3.11+ and Node.js runtimes.
2. Checks port 8000 (FastAPI) and port 3000 (Next.js), safely reusing active services without duplicate spawns.
3. Polls `/api/health` until ML inference runtimes are loaded.
4. Verifies the Next.js API proxy to FastAPI.
5. Displays the active Remote-Sensing ML Capabilities matrix.
6. Automatically opens `http://localhost:3000/query` in your default browser.

### 2. Standard Concurrent Development
```bash
npm run dev
```

### 3. Dedicated Microservice Dev Commands
```bash
# Frontend only (Port 3000)
npm run dev:next

# FastAPI ML Backend only (Port 8000)
npm run dev:ml
```

---

## 🛰️ Remote-Sensing ML Capabilities Matrix

| Capability | Model / Engine | Real vs Baseline vs Fallback | Status |
| :--- | :--- | :--- | :--- |
| **Domain-Adapted Vision Core** | `flax-community/clip-rsicd` | **REAL RS-ADAPTED MODEL** | `LOADED` |
| **Adaptation Pipeline (PEFT/LoRA)** | PyTorch Contrastive Adaptation | **TRAINING PIPELINE READY** | `REPRODUCIBLE` |
| **Remote-Sensing VQA** | `rs_vqa_engine.py` (RSICD + DINO + ExG) | **REAL / DOMAIN REASONING** | `LOADED` |
| **Zero-Shot Grounding** | `IDEA-Research/grounding-dino-tiny` | **REAL ZERO-SHOT MODEL** | `LOADED` |
| **Optical Scene Captioning** | `Salesforce/blip-image-captioning-base` | **GENERIC PRETRAINED MODEL** | `LOADED` |
| **Semantic Change Understanding** | `semantic_change.py` | **RESEARCH BASELINE** | `LOADED` |
| **Optical + SAR Physical Fusion** | `optical_sar_fusion.py` (ExG + Microwave) | **RESEARCH BASELINE** | `LOADED` |
| **Rigorous Co-Registration** | `coregistration.py` (CRS/Affine/Overlap) | **REAL GEOSPATIAL VALIDATION** | `LOADED` |
| **High-Res Sliding Tiling + NMS** | `tiling.py` (Normalized [0, 1000] Boxes) | **REAL TILING ENGINE** | `LOADED` |
| **Honest Confidence Scoring** | `confidence_system.py` | **UNCALIBRATED / CALIBRATED** | `ACTIVE` |
| **Benchmark Framework** | `benchmarks/evaluator.py` (RSVQA, VRSBench, etc.) | **EVALUATION READY** | `ACTIVE` |
| **ISRO/SAC Adapter** | `isro_evaluation.py` (Cartosat-2S + RISAT) | **READY FOR EVALUATION** | `ACTIVE` |

---

## 🧪 Testing & Verification Suites

```bash
# Run all backend regression unit and integration tests
python ml_backend/run_all_backend_tests.py

# Run frontend schema & API contract validation tests
node test_frontend_schemas.js

# Run Next.js production build verification
npm run build

# TypeScript validation
npx tsc --noEmit
```

---

## 📚 Technical Documentation

Detailed architectural and scientific disclosures are available in the [`docs/`](docs/) directory:
- [Model Architecture & Provenance](docs/model-architecture.md)
- [PEFT / Adaptation Pipeline](docs/training.md)
- [Benchmark Evaluation Framework](docs/evaluation.md)
- [Supported Datasets Reference](docs/datasets.md)
- [Semantic Change & Change-VQA](docs/change-vqa.md)
- [Multimodal Optical + SAR Fusion](docs/optical-sar.md)
- [Model Quality & Scientific Honesty](ml_backend/MODEL_QUALITY.md)
