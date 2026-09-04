# 🌌 NexSpace — Multi-Modal Geospatial Intelligence Platform

NexSpace is a production-grade Agentic Multi-Modal Remote Sensing Intelligence Platform integrating Vision-Language Models (VLM), Zero-Shot Grounding, Bi-Temporal Change Detection, Optical+SAR Multimodal Fusion, and Geospatial Intelligence into an auditable investigation terminal.

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
5. Initializes `.env.local` from `.env.example` if not already present (strictly preserving existing `.env.local`).
6. Verifies system integrity, required file trees, and Python backend module imports.

---

### Option B: Manual Cross-Platform Setup (Windows / macOS / Linux)

If you prefer manual configuration or are on Linux/macOS:

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

# 5. Initialize environment variables (only if .env.local does not exist)
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

## 🛰️ Predefined Live Investigation Scenarios

Inside the NexSpace NLP Terminal (`/query`), click any scenario button for real-time model execution:

| Scenario | Objective | Specialist Model / Tool | Expected Route |
| :--- | :--- | :--- | :--- |
| **1. Satellite Scene Analysis** | Overview of nadir scene infrastructure | Salesforce BLIP Base | `Optical_Caption` |
| **2. Building Detection** | Zero-shot visual bounding box localization | IDEA Grounding DINO Tiny | `Grounding` |
| **3. Remote Sensing VQA** | Natural language land-cover question | PaliGemma / RSVQA Adapter | `VQA` |
| **4. Combined Investigation** | Composite captioning + building detection | BLIP + Grounding DINO | `Optical_Caption` + `Grounding` |
| **5. Temporal Change Analysis** | Bi-temporal change detection & Otsu anomalies | Classical OpenCV Difference | `Change_Analysis` + `Anomaly_Extraction` |
| **6. Optical + SAR Fusion** | Cross-modal 1536-dim joint feature fusion | Dual-Backbone Feature Baseline | `Optical_SAR_Analysis` |
| **7. Geospatial Intelligence** | Real CRS world bounds & ground surface area | GeoTIFF / Affine Coordinate Engine | `Geospatial Engine` |

---

## 🧪 Testing & Verification Suites

```bash
# TypeScript compiler type check
npx tsc --noEmit

# Core ML Controller & Hardening unit tests
python ml_backend/test_step10_hardening.py
python ml_backend/test_step14_caption_safety.py

# End-to-end live API proxy & route integration tests
node test_e2e_live_integration.js
node test_frontend_schemas.js
node test_backend.js
```

---

## 🔬 Scientific Honesty & Methodology Disclosures

Detailed model audits, empirical benchmark metrics, and domain-shift disclosures are documented in [`ml_backend/MODEL_QUALITY.md`](ml_backend/MODEL_QUALITY.md) and [`ml_backend/PRD_COMPLIANCE.md`](ml_backend/PRD_COMPLIANCE.md).
