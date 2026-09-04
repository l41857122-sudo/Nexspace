# 🌌 NexSpace — Multi-Modal Geospatial Intelligence Platform

NexSpace is a production-grade Agentic Multi-Modal Remote Sensing Intelligence Platform integrating Vision-Language Models (VLM), Zero-Shot Grounding, Bi-Temporal Change Detection, Optical+SAR Multimodal Fusion, and Geospatial Intelligence into an auditable investigation terminal.

---

## ⚡ Quick Start: One-Click Live Demo

To launch the full live stack with both FastAPI backend and Next.js frontend and automatically open the investigation terminal in your browser:

```bash
npm run demo
```

The demo runner automatically:
1. Validates Python 3.11+ and Node.js environments.
2. Checks port 8000 (FastAPI) and port 3000 (Next.js), safely reusing active services without duplicate spawns.
3. Polls `/api/health` until backend inference runtimes are ready.
4. Verifies the Next.js API proxy to FastAPI.
5. Displays the active Remote-Sensing ML Capabilities table.
6. Opens `http://localhost:3000/query` in your default browser.

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

## 🛠️ Developer Scripts

```bash
# Start Next.js and FastAPI concurrently
npm run dev

# Launch one-click demo launcher
npm run demo

# Build production Next.js bundle
npm run build

# Run TypeScript type check
npx tsc --noEmit

# Run Next.js and ESLint checks
npm run lint

# Run Backend Python test suites
python ml_backend/test_step10_hardening.py
python ml_backend/test_step13_model_quality.py

# Run Live Integration & Schema validation tests
node test_e2e_live_integration.js
node test_frontend_schemas.js
```

---

## 🔬 Scientific Honesty & Methodology Disclosures

Detailed model audits, empirical benchmark metrics, and domain-shift disclosures are documented in [`ml_backend/MODEL_QUALITY.md`](ml_backend/MODEL_QUALITY.md) and [`ml_backend/PRD_COMPLIANCE.md`](ml_backend/PRD_COMPLIANCE.md).
