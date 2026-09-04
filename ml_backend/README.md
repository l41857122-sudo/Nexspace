# Geospatial Vision-Language Interface Controller (ML CV Core)

Agentic controller & ML CV pipeline for the NexSpace system.

## Modules Overview

- `router.py` — Pure logic query router. Classifies open-ended vs. counting vs. closed-ended queries, normalizes phrasing into RSVQA style, detects low-confidence counting queries, and builds `target_tools` decision JSON.
- `tools.py` — Hugging Face pipeline wrappers around `google/paligemma-3b-ft-rsvqa-lr-224` (VQA) and `Salesforce/blip-image-captioning-base` (Optical & SAR scene captioning).
- `change_analysis.py` — Classical pixel-diff pipeline generating change heatmaps, pixel deltas, and textual change summaries for co-registered bi-temporal pairs.
- `orchestrator.py` — `GeoVLMController` linking router, model wrappers, change analysis, and response synthesis into auditable execution traces.
- `server.py` — FastAPI REST application providing `/api/query`, `/api/change-analysis`, and `/api/health`.
- `example.py` — Runnable CLI test script covering all PRD scenarios.

## Setup & Running

```bash
cd ml_backend
pip install -r requirements.txt
python server.py
# Server runs at http://localhost:8000
```
