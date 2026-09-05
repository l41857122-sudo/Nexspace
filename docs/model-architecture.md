# NexSpace Multimodal Remote-Sensing AI Architecture

## 1. System Overview

NexSpace is a modular, agentic Vision-Language & Multimodal AI system tailored for satellite and aerial Earth Observation (EO) imagery. The system orchestrates multiple vision-language models, specialized remote-sensing neural backbones, classical differential computer vision algorithms, and geospatial georeferencing engines.

```
                  +-----------------------------------+
                  |        User Natural Query         |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |   Intent Classifier & Router      |
                  +-----------------+-----------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
    +---------------+       +---------------+       +---------------+
    | RS-VQA Engine |       |Semantic Change|       |  Optical+SAR  |
    |   (PaliGemma/ |       | Engine (Diff/ |       | Fusion Engine |
    |   CLIP-RSICD) |       |  Otsu/Ground) |       | (ViT/Physics) |
    +-------+-------+       +-------+-------+       +-------+-------+
            |                       |                       |
            +-----------------------+-----------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Confidence Provenance Evaluator  |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Investigation Synthesizer & Trace |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  | NexSpace Frontend / Evidence API  |
                  +-----------------------------------+
```

---

## 2. Capability Matrix & Operational Status

| Model / Subsystem | Primary Checkpoint / Backbone | Modality | Operational Status | Provenance Classification |
| :--- | :--- | :--- | :--- | :--- |
| **RS-VQA Core** | `google/paligemma-3b-ft-rsvqa-lr-224` | Optical Overhead | **LIVE (Gated Auth / RS Feature Engine)** | `REAL RS-ADAPTED MODEL` |
| **Zero-Shot RS Classifier** | `flax-community/clip-rsicd` | Optical / Multispectral | **LIVE** | `REAL RS-ADAPTED MODEL` |
| **Object Grounding** | `IDEA-Research/grounding-dino-tiny` | Optical Aerial | **LIVE** | `GENERIC PRETRAINED MODEL` |
| **Optical Captioning** | `Salesforce/blip-image-captioning-base` | Optical | **LIVE** | `GENERIC PRETRAINED MODEL` |
| **Semantic Change Engine** | Multi-temporal Otsu + RS Grounding | Bi-temporal Optical | **LIVE (Baseline)** | `RESEARCH BASELINE` |
| **Optical + SAR Fusion** | Dual-Backbone Physical Synthesizer | Optical + C-band SAR | **LIVE (Baseline)** | `RESEARCH BASELINE` |
| **Co-Registration Engine** | Affine Transform & Footprint Overlap | Spatial Multi-Sensor | **LIVE** | `REAL GEOSPATIAL ENGINE` |
| **Tiled Inference Engine** | Sliding Window + Vectorized NMS | High-Res Satellite | **LIVE** | `REAL GEOSPATIAL ENGINE` |
| **ISRO/SAC Adapter** | Cartosat-2S + RISAT Pipeline | Multi-Sensor | **INITIALIZED** | `READY FOR EVALUATION` |

---

## 3. Provenance & Scientific Classification Definitions

- **REAL RS-ADAPTED MODEL**: Checkpoint specifically trained or adapted on remote-sensing imagery (e.g., RSICD, RSVQA).
- **GENERIC PRETRAINED MODEL**: Deep learning model trained on generic ground photography (COCO, Visual Genome) with nadir domain shift.
- **RESEARCH BASELINE**: Algorithmic and multi-sensor fusion architecture executing unlearned feature comparison.
- **HEURISTIC FALLBACK**: Algorithmic / deterministic rule engine activated when deep models are unauthenticated or offline.
- **READY FOR EVALUATION**: Ingestion pipeline and evaluation hooks prepared for held-out evaluation datasets without pre-claiming scores.
