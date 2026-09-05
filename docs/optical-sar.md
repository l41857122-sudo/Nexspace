# Multimodal Optical + SAR Joint Analysis

This document describes the physical microwave radar and multi-spectral optical fusion architecture implemented in `ml_backend/optical_sar_fusion.py` and wrapped by `OpticalSARAnalysisTool` in `ml_backend/tools.py`.

---

## 1. Physical Modality Principles

NexSpace does **not** simply concatenate generic captions from optical and SAR images. Instead, it processes the complementary electromagnetic properties of both sensors:

| Attribute | Optical Remote Sensing (e.g., Cartosat, Sentinel-2) | Synthetic Aperture Radar (SAR, e.g., RISAT, Sentinel-1) |
|---|---|---|
| **Wavelength / Band** | Visible / Near-Infrared ($\lambda \approx 0.4 - 1.0\,\mu\text{m}$) | Microwave C-band / X-band / L-band ($\lambda \approx 3 - 25\,\text{cm}$) |
| **Physical Interaction** | Surface reflectance, pigment absorption, cloud sensitive | Dielectric constant, surface roughness, structural geometry |
| **Dominant Signatures** | Spectral coloration, shadow definition, vegetation vigor (ExG, NDVI) | Double-bounce dihedral reflections (buildings, ships), specular reflection (smooth water) |
| **Atmospheric Sensitivity**| Cloud, haze, illumination dependent (day-only) | All-weather, day/night penetration |

---

## 2. Fusion Pipeline Architecture

```text
    OPTICAL SENSOR (TIF/PNG)                    SAR SENSOR (TIF/PNG)
               │                                         │
               ▼                                         ▼
   1. OPTICAL EXTRACTION                     1. SAR BACKSCATTER ANALYSIS
   • RSICD Zero-Shot Land Cover              • Speckle noise & intensity stats
   • Excess Green (ExG) Vegetation           • Dynamic range & peak backscatter
   • Solar Shadow / Reflectance              • Double-bounce dihedral identification
               │                                         │
               └────────────────────┬────────────────────┘
                                    ▼
                         2. MULTIMODAL FUSION ENGINE
                         • Modality-specific evidence generation
                         • Cross-modal consistency / discrepancy check
                         • Joint Physical Reasoning
                                    ▼
                         3. STRUCTURED OUTPUT REPORT
                         • OPTICAL EVIDENCE
                         • SAR EVIDENCE
                         • FUSED CONCLUSION
```

---

## 3. Discrepancy & Fusion Logic

The engine resolves common real-world earth-observation phenomena:

1. **Cloud Penetration / Camouflage:**
   - *Optical Observation:* Obscured by cloud cover or haze.
   - *SAR Observation:* Strong high-intensity backscatter with structural patterns.
   - *Fused Conclusion:* "Sub-canopy / cloud-penetrated built-up structures detected despite optical occlusion."

2. **Specular Water vs. Smooth Pavement:**
   - *Optical Observation:* Dark region / absorption signature.
   - *SAR Observation:* Specular microwave scatter (< -20 dB).
   - *Fused Conclusion:* "Confirmed open calm water body with high certainty."

3. **Vegetation Canopy vs. Rough Soil:**
   - *Optical Observation:* High greenness / ExG index.
   - *SAR Observation:* Volumetric diffuse scattering.
   - *Fused Conclusion:* "Dense vegetative canopy / agricultural biomass."

---

## 4. Output Contract

Every multimodal execution outputs explicitly separated sections:

```json
{
  "optical_evidence": "Visible roof structures, asphalt pathways, and adjacent green vegetation detected via spectral reflectance.",
  "sar_evidence": "Strong double-bounce microwave backscatter (dynamic range 19.8 dB) indicating prominent vertical dihedral structures.",
  "fused_conclusion": "Both modalities consistently support the presence of a reinforced built-up industrial complex with surrounding vegetation.",
  "modality_agreement": "HIGH",
  "confidence": 0.88,
  "confidence_source": "uncalibrated_model",
  "model_id": "flax-community/clip-rsicd + sar-backscatter-fusion-baseline",
  "status": "RESEARCH BASELINE"
}
```
In accordance with the **Scientific Honesty Rule**, where a fully end-to-end trained optical-SAR cross-attention transformer checkpoint is unavailable, it is explicitly tagged as `RESEARCH BASELINE — trained multimodal fusion checkpoint unavailable`.
