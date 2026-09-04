# NexSpace ML Model Runtime Documentation

This document describes the model runtime architecture, supported models, execution pipelines, evidence schemas, and fallback behavior for the NexSpace GeoVLM system.

---

## 1. Supported Models & Checkpoints

| Task | Primary Checkpoint | Architecture / Class | Runtime Class | Fallback Mechanism |
|---|---|---|---|---|
| **VQA** | `google/paligemma-3b-ft-rsvqa-lr-224` | `PaliGemmaForConditionalGeneration` | `PaliGemmaVQARuntime` | `rsvqa_heuristic_adapter` (`fallback: true`) |
| **Optical Captioning** | `Salesforce/blip-image-captioning-base` | `BlipForConditionalGeneration` | `BLIPCaptioningRuntime` | `optical_caption_fallback` |
| **SAR Captioning** | `Salesforce/blip-image-captioning-base` | `BlipForConditionalGeneration` | `BLIPCaptioningRuntime` | `sar_caption_fallback` |
| **Spatial Grounding** | `IDEA-Research/grounding-dino-tiny` / `IDEA-Research/grounding-dino-base` | `AutoModelForZeroShotObjectDetection` | `GroundingDINORuntime` | Explicit `unavailable` status (`detections: []`, `evidence: []`) |
| **Change Analysis** | Classical Pixel Diff (NumPy/PIL) | Algorithmic Delta Engine | `ChangeAnalysisTool` | N/A (Native execution) |
| **Bi-Temporal VQA** | `nexspace/bitemporal-change-vlm` | Adapter | `ChangeVQARuntime` | `bitemporal_diff_heuristic` |
| **Optical-SAR Fusion** | `nexspace/optical-sar-cross-fusion` | Multi-stream adapter | `OpticalSARFusionRuntime` | Multi-sensor stream separation with `fusion.status="not_implemented"` |

---

## 2. PaliGemma Remote Sensing VQA Runtime

### Architecture & Runtime Details
- **Classes**: `transformers.PaliGemmaProcessor`, `transformers.PaliGemmaForConditionalGeneration`
- **Default Checkpoint**: `google/paligemma-3b-ft-rsvqa-lr-224` (configurable via `PALIGEMMA_MODEL_ID` or `VQA_MODEL_ID`)
- **Authentication & License**: Gated checkpoint on Hugging Face Hub requiring accepted license terms and a valid `HF_TOKEN` environment variable.
- **Hardware & Device Support**:
  - `CPU`: `torch.float32` full precision execution.
  - `CUDA`: `torch.bfloat16` / `torch.float16` execution when available.
- **Prompt Format**:
  Standard PaliGemma task prompt prefix:
  ```
  "answer en {question}"
  ```
  Examples:
  - `"answer en is there water in this image?"`
  - `"answer en are there buildings visible in this image?"`
  - `"answer en how many buildings are visible?"`
  - `"answer en what type of land cover is visible?"`

### VQA Output Schema & Evidence
```json
{
  "answer": "yes",
  "question": "Is there water in this image?",
  "confidence": null,
  "confidence_type": "unavailable",
  "confidence_source": "google/paligemma-3b-ft-rsvqa-lr-224",
  "model": "google/paligemma-3b-ft-rsvqa-lr-224",
  "device": "cpu",
  "inference_time_ms": 1420.5,
  "fallback": false,
  "success": true
}
```

### Evidence Schema:
```json
{
  "type": "vqa_answer",
  "question": "Is there water in this image?",
  "answer": "yes",
  "model": "google/paligemma-3b-ft-rsvqa-lr-224",
  "confidence": null,
  "confidence_type": "unavailable",
  "confidence_source": "google/paligemma-3b-ft-rsvqa-lr-224",
  "fallback": false,
  "fallback_reason": null
}
```

### Fallback Behavior
When `HF_TOKEN` is not configured or the gated checkpoint is inaccessible, the runtime safely falls back:
```json
{
  "type": "vqa_answer",
  "question": "Is there water in this image?",
  "answer": "yes",
  "model": "rsvqa_heuristic_adapter",
  "confidence": null,
  "confidence_type": "heuristic",
  "confidence_source": "rsvqa_heuristic_adapter",
  "fallback": true,
  "fallback_reason": "MODEL UNAVAILABLE — AUTHENTICATION REQUIRED (Gated checkpoint requires HF_TOKEN with accepted license agreement)"
}
```
No fake model confidence scores or simulated PaliGemma outputs are fabricated.

---

## 3. Grounding DINO Zero-Shot Object Grounding

- **Classes**: `transformers.AutoProcessor`, `transformers.AutoModelForZeroShotObjectDetection`
- **Default Checkpoint**: `IDEA-Research/grounding-dino-tiny` (680 MB) / `IDEA-Research/grounding-dino-base` (1.7 GB)
- **Prompt Format**: Lowercase target phrase with trailing period (e.g. `"buildings."`, `"roads."`).
- **Coordinate Convention**: Cartesian pixel bounding boxes `[x1, y1, x2, y2]`.
- **Evidence Schema**: `type: "bounding_box"`, `source: "Grounding_DINO"`, `label`, `box`, `score`, `image_dimensions`.

---

---

## 4. Optical + SAR Multimodal Feature Fusion Baseline

### Architecture & Numerical Fusion
- **Optical Backbone**: `Salesforce/blip-image-captioning-base.vision_model` (ViT-B, 768-dim pooled feature vector $\mathbf{f}_{opt}$).
- **SAR Backbone**: `generic_vision_encoder_baseline` (ViT-B on 3-channel intensity-preserving inputs, 768-dim pooled feature vector $\mathbf{f}_{sar}$).
- **Fusion Type**: Explicitly labeled `feature_fusion_baseline`.
- **Trained Weights**: `False` (untrained deterministic feature projection & concatenation; no fake trained multimodal weights are claimed).
- **Numerical Operations**:
  - Normalized Unit Vectors: $\hat{\mathbf{f}}_{opt} = \mathbf{f}_{opt} / \|\mathbf{f}_{opt}\|_2$, $\hat{\mathbf{f}}_{sar} = \mathbf{f}_{sar} / \|\mathbf{f}_{sar}\|_2$.
  - Cross-Modal Cosine Similarity: $\rho = \hat{\mathbf{f}}_{opt} \cdot \hat{\mathbf{f}}_{sar} \in [-1, 1]$.
  - Cross-Modal Discrepancy Norm: $d = \|\hat{\mathbf{f}}_{opt} - \hat{\mathbf{f}}_{sar}\|_2$.
  - Fused Joint Representation: $\mathbf{f}_{joint} = [\hat{\mathbf{f}}_{opt}; \hat{\mathbf{f}}_{sar}] \in \mathbb{R}^{1536}$.
- **Alignment Behavior**:
  - `dimension_match_only` when dimensions match (`alignment_warning = False`).
  - `dimension_mismatch_rescaled` when dimensions differ (`alignment_warning = True`, SAR resized to Optical dimensions).
- **Confidence Provenance**: `confidence = None`, `confidence_type = "unavailable"`, `confidence_source = "optical_sar_feature_fusion_baseline"`.

---

## 5. Dynamic Anomaly Extraction & Spatial Evidence Engine

### Architecture & Pipeline
- **Module**: `anomaly_engine.py` (`AnomalyEngine`)
- **Algorithm**:
  1. Multichannel/Luminance difference computation: $\Delta(x, y) = \max_{c \in \{R,G,B\}} |I_B(x, y, c) - I_A(x, y, c)|$.
  2. Dynamic Thresholding:
     - **Otsu's Global Optimal Threshold**: Minimizes intra-class intensity variance $\sigma_w^2(t)$.
     - **Percentile Threshold**: 95th percentile of non-zero delta distribution.
     - **Fixed Threshold**: User-defined intensity cutoff (e.g. 15%).
  3. Connected Component Segmentation: 8-connectivity labeling via `scipy.ndimage.label`.
  4. Noise Filtering: Suppresses isolated speckles with `area_pixels < min_pixel_area` (default: 20 px) or `area_fraction < min_area_fraction`.
  5. Geometric Localization:
     - Pixel Bounding Boxes: `bbox_pixel = [x1, y1, x2, y2]` ($x_1 < x_2$, $y_1 < y_2$, clamped to image dimensions).
     - Normalized Bounding Boxes: `bbox_normalized = [x1/W, y1/H, x2/W, y2/H]` ($\in [0, 1]$).
     - Heuristic Severity: $\text{severity} = \text{clip}(0.6 \cdot \text{change\_score} + 0.4 \cdot \min(1.0, 20 \cdot \text{area\_fraction}), 0.0, 1.0)$ with `severity_score_type = "heuristic"`.
- **Georeferencing**:
  - Inspects GeoTIFF tags (`33550`, `33922`, `34735`).
  - If unavailable: explicitly sets `geospatial_coordinates_available = False` (never fabricates lat/long).
- **Evidence Schema**:
  ```json
  {
    "type": "change_region",
    "id": "change_region_001",
    "bbox_pixel": [100.0, 150.0, 200.0, 250.0],
    "bbox_normalized": [0.1953, 0.293, 0.3906, 0.4883],
    "area_pixels": 10000,
    "area_fraction": 0.038147,
    "mean_intensity_delta": 185.4,
    "max_intensity_delta": 240.0,
    "change_score": 0.7271,
    "severity_score": 0.5898,
    "severity_score_type": "heuristic",
    "source": "classical_change_analysis",
    "geospatial_coordinates_available": false
  }
  ```

---

## 6. Hardware Management & Device Selection

The `DeviceManager` probes available hardware safely:
1. Probes `torch.cuda.is_available()`. If CUDA is available and functional, uses `cuda:0`.
2. Otherwise safely defaults to `cpu`.
3. Hardware load and inference timestamps are measured and recorded in millisecond precision within the `execution_trace`.

---

---

## 8. Execution Telemetry & Lifecycle Tracing

### Architecture & Stage Lifecycle
- **Module**: `telemetry.py` (`ExecutionTrace`, `TraceStage`)
- **Monotonic Timing**: Microsecond resolution using Python `time.perf_counter()` for all duration measurements.
- **Timestamps**: Real chronological ISO-8601 UTC timestamps via `datetime.now(timezone.utc).isoformat()`.
- **Pipeline Stages**:
  1. `request_received`: Query arrival and payload metadata inspection.
  2. `input_validation`: Strict structural and dimensional checks for queries and imagery.
  3. `intent_classification`: Intent rules and regex disambiguation.
  4. `task_planning`: Explicit deterministic `QueryPlan` construction (`reasoning_basis: "intent_rules"`).
  5. `tool_selection`: Specialist tool resolution with multi-tool chaining.
  6. `parameter_extraction`: Natural language parameter parsing.
  7. `tool_execution`: Error-isolated concurrent execution with sub-stage tracking.
  8. `evidence_extraction`: Node extraction and provenance tagging.
  9. `evidence_validation`: Mathematical and geometric validity checks.
  10. `result_synthesis`: Objective observation synthesis and limitation aggregation.
  11. `response_validation`: Output serialization and contract verification.
  12. `response_completed`: Final duration recording.

---

## 9. Evidence Graph & Provenance Validation

### Architecture
- **Module**: `evidence_graph.py` (`EvidenceGraph`, `EvidenceNode`)
- **Traceability Link**:
  ```
  Query (ID) ──► Task ──► Tool ──► Model/Algorithm ──► Evidence Node (ev_xxx) ──► Response Claim
  ```
- **Validation Rules**:
  - **Unique IDs**: Rejects duplicate node identifiers.
  - **Numeric Sanity**: Rejects `NaN`, `Infinity`, or out-of-bounds probabilities.
  - **Geometric Validity**: Strict bounding box coordinates ($x_1 < x_2$, $y_1 < y_2$, $0 \le \text{norm} \le 1$).
  - **Provenance Integrity**: Requires verified `source_tool` and `source_model`.

---

## 10. Investigation Report Schema & Result Synthesis

### Canonical Investigation Object
```json
{
  "investigation_report": {
    "query": "Compare these images, locate buildings, and tell me if there is water",
    "task": "MULTI_TASK",
    "plan": {
      "task_type": "MULTI_TASK",
      "selected_tools": ["Change_Analysis", "Anomaly_Extraction", "Grounding", "VQA"],
      "reasoning_basis": "intent_rules",
      "parameters": {
        "query": "...",
        "grounding_targets": ["buildings"],
        "cross_modal": false
      }
    },
    "observations": [
      "Bi-temporal change analysis measured 12.4% pixel perturbation...",
      "Grounding DINO detected 5 candidate region(s) matching 'buildings'...",
      "PaliGemma VQA inferred answer: 'water body present'."
    ],
    "evidence": [...],
    "limitations": [
      "VQA operated in fallback mode (deterministic heuristic adapter).",
      "Geospatial coordinates (lat/lon) are unavailable from the source image metadata."
    ],
    "execution_summary": {
      "tools_attempted": 4,
      "tools_completed": 4,
      "tools_failed": 0,
      "evidence_count": 6,
      "anomaly_count": 2,
      "detection_count": 5,
      "fallback_count": 1,
      "total_duration_ms": 15405.05
    },
    "trace": [...]
  }
}
```

---

## 11. Confidence Normalization & Scientific Transparency

### Allowed Confidence Types
1. **`model`**: Genuine, calibrated model output probabilities (e.g. Grounding DINO box detection scores).
2. **`heuristic`**: Algorithmic rules, area-intensity deltas, or fallback adapters (e.g. Otsu change anomaly severity).
3. **`unavailable`**: Uncalibrated raw logits, greedy text token strings, or untrained baselines (e.g. BLIP, PaliGemma, Optical-SAR cosine alignment).

### Scientific Distinctions
- **OBSERVATION**: Factual computational output (e.g. *"Grounding DINO identified 5 candidate boxes"*).
- **INTERPRETATION**: Carefully qualified deduction (e.g. *"Feature difference indicates possible surface perturbation"*).
- **UNSUPPORTED CLAIM (FORBIDDEN)**: Speculative semantic leaps (e.g. *"A missile struck 5 buildings"*).

---

## 12. Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PALIGEMMA_MODEL_ID` / `VQA_MODEL_ID` | `google/paligemma-3b-ft-rsvqa-lr-224` | Model checkpoint for PaliGemma VQA |
| `CAPTIONING_MODEL_ID` | `Salesforce/blip-image-captioning-base` | Model checkpoint for BLIP Captioning |
| `GROUNDING_MODEL_ID` | `IDEA-Research/grounding-dino-tiny` | Model checkpoint for Grounding DINO |
| `FUSION_BACKBONE_ID` | `Salesforce/blip-image-captioning-base` | Vision backbone for Optical-SAR Feature Fusion |
| `DEVICE` | Auto (`cuda` or `cpu`) | Force execution device |
| `HF_TOKEN` | *None* | Hugging Face access token for gated models (e.g., PaliGemma) |

---

## 13. Geospatial Intelligence & GeoTIFF / CRS / GSD Engine (STEP 9)

### Architecture
- **Modules**:
  - `ml_backend/geospatial.py`: `GeoMetadata`, `GeoTransform`, `GeospatialEngine`.
  - `ml_backend/geojson_export.py`: `export_evidence_to_geojson`.
- **Mathematical Transformation**:
  $$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
  where $(c, f)$ is the top-left tiepoint origin, $a = s_x$ (Easting scale / GSD), and $e = -s_y$ (Northing scale).
- **Coordinate Conversion & Metrics**:
  - **Pixel $\to$ World**: Maps pixel $(x, y)$ to ground coordinate $(X, Y)$ via forward affine matrix.
  - **World $\to$ Pixel**: Inverse transformation mapping $(X, Y)$ back to pixel coordinates $(x, y)$.
  - **Ground Area**: Computes physical surface area in $\text{m}^2$ (for projected CRS) or geodesic latitude-scaled area (for geographic systems).
  - **Geodesic Distance**: WGS84 geodesic (`pyproj.Geod`) or haversine great-circle calculation between coordinates.
- **Interoperability & Standards**:
  - **RFC 7946 GeoJSON**: Exports spatial evidence nodes as `FeatureCollection` with `Polygon` bounding footprints and `Point` centroids.
  - **Projected CRS Reprojection**: Automatically converts local coordinates (e.g. UTM) to WGS84 (`EPSG:4326`) for GeoJSON mapping while preserving native CRS in feature properties.
- **Scientific Integrity Rules**:
  - Never fabricate latitude/longitude from arbitrary pixel indices when geospatial metadata is absent.
  - Plain PNG/JPEG imagery explicitly returns `geospatial_available: false` and maintains `geospatial_coordinates_available = False`.
  - Projected CRS measurements are labeled in native units ($\text{m}^2$ / $\text{m}$) rather than blindly assuming degrees.

---

## 14. Production Hardening, Security & Reliability (STEP 10)

### 1. Security Defenses & Input Validation
- **Central Configuration (`config.py`)**: Environment-driven thresholds with safe defaults (`MAX_UPLOAD_SIZE_MB=25.0`, `MAX_IMAGE_PIXELS=50,000,000`, `MAX_QUERY_LENGTH=1000`).
- **Decompression Bomb Protection**: Pillow `Image.MAX_IMAGE_PIXELS` bound prevents memory exhaustion from malicious image dimensions.
- **Upload Validation**: Decodes base64 with byte length checking, format verification, and safe image loading.
- **Secret Sanitization**: Automated regex scrubbing of `HF_TOKEN`, API keys, passwords, and private tokens from all traces, logs, and error responses (`settings.sanitize_secrets`).
- **No Stack Trace Leakage**: Global exception handlers redact internal paths and stack traces, returning structured JSON error bodies (`HTTP_400`, `HTTP_413`, `HTTP_422`, `HTTP_500`).

### 2. Request-Scoped IDs & Concurrency Safety
- **Unique Request IDs**: Every request receives a unique identifier (`req_<hex12>`) threaded through execution traces, investigation reports, and API responses.
- **Thread-Safe Lazy Loading**: Singletons in `model_runtime.py` utilize `threading.Lock()` to prevent race conditions or duplicate model initialization under concurrent workloads.
- **Request-Scoped Evidence Graphs**: Evidence node IDs are dynamically scoped to each request, eliminating cross-request collision.

### 3. Error Handling & Failure Containment
- **Specialist Tool Failure Isolation**: Each specialist tool executes inside isolated exception barriers. A runtime failure in one tool (e.g. out of memory, gated checkpoint) does not crash the multi-tool pipeline.
- **Factual Fallback Reporting**: When a tool operates in fallback mode, it is transparently recorded in `investigation_report.limitations` and `execution_trace`.

### 4. Standardized HTTP Status Codes
| HTTP Status | Trigger Condition |
|---|---|
| `200 OK` | Query, change analysis, or health check executed successfully |
| `400 Bad Request` | Malformed base64 image data, unsupported file format, or excessive query length |
| `413 Payload Too Large` | Uploaded image file size exceeds `MAX_UPLOAD_SIZE_MB` or image dimensions exceed `MAX_IMAGE_PIXELS` |
| `422 Unprocessable Content` | Schema or Pydantic validation failure |
| `500 Internal Server Error` | Unexpected backend exception (sanitized, stack trace hidden) |
| `503 Service Unavailable` | Required backend model service unreachable |

### 5. Configurable CORS Policy
- Configurable via `ALLOWED_ORIGINS` environment variable (defaults to `*` for development).


