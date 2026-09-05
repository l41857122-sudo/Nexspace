# Benchmark Evaluation & ISRO/SAC Readiness

## 1. Supported Public Benchmarks

NexSpace incorporates a unified benchmark evaluation suite (`ml_backend/benchmarks/evaluator.py`):

| Benchmark | Target Task | Primary Evaluation Metrics |
| :--- | :--- | :--- |
| **RSVQA** | Remote-Sensing Visual QA | Top-1 Accuracy, Presence Accuracy, Counting Accuracy |
| **VRSBench** | VQA & Object Grounding | VQA Accuracy, Mean IoU @ 0.50 |
| **CDVQA** | Bi-temporal Change VQA | Change QA Accuracy, Segmentation F1 & IoU |
| **BigEarthNet** | Multi-Label Classification | Mean Average Precision (mAP), Macro/Micro F1 |

## 2. Evaluation Commands

```bash
# Evaluate on RSVQA
python ml_backend/benchmarks/evaluator.py --benchmark RSVQA --data-dir ml_backend/datasets/rsvqa

# Evaluate on CDVQA
python ml_backend/benchmarks/evaluator.py --benchmark CDVQA --data-dir ml_backend/datasets/cdvqa

# Evaluate on VRSBench
python ml_backend/benchmarks/evaluator.py --benchmark VRSBench --data-dir ml_backend/datasets/vrsbench
```

## 3. Scientific Honesty Standards

When dataset files are not located in the target directory, the evaluator generates a structured report with status:
`NOT RUN — DATASET NOT AVAILABLE`

The system never fabricates scores, accuracy percentages, or detection IoU values.

## 4. ISRO / SAC Evaluation Readiness

For the Cartosat-2S and RISAT multi-sensor evaluation track:
- **Adapter**: `ml_backend/isro_evaluation.py`
- **Supported Sensors**: Cartosat-2S (0.65m PAN / 2.0m Multi-Spectral), RISAT-1/2B (C-band SAR)
- **Status**: `READY FOR EVALUATION`
- **Output Submission**: Structured JSON and GeoJSON predictions exported to `ml_backend/isro_evaluation_exports/`.
