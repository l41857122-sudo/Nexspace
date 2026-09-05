# Remote-Sensing Adaptation & Fine-Tuning Pipeline

## 1. Overview

NexSpace includes a reproducible PEFT / LoRA and Linear Residual Adapter training pipeline (`ml_backend/train_rs_adaptation.py`) designed for Earth Observation datasets.

## 2. Dataset Ingestion

Supported dataset formats:
- **RSICD**: Remote Sensing Image Captioning & Classification Dataset
- **BigEarthNet**: Sentinel-2 / Landsat-8 Multi-Spectral Surface Reflectance
- **PatternNet**: High-resolution Aerial Scene Classification

Expected directory structure:
```
ml_backend/datasets/rsicd/
  ├── train.json
  ├── val.json
  └── images/
      ├── train/
      └── val/
```

## 3. Training Command

To execute adaptation training on your remote-sensing dataset:

```bash
# Verify pipeline integrity (dry-run mode)
python ml_backend/train_rs_adaptation.py --dry-run

# Run full fine-tuning with LoRA
python ml_backend/train_rs_adaptation.py \
  --base-model flax-community/clip-rsicd \
  --data-dir ml_backend/datasets/rsicd \
  --epochs 5 \
  --batch-size 16 \
  --lr 1e-4 \
  --lora-rank 8
```

## 4. Checkpoint Export & Loading

Trained adapter weights are automatically serialized to:
- `ml_backend/weights/rs_adapter/rs_adapter_weights.pt`
- `ml_backend/weights/rs_adapter/adapter_metadata.json`

When these weights are present, `RemoteSensingVisionRuntime` automatically loads them and updates model provenance to `TRAINED MODEL`.
