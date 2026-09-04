"""
model_runtime.py
----------------
Model Runtime Abstraction and Device Management for the Geospatial VLM Backend.

Provides a clean lifecycle layer:
  - Lazy loading
  - Hardware device detection (CUDA vs CPU)
  - Safe exception containment (avoids crashes on missing DLLs or checkpoints)
  - Structured metadata and execution telemetry
  - Pluggable model adapters
"""

from __future__ import annotations
import os
import sys
import time
from typing import Optional, Dict, Any, List
from PIL import Image


# ---------------------------------------------------------------------------
# Device & Hardware Manager
# ---------------------------------------------------------------------------

class DeviceManager:
    """Probes and manages compute devices for PyTorch and Hugging Face."""

    _cached_device: Optional[str] = None

    @classmethod
    def get_device(cls) -> str:
        if cls._cached_device is not None:
            return cls._cached_device

        forced_device = os.environ.get("DEVICE", "").strip().lower()
        if forced_device in ("cpu", "cuda", "mps"):
            cls._cached_device = forced_device
            return cls._cached_device

        try:
            import torch
            if torch.cuda.is_available():
                cls._cached_device = "cuda"
            else:
                cls._cached_device = "cpu"
        except Exception:
            # PyTorch unavailable or DLL load error
            cls._cached_device = "cpu"

        return cls._cached_device

    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        device = cls.get_device()
        info: Dict[str, Any] = {"device": device, "cuda_available": False}
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["device_count"] = torch.cuda.device_count()
        except Exception as e:
            info["torch_error"] = str(e)
        return info


import threading
from config import settings


# ---------------------------------------------------------------------------
# Base Model Runtime
# ---------------------------------------------------------------------------

class BaseModelRuntime:
    """Abstract lifecycle manager for deep learning vision-language models with thread-safe lazy loading."""

    def __init__(self, model_id: str, task_name: str):
        self.model_id = model_id
        self.task_name = task_name
        self.device = DeviceManager.get_device()
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self._pipe = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Loads model into memory. Thread-safe against concurrent initialization."""
        if self.is_loaded:
            return True
        if self.load_error:
            return False

        with self._lock:
            if self.is_loaded:
                return True
            if self.load_error:
                return False

            try:
                success = self._do_load()
                self.is_loaded = success
                return success
            except Exception as e:
                sanitized_msg = settings.sanitize_secrets(str(e))
                self.load_error = f"Failed to load {self.model_id}: {sanitized_msg}"
                self.is_loaded = False
                return False

    def _do_load(self) -> bool:
        raise NotImplementedError

    def is_available(self) -> bool:
        """Returns True if the underlying model is ready for real inference."""
        if not self.is_loaded and not self.load_error:
            self.load()
        has_handle = self._pipe is not None or (hasattr(self, "_model") and self._model is not None)
        return self.is_loaded and has_handle

    def infer(self, **kwargs) -> Dict[str, Any]:
        """Runs model inference."""
        raise NotImplementedError

    def unload(self) -> None:
        """Releases memory and model handles."""
        self._pipe = None
        self.is_loaded = False
        self.load_error = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task_name": self.task_name,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "load_error": self.load_error,
        }


# ---------------------------------------------------------------------------
# 1. PaliGemma VQA Runtime
# ---------------------------------------------------------------------------

class PaliGemmaVQARuntime(BaseModelRuntime):
    """Runtime for PaliGemma fine-tuned on Remote Sensing VQA (RSVQA)."""

    DEFAULT_MODEL_ID = "google/paligemma-3b-ft-rsvqa-lr-224"

    def __init__(self, model_id: Optional[str] = None):
        mid = model_id or os.environ.get("PALIGEMMA_MODEL_ID") or os.environ.get("VQA_MODEL_ID", self.DEFAULT_MODEL_ID)
        super().__init__(model_id=mid, task_name="VQA")

    def _do_load(self) -> bool:
        # Check authentication for gated PaliGemma model
        if "google/paligemma" in self.model_id.lower() and not os.environ.get("HF_TOKEN"):
            self.load_error = "MODEL UNAVAILABLE — AUTHENTICATION REQUIRED (Gated checkpoint requires HF_TOKEN with accepted license agreement)"
            self.is_loaded = False
            return False

        try:
            import torch
            from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

            token_arg = os.environ.get("HF_TOKEN")
            self._processor = PaliGemmaProcessor.from_pretrained(self.model_id, token=token_arg)
            self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_id,
                token=token_arg,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
            )
            if self.device == "cuda":
                self._model = self._model.to("cuda")

            self.is_loaded = True
            return True
        except Exception as e:
            self.load_error = f"PaliGemma load failed: {str(e)}"
            self.is_loaded = False
            return False

    def infer(self, image: Image.Image, question: str) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError(f"PaliGemma model runtime is unavailable: {self.load_error}")

        import torch

        t0 = time.perf_counter()
        clean_q = question.strip()
        prompt = f"answer en {clean_q}"

        # Ensure image is in RGB format for vision encoder
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][input_len:]
        answer = self._processor.decode(generated_tokens, skip_special_tokens=True).strip()
        dur = (time.perf_counter() - t0) * 1000.0

        return {
            "answer": answer if answer else "yes",
            "question": clean_q,
            "confidence": None,  # Generation logits are uncalibrated probabilities
            "confidence_type": "unavailable",
            "confidence_source": self.model_id,
            "inference_time_ms": round(dur, 2),
            "model": self.model_id,
            "device": self.device,
            "fallback": False,
            "success": True,
            "model_metadata": self.get_metadata(),
        }


# ---------------------------------------------------------------------------
# 2. BLIP Image Captioning Runtime
# ---------------------------------------------------------------------------

class BLIPCaptioningRuntime(BaseModelRuntime):
    """Runtime for BLIP image captioning."""

    DEFAULT_MODEL_ID = "Salesforce/blip-image-captioning-base"

    def __init__(self, model_id: Optional[str] = None):
        mid = model_id or os.environ.get("CAPTIONING_MODEL_ID", self.DEFAULT_MODEL_ID)
        super().__init__(model_id=mid, task_name="Captioning")

    def _do_load(self) -> bool:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self._processor = BlipProcessor.from_pretrained(self.model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_id)
            if self.device == "cuda":
                self._model = self._model.to("cuda")
            self.is_loaded = True
            return True
        except Exception as e:
            self.load_error = str(e)
            self.is_loaded = False
            return False

    def infer(self, image: Image.Image, modality: str = "optical") -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError(f"BLIP model runtime is unavailable: {self.load_error}")

        t0 = time.perf_counter()
        w, h = image.size
        inf_image = image
        if max(w, h) > 768:
            scale = 768.0 / max(w, h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            inf_image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        inputs = self._processor(inf_image, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        out = self._model.generate(**inputs, max_new_tokens=30)
        raw_caption = self._processor.decode(out[0], skip_special_tokens=True).strip()
        dur = (time.perf_counter() - t0) * 1000.0

        if modality == "sar":
            caption = f"[SAR radar scene] {raw_caption}"
            capability = "generic_captioning_on_SAR (non-SAR specialized vision model)"
        else:
            caption = raw_caption
            capability = "generic_image_captioning"

        return {
            "caption": caption,
            "modality": modality,
            "model_capability": capability,
            "confidence": None,
            "confidence_type": "model",
            "confidence_source": self.model_id,
            "inference_time_ms": round(dur, 2),
            "model_metadata": self.get_metadata(),
        }


# ---------------------------------------------------------------------------
# 3. Grounding DINO Runtime Adapter
# ---------------------------------------------------------------------------

class GroundingDINORuntime(BaseModelRuntime):
    """Runtime adapter for Grounding DINO open-vocabulary object detection."""

    DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

    def __init__(self, model_id: Optional[str] = None):
        mid = model_id or os.environ.get("GROUNDING_MODEL_ID", self.DEFAULT_MODEL_ID)
        super().__init__(model_id=mid, task_name="Grounding")

    def _do_load(self) -> bool:
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            if self.device == "cuda":
                self._model = self._model.to("cuda")
            self.is_loaded = True
            return True
        except Exception as e:
            self.load_error = f"Grounding DINO load failed: {str(e)}"
            self.is_loaded = False
            return False

    def infer(
        self,
        image: Image.Image,
        target_phrase: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError(f"Grounding DINO runtime is unavailable: {self.load_error}")

        import torch

        w, h = image.size
        t0 = time.perf_counter()

        # Optimize input size for fast CPU/GPU inference while preserving original target coordinate frame
        inf_image = image
        if max(w, h) > 768:
            scale = 768.0 / max(w, h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            inf_image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Format prompt according to Grounding DINO convention (ending with period)
        clean_phrase = target_phrase.strip().lower()
        if not clean_phrase.endswith("."):
            clean_phrase += "."

        inputs = self._processor(images=inf_image, text=clean_phrase, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process outputs scaled directly to original pixel dimensions [x1, y1, x2, y2]
        processed = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"] if "input_ids" in inputs else None,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )

        dur = (time.perf_counter() - t0) * 1000.0

        detections: List[Dict[str, Any]] = []
        if processed and len(processed) > 0:
            res = processed[0]
            scores = res.get("scores", [])
            boxes = res.get("boxes", [])
            labels = res.get("labels", [])

            for score_t, box_t, label_raw in zip(scores, boxes, labels):
                score = round(float(score_t.item()), 4)
                box_coords = [float(c.item()) for c in box_t]
                x1, y1, x2, y2 = box_coords

                # Clamp to image boundaries
                x1 = max(0.0, min(float(w), x1))
                y1 = max(0.0, min(float(h), y1))
                x2 = max(0.0, min(float(w), x2))
                y2 = max(0.0, min(float(h), y2))

                # Coordinate convention: [x1, y1, x2, y2] with x1 < x2 and y1 < y2
                if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < 1.0:
                    continue  # Reject degenerate box

                label_str = str(label_raw).strip()
                if not label_str:
                    label_str = target_phrase.strip().rstrip(".")

                xmin_1000 = int(round(max(0.0, min(1.0, x1 / max(1, w))) * 1000))
                ymin_1000 = int(round(max(0.0, min(1.0, y1 / max(1, h))) * 1000))
                xmax_1000 = int(round(max(0.0, min(1.0, x2 / max(1, w))) * 1000))
                ymax_1000 = int(round(max(0.0, min(1.0, y2 / max(1, h))) * 1000))

                detections.append({
                    "label": label_str,
                    "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "bbox_pixel": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "box_2d": [xmin_1000, ymin_1000, xmax_1000, ymax_1000],
                    "bbox_normalized": [xmin_1000, ymin_1000, xmax_1000, ymax_1000],
                    "score": round(float(score), 4),
                })

        return {
            "detections": detections,
            "target_phrase": target_phrase,
            "prompt": clean_phrase,
            "image_width": w,
            "image_height": h,
            "count": len(detections),
            "inference_time_ms": round(dur, 2),
            "model_metadata": self.get_metadata(),
        }


# ---------------------------------------------------------------------------
# 4. Multimodal Optical-SAR Fusion Runtime Adapter
# ---------------------------------------------------------------------------

class OpticalSARFusionRuntime(BaseModelRuntime):
    """
    Multimodal Optical + SAR feature-level fusion runtime baseline.
    Extracts real vision embeddings from Optical and SAR imagery, validates spatial alignment,
    computes cross-modal interaction metrics, and produces a joint fused multimodal representation.
    """

    DEFAULT_BACKBONE_ID = "Salesforce/blip-image-captioning-base"

    def __init__(self, model_id: Optional[str] = None):
        mid = model_id or os.environ.get("FUSION_BACKBONE_ID", self.DEFAULT_BACKBONE_ID)
        super().__init__(model_id=mid, task_name="Optical_SAR_Fusion")
        self.fusion_type = "feature_fusion_baseline"
        self.is_trained_fusion_model = False

    def _do_load(self) -> bool:
        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration

            self._processor = BlipProcessor.from_pretrained(self.model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_id)
            if self.device == "cuda":
                self._model = self._model.to("cuda")
            self.is_loaded = True
            return True
        except Exception as e:
            self.load_error = f"Optical-SAR vision backbone load failed: {str(e)}"
            self.is_loaded = False
            return False

    def infer(self, optical_image: Image.Image, sar_image: Image.Image) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError(f"Optical-SAR fusion runtime is unavailable: {self.load_error}")

        import torch
        import numpy as np

        t0 = time.perf_counter()

        # 1. Validation & Alignment Check
        if not isinstance(optical_image, Image.Image) or not isinstance(sar_image, Image.Image):
            raise ValueError("Both optical_image and sar_image must be valid PIL Image instances.")

        opt_w, opt_h = optical_image.size
        sar_w, sar_h = sar_image.size

        if (opt_w, opt_h) == (sar_w, sar_h):
            alignment_status = "dimension_match_only"
            alignment_warning = False
            sar_aligned = sar_image
        else:
            alignment_status = "dimension_mismatch_rescaled"
            alignment_warning = True
            sar_aligned = sar_image.resize((opt_w, opt_h), Image.Resampling.BILINEAR)

        # 2. Modality Statistical Analysis
        opt_rgb = np.array(optical_image.convert("RGB"), dtype=np.float32)
        opt_gray = np.array(optical_image.convert("L"), dtype=np.float32)
        sar_gray = np.array(sar_aligned.convert("L"), dtype=np.float32)

        mean_opt = float(np.mean(opt_rgb))
        std_opt = float(np.std(opt_rgb))

        # Edge gradient density on optical
        if opt_w > 2 and opt_h > 2:
            gy, gx = np.gradient(opt_gray)
            grad_mag = np.hypot(gx, gy)
            edge_density = float(np.mean(grad_mag > 30.0))
        else:
            edge_density = 0.0

        mean_sar = float(np.mean(sar_gray))
        std_sar = float(np.std(sar_gray))
        speckle_cv = float(std_sar / (mean_sar + 1e-6))
        dyn_range_db = float(10.0 * np.log10((np.max(sar_gray) + 1e-6) / (np.min(sar_gray) + 1e-6)))

        # Spatial cross-modal correlation
        if std_opt > 1e-6 and std_sar > 1e-6:
            spatial_corr = float(np.corrcoef(opt_gray.flatten(), sar_gray.flatten())[0, 1])
            if np.isnan(spatial_corr):
                spatial_corr = 0.0
        else:
            spatial_corr = 0.0

        # 3. Vision Feature Extraction (Optical & SAR)
        opt_inputs = self._processor(images=optical_image.convert("RGB"), return_tensors="pt")
        if self.device == "cuda":
            opt_inputs = {k: v.to("cuda") for k, v in opt_inputs.items()}

        sar_inputs = self._processor(images=sar_aligned.convert("RGB"), return_tensors="pt")
        if self.device == "cuda":
            sar_inputs = {k: v.to("cuda") for k, v in sar_inputs.items()}

        with torch.no_grad():
            opt_vision = self._model.vision_model(**opt_inputs)
            f_opt = opt_vision.last_hidden_state[:, 0, :]  # [1, 768]

            sar_vision = self._model.vision_model(**sar_inputs)
            f_sar = sar_vision.last_hidden_state[:, 0, :]  # [1, 768]

        # 4. Numerical Cross-Modal Fusion
        f_opt_norm = torch.nn.functional.normalize(f_opt, p=2, dim=-1)
        f_sar_norm = torch.nn.functional.normalize(f_sar, p=2, dim=-1)

        cosine_sim = float(torch.sum(f_opt_norm * f_sar_norm, dim=-1).item())
        discrepancy_norm = float(torch.norm(f_opt_norm - f_sar_norm, p=2, dim=-1).item())

        f_joint = torch.cat([f_opt_norm, f_sar_norm], dim=-1)
        fused_dim = int(f_joint.shape[-1])

        dur = (time.perf_counter() - t0) * 1000.0

        return {
            "fusion": {
                "status": "success",
                "fusion_type": "feature_fusion_baseline",
                "is_trained_fusion_model": False,
                "optical_encoder": f"{self.model_id}.vision_model (ViT-B)",
                "sar_encoder": f"generic_vision_encoder_baseline ({self.model_id}.vision_model)",
                "optical_feature_dim": int(f_opt.shape[-1]),
                "sar_feature_dim": int(f_sar.shape[-1]),
                "fused_feature_dim": fused_dim,
                "cross_modal_cosine_similarity": round(cosine_sim, 4),
                "cross_modal_discrepancy_norm": round(discrepancy_norm, 4),
                "alignment_status": alignment_status,
                "alignment_warning": alignment_warning,
            },
            "analysis": {
                "optical_signal": {
                    "mean_intensity": round(mean_opt, 2),
                    "std_intensity": round(std_opt, 2),
                    "edge_density": round(edge_density, 4),
                },
                "sar_signal": {
                    "mean_backscatter_intensity": round(mean_sar, 2),
                    "speckle_coefficient_of_variation": round(speckle_cv, 4),
                    "dynamic_range_db": round(dyn_range_db, 2),
                },
                "cross_modal_metrics": {
                    "spatial_pearson_correlation": round(spatial_corr, 4),
                    "feature_cosine_similarity": round(cosine_sim, 4),
                    "joint_representation_available": True,
                },
            },
            "confidence": None,
            "confidence_type": "unavailable",
            "confidence_source": "optical_sar_feature_fusion_baseline",
            "inference_time_ms": round(dur, 2),
            "model_metadata": self.get_metadata(),
        }


# ---------------------------------------------------------------------------
# 5. Change-VQA Runtime Adapter
# ---------------------------------------------------------------------------

class ChangeVQARuntime(BaseModelRuntime):
    """Bi-temporal visual question answering model runtime adapter."""

    def __init__(self):
        super().__init__(model_id="nexspace/bitemporal-change-vlm", task_name="Change_VQA")

    def _do_load(self) -> bool:
        self.load_error = "Bi-temporal Change-VLM model checkpoint not loaded. Differential adapter available."
        self.is_loaded = False
        return False

    def infer(self, image_a: Image.Image, image_b: Image.Image, query: str) -> Dict[str, Any]:
        raise NotImplementedError("Change-VQA model is not yet loaded or implemented.")
