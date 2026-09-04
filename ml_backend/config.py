"""
config.py
---------
Production configuration, environment variables, security thresholds,
and resource limits for the SatQuery AI / NexSpace ML Backend.
"""

from __future__ import annotations
import os
import re
from typing import List, Set


class Settings:
    """Centralized production settings with safe defaults and environment overrides."""

    # Service Metadata
    SERVICE_NAME: str = "SatQuery AI ML CV Engine"
    SERVICE_VERSION: str = "1.0.0"

    # Security & Resource Limits
    # Max image payload size in MB before decoding
    MAX_UPLOAD_SIZE_MB: float = float(os.environ.get("MAX_UPLOAD_SIZE_MB", "25.0"))
    # Max image dimensions in pixels (e.g. 50 Megapixels) to protect against decompression bombs
    MAX_IMAGE_PIXELS: int = int(os.environ.get("MAX_IMAGE_PIXELS", "50000000"))
    # Max query question character length
    MAX_QUERY_LENGTH: int = int(os.environ.get("MAX_QUERY_LENGTH", "1000"))
    # Max number of spatial evidence features allowed in a single response
    MAX_EVIDENCE_ITEMS: int = int(os.environ.get("MAX_EVIDENCE_ITEMS", "500"))

    # Timeouts (in seconds)
    REQUEST_TIMEOUT_SECONDS: float = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "120.0"))
    MODEL_INFERENCE_TIMEOUT_SECONDS: float = float(os.environ.get("MODEL_INFERENCE_TIMEOUT_SECONDS", "60.0"))

    # CORS
    _cors_env = os.environ.get("ALLOWED_ORIGINS", "*")
    ALLOWED_ORIGINS: List[str] = [orig.strip() for orig in _cors_env.split(",") if orig.strip()]

    # Model Checkpoints
    PALIGEMMA_MODEL_ID: str = os.environ.get("PALIGEMMA_MODEL_ID") or os.environ.get("VQA_MODEL_ID") or "google/paligemma-3b-ft-rsvqa-lr-224"
    CAPTIONING_MODEL_ID: str = os.environ.get("CAPTIONING_MODEL_ID", "Salesforce/blip-image-captioning-base")
    GROUNDING_MODEL_ID: str = os.environ.get("GROUNDING_MODEL_ID", "IDEA-Research/grounding-dino-tiny")
    FUSION_BACKBONE_ID: str = os.environ.get("FUSION_BACKBONE_ID", "Salesforce/blip-image-captioning-base")

    # Hardware Device
    DEVICE: str = os.environ.get("DEVICE", "").lower()

    # Secret Sanitization Patterns
    SECRET_PATTERNS = [
        re.compile(r"hf_[A-Za-z0-9]{20,}", re.IGNORECASE),
        re.compile(r"(api[_-]?key|token|secret|password)[\"'\s:=]+([A-Za-z0-9_\-\.]{8,})", re.IGNORECASE),
    ]

    @classmethod
    def sanitize_secrets(cls, text: str) -> str:
        """Sanitizes potential API keys or tokens from logs, error messages, and traces."""
        if not isinstance(text, str):
            return text
        sanitized = text
        for pattern in cls.SECRET_PATTERNS:
            sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
        return sanitized


settings = Settings()
