"""
server.py
---------
Production-hardened FastAPI REST server exposing the SatQuery AI ML CV Controller.

Features:
- Centralized configuration and resource thresholds (MAX_UPLOAD_SIZE_MB, MAX_IMAGE_PIXELS)
- Robust input validation and defense against decompression bombs / malformed payloads
- Request-scoped IDs and auditable execution provenance
- Strict error containment and secret redaction (no stack traces leaked)
- Standardized HTTP status codes (200, 400, 413, 422, 500)
- Configurable CORS policy

Endpoints:
  GET  /api/health
  POST /api/query
  POST /api/change-analysis
  POST /api/geojson
"""

import base64
import io
import os
import sys
import uuid
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from PIL import Image

# Ensure ml_backend folder is in sys.path
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from config import settings
from orchestrator import GeoVLMController
import change_analysis
from anomaly_engine import anomaly_engine
from geospatial import GeospatialEngine, GeoMetadata
from geojson_export import export_evidence_to_geojson

# Protect against PIL decompression bomb vulnerabilities
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS

app = FastAPI(
    title=settings.SERVICE_NAME,
    description="Production-Grade Agentic Vision-Language Controller API for Geospatial Remote Sensing",
    version=settings.SERVICE_VERSION,
)

# Configurable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS if settings.ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = GeoVLMController()


# ---------------------------------------------------------------------------
# Global Exception Handlers (Prevent Stack Trace & Secret Leakage)
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    req_id = getattr(request.state, "request_id", f"req_{uuid.uuid4().hex[:12]}")
    sanitized_detail = settings.sanitize_secrets(str(exc.detail))
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": req_id,
            "status": "error",
            "detail": sanitized_detail,
            "error_code": f"HTTP_{exc.status_code}",
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", f"req_{uuid.uuid4().hex[:12]}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "request_id": req_id,
            "status": "validation_error",
            "detail": "Request body validation failed.",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", f"req_{uuid.uuid4().hex[:12]}")
    # Log internally (with sanitization), but never expose raw trace to client
    print(f"[{req_id}] Internal Server Error: {settings.sanitize_secrets(str(exc))}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "request_id": req_id,
            "status": "internal_error",
            "detail": "An internal error occurred while processing the geospatial query.",
        },
    )


# ---------------------------------------------------------------------------
# Safe Base64 Image Processing
# ---------------------------------------------------------------------------

def decode_b64_image(b64_str: Optional[str], field_name: str = "image") -> Optional[Image.Image]:
    """
    Decodes and validates base64 image strings with strict size, format,
    and decompression protection checks.
    """
    if not b64_str:
        return None

    if not isinstance(b64_str, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid payload for '{field_name}': expected base64 string.",
        )

    # Check raw string length to reject enormous payloads before decoding
    max_b64_len = int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 * 1.4)
    if len(b64_str) > max_b64_len:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image in field '{field_name}' exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    raw_payload = b64_str.split(",")[1] if "," in b64_str else b64_str
    # Strip all whitespace, newlines, and trailing characters from browser FileReader
    clean_str = "".join(raw_payload.split())
    # Add missing base64 padding if needed
    pad_len = len(clean_str) % 4
    if pad_len != 0:
        clean_str += "=" * (4 - pad_len)

    try:
        data = base64.b64decode(clean_str)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed base64 encoding in field '{field_name}'.",
        )

    if len(data) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Decoded image in field '{field_name}' exceeds {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    try:
        bio = io.BytesIO(data)
        img = Image.open(bio)
        # Check image pixel dimensions
        w, h = img.size
        if w * h > settings.MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image dimensions ({w}x{h} = {w*h} pixels) exceed safety limit of {settings.MAX_IMAGE_PIXELS} pixels.",
            )
        # Load image data into memory
        img.load()
        return img.convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to parse image in field '{field_name}': corrupted or unsupported file format.",
        )


def encode_image_b64(img: Image.Image) -> str:
    """Encodes a PIL Image to a base64 PNG data URI string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# API Request Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(default="", max_length=1000)
    optical_image: Optional[str] = None
    sar_image: Optional[str] = None
    change_image_a: Optional[str] = None
    change_image_b: Optional[str] = None
    probe_features: Optional[List[str]] = None


class ChangeRequest(BaseModel):
    image_a: str
    image_b: str
    change_threshold: float = Field(default=0.15, ge=0.0, le=1.0)


class GeoJSONExportRequest(BaseModel):
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    image: Optional[str] = None


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health(request: Request):
    """
    Lightweight health and capability check.
    Does NOT load heavy model checkpoints into memory.
    """
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    hf_token_set = bool(os.environ.get("HF_TOKEN"))
    return {
        "request_id": req_id,
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "capabilities": {
            "captioning": "available",
            "grounding": "available",
            "vqa": "available" if hf_token_set else "adapter_available",
            "change_analysis": "available",
            "anomaly_extraction": "available",
            "optical_sar_fusion": "baseline_available",
            "geospatial": "available",
        },
    }


@app.post("/api/query")
def process_query(req: QueryRequest, request: Request):
    """
    Executes an end-to-end agentic remote sensing query with multi-specialist orchestration.
    """
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = req_id

    # Validate query length
    if len(req.query) > settings.MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query exceeds maximum character limit of {settings.MAX_QUERY_LENGTH}.",
        )

    # Safe decoding with format and size checking
    optical_img = decode_b64_image(req.optical_image, field_name="optical_image")
    sar_img = decode_b64_image(req.sar_image, field_name="sar_image")
    change_a_img = decode_b64_image(req.change_image_a, field_name="change_image_a")
    change_b_img = decode_b64_image(req.change_image_b, field_name="change_image_b")

    res = controller.handle_request(
        query=req.query,
        optical_image=optical_img,
        sar_image=sar_img,
        change_image_a=change_a_img,
        change_image_b=change_b_img,
        probe_features=req.probe_features,
        request_id=req_id,
    )

    # Geospatial Metadata & GeoJSON Export
    target_img = optical_img or change_b_img or change_a_img or sar_img
    geo_meta = GeospatialEngine.extract_metadata(target_img) if target_img else GeoMetadata(geospatial_available=False)
    res["request_id"] = req_id
    res["geospatial_metadata"] = geo_meta.to_dict()
    res["geojson"] = export_evidence_to_geojson(res.get("evidence", []), geo_meta)

    # If change analysis was performed, include heatmap overlay as base64
    if change_a_img and change_b_img:
        try:
            chg = change_analysis.analyze(change_a_img, change_b_img)
            if res.get("change_analysis"):
                res["change_analysis"]["overlay_image"] = encode_image_b64(chg.overlay)
                res["change_analysis"]["heatmap_image"] = encode_image_b64(chg.heatmap)
        except Exception as e:
            print(f"[{req_id}] Error encoding change overlay: {e}")

    return res


@app.post("/api/change-analysis")
def process_change(req: ChangeRequest, request: Request):
    """
    Executes bi-temporal differential change analysis and dynamic anomaly extraction.
    """
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = req_id

    img_a = decode_b64_image(req.image_a, field_name="image_a")
    img_b = decode_b64_image(req.image_b, field_name="image_b")

    if img_a is None or img_b is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both 'image_a' and 'image_b' must be provided and valid base64 image strings.",
        )

    res = change_analysis.analyze(img_a, img_b, change_threshold=req.change_threshold)
    anom = anomaly_engine.extract_change_anomalies(
        img_a,
        img_b,
        threshold_strategy="otsu",
        custom_threshold=req.change_threshold,
    )
    regions = anom.get("regions", [])

    geo_meta = GeospatialEngine.extract_metadata(img_b or img_a)
    geojson_data = export_evidence_to_geojson(regions, geo_meta)

    return {
        "request_id": req_id,
        "status": "success",
        "summary": res.summary,
        "changed_fraction": res.changed_fraction,
        "mean_intensity_delta": res.mean_intensity_delta,
        "overlay_image": encode_image_b64(res.overlay),
        "heatmap_image": encode_image_b64(res.heatmap),
        "anomalies": regions,
        "evidence": regions,
        "geospatial_coordinates_available": geo_meta.geospatial_available,
        "geospatial_metadata": geo_meta.to_dict(),
        "geojson": geojson_data,
        "anomaly_summary": {
            "total_regions": len(regions),
            "total_changed_pixels": anom.get("total_changed_pixels", 0),
            "changed_fraction": anom.get("changed_fraction", 0.0),
            "threshold_method": anom.get("threshold_method", "otsu_optimal_variance"),
            "threshold_value_255": anom.get("threshold_value_255", 38.25),
        },
    }


@app.post("/api/geojson")
def export_geojson_endpoint(req: GeoJSONExportRequest, request: Request):
    """
    Converts arbitrary spatial evidence nodes into an RFC 7946 GeoJSON FeatureCollection.
    """
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    request.state.request_id = req_id

    img = decode_b64_image(req.image, field_name="image") if req.image else None
    geo_meta = GeospatialEngine.extract_metadata(img) if img else GeoMetadata(geospatial_available=False)
    geojson_res = export_evidence_to_geojson(req.evidence, geo_meta)
    geojson_res["request_id"] = req_id
    return geojson_res


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
