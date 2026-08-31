"""
server.py
---------
FastAPI REST server exposing the SatQuery AI ML CV Controller.

Endpoints:
  POST /api/query
  POST /api/change-analysis
  GET  /api/health
"""

import base64
import io
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from orchestrator import GeoVLMController
import change_analysis

app = FastAPI(
    title="SatQuery AI ML CV Engine",
    description="Agentic Vision-Language Controller API for Geospatial Remote Sensing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = GeoVLMController()


def decode_b64_image(b64_str: Optional[str]) -> Optional[Image.Image]:
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"Image decode error: {e}")
        return None


def encode_image_b64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class QueryRequest(BaseModel):
    query: str = ""
    optical_image: Optional[str] = None
    sar_image: Optional[str] = None
    change_image_a: Optional[str] = None
    change_image_b: Optional[str] = None
    probe_features: Optional[List[str]] = None


class ChangeRequest(BaseModel):
    image_a: str
    image_b: str
    change_threshold: float = 0.15


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "SatQuery AI ML CV Backend", "version": "1.0.0"}


@app.post("/api/query")
def process_query(req: QueryRequest):
    optical_img = decode_b64_image(req.optical_image)
    sar_img = decode_b64_image(req.sar_image)
    change_a_img = decode_b64_image(req.change_image_a)
    change_b_img = decode_b64_image(req.change_image_b)

    res = controller.handle_request(
        query=req.query,
        optical_image=optical_img,
        sar_image=sar_img,
        change_image_a=change_a_img,
        change_image_b=change_b_img,
        probe_features=req.probe_features,
    )

    # If change analysis was performed, include heatmap overlay as base64
    if change_a_img and change_b_img:
        try:
            chg = change_analysis.analyze(change_a_img, change_b_img)
            if res.get("change_analysis"):
                res["change_analysis"]["overlay_image"] = encode_image_b64(chg.overlay)
                res["change_analysis"]["heatmap_image"] = encode_image_b64(chg.heatmap)
        except Exception as e:
            print(f"Error encoding change overlay: {e}")

    return res


@app.post("/api/change-analysis")
def process_change(req: ChangeRequest):
    img_a = decode_b64_image(req.image_a)
    img_b = decode_b64_image(req.image_b)

    if not img_a or not img_b:
        raise HTTPException(status_code=400, detail="Invalid base64 image data for image_a or image_b")

    res = change_analysis.analyze(img_a, img_b, change_threshold=req.change_threshold)

    return {
        "summary": res.summary,
        "changed_fraction": res.changed_fraction,
        "mean_intensity_delta": res.mean_intensity_delta,
        "overlay_image": encode_image_b64(res.overlay),
        "heatmap_image": encode_image_b64(res.heatmap),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
