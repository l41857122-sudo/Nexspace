"""
NexSpace Gate 3 Step 3: Multispectral Runtime Proof Engine
Executes Phase 10 (Real Data Runtime Proof) and Phase 11 (Negative Runtime Proofs).
Outputs verified live data, authentic statistics, and truthful provenance contracts.
"""

import os
import sys
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from imagery_provider import Element84STACProvider, ImagerySearchQuery, SceneDownloadResult
from multispectral import MultispectralScene, SpectralEngine
from tools import MultispectralAnalysisTool
from investigation_pipeline import global_investigation_store, InvestigationPipeline


def run_proofs():
    print("=" * 70)
    print("NEXSPACE GATE 3 STEP 3 — RUNTIME PROOF EXECUTION")
    print("=" * 70)

    provider = Element84STACProvider()
    results = {}

    # -------------------------------------------------------------------
    # PHASE 10: REAL DATA RUNTIME PROOF
    # -------------------------------------------------------------------
    print("\n--- PHASE 10: REAL DATA RUNTIME PROOF ---")
    # Search for an actual recent Sentinel-2 scene over Lucknow (26.8467° N, 80.9462° E)
    lucknow_bbox = [80.90, 26.80, 81.00, 26.90]
    search_q = ImagerySearchQuery(
        bbox=lucknow_bbox,
        max_cloud_cover=20.0,
        limit=5,
    )
    scenes = provider.search(search_q)
    if not scenes:
        # Fallback to general valid S2 bbox
        search_q.bbox = [77.10, 28.50, 77.30, 28.70] # Delhi
        scenes = provider.search(search_q)

    assert len(scenes) > 0, "No real Sentinel-2 scenes found in Element84 STAC!"
    target_scene = scenes[0]
    print(f"Discovered Live S2 Scene: {target_scene.scene_id}")
    print(f"Acquisition Date: {target_scene.acquisition_date}")
    print(f"Cloud Cover: {target_scene.cloud_cover}%")
    print(f"Available Assets in STAC Catalog: {list(target_scene.assets.keys())}")

    # Ingest actual bands B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR-1) using download_multispectral_scene
    requested_bands = ["B03", "B04", "B08", "B11"]
    print(f"Requesting actual spectral band ingestion: {requested_bands} ...")
    ms_scene = provider.download_multispectral_scene(
        scene=target_scene,
        bands=requested_bands,
        window_size=256,
    )

    print("\n[LIVE INGESTION PROOF]")
    print(f"SOURCE: {ms_scene.source}")
    print(f"SCENE ID: {ms_scene.scene_id}")
    print(f"BANDS LOADED: {ms_scene.loaded_bands}")
    print(f"AVAILABLE BANDS: {ms_scene.available_bands[:8]} ... (total {len(ms_scene.available_bands)})")
    print(f"SPECTRAL SOURCE: {ms_scene.spectral_source}")
    print(f"CRS: {ms_scene.crs}")
    print(f"TRANSFORM: {ms_scene.transform}")
    print(f"DIMENSIONS: {ms_scene.width}x{ms_scene.height}")
    print(f"RESOLUTION: {ms_scene.resolution}")
    print(f"DATA TYPES: { {b: ms_scene.bands[b].dtype for b in ms_scene.bands} }")
    print(f"NATIVE RESOLUTIONS: { {b: f'{ms_scene.bands[b].native_resolution_m}m' for b in ms_scene.bands} }")

    # Compute genuine spectral indices
    ndvi_res = SpectralEngine.compute_ndvi(ms_scene)
    ndwi_res = SpectralEngine.compute_ndwi(ms_scene)
    ndbi_res = SpectralEngine.compute_ndbi(ms_scene)

    print("\n[AUTHENTIC SPECTRAL INDICES]")
    print(f"NDVI Status: {ndvi_res.status}, Provenance: {ndvi_res.provenance}")
    print(f"NDVI Formula: {ndvi_res.formula}")
    print(f"NDVI Valid Pixels: {ndvi_res.stats.valid_pixel_count}/{ndvi_res.stats.total_pixel_count} (nodata: {ndvi_res.stats.nodata_pixel_count})")
    print(f"NDVI Range: [{ndvi_res.stats.min} .. {ndvi_res.stats.max}], Mean: {ndvi_res.stats.mean}, Median: {ndvi_res.stats.median}")

    print(f"NDWI Status: {ndwi_res.status}, Provenance: {ndwi_res.provenance}")
    print(f"NDWI Formula: {ndwi_res.formula}")
    print(f"NDWI Valid Pixels: {ndwi_res.stats.valid_pixel_count}/{ndwi_res.stats.total_pixel_count}")
    print(f"NDWI Range: [{ndwi_res.stats.min} .. {ndwi_res.stats.max}], Mean: {ndwi_res.stats.mean}")

    print(f"NDBI Status: {ndbi_res.status}, Provenance: {ndbi_res.provenance}")
    print(f"NDBI Formula: {ndbi_res.formula}")
    print(f"NDBI Resampling Applied: {ndbi_res.resampling_applied} ({ndbi_res.resampling_details})")
    print(f"NDBI Valid Pixels: {ndbi_res.stats.valid_pixel_count}/{ndbi_res.stats.total_pixel_count}")
    print(f"NDBI Range: [{ndbi_res.stats.min} .. {ndbi_res.stats.max}], Mean: {ndbi_res.stats.mean}")

    phase10_data = {
        "source": ms_scene.source,
        "scene_id": ms_scene.scene_id,
        "loaded_bands": ms_scene.loaded_bands,
        "available_bands": ms_scene.available_bands,
        "spectral_source": ms_scene.spectral_source,
        "crs": ms_scene.crs,
        "transform": list(ms_scene.transform),
        "dimensions": f"{ms_scene.width}x{ms_scene.height}",
        "resolution": ms_scene.resolution,
        "ndvi": {
            "status": ndvi_res.status,
            "provenance": ndvi_res.provenance,
            "valid_pixels": ndvi_res.stats.valid_pixel_count,
            "min": ndvi_res.stats.min,
            "max": ndvi_res.stats.max,
            "mean": ndvi_res.stats.mean,
        },
        "ndwi": {
            "status": ndwi_res.status,
            "provenance": ndwi_res.provenance,
            "valid_pixels": ndwi_res.stats.valid_pixel_count,
            "min": ndwi_res.stats.min,
            "max": ndwi_res.stats.max,
            "mean": ndwi_res.stats.mean,
        },
        "ndbi": {
            "status": ndbi_res.status,
            "provenance": ndbi_res.provenance,
            "resampling_applied": ndbi_res.resampling_applied,
            "resampling_details": ndbi_res.resampling_details,
            "valid_pixels": ndbi_res.stats.valid_pixel_count,
            "min": ndbi_res.stats.min,
            "max": ndbi_res.stats.max,
            "mean": ndbi_res.stats.mean,
        },
    }
    results["phase_10_real_data_proof"] = phase10_data

    # -------------------------------------------------------------------
    # PHASE 11: NEGATIVE RUNTIME PROOF
    # -------------------------------------------------------------------
    print("\n--- PHASE 11: NEGATIVE RUNTIME PROOF ---")
    # Proof 11.1: RGB-only image submitted to Multispectral Analysis
    tool = MultispectralAnalysisTool()
    rgb_img = Image.new("RGB", (256, 256), color=(60, 120, 40))
    neg_res = tool.execute({
        "image": rgb_img,
        "multispectral_scene": None,
        "indices": ["NDVI", "NDWI"],
        "query": "Calculate NDVI vegetation index",
    })

    print("\n[NEGATIVE PROOF 1: RGB-Only Image]")
    print(f"Tool Status: {neg_res.status}")
    print(f"Provenance: {neg_res.provenance}")
    print(f"Error Truthfully Reported: {neg_res.error}")
    print(f"Spectral Provenance in Data: {neg_res.data['spectral']['provenance']}")
    assert neg_res.status == "unavailable", "RGB-only must yield status 'unavailable'!"
    assert neg_res.provenance == "UNAVAILABLE", "RGB-only must carry provenance 'UNAVAILABLE'!"
    assert "B04" in neg_res.error and "B08" in neg_res.error, "Error must explain B04 and B08 requirements!"

    # Proof 11.2: Provider metadata claims Sentinel-2 bands B02..B12 BUT only RGB preview is downloaded
    download_res_rgb = provider.download_scene(
        scene=target_scene,
        prefer_preview=True,
    )
    print("\n[NEGATIVE PROOF 2: Provider Metadata vs Downloaded Asset Mismatch]")
    print(f"Scene Available Bands (from catalog): {target_scene.available_bands[:6]}...")
    print(f"Scene Catalog Spectral Source: {target_scene.spectral_source}")
    print(f"Downloaded Image Path: {download_res_rgb.image_path}")
    print(f"Downloaded Spectral Source: {download_res_rgb.spectral_source}")
    print(f"Downloaded Loaded Bands: {download_res_rgb.loaded_bands}")

    assert download_res_rgb.spectral_source != "REAL_MULTISPECTRAL", "Visual asset must NOT be labelled REAL_MULTISPECTRAL!"
    assert download_res_rgb.spectral_source == "RGB_VISUAL_ASSET", "Visual asset must be labelled RGB_VISUAL_ASSET!"
    assert download_res_rgb.loaded_bands == ["R", "G", "B"], "Loaded bands must reflect visual RGB asset!"
    assert download_res_rgb.multispectral_scene is None, "multispectral_scene must be None for RGB preview!"

    # Proof 11.3: End-to-end investigation pipeline on RGB-only NDVI query
    pipeline = InvestigationPipeline()
    pipe_res = pipeline.execute(
        query="Calculate NDVI spectral index",
        optical_image=rgb_img,
    )
    print("\n[NEGATIVE PROOF 3: End-to-End Pipeline Contract on RGB-Only Query]")
    print(f"Pipeline Result Category: {pipe_res['result_category']}")
    print(f"Pipeline Status: {pipe_res['status']}")
    print(f"Pipeline Provenance: {pipe_res['provenance']}")
    print(f"Pipeline Spectral Payload: {pipe_res.get('spectral')}")
    print(f"Pipeline Response Text: {pipe_res['response_text']}")

    assert pipe_res["result_category"] == "unavailable_result", "Must categorize as unavailable_result!"
    assert pipe_res["provenance"] == "UNAVAILABLE", "Overall provenance must be UNAVAILABLE!"
    assert "B04" in pipe_res["response_text"] and "B08" in pipe_res["response_text"], "Response text must truthfully explain missing sensor bands!"

    phase11_data = {
        "rgb_only_tool": {
            "status": neg_res.status,
            "provenance": neg_res.provenance,
            "error": neg_res.error,
            "spectral_provenance": neg_res.data["spectral"]["provenance"],
        },
        "catalog_vs_download_mismatch": {
            "catalog_spectral_source": target_scene.spectral_source,
            "catalog_available_bands": target_scene.available_bands,
            "downloaded_spectral_source": download_res_rgb.spectral_source,
            "downloaded_loaded_bands": download_res_rgb.loaded_bands,
            "is_real_multispectral": download_res_rgb.spectral_source == "REAL_MULTISPECTRAL",
        },
        "e2e_pipeline_rgb_rejection": {
            "result_category": pipe_res["result_category"],
            "status": pipe_res["status"],
            "provenance": pipe_res["provenance"],
            "response_text": pipe_res["response_text"],
        },
    }
    results["phase_11_negative_proof"] = phase11_data

    # Save to scratch
    out_path = os.path.join(os.path.dirname(__file__), "gate3_step3_runtime_proof.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved proof results to {out_path}")
    print("=" * 70)
    print("PHASE 10 & PHASE 11 PROOFS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_proofs()
