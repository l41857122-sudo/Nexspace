import io
import math
import numpy as np
from PIL import Image

def compute_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Computes Normalized Difference Vegetation Index:
    NDVI = (NIR - Red) / (NIR + Red)
    """
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    denominator = nir + red
    denominator[denominator == 0] = 1e-5
    ndvi = (nir - red) / denominator
    return np.clip(ndvi, -1.0, 1.0)

def compute_cloud_mask(image_rgb: np.ndarray, threshold: float = 0.85) -> tuple[np.ndarray, float]:
    """
    Computes simple cloud mask threshold on high-brightness / low-contrast pixels.
    Returns (mask_bool_array, cloud_cover_percentage).
    """
    norm = image_rgb.astype(np.float32) / 255.0
    brightness = np.mean(norm, axis=2)
    cloud_mask = brightness > threshold
    cloud_pct = float(np.mean(cloud_mask) * 100.0)
    return cloud_mask, round(cloud_pct, 2)

def generate_pixel_diff_heatmap(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, float, list[dict]]:
    """
    Co-registers two image rasters and calculates pixel difference heatmap + anomalies.
    Returns (heatmap_rgb, delta_percentage, anomaly_list).
    """
    if img_a.shape != img_b.shape:
        # Resize B to match A
        pil_b = Image.fromarray(img_b)
        pil_b = pil_b.resize((img_a.shape[1], img_a.shape[0]))
        img_b = np.array(pil_b)

    diff = np.abs(img_a.astype(np.float32) - img_b.astype(np.float32))
    diff_gray = np.mean(diff, axis=2) / 255.0
    
    threshold_val = 0.25
    change_pixels = diff_gray > threshold_val
    delta_pct = round(float(np.mean(change_pixels) * 100.0), 2)

    # Simple heatmap coloring (red for change)
    heatmap = np.zeros_like(img_a)
    heatmap[:, :, 0] = (diff_gray * 255).astype(np.uint8)
    heatmap[:, :, 1] = (diff_gray * 100).astype(np.uint8)
    heatmap[:, :, 2] = 0

    # Extract synthetic / thresholded anomaly regions
    anomalies = [
        {
            "id": "veg",
            "label": "ANOMALY #01",
            "type": "Vegetation Loss",
            "confidence": 0.96,
            "area_km2": 4.2,
            "bbox": [20, 60, 32, 24],
            "location": "LAT: 34.951° LON: -118.241°"
        },
        {
            "id": "struct",
            "label": "ANOMALY #02",
            "type": "New Structure",
            "confidence": 0.87,
            "area_km2": 0.8,
            "bbox": [70, 15, 24, 20],
            "location": "LAT: 34.958° LON: -118.239°"
        }
    ]

    return heatmap, delta_pct, anomalies

def detect_objects_placeholder(image_array: np.ndarray) -> list[dict]:
    """
    Detects maritime vessels and storage infrastructure on aerial rasters.
    """
    return [
        {
            "id": "entity-1",
            "name": "Vessel_Panamax_01",
            "type": "Cargo/Container",
            "meta": "294 × 32m",
            "confidence": "94%",
            "statusColor": "bg-emerald-400",
            "badgeBorder": "border-emerald-500/30",
            "badgeBg": "bg-emerald-500/10",
            "badgeText": "text-emerald-400",
            "x": 48,
            "y": 38,
            "width": 140,
            "height": 48
        },
        {
            "id": "entity-2",
            "name": "Infra_Tank_Farm_B",
            "type": "Storage/Liquid",
            "meta": "14,500 m²",
            "confidence": "82%",
            "statusColor": "bg-amber-400",
            "badgeBorder": "border-amber-500/30",
            "badgeBg": "bg-amber-500/10",
            "badgeText": "text-amber-300",
            "x": 64,
            "y": 56,
            "width": 110,
            "height": 60
        },
        {
            "id": "entity-3",
            "name": "Vessel_Feeder_12",
            "type": "Cargo/Breakbulk",
            "meta": "142 × 22m",
            "confidence": "91%",
            "statusColor": "bg-emerald-400",
            "badgeBorder": "border-emerald-500/30",
            "badgeBg": "bg-emerald-500/10",
            "badgeText": "text-emerald-400",
            "x": 28,
            "y": 28,
            "width": 120,
            "height": 44
        }
    ]
