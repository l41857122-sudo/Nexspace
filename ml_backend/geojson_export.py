"""
geojson_export.py
-----------------
GeoJSON Exporter for Geospatial Evidence & Ground Footprints.

Converts enriched spatial evidence nodes (change regions, object detections)
into standard RFC 7946 GeoJSON FeatureCollections for GIS interoperability.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from geospatial import GeoMetadata


def export_evidence_to_geojson(
    evidence_list: List[Dict[str, Any]],
    geo_meta: Optional[GeoMetadata] = None,
) -> Dict[str, Any]:
    """
    Exports a list of spatial evidence items to a standard GeoJSON FeatureCollection.
    If geospatial metadata is unavailable, returns a clean structured unavailable response.
    """
    if geo_meta is None or not geo_meta.geospatial_available:
        return {
            "type": "FeatureCollection",
            "features": [],
            "geospatial_available": False,
            "reason": geo_meta.reason if geo_meta else "No geospatial metadata available",
        }

    src_crs = geo_meta.crs or "EPSG:4326"
    transformer: Optional[Any] = None

    # Standard GeoJSON requires coordinates in WGS 84 (EPSG:4326) [lon, lat]
    if HAS_PYPROJ and src_crs != "EPSG:4326":
        try:
            transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        except Exception:
            transformer = None

    features: List[Dict[str, Any]] = []

    for item in evidence_list:
        world_info = item.get("bbox_world")
        if not world_info:
            continue

        raw_polygon = world_info.get("polygon_world")
        if not raw_polygon:
            min_x = world_info.get("min_x")
            min_y = world_info.get("min_y")
            max_x = world_info.get("max_x")
            max_y = world_info.get("max_y")
            if min_x is None or min_y is None or max_x is None or max_y is None:
                continue
            raw_polygon = [
                (min_x, max_y),
                (max_x, max_y),
                (max_x, min_y),
                (min_x, min_y),
                (min_x, max_y),
            ]

        # Reproject coordinates to [lon, lat] if projected and validate finiteness
        geojson_coords: List[List[float]] = []
        is_geom_valid = True
        for x, y in raw_polygon:
            if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
                is_geom_valid = False
                break

            if transformer:
                try:
                    lon, lat = transformer.transform(x, y)
                    if math.isfinite(lon) and math.isfinite(lat):
                        geojson_coords.append([round(lon, 7), round(lat, 7)])
                    else:
                        is_geom_valid = False
                        break
                except Exception:
                    geojson_coords.append([round(x, 7), round(y, 7)])
            else:
                geojson_coords.append([round(x, 7), round(y, 7)])

        if not is_geom_valid or len(geojson_coords) < 3:
            continue

        feature_id = item.get("id") or item.get("evidence_id") or f"feat_{len(features) + 1:03d}"
        feature_type = item.get("type") or item.get("evidence_type") or "spatial_evidence"

        properties = {
            "evidence_id": feature_id,
            "type": feature_type,
            "label": item.get("label", feature_type),
            "source": item.get("source", "NexSpace"),
            "confidence": item.get("score") or item.get("confidence"),
            "severity_score": item.get("severity_score"),
            "ground_area": item.get("ground_area"),
            "ground_area_unit": item.get("ground_area_unit", "m2"),
            "native_crs": src_crs,
            "bbox_pixel": item.get("bbox_pixel"),
        }

        # Filter None values in properties
        clean_properties = {k: v for k, v in properties.items() if v is not None}

        features.append({
            "type": "Feature",
            "id": feature_id,
            "geometry": {
                "type": "Polygon",
                "coordinates": [geojson_coords],
            },
            "properties": clean_properties,
        })

        if len(features) >= 500:
            break

    return {
        "type": "FeatureCollection",
        "geospatial_available": True,
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84" if transformer or src_crs == "EPSG:4326" else src_crs
            }
        },
        "features": features,
    }
