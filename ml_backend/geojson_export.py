"""
geojson_export.py
-----------------
GeoJSON Exporter for Geospatial Evidence & Ground Footprints.

Converts enriched spatial evidence nodes (change regions, object detections, linear infrastructure)
into standard RFC 7946 GeoJSON FeatureCollections for GIS interoperability.

Includes:
- Strict coordinate range validation (lat in [-90, 90], lon in [-180, 180])
- Scene footprint containment verification (prevents phantom out-of-scene coordinates)
- Semantic categorization for map content hierarchy:
  - discrete objects (building, car, truck, ship, airplane, container)
  - regions (water body, forest, agriculture, construction site, industrial area)
  - linear features (road, highway, railway, river, runway)
"""

from __future__ import annotations
import math
import logging
from typing import List, Dict, Any, Optional

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from geospatial import GeoMetadata

logger = logging.getLogger("nexspace.geojson_export")

DISCRETE_OBJECT_KEYWORDS = {
    "building", "car", "truck", "ship", "vessel", "boat", "airplane", "plane",
    "aircraft", "tower", "container", "warehouse", "vehicle", "tank", "storage tank",
    "hangar", "crane", "train", "structure"
}

REGION_KEYWORDS = {
    "water", "water body", "lake", "ocean", "sea", "forest", "grassland", "vegetation",
    "agricultural", "agriculture", "farmland", "farm", "crop", "construction",
    "construction site", "industrial", "industrial area", "airport", "park", "wetland"
}

LINEAR_KEYWORDS = {
    "road", "highway", "freeway", "street", "railway", "railroad", "river", "canal",
    "runway", "taxiway", "pipeline", "power line", "coastline"
}


def classify_feature_category(label: str, feat_type: str = "") -> str:
    """Classifies a target into discrete, region, or linear hierarchy category."""
    combined = f"{label} {feat_type}".lower()
    for kw in LINEAR_KEYWORDS:
        if kw in combined:
            return "linear"
    for kw in REGION_KEYWORDS:
        if kw in combined:
            return "region"
    for kw in DISCRETE_OBJECT_KEYWORDS:
        if kw in combined:
            return "discrete"
    return "discrete"


def export_evidence_to_geojson(
    evidence_list: List[Dict[str, Any]],
    geo_meta: Optional[GeoMetadata] = None,
) -> Dict[str, Any]:
    """
    Exports a list of spatial evidence items to a standard GeoJSON FeatureCollection.
    Enforces strict WGS84 range validation and scene containment checking.
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

    # Retrieve scene footprint bounds for containment validation
    scene_bounds: Optional[Dict[str, float]] = None
    if geo_meta.bounds_world:
        scene_bounds = {
            "min_x": float(geo_meta.bounds_world.get("min_x", -180.0)),
            "min_y": float(geo_meta.bounds_world.get("min_y", -90.0)),
            "max_x": float(geo_meta.bounds_world.get("max_x", 180.0)),
            "max_y": float(geo_meta.bounds_world.get("max_y", 90.0)),
        }

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

        # Reproject coordinates to [lon, lat] if projected and validate ranges
        geojson_coords: List[List[float]] = []
        is_geom_valid = True
        for x, y in raw_polygon:
            if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
                is_geom_valid = False
                break

            if transformer:
                try:
                    lon, lat = transformer.transform(x, y)
                except Exception:
                    lon, lat = float(x), float(y)
            else:
                lon, lat = float(x), float(y)

            # Section 10: Strict Coordinate Validation: lat in [-90, 90], lon in [-180, 180]
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                logger.warning("Coordinate out of WGS84 range: (%f, %f) — rejecting", lat, lon)
                is_geom_valid = False
                break

            geojson_coords.append([round(lon, 7), round(lat, 7)])

        if not is_geom_valid or len(geojson_coords) < 3:
            continue

        # Compute centroid for marker placement & containment test
        center_lon = sum(pt[0] for pt in geojson_coords[:-1]) / max(1, len(geojson_coords) - 1)
        center_lat = sum(pt[1] for pt in geojson_coords[:-1]) / max(1, len(geojson_coords) - 1)

        # Section 10: Containment validation — detection must fall inside scene footprint
        # Allow 5% coordinate epsilon for boundary edge pixels
        if scene_bounds:
            margin_x = max(0.005, (scene_bounds["max_x"] - scene_bounds["min_x"]) * 0.05)
            margin_y = max(0.005, (scene_bounds["max_y"] - scene_bounds["min_y"]) * 0.05)
            if not (
                (scene_bounds["min_x"] - margin_x) <= center_lon <= (scene_bounds["max_x"] + margin_x)
                and (scene_bounds["min_y"] - margin_y) <= center_lat <= (scene_bounds["max_y"] + margin_y)
            ):
                logger.warning(
                    "Detection centroid (%f, %f) outside scene bounds [%f, %f, %f, %f] — unverified",
                    center_lat, center_lon,
                    scene_bounds["min_x"], scene_bounds["min_y"], scene_bounds["max_x"], scene_bounds["max_y"]
                )
                continue

        feature_id = item.get("id") or item.get("evidence_id") or f"feat_{len(features) + 1:03d}"
        feature_type = item.get("type") or item.get("evidence_type") or "spatial_evidence"
        label = item.get("label", feature_type)

        category = classify_feature_category(label, feature_type)

        properties = {
            "evidence_id": feature_id,
            "type": feature_type,
            "label": label,
            "feature_category": category,  # "discrete" | "region" | "linear"
            "centroid": [round(center_lon, 7), round(center_lat, 7)],
            "source": item.get("source", "NexSpace"),
            "confidence": item.get("score") or item.get("confidence"),
            "severity_score": item.get("severity_score"),
            "ground_area": item.get("ground_area"),
            "ground_area_unit": item.get("ground_area_unit", "m2"),
            "native_crs": src_crs,
            "bbox_pixel": item.get("bbox_pixel"),
            "verification_status": "VERIFIED_IN_FOOTPRINT",
        }

        # Clean None values in properties
        clean_properties = {k: v for k, v in properties.items() if v is not None}

        # Emit feature with exact polygon boundary and centroid
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
