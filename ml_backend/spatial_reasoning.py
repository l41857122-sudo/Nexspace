"""
spatial_reasoning.py
--------------------
Gate 3 Step 7: Mathematically Rigorous, Geospatially Grounded Spatial Intelligence Engine.

Evaluates 2D and 3D spatial predicates over points, bounding boxes, polygons, and linear features
across both normalized pixel space [0..1000] and geographic WGS84 (EPSG:4326) / projected coordinate spaces.

Supported 11 Core Spatial Relations:
1. near (distance <= threshold)
2. inside (true geometric containment)
3. outside (true geometric disjointness)
4. adjacent (topological touching or bounded buffer contact)
5. crossing (geometric line/polygon or line/line intersection)
6. along (corridor proximity to linear feature)
7. closest_to (nearest-neighbor distance ranking among candidates)
8. north_of (geographic northing or azimuth bearing in [315, 45])
9. south_of (geographic southing or azimuth bearing in [135, 225])
10. east_of (geographic easting or azimuth bearing in [45, 135])
11. west_of (geographic westing or azimuth bearing in [225, 315])

Also supports:
- Multi-target relationship decomposition
- Object <-> Region relationships
- Object <-> Object relationships
- Detection <-> Geographic coordinate mapping (WGS84)
- Detection <-> GeoJSON geometry consistency (RFC 7946)
- AOI <-> Detection containment & intersection
- Scene footprint <-> Target boundary validation
- Metric distance computation (Haversine in meters, Euclidean in projected CRS)
- Geodetic directional bearing computation
- Strict Anti-Fabrication: confidence=None, confidence_type="UNAVAILABLE"
- Provenance: "ALGORITHMIC"
"""

from __future__ import annotations
import math
import re
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np

try:
    import shapely
    import shapely.geometry as sgeom
    import shapely.ops as sops
    from shapely.geometry import (
        Point,
        Polygon,
        LineString,
        MultiPolygon,
        MultiLineString,
        box as shapely_box,
        shape as shapely_shape,
    )
    from shapely.validation import explain_validity
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

logger = logging.getLogger("nexspace.spatial_reasoning")


# ============================================================================
# CANONICAL ENUMS & TAXONOMY
# ============================================================================

class SpatialRelationType(str, Enum):
    NEAR = "near"
    INSIDE = "inside"
    OUTSIDE = "outside"
    ADJACENT = "adjacent"
    CROSSING = "crossing"
    ALONG = "along"
    CLOSEST_TO = "closest_to"
    NORTH_OF = "north_of"
    SOUTH_OF = "south_of"
    EAST_OF = "east_of"
    WEST_OF = "west_of"


class SpatialEvaluationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    UNAVAILABLE = "UNAVAILABLE"
    AMBIGUOUS = "AMBIGUOUS"
    NO_RESULT = "NO_RESULT"


# Synonyms and natural language mapping to canonical relations
SPATIAL_SYNONYM_MAP: Dict[str, SpatialRelationType] = {
    "near": SpatialRelationType.NEAR,
    "close_to": SpatialRelationType.NEAR,
    "close to": SpatialRelationType.NEAR,
    "around": SpatialRelationType.NEAR,
    "in_proximity_to": SpatialRelationType.NEAR,
    "in proximity to": SpatialRelationType.NEAR,
    "vicinity_of": SpatialRelationType.NEAR,
    "vicinity of": SpatialRelationType.NEAR,
    "inside": SpatialRelationType.INSIDE,
    "within": SpatialRelationType.INSIDE,
    "in": SpatialRelationType.INSIDE,
    "in_the_interior_of": SpatialRelationType.INSIDE,
    "in the interior of": SpatialRelationType.INSIDE,
    "located_inside": SpatialRelationType.INSIDE,
    "located inside": SpatialRelationType.INSIDE,
    "outside": SpatialRelationType.OUTSIDE,
    "outside_of": SpatialRelationType.OUTSIDE,
    "outside of": SpatialRelationType.OUTSIDE,
    "exterior_to": SpatialRelationType.OUTSIDE,
    "exterior to": SpatialRelationType.OUTSIDE,
    "adjacent": SpatialRelationType.ADJACENT,
    "adjacent_to": SpatialRelationType.ADJACENT,
    "adjacent to": SpatialRelationType.ADJACENT,
    "beside": SpatialRelationType.ADJACENT,
    "next_to": SpatialRelationType.ADJACENT,
    "next to": SpatialRelationType.ADJACENT,
    "flanking": SpatialRelationType.ADJACENT,
    "bordering": SpatialRelationType.ADJACENT,
    "crossing": SpatialRelationType.CROSSING,
    "intersecting": SpatialRelationType.CROSSING,
    "across": SpatialRelationType.CROSSING,
    "along": SpatialRelationType.ALONG,
    "alongside": SpatialRelationType.ALONG,
    "following": SpatialRelationType.ALONG,
    "closest_to": SpatialRelationType.CLOSEST_TO,
    "closest to": SpatialRelationType.CLOSEST_TO,
    "nearest_to": SpatialRelationType.CLOSEST_TO,
    "nearest to": SpatialRelationType.CLOSEST_TO,
    "closest": SpatialRelationType.CLOSEST_TO,
    "nearest": SpatialRelationType.CLOSEST_TO,
    "north_of": SpatialRelationType.NORTH_OF,
    "north of": SpatialRelationType.NORTH_OF,
    "to_the_north_of": SpatialRelationType.NORTH_OF,
    "to the north of": SpatialRelationType.NORTH_OF,
    "above": SpatialRelationType.NORTH_OF,
    "south_of": SpatialRelationType.SOUTH_OF,
    "south of": SpatialRelationType.SOUTH_OF,
    "to_the_south_of": SpatialRelationType.SOUTH_OF,
    "to the south of": SpatialRelationType.SOUTH_OF,
    "below": SpatialRelationType.SOUTH_OF,
    "east_of": SpatialRelationType.EAST_OF,
    "east of": SpatialRelationType.EAST_OF,
    "to_the_east_of": SpatialRelationType.EAST_OF,
    "to the east of": SpatialRelationType.EAST_OF,
    "right_of": SpatialRelationType.EAST_OF,
    "right of": SpatialRelationType.EAST_OF,
    "west_of": SpatialRelationType.WEST_OF,
    "west of": SpatialRelationType.WEST_OF,
    "to_the_west_of": SpatialRelationType.WEST_OF,
    "to the west of": SpatialRelationType.WEST_OF,
    "left_of": SpatialRelationType.WEST_OF,
    "left of": SpatialRelationType.WEST_OF,
}


# ============================================================================
# CANONICAL GEOMETRIC & GEODESIC MATH UTILITIES
# ============================================================================

def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes exact geodesic great-circle distance between two WGS84 points in meters."""
    R = 6371000.0  # WGS84 mean Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(R * c, 3)


def calculate_geodetic_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes initial geodetic bearing (forward azimuth) from point 1 to point 2 in degrees [0..360).
    0 = North, 90 = East, 180 = South, 270 = West.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return round(bearing, 2)


def calculate_pixel_bearing_deg(cx1: float, cy1: float, cx2: float, cy2: float) -> float:
    """
    Computes screen/pixel bearing where North is up (-y axis).
    Returns degrees in [0..360).
    """
    dx = cx2 - cx1
    dy = cy2 - cy1  # Screen: down is positive y
    # Angle from North (-y):
    bearing = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
    return round(bearing, 2)


def classify_direction_from_bearing(bearing_deg: float) -> str:
    """Classifies bearing in degrees into primary cardinal direction (North, East, South, West)."""
    b = (bearing_deg + 360.0) % 360.0
    if b >= 315.0 or b < 45.0:
        return "north_of"
    elif 45.0 <= b < 135.0:
        return "east_of"
    elif 135.0 <= b < 225.0:
        return "south_of"
    else:
        return "west_of"


# ============================================================================
# CANONICAL SPATIAL CONTRACT (STEP 2)
# ============================================================================

@dataclass
class SpatialContractResult:
    """
    Standardized, machine-readable output contract for every spatial relation conclusion.
    Adheres strictly to Gate 3 Step 7 Anti-Fabrication Invariants:
    - provenance: ALGORITHMIC
    - confidence: null
    - confidence_type: "UNAVAILABLE"
    """
    relation: str
    subject: Dict[str, Any]
    object: Dict[str, Any]
    result: bool
    distance: Dict[str, Any]  # {"value": float, "unit": "meters" | "normalized_pixels"}
    bearing_deg: Optional[float] = None
    direction_label: Optional[str] = None
    method: str = "SHAPELY_GEOMETRIC_PREDICATE"
    provenance: str = "ALGORITHMIC"
    confidence: Optional[float] = None
    confidence_type: str = "UNAVAILABLE"
    evidence: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    status: str = "SUCCESS"  # SUCCESS, INVALID_INPUT, UNAVAILABLE, AMBIGUOUS, NO_RESULT
    error_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation": self.relation,
            "subject": self.subject,
            "object": self.object,
            "result": self.result,
            "distance": self.distance,
            "bearing_deg": self.bearing_deg,
            "direction_label": self.direction_label,
            "method": self.method,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "confidence_type": self.confidence_type,
            "evidence": self.evidence,
            "validation": self.validation,
            "status": self.status,
            "error_reason": self.error_reason,
        }


# ============================================================================
# GEOMETRY CONVERSION & PARSING HELPERS
# ============================================================================

def to_shapely_geometry(
    item: Dict[str, Any],
    geo_transform: Optional[Any] = None,
    crs: str = "EPSG:4326",
) -> Tuple[Optional[Any], str, Optional[Tuple[float, float]]]:
    """
    Extracts or constructs a Shapely geometry from an entity item.
    Returns: (geometry, coordinate_space, (center_x, center_y))
    where coordinate_space is 'geographic', 'projected', or 'pixel'.
    """
    if not isinstance(item, dict):
        return None, "invalid", None

    # Check for direct GeoJSON geometry
    if "geometry" in item and isinstance(item["geometry"], dict):
        g_dict = item["geometry"]
        try:
            geom = shapely_shape(g_dict)
            c = geom.centroid
            return geom, "geographic", (c.x, c.y)
        except Exception as e:
            logger.debug(f"Failed to parse GeoJSON geometry: {e}")

    # Check for explicit polygon coordinates (takes precedence over point centroid)
    if "polygon" in item and item["polygon"]:
        pts = item["polygon"]
        if len(pts) >= 3:
            is_geo = all(-180.0 <= p[0] <= 180.0 and -90.0 <= p[1] <= 90.0 for p in pts)
            if not is_geo and geo_transform and hasattr(geo_transform, "pixel_to_world"):
                world_pts = [geo_transform.pixel_to_world(p[0], p[1]) for p in pts]
                poly = Polygon(world_pts)
                c = poly.centroid
                return poly, "geographic", (c.x, c.y)
            else:
                poly = Polygon(pts)
                c = poly.centroid
                space = "geographic" if is_geo else "pixel"
                return poly, space, (c.x, c.y)

    # Check for linear feature coordinates
    if "line" in item and item["line"]:
        pts = item["line"]
        if len(pts) >= 2:
            is_geo = all(-180.0 <= p[0] <= 180.0 and -90.0 <= p[1] <= 90.0 for p in pts)
            line = LineString(pts)
            c = line.centroid
            space = "geographic" if is_geo else "pixel"
            return line, space, (c.x, c.y)

    # Check for WGS84 coordinates dictionary (Point)
    if "coordinates" in item and isinstance(item["coordinates"], dict):
        coords = item["coordinates"]
        lat = coords.get("lat")
        lon = coords.get("lon")
        if lat is not None and lon is not None and not (math.isnan(lat) or math.isnan(lon)):
            pt = Point(float(lon), float(lat))
            return pt, "geographic", (float(lon), float(lat))

    # Check for bounding box
    bbox = item.get("bbox") or item.get("box") or item.get("bbox_normalized")
    if bbox and len(bbox) == 4:
        xmin, ymin, xmax, ymax = [float(v) for v in bbox]
        is_geo = (-180.0 <= xmin <= 180.0 and -180.0 <= xmax <= 180.0 and
                  -90.0 <= ymin <= 90.0 and -90.0 <= ymax <= 90.0 and
                  abs(xmax - xmin) < 30.0 and abs(ymax - ymin) < 30.0)

        if not is_geo and geo_transform and hasattr(geo_transform, "bbox_pixel_to_world"):
            b_dict = geo_transform.bbox_pixel_to_world([xmin, ymin, xmax, ymax], crs)
            poly = Polygon(b_dict["polygon_world"])
            c = poly.centroid
            return poly, "geographic", (c.x, c.y)
        else:
            b_geom = shapely_box(min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))
            c = b_geom.centroid
            space = "geographic" if is_geo else "pixel"
            return b_geom, space, (c.x, c.y)

    return None, "empty", None


# ============================================================================
# CORE SPATIAL RELATIONSHIP ENGINE (BACKWARD COMPATIBLE & ENHANCED)
# ============================================================================

class SpatialRelationshipEngine:
    """
    Evaluates 2D geometric predicates between detected candidate entities and reference features.
    Provides complete backward compatibility with previous gates while implementing
    strict Gate 3 Step 7 spatial intelligence invariants.
    """

    DEFAULT_NEAR_THRESHOLD = 180.0        # Max distance in [0..1000] pixel space
    DEFAULT_NEAR_THRESHOLD_METERS = 250.0  # Max distance in real-world metric space
    DEFAULT_ADJACENT_THRESHOLD = 80.0     # Max distance for adjacency in [0..1000] space
    DEFAULT_ADJACENT_METERS = 15.0        # Max distance for real-world adjacency
    DEFAULT_CORRIDOR_METERS = 40.0        # Corridor width buffer for 'along' relation
    DEFAULT_CORRIDOR_PIXELS = 60.0

    @classmethod
    def get_centroid(cls, item: Dict[str, Any]) -> Tuple[float, float]:
        """Calculates (cx, cy) in entity coordinate space from polygon, bbox, or coordinates."""
        if "coordinates" in item and isinstance(item["coordinates"], dict):
            coords = item["coordinates"]
            if "lat" in coords and "lon" in coords:
                return float(coords["lon"]), float(coords["lat"])

        if "polygon" in item and item["polygon"]:
            pts = item["polygon"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

        bbox = item.get("bbox") or item.get("box") or item.get("bbox_normalized") or [0, 0, 1000, 1000]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return float(cx), float(cy)

    @classmethod
    def get_bbox(cls, item: Dict[str, Any]) -> List[float]:
        """Returns [xmin, ymin, xmax, ymax] in entity coordinate space."""
        if "bbox" in item:
            return item["bbox"]
        if "box" in item:
            return item["box"]
        if "bbox_normalized" in item:
            return item["bbox_normalized"]
        if "polygon" in item and item["polygon"]:
            pts = item["polygon"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return [min(xs), min(ys), max(xs), max(ys)]
        return [0.0, 0.0, 1000.0, 1000.0]

    @classmethod
    def euclidean_distance(cls, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    @classmethod
    def evaluate_predicate(
        cls,
        target_item: Dict[str, Any],
        reference_items: List[Dict[str, Any]],
        predicate: str,
        threshold: Optional[float] = None,
        geo_transform: Optional[Any] = None,
        crs: str = "EPSG:4326",
    ) -> Tuple[bool, float, Optional[Dict[str, Any]]]:
        """
        Evaluates a spatial predicate between a target entity and candidate reference entities.
        Returns: (is_satisfied, match_score, matched_reference_item)
        Maintains backward compatibility with Gate 3.2, 3.3, 3.4, 3.5, 3.6 pipelines.
        """
        pred_key = predicate.strip().lower().replace(" ", "_")
        canon_rel = SPATIAL_SYNONYM_MAP.get(pred_key)
        if not canon_rel:
            # Fallback if unknown predicate
            canon_rel = SpatialRelationType.NEAR

        if not reference_items:
            # If no reference items present in the scene, return default True with nominal score
            return True, 0.5, None

        # Prepare target geometry
        t_geom, t_space, t_c = to_shapely_geometry(target_item, geo_transform, crs)
        if t_geom is None or t_c is None:
            t_cx, t_cy = cls.get_centroid(target_item)
            t_geom = Point(t_cx, t_cy)
            t_space = "pixel"
            t_c = (t_cx, t_cy)

        best_score = 0.0
        matched_ref: Optional[Dict[str, Any]] = None

        # Special handling for closest_to: rank all reference items by distance
        if canon_rel == SpatialRelationType.CLOSEST_TO:
            min_dist = float("inf")
            for ref in reference_items:
                r_geom, r_space, r_c = to_shapely_geometry(ref, geo_transform, crs)
                if r_geom is None or r_c is None:
                    r_cx, r_cy = cls.get_centroid(ref)
                    r_geom = Point(r_cx, r_cy)
                    r_c = (r_cx, r_cy)

                if t_space == "geographic" and r_space == "geographic":
                    d = haversine_distance_meters(t_c[1], t_c[0], r_c[1], r_c[0])
                else:
                    d = cls.euclidean_distance(t_c, r_c)

                if d < min_dist:
                    min_dist = d
                    matched_ref = ref

            is_sat = matched_ref is not None
            score = 1.0 if is_sat else 0.0
            return is_sat, score, matched_ref

        # Standard predicate evaluation across reference candidates
        for ref in reference_items:
            r_geom, r_space, r_c = to_shapely_geometry(ref, geo_transform, crs)
            if r_geom is None or r_c is None:
                r_cx, r_cy = cls.get_centroid(ref)
                r_geom = Point(r_cx, r_cy)
                r_c = (r_cx, r_cy)
                r_space = "pixel"

            # Compute distance between centroids or geometries
            is_geo = (t_space == "geographic" and r_space == "geographic")
            if is_geo:
                dist = haversine_distance_meters(t_c[1], t_c[0], r_c[1], r_c[0])
                dist_unit = "meters"
            else:
                dist = cls.euclidean_distance(t_c, r_c)
                dist_unit = "normalized_pixels"

            # Evaluate specific canonical relation
            if canon_rel == SpatialRelationType.NEAR:
                max_dist = threshold if threshold is not None else (
                    cls.DEFAULT_NEAR_THRESHOLD_METERS if is_geo else cls.DEFAULT_NEAR_THRESHOLD
                )
                if dist <= max_dist:
                    score = max(0.01, 1.0 - (dist / max(1.0, max_dist)))
                    if score > best_score:
                        best_score = score
                        matched_ref = ref

            elif canon_rel == SpatialRelationType.ADJACENT:
                max_dist = threshold if threshold is not None else (
                    cls.DEFAULT_ADJACENT_METERS if is_geo else cls.DEFAULT_ADJACENT_THRESHOLD
                )
                # Topological touch or close distance
                touches = HAS_SHAPELY and r_geom.touches(t_geom)
                if touches or dist <= max_dist:
                    score = 0.95 if touches else max(0.01, 1.0 - (dist / max(1.0, max_dist)))
                    if score > best_score:
                        best_score = score
                        matched_ref = ref

            elif canon_rel == SpatialRelationType.INSIDE:
                # Geometric containment: target inside reference
                inside = False
                if HAS_SHAPELY:
                    inside = r_geom.contains(t_geom) or r_geom.contains(Point(t_c[0], t_c[1]))
                if not inside:
                    r_box = cls.get_bbox(ref)
                    inside = (r_box[0] <= t_c[0] <= r_box[2]) and (r_box[1] <= t_c[1] <= r_box[3])
                if inside:
                    score = 0.95
                    if score > best_score:
                        best_score = score
                        matched_ref = ref

            elif canon_rel == SpatialRelationType.OUTSIDE:
                # Geometric disjointness: target not inside reference
                inside = False
                if HAS_SHAPELY:
                    inside = r_geom.contains(t_geom) or r_geom.contains(Point(t_c[0], t_c[1]))
                if not inside:
                    r_box = cls.get_bbox(ref)
                    inside = (r_box[0] <= t_c[0] <= r_box[2]) and (r_box[1] <= t_c[1] <= r_box[3])
                if not inside:
                    score = 0.90
                    if score > best_score:
                        best_score = score
                        matched_ref = ref

            elif canon_rel == SpatialRelationType.CROSSING:
                # Geometric crossing or intersection
                crossing = False
                if HAS_SHAPELY:
                    crossing = r_geom.crosses(t_geom) or r_geom.intersects(t_geom)
                if not crossing:
                    t_box = cls.get_bbox(target_item)
                    r_box = cls.get_bbox(ref)
                    ixmin = max(t_box[0], r_box[0])
                    iymin = max(t_box[1], r_box[1])
                    ixmax = min(t_box[2], r_box[2])
                    iymax = min(t_box[3], r_box[3])
                    crossing = (ixmax > ixmin and iymax > iymin)
                if crossing:
                    score = 0.92
                    if score > best_score:
                        best_score = score
                        matched_ref = ref

            elif canon_rel == SpatialRelationType.ALONG:
                # Target lies within linear corridor buffer of reference
                corridor_width = threshold if threshold is not None else (
                    cls.DEFAULT_CORRIDOR_METERS if is_geo else cls.DEFAULT_CORRIDOR_PIXELS
                )
                along = False
                if HAS_SHAPELY:
                    corridor_buf = r_geom.buffer(corridor_width)
                    along = corridor_buf.intersects(t_geom)
                if not along:
                    along = dist <= corridor_width
                if along:
                    score = max(0.01, 1.0 - (dist / max(1.0, corridor_width * 2.0)))
                    if score > best_score:
                        best_score = score
                        matched_ref = ref

            elif canon_rel == SpatialRelationType.NORTH_OF:
                # North: t_lat > r_lat (geographic) or t_cy < r_cy (pixel North is top)
                if is_geo:
                    dy = t_c[1] - r_c[1]  # positive if target lat > ref lat
                    is_north = dy > 0.0
                    bearing = calculate_geodetic_bearing_deg(r_c[1], r_c[0], t_c[1], t_c[0])
                    # Bearing within [315, 360) or [0, 45]
                    is_dir = (bearing >= 315.0 or bearing <= 45.0) or (is_north and abs(t_c[0] - r_c[0]) < abs(dy) * 1.5)
                else:
                    dy = r_c[1] - t_c[1]  # in pixel space, smaller y is North
                    is_dir = dy > 0.0

                if is_dir:
                    score = min(1.0, max(0.1, abs(dy) / (0.01 if is_geo else 200.0)))
                    if score > best_score:
                        best_score = score
                        matched_ref = ref

            elif canon_rel == SpatialRelationType.SOUTH_OF:
                if is_geo:
                    dy = r_c[1] - t_c[1]  # positive if target lat < ref lat
                    is_south = dy > 0.0
                    bearing = calculate_geodetic_bearing_deg(r_c[1], r_c[0], t_c[1], t_c[0])
                    is_dir = (135.0 <= bearing <= 225.0) or (is_south and abs(t_c[0] - r_c[0]) < abs(dy) * 1.5)
                else:
                    dy = t_c[1] - r_c[1]
                    is_dir = dy > 0.0

                if is_dir:
                    score = min(1.0, max(0.1, abs(dy) / (0.01 if is_geo else 200.0)))
                    if score > best_score:
                        best_score = score
                        matched_ref = ref

            elif canon_rel == SpatialRelationType.EAST_OF:
                if is_geo:
                    dx = t_c[0] - r_c[0]  # positive if target lon > ref lon
                    is_east = dx > 0.0
                    bearing = calculate_geodetic_bearing_deg(r_c[1], r_c[0], t_c[1], t_c[0])
                    is_dir = (45.0 <= bearing <= 135.0) or (is_east and abs(t_c[1] - r_c[1]) < abs(dx) * 1.5)
                else:
                    dx = t_c[0] - r_c[0]
                    is_dir = dx > 0.0

                if is_dir:
                    score = min(1.0, max(0.1, abs(dx) / (0.01 if is_geo else 200.0)))
                    if score > best_score:
                        best_score = score
                        matched_ref = ref

            elif canon_rel == SpatialRelationType.WEST_OF:
                if is_geo:
                    dx = r_c[0] - t_c[0]  # positive if target lon < ref lon
                    is_west = dx > 0.0
                    bearing = calculate_geodetic_bearing_deg(r_c[1], r_c[0], t_c[1], t_c[0])
                    is_dir = (225.0 <= bearing <= 315.0) or (is_west and abs(t_c[1] - r_c[1]) < abs(dx) * 1.5)
                else:
                    dx = r_c[0] - t_c[0]
                    is_dir = dx > 0.0

                if is_dir:
                    score = min(1.0, max(0.1, abs(dx) / (0.01 if is_geo else 200.0)))
                    if score > best_score:
                        best_score = score
                        matched_ref = ref

        is_satisfied = best_score > 0.0
        return is_satisfied, round(best_score, 4), matched_ref

    @classmethod
    def filter_by_spatial_relations(
        cls,
        candidate_detections: List[Dict[str, Any]],
        reference_detections: List[Dict[str, Any]],
        relations: List[Any],
        geo_transform: Optional[Any] = None,
        crs: str = "EPSG:4326",
    ) -> List[Dict[str, Any]]:
        """Filters target detections based on required spatial relations."""
        if not candidate_detections or not relations:
            return candidate_detections

        filtered = []
        for det in candidate_detections:
            all_satisfied = True
            matched_refs = []

            for rel in relations:
                pred = rel.relation if hasattr(rel, "relation") else rel.get("relation", "near")
                thresh = rel.distance_threshold if hasattr(rel, "distance_threshold") else rel.get("distance_threshold")
                satisfied, score, matched = cls.evaluate_predicate(
                    det, reference_detections, pred, thresh, geo_transform, crs
                )
                if not satisfied:
                    all_satisfied = False
                    break
                if matched:
                    matched_refs.append(matched.get("id") or matched.get("label"))

            if all_satisfied:
                enriched_det = dict(det)
                if matched_refs:
                    enriched_det["spatial_relation_satisfaction"] = {
                        "satisfied_predicates": [getattr(r, "relation", "near") for r in relations],
                        "matched_references": matched_refs,
                    }
                filtered.append(enriched_det)

        return filtered


# ============================================================================
# COMPREHENSIVE GEOSPATIAL SPATIAL ENGINE (GATE 3 STEP 7 SPECIFICATION)
# ============================================================================

class GeospatialSpatialEngine:
    """
    Advanced Spatial Intelligence & Geospatial Reasoning Engine for SatQuery AI.
    Executes verifiable geometric predicates with canonical contracts,
    traceability, failure handling, and multi-target support.
    """

    @classmethod
    def evaluate_contract(
        cls,
        subject: Dict[str, Any],
        object_ref: Dict[str, Any],
        relation: str,
        threshold: Optional[float] = None,
        geo_transform: Optional[Any] = None,
        crs: str = "EPSG:4326",
        scene_bounds: Optional[Dict[str, float]] = None,
    ) -> SpatialContractResult:
        """
        Evaluates a pair of entities under the canonical spatial contract.
        Produces full geometric validation, exact distances, bearings, and evidence.
        """
        # Validate inputs
        if not subject:
            return SpatialContractResult(
                relation=relation,
                subject={},
                object=object_ref or {},
                result=False,
                distance={"value": -1.0, "unit": "unavailable"},
                status=SpatialEvaluationStatus.INVALID_INPUT.value,
                error_reason="Missing subject entity",
            )

        if not object_ref:
            return SpatialContractResult(
                relation=relation,
                subject=subject,
                object={},
                result=False,
                distance={"value": -1.0, "unit": "unavailable"},
                status=SpatialEvaluationStatus.INVALID_INPUT.value,
                error_reason="Missing reference object entity",
            )

        # Parse relation
        norm_rel = relation.strip().lower().replace(" ", "_")
        canon_rel = SPATIAL_SYNONYM_MAP.get(norm_rel)
        if not canon_rel:
            return SpatialContractResult(
                relation=relation,
                subject=subject,
                object=object_ref,
                result=False,
                distance={"value": -1.0, "unit": "unavailable"},
                status=SpatialEvaluationStatus.UNAVAILABLE.value,
                error_reason=f"Unsupported spatial relation: '{relation}'",
            )

        # Extract Shapely geometries
        s_geom, s_space, s_c = to_shapely_geometry(subject, geo_transform, crs)
        o_geom, o_space, o_c = to_shapely_geometry(object_ref, geo_transform, crs)

        if s_geom is None or s_c is None:
            return SpatialContractResult(
                relation=canon_rel.value,
                subject=subject,
                object=object_ref,
                result=False,
                distance={"value": -1.0, "unit": "unavailable"},
                status=SpatialEvaluationStatus.INVALID_INPUT.value,
                error_reason="Subject geometry could not be constructed",
            )

        if o_geom is None or o_c is None:
            return SpatialContractResult(
                relation=canon_rel.value,
                subject=subject,
                object=object_ref,
                result=False,
                distance={"value": -1.0, "unit": "unavailable"},
                status=SpatialEvaluationStatus.INVALID_INPUT.value,
                error_reason="Reference object geometry could not be constructed",
            )

        # Check geometry validity
        s_valid = s_geom.is_valid if HAS_SHAPELY else True
        o_valid = o_geom.is_valid if HAS_SHAPELY else True
        if not s_valid or not o_valid:
            return SpatialContractResult(
                relation=canon_rel.value,
                subject=subject,
                object=object_ref,
                result=False,
                distance={"value": -1.0, "unit": "unavailable"},
                status=SpatialEvaluationStatus.INVALID_INPUT.value,
                error_reason="Invalid geometry detected (self-intersecting or degenerate)",
            )

        # Determine coordinate space and compute distance / bearing
        is_geo = (s_space == "geographic" and o_space == "geographic")
        if is_geo:
            if HAS_SHAPELY:
                if s_geom.intersects(o_geom):
                    dist_val = 0.0
                else:
                    p1, p2 = sops.nearest_points(s_geom, o_geom)
                    dist_val = haversine_distance_meters(p1.y, p1.x, p2.y, p2.x)
            else:
                dist_val = haversine_distance_meters(s_c[1], s_c[0], o_c[1], o_c[0])
            dist_unit = "meters"
            bearing_val = calculate_geodetic_bearing_deg(o_c[1], o_c[0], s_c[1], s_c[0])
        else:
            if HAS_SHAPELY:
                if s_geom.intersects(o_geom):
                    dist_val = 0.0
                else:
                    p1, p2 = sops.nearest_points(s_geom, o_geom)
                    dist_val = round(SpatialRelationshipEngine.euclidean_distance((p1.x, p1.y), (p2.x, p2.y)), 3)
            else:
                dist_val = round(SpatialRelationshipEngine.euclidean_distance(s_c, o_c), 3)
            dist_unit = "normalized_pixels"
            bearing_val = calculate_pixel_bearing_deg(o_c[0], o_c[1], s_c[0], s_c[1])

        dir_label = classify_direction_from_bearing(bearing_val)

        # Validate against scene bounds if provided
        within_scene = True
        if scene_bounds:
            within_scene = (
                scene_bounds.get("min_x", -180.0) <= s_c[0] <= scene_bounds.get("max_x", 180.0) and
                scene_bounds.get("min_y", -90.0) <= s_c[1] <= scene_bounds.get("max_y", 90.0) and
                scene_bounds.get("min_x", -180.0) <= o_c[0] <= scene_bounds.get("max_x", 180.0) and
                scene_bounds.get("min_y", -90.0) <= o_c[1] <= scene_bounds.get("max_y", 90.0)
            )

        # Evaluate exact relation
        result = False
        evidence = []
        method = "SHAPELY_GEOMETRIC_PREDICATE"

        if canon_rel == SpatialRelationType.NEAR:
            effective_thresh = threshold if threshold is not None else (
                SpatialRelationshipEngine.DEFAULT_NEAR_THRESHOLD_METERS if is_geo else SpatialRelationshipEngine.DEFAULT_NEAR_THRESHOLD
            )
            result = (dist_val <= effective_thresh)
            evidence.append(f"Separation distance ({dist_val} {dist_unit}) {'<=' if result else '>'} threshold ({effective_thresh} {dist_unit})")

        elif canon_rel == SpatialRelationType.INSIDE:
            if HAS_SHAPELY:
                result = o_geom.contains(s_geom) or o_geom.contains(Point(s_c[0], s_c[1]))
            else:
                o_box = SpatialRelationshipEngine.get_bbox(object_ref)
                result = (o_box[0] <= s_c[0] <= o_box[2]) and (o_box[1] <= s_c[1] <= o_box[3])
            evidence.append(f"Subject {'is fully spatially contained within' if result else 'lies outside of'} reference geometry")

        elif canon_rel == SpatialRelationType.OUTSIDE:
            if HAS_SHAPELY:
                inside = o_geom.contains(s_geom) or o_geom.contains(Point(s_c[0], s_c[1]))
            else:
                o_box = SpatialRelationshipEngine.get_bbox(object_ref)
                inside = (o_box[0] <= s_c[0] <= o_box[2]) and (o_box[1] <= s_c[1] <= o_box[3])
            result = not inside
            evidence.append(f"Subject {'is outside / disjoint from' if result else 'is contained inside'} reference geometry")

        elif canon_rel == SpatialRelationType.ADJACENT:
            effective_thresh = threshold if threshold is not None else (
                SpatialRelationshipEngine.DEFAULT_ADJACENT_METERS if is_geo else SpatialRelationshipEngine.DEFAULT_ADJACENT_THRESHOLD
            )
            touches = HAS_SHAPELY and (o_geom.touches(s_geom) or s_geom.touches(o_geom))
            result = touches or (dist_val <= effective_thresh)
            evidence.append(f"Subject {'touches boundary or satisfies adjacency buffer' if result else 'does not contact'} reference geometry")

        elif canon_rel == SpatialRelationType.CROSSING:
            if HAS_SHAPELY:
                result = o_geom.crosses(s_geom) or s_geom.crosses(o_geom) or (
                    o_geom.intersects(s_geom) and not (o_geom.contains(s_geom) or s_geom.contains(o_geom))
                )
            else:
                result = dist_val <= 20.0
            evidence.append(f"Subject geometry {'crosses/intersects interior of' if result else 'does not intersect'} reference geometry")

        elif canon_rel == SpatialRelationType.ALONG:
            effective_corridor = threshold if threshold is not None else (
                SpatialRelationshipEngine.DEFAULT_CORRIDOR_METERS if is_geo else SpatialRelationshipEngine.DEFAULT_CORRIDOR_PIXELS
            )
            if HAS_SHAPELY:
                corridor_buf = (effective_corridor / 111320.0) if is_geo else effective_corridor
                corridor = o_geom.buffer(corridor_buf)
                result = corridor.intersects(s_geom) or (dist_val <= effective_corridor)
            else:
                result = dist_val <= effective_corridor
            evidence.append(f"Subject {'lies within the corridor buffer of' if result else 'is separated from'} reference corridor ({effective_corridor} {dist_unit})")

        elif canon_rel == SpatialRelationType.NORTH_OF:
            if is_geo:
                # North: s_lat > o_lat or bearing in [315, 360) U [0, 45]
                result = (s_c[1] > o_c[1]) and (bearing_val >= 315.0 or bearing_val <= 45.0 or abs(s_c[0] - o_c[0]) < abs(s_c[1] - o_c[1]) * 1.5)
            else:
                result = (s_c[1] < o_c[1])  # Screen pixel North is smaller y
            evidence.append(f"Subject centroid ({s_c}) is {'north of' if result else 'not north of'} reference centroid ({o_c}) with bearing {bearing_val}°")

        elif canon_rel == SpatialRelationType.SOUTH_OF:
            if is_geo:
                result = (s_c[1] < o_c[1]) and (135.0 <= bearing_val <= 225.0 or abs(s_c[0] - o_c[0]) < abs(s_c[1] - o_c[1]) * 1.5)
            else:
                result = (s_c[1] > o_c[1])
            evidence.append(f"Subject centroid ({s_c}) is {'south of' if result else 'not south of'} reference centroid ({o_c}) with bearing {bearing_val}°")

        elif canon_rel == SpatialRelationType.EAST_OF:
            if is_geo:
                result = (s_c[0] > o_c[0]) and (45.0 <= bearing_val <= 135.0 or abs(s_c[1] - o_c[1]) < abs(s_c[0] - o_c[0]) * 1.5)
            else:
                result = (s_c[0] > o_c[0])
            evidence.append(f"Subject centroid ({s_c}) is {'east of' if result else 'not east of'} reference centroid ({o_c}) with bearing {bearing_val}°")

        elif canon_rel == SpatialRelationType.WEST_OF:
            if is_geo:
                result = (s_c[0] < o_c[0]) and (225.0 <= bearing_val <= 315.0 or abs(s_c[1] - o_c[1]) < abs(s_c[0] - o_c[0]) * 1.5)
            else:
                result = (s_c[0] < o_c[0])
            evidence.append(f"Subject centroid ({s_c}) is {'west of' if result else 'not west of'} reference centroid ({o_c}) with bearing {bearing_val}°")

        elif canon_rel == SpatialRelationType.CLOSEST_TO:
            # Pairwise closest evaluation (single pair comparison)
            result = True
            evidence.append(f"Calculated pairwise candidate distance: {dist_val} {dist_unit}")

        # Structure subject and object summaries
        subj_summary = {
            "id": subject.get("id") or subject.get("evidence_id") or "subject_01",
            "type": subject.get("category") or subject.get("geometry_type") or "discrete_object",
            "label": subject.get("label") or subject.get("concept") or "target",
            "centroid": [round(s_c[0], 6), round(s_c[1], 6)],
            "coordinate_space": s_space,
        }
        obj_summary = {
            "id": object_ref.get("id") or object_ref.get("evidence_id") or "object_01",
            "type": object_ref.get("category") or object_ref.get("geometry_type") or "region",
            "label": object_ref.get("label") or object_ref.get("concept") or "reference",
            "centroid": [round(o_c[0], 6), round(o_c[1], 6)],
            "coordinate_space": o_space,
        }

        validation_dict = {
            "crs": crs if is_geo else "PIXEL_NORMALIZED",
            "coordinate_space": "WGS84_GEOGRAPHIC" if is_geo else "NORMALIZED_IMAGE_COORDINATES",
            "subject_valid": s_valid,
            "object_valid": o_valid,
            "within_scene_footprint": within_scene,
        }

        return SpatialContractResult(
            relation=canon_rel.value,
            subject=subj_summary,
            object=obj_summary,
            result=result,
            distance={"value": dist_val, "unit": dist_unit},
            bearing_deg=bearing_val,
            direction_label=dir_label,
            method=method,
            provenance="ALGORITHMIC",
            confidence=None,
            confidence_type="UNAVAILABLE",
            evidence=evidence,
            validation=validation_dict,
            status=SpatialEvaluationStatus.SUCCESS.value,
            error_reason=None,
        )

    @classmethod
    def evaluate_closest_to(
        cls,
        candidates: List[Dict[str, Any]],
        reference: Dict[str, Any],
        geo_transform: Optional[Any] = None,
        crs: str = "EPSG:4326",
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Ranks candidate entities by exact distance to reference entity.
        Returns: (closest_candidate, ranked_candidates_with_distances)
        """
        if not candidates or not reference:
            return None, []

        ranked = []
        for cand in candidates:
            contract = cls.evaluate_contract(cand, reference, "near", threshold=1e9, geo_transform=geo_transform, crs=crs)
            d_val = contract.distance.get("value", float("inf"))
            d_unit = contract.distance.get("unit", "meters")
            item_copy = dict(cand)
            item_copy["evaluated_distance"] = d_val
            item_copy["distance_unit"] = d_unit
            ranked.append(item_copy)

        ranked.sort(key=lambda x: x["evaluated_distance"])
        closest = ranked[0] if ranked else None
        return closest, ranked

    @classmethod
    def evaluate_multi_target_query(
        cls,
        subjects: List[Dict[str, Any]],
        objects: List[Dict[str, Any]],
        relation: str,
        threshold: Optional[float] = None,
        geo_transform: Optional[Any] = None,
        crs: str = "EPSG:4326",
    ) -> List[SpatialContractResult]:
        """
        Evaluates full Cartesian product (S x O) of multi-target queries,
        guaranteeing pairing by geometric relationship, not array index.
        """
        results = []
        for s in subjects:
            for o in objects:
                contract = cls.evaluate_contract(s, o, relation, threshold, geo_transform, crs)
                results.append(contract)
        return results

    @classmethod
    def evaluate_aoi_containment(
        cls,
        detection: Dict[str, Any],
        aoi: Dict[str, Any],
        geo_transform: Optional[Any] = None,
        crs: str = "EPSG:4326",
    ) -> Dict[str, Any]:
        """Validates whether a detection lies within an explicit AOI geometry."""
        contract = cls.evaluate_contract(detection, aoi, "inside", geo_transform=geo_transform, crs=crs)
        return {
            "detection_id": detection.get("id") or detection.get("evidence_id"),
            "aoi_id": aoi.get("id") or "aoi_01",
            "is_contained": contract.result,
            "distance": contract.distance,
            "provenance": "ALGORITHMIC",
            "confidence": None,
        }

    @classmethod
    def evaluate_scene_containment(
        cls,
        item: Dict[str, Any],
        scene_bounds: Dict[str, float],
    ) -> bool:
        """Validates that entity coordinates lie within the geographic raster footprint."""
        cx, cy = SpatialRelationshipEngine.get_centroid(item)
        min_x = scene_bounds.get("min_x", -180.0)
        max_x = scene_bounds.get("max_x", 180.0)
        min_y = scene_bounds.get("min_y", -90.0)
        max_y = scene_bounds.get("max_y", 90.0)
        return (min_x <= cx <= max_x) and (min_y <= cy <= max_y)

    @classmethod
    def map_detection_to_geojson(
        cls,
        item: Dict[str, Any],
        geo_transform: Optional[Any] = None,
        crs: str = "EPSG:4326",
    ) -> Dict[str, Any]:
        """
        Maps a detection bounding box or polygon to RFC 7946 GeoJSON Feature format.
        Verifies coordinate finite bounds and CRS.
        """
        geom, space, center = to_shapely_geometry(item, geo_transform, crs)
        if geom is None or HAS_SHAPELY is False:
            cx, cy = SpatialRelationshipEngine.get_centroid(item)
            geojson_geom = {"type": "Point", "coordinates": [cx, cy]}
        else:
            geojson_geom = sgeom.mapping(geom)

        return {
            "type": "Feature",
            "geometry": geojson_geom,
            "properties": {
                "id": item.get("id") or item.get("evidence_id") or "det_01",
                "label": item.get("label") or "entity",
                "score": item.get("score"),
                "provenance": "ALGORITHMIC",
                "crs": crs,
            }
        }

    @classmethod
    def generate_execution_trace(
        cls,
        query: str,
        parsed_subject_target: str,
        parsed_reference_target: str,
        parsed_relation: str,
        candidate_count: int,
        matched_results: List[SpatialContractResult],
    ) -> Dict[str, Any]:
        """
        Generates structured, step-by-step audit trace for spatial query explanation.
        """
        trace = {
            "query": query,
            "parsed_targets": {
                "subject": parsed_subject_target,
                "reference": parsed_reference_target,
            },
            "parsed_relation": parsed_relation,
            "candidate_geometries_count": candidate_count,
            "geometric_operation": f"evaluate_predicate({parsed_relation})",
            "provenance": "ALGORITHMIC",
            "confidence": None,
            "confidence_type": "UNAVAILABLE",
            "results_summary": {
                "total_evaluated_pairs": len(matched_results),
                "satisfied_count": sum(1 for r in matched_results if r.result),
                "satisfied_ids": [r.subject.get("id") for r in matched_results if r.result],
            },
            "trace_steps": [
                f"1. Parsed natural-language query '{query}' into subject '{parsed_subject_target}', relation '{parsed_relation}', reference '{parsed_reference_target}'",
                f"2. Retrieved {candidate_count} candidates for subject and reference entities",
                f"3. Projected geometries to canonical coordinate space and evaluated 2D Shapely/Geodesic predicate",
                f"4. Filtered candidates based on geometric satisfaction with zero statistical fabrication",
            ]
        }
        return trace


# Canonical singleton instance
spatial_engine = GeospatialSpatialEngine()
spatial_relationship_engine = SpatialRelationshipEngine()
