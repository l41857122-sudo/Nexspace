"""
overpass_client.py
------------------
OpenStreetMap Overpass API Integration Layer for NexSpace.

Provides robust, rate-limited, cached geospatial feature discovery for:
- Building footprints and structures
- Roads, highways, and transport paths
- Bridges and elevated infrastructure
- Waterways and natural water bodies
- Industrial and port facilities

Guarantees:
- Never invents or hallucinates features if Overpass returns empty or fails.
- Adheres to OpenStreetMap usage policies with user-agent, timeouts, and rate limits.
- Provides standard GeoJSON FeatureCollections in WGS84 (EPSG:4326).
- Strict attribution to OpenStreetMap contributors.
"""

from __future__ import annotations

import os
import json
import time
import math
import hashlib
import logging
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Set

logger = logging.getLogger("nexspace.overpass")

# Provider Endpoints (with fallback mirrors)
DEFAULT_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

DEFAULT_USER_AGENT = "NexSpace-Geospatial-Intelligence/2.0 (satellite-analysis-platform)"
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "overpass")
CACHE_TTL_SECONDS = 86400  # 24 hours
MAX_BBOX_AREA_DEG2 = 0.25  # ~30km x 30km safety bounding limit


@dataclass
class OverpassQueryResult:
    """Canonical result of an Overpass API feature query."""
    status: str  # "success" | "empty" | "timeout" | "rate_limited" | "error" | "invalid_bbox"
    geojson: Dict[str, Any]  # GeoJSON FeatureCollection
    total_features: int
    feature_counts: Dict[str, int]
    attribution: str
    cached: bool
    execution_duration_ms: float
    query_bbox: List[float]  # [min_lon, min_lat, max_lon, max_lat]
    error_message: Optional[str] = None
    actionable_reason: Optional[str] = None
    endpoint_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OverpassClient:
    """Production client for querying OpenStreetMap features via Overpass API."""

    def __init__(
        self,
        endpoints: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        user_agent: str = DEFAULT_USER_AGENT,
        request_interval_s: float = 1.0,
    ):
        self.endpoints = endpoints or DEFAULT_OVERPASS_ENDPOINTS
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.user_agent = user_agent
        self.request_interval_s = request_interval_s
        self._last_request_time: float = 0.0

        os.makedirs(self.cache_dir, exist_ok=True)

    def validate_bbox(self, bbox: Any) -> Tuple[bool, Optional[List[float]], Optional[str]]:
        """
        Validates and normalizes bounding box into [min_lon, min_lat, max_lon, max_lat].
        """
        if not bbox:
            return False, None, "Bounding box is empty or missing."

        if isinstance(bbox, dict):
            try:
                min_lon = float(bbox.get("west", bbox.get("min_lon", 0.0)))
                min_lat = float(bbox.get("south", bbox.get("min_lat", 0.0)))
                max_lon = float(bbox.get("east", bbox.get("max_lon", 0.0)))
                max_lat = float(bbox.get("north", bbox.get("max_lat", 0.0)))
                box = [min_lon, min_lat, max_lon, max_lat]
            except (ValueError, TypeError):
                return False, None, "Invalid dictionary coordinates in bounding box."
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                box = [float(c) for c in bbox]
            except (ValueError, TypeError):
                return False, None, "Invalid numeric values in bounding box list."
        else:
            return False, None, f"Expected 4 coordinates [min_lon, min_lat, max_lon, max_lat], got {type(bbox)}"

        min_lon, min_lat, max_lon, max_lat = box

        # Spatial range validation
        if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
            return False, None, f"Longitude out of WGS84 range [-180, 180]: min_lon={min_lon}, max_lon={max_lon}"
        if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
            return False, None, f"Latitude out of WGS84 range [-90, 90]: min_lat={min_lat}, max_lat={max_lat}"

        if min_lon > max_lon:
            return False, None, f"min_lon ({min_lon}) cannot be greater than max_lon ({max_lon})"
        if min_lat > max_lat:
            return False, None, f"min_lat ({min_lat}) cannot be greater than max_lat ({max_lat})"

        # Check for zero or degenerate area
        if min_lon == max_lon and min_lat == max_lat:
            return False, None, "Degenerate point bounding box provided without area."

        return True, box, None

    def build_query(self, bbox: List[float], feature_types: Optional[List[str]] = None, timeout_s: int = 25) -> str:
        """
        Constructs Overpass QL with `out geom;` for direct geometry extraction.
        bbox is [min_lon, min_lat, max_lon, max_lat].
        Overpass expects (min_lat, min_lon, max_lat, max_lon) in South, West, North, East order.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        s, w, n, e = min_lat, min_lon, max_lat, max_lon

        if not feature_types:
            feature_types = ["buildings", "roads", "bridges", "waterways"]

        parts: List[str] = []

        if "buildings" in feature_types or "all" in feature_types:
            parts.append(f'way["building"]({s},{w},{n},{e});')
            parts.append(f'relation["building"]({s},{w},{n},{e});')

        if "roads" in feature_types or "all" in feature_types:
            parts.append(f'way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|service|unclassified"]({s},{w},{n},{e});')

        if "bridges" in feature_types or "all" in feature_types:
            parts.append(f'way["bridge"="yes"]({s},{w},{n},{e});')

        if "waterways" in feature_types or "water" in feature_types or "all" in feature_types:
            parts.append(f'way["waterway"]({s},{w},{n},{e});')
            parts.append(f'way["natural"="water"]({s},{w},{n},{e});')
            parts.append(f'relation["natural"="water"]({s},{w},{n},{e});')

        if "infrastructure" in feature_types or "all" in feature_types:
            parts.append(f'way["aeroway"]({s},{w},{n},{e});')
            parts.append(f'way["railway"]({s},{w},{n},{e});')
            parts.append(f'way["industrial"]({s},{w},{n},{e});')

        body = "\n  ".join(parts)
        query = f"[out:json][timeout:{timeout_s}];\n(\n  {body}\n);\nout body geom qt;"
        return query

    def _get_cache_path(self, bbox: List[float], feature_types: List[str]) -> str:
        key_str = f"{[round(c, 5) for c in bbox]}_{sorted(feature_types)}"
        hash_str = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"overpass_{hash_str}.json")

    def _read_cache(self, cache_path: str) -> Optional[OverpassQueryResult]:
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_time = data.get("cached_at", 0)
            if (time.time() - cached_time) < CACHE_TTL_SECONDS:
                res_dict = data.get("result", {})
                res_dict["cached"] = True
                return OverpassQueryResult(**res_dict)
        except Exception as e:
            logger.debug(f"Failed to read cache {cache_path}: {e}")
        return None

    def _write_cache(self, cache_path: str, result: OverpassQueryResult) -> None:
        try:
            data = {
                "cached_at": time.time(),
                "result": result.to_dict(),
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Failed to write cache {cache_path}: {e}")

    def parse_elements_to_geojson(self, elements: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Parses Overpass JSON elements (with geom coordinates) into GeoJSON FeatureCollection.
        """
        features: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {
            "buildings": 0,
            "roads": 0,
            "bridges": 0,
            "waterways": 0,
            "infrastructure": 0,
            "other": 0,
        }

        seen_ids: Set[str] = set()

        for el in elements:
            el_type = el.get("type")
            el_id = el.get("id")
            global_id = f"{el_type}/{el_id}"
            if global_id in seen_ids:
                continue
            seen_ids.add(global_id)

            tags = el.get("tags", {})
            geom = el.get("geometry", [])

            # Categorize
            feature_class = "other"
            if "building" in tags:
                feature_class = "buildings"
            elif tags.get("bridge") == "yes":
                feature_class = "bridges"
            elif "highway" in tags:
                feature_class = "roads"
            elif "waterway" in tags or tags.get("natural") == "water":
                feature_class = "waterways"
            elif "aeroway" in tags or "railway" in tags or "industrial" in tags:
                feature_class = "infrastructure"

            counts[feature_class] = counts.get(feature_class, 0) + 1

            geometry = None

            # 1. Point (node)
            if el_type == "node" and "lat" in el and "lon" in el:
                geometry = {
                    "type": "Point",
                    "coordinates": [float(el["lon"]), float(el["lat"])],
                }

            # 2. Way with geometry points
            elif el_type == "way" and isinstance(geom, list) and len(geom) >= 2:
                coords = [[float(pt["lon"]), float(pt["lat"])] for pt in geom if "lon" in pt and "lat" in pt]
                if len(coords) >= 2:
                    # Check if polygon (closed ring with >= 4 points)
                    is_closed = (coords[0] == coords[-1]) and (len(coords) >= 4)
                    if is_closed and feature_class in ("buildings", "waterways", "infrastructure"):
                        geometry = {
                            "type": "Polygon",
                            "coordinates": [coords],
                        }
                    else:
                        geometry = {
                            "type": "LineString",
                            "coordinates": coords,
                        }

            # 3. Relation with members
            elif el_type == "relation" and "members" in el:
                # Basic multi-geometry extraction from way members
                polygon_rings: List[List[List[float]]] = []
                for m in el["members"]:
                    m_geom = m.get("geometry", [])
                    if isinstance(m_geom, list) and len(m_geom) >= 4:
                        m_coords = [[float(pt["lon"]), float(pt["lat"])] for pt in m_geom if "lon" in pt and "lat" in pt]
                        if m_coords and m_coords[0] == m_coords[-1]:
                            polygon_rings.append(m_coords)

                if polygon_rings:
                    geometry = {
                        "type": "MultiPolygon",
                        "coordinates": [[[ring] for ring in polygon_rings]],
                    }

            if geometry:
                feature = {
                    "type": "Feature",
                    "id": global_id,
                    "geometry": geometry,
                    "properties": {
                        "osm_id": el_id,
                        "osm_type": el_type,
                        "feature_class": feature_class,
                        "name": tags.get("name"),
                        "tags": tags,
                        "attribution": "© OpenStreetMap contributors",
                    },
                }
                features.append(feature)

        geojson_coll = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "attribution": "© OpenStreetMap contributors",
                "osm_license": "Open Data Commons Open Database License (ODbL)",
                "feature_count": len(features),
            },
        }

        return geojson_coll, counts

    def query_features(
        self,
        bbox: Any,
        feature_types: Optional[List[str]] = None,
        timeout_s: int = 25,
        use_cache: bool = True,
    ) -> OverpassQueryResult:
        """
        Queries OpenStreetMap for features inside bbox.
        Does not fabricate any data. Returns empty or structured error if unavailable.
        """
        start_time = time.time()
        feature_types = feature_types or ["buildings", "roads", "bridges", "waterways"]

        # Validate bounding box
        valid, norm_bbox, err = self.validate_bbox(bbox)
        if not valid or not norm_bbox:
            return OverpassQueryResult(
                status="invalid_bbox",
                geojson={"type": "FeatureCollection", "features": []},
                total_features=0,
                feature_counts={},
                attribution="© OpenStreetMap contributors",
                cached=False,
                execution_duration_ms=(time.time() - start_time) * 1000.0,
                query_bbox=bbox if isinstance(bbox, list) else [],
                error_message=err,
                actionable_reason="Provide a valid [min_lon, min_lat, max_lon, max_lat] bounding box in WGS84.",
            )

        # Validate area size to avoid slamming public Overpass infrastructure
        delta_lon = norm_bbox[2] - norm_bbox[0]
        delta_lat = norm_bbox[3] - norm_bbox[1]
        area = delta_lon * delta_lat
        if area > MAX_BBOX_AREA_DEG2:
            # Clamp bbox around center to safe window
            c_lon = (norm_bbox[0] + norm_bbox[2]) / 2.0
            c_lat = (norm_bbox[1] + norm_bbox[3]) / 2.0
            half_d = math.sqrt(MAX_BBOX_AREA_DEG2) / 2.0
            norm_bbox = [
                round(c_lon - half_d, 6),
                round(c_lat - half_d, 6),
                round(c_lon + half_d, 6),
                round(c_lat + half_d, 6),
            ]
            logger.info(f"Query bbox exceeded {MAX_BBOX_AREA_DEG2} deg2. Clamped to centered AOI: {norm_bbox}")

        # Check local cache
        cache_path = self._get_cache_path(norm_bbox, feature_types)
        if use_cache:
            cached_res = self._read_cache(cache_path)
            if cached_res:
                logger.info(f"Overpass cache hit for bbox={norm_bbox}: {cached_res.total_features} features")
                return cached_res

        # Rate limiting: wait if needed
        now = time.time()
        elapsed_since_last = now - self._last_request_time
        if elapsed_since_last < self.request_interval_s:
            time.sleep(self.request_interval_s - elapsed_since_last)

        query_ql = self.build_query(norm_bbox, feature_types, timeout_s=timeout_s)
        encoded_data = urllib.parse.urlencode({"data": query_ql}).encode("utf-8")

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        last_error = ""
        last_reason = ""
        raw_json = None
        successful_endpoint = None

        for endpoint in self.endpoints:
            try:
                self._last_request_time = time.time()
                req = urllib.request.Request(endpoint, data=encoded_data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout_s) as response:
                    status_code = response.getcode()
                    if status_code == 200:
                        content = response.read().decode("utf-8")
                        raw_json = json.loads(content)
                        successful_endpoint = endpoint
                        break
                    elif status_code == 429:
                        last_error = "HTTP 429: Rate limited by Overpass API."
                        last_reason = "Overpass public rate limit exceeded. Please back off query frequency."
                        continue
            except urllib.error.HTTPError as he:
                last_error = f"HTTP {he.code}: {he.reason}"
                if he.code == 429:
                    last_reason = "Overpass public server rate limited. Switch or back off."
                elif he.code == 504:
                    last_reason = "Overpass query timed out on server side. Reduce query bounding box size."
                continue
            except (urllib.error.URLError, TimeoutError, OSError) as ue:
                last_error = f"Network or timeout error contacting Overpass ({endpoint}): {ue}"
                last_reason = "Overpass API mirror unreachable or timed out."
                continue
            except Exception as ex:
                last_error = f"Unexpected error during Overpass query: {ex}"
                last_reason = "Could not parse or execute Overpass response."
                continue

        duration_ms = (time.time() - start_time) * 1000.0

        if raw_json is None:
            # Overpass service is unavailable or rate limited - graceful fallback without crashing
            logger.warning(f"Overpass query failed: {last_error}")
            return OverpassQueryResult(
                status="rate_limited" if "429" in last_error else "timeout" if "timed out" in last_error else "error",
                geojson={"type": "FeatureCollection", "features": []},
                total_features=0,
                feature_counts={},
                attribution="© OpenStreetMap contributors",
                cached=False,
                execution_duration_ms=duration_ms,
                query_bbox=norm_bbox,
                error_message=last_error,
                actionable_reason=last_reason,
            )

        elements = raw_json.get("elements", [])
        if not elements:
            res = OverpassQueryResult(
                status="empty",
                geojson={"type": "FeatureCollection", "features": []},
                total_features=0,
                feature_counts={t: 0 for t in feature_types},
                attribution="© OpenStreetMap contributors",
                cached=False,
                execution_duration_ms=duration_ms,
                query_bbox=norm_bbox,
                endpoint_used=successful_endpoint,
            )
            if use_cache:
                self._write_cache(cache_path, res)
            return res

        geojson_coll, counts = self.parse_elements_to_geojson(elements)
        total_feats = len(geojson_coll.get("features", []))

        res = OverpassQueryResult(
            status="success",
            geojson=geojson_coll,
            total_features=total_feats,
            feature_counts=counts,
            attribution="© OpenStreetMap contributors",
            cached=False,
            execution_duration_ms=duration_ms,
            query_bbox=norm_bbox,
            endpoint_used=successful_endpoint,
        )

        if use_cache and total_feats > 0:
            self._write_cache(cache_path, res)

        return res


# Global singleton instance
_overpass_client_instance: Optional[OverpassClient] = None

def get_overpass_client() -> OverpassClient:
    global _overpass_client_instance
    if _overpass_client_instance is None:
        _overpass_client_instance = OverpassClient()
    return _overpass_client_instance
