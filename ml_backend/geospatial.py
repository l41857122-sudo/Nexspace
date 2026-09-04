"""
geospatial.py
-------------
Real Geospatial Intelligence, GeoTIFF / CRS / GSD Support & Pixel-to-World Coordinate Engine.

Extracts genuine georeferencing metadata from TIFF/GeoTIFF raster headers, constructs affine transforms,
performs bi-directional Pixel <-> World conversions, computes ground footprint areas and geodesic distances,
and enriches spatial evidence without fabricating coordinates when metadata is absent.
"""

from __future__ import annotations
import math
import io
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from PIL import Image
import numpy as np

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


# Standard TIFF / GeoTIFF Tag IDs
TAG_MODEL_PIXEL_SCALE = 33550       # (ScaleX, ScaleY, ScaleZ)
TAG_MODEL_TIEPOINT = 33922          # (I, J, K, X, Y, Z)
TAG_MODEL_TRANSFORMATION = 34264    # 4x4 matrix
TAG_GEO_KEY_DIRECTORY = 34735       # GeoKey directory array
TAG_GEO_DOUBLE_PARAMS = 34736       # GeoDouble parameters
TAG_GEO_ASCII_PARAMS = 34737        # GeoAscii parameters

# GeoTIFF Key IDs
GEOKEY_GT_MODEL_TYPE = 1024         # 1 = Projected, 2 = Geographic
GEOKEY_GT_RASTER_TYPE = 1025        # 1 = Area (PixelIsArea), 2 = Point (PixelIsPoint)
GEOKEY_GEOGRAPHIC_TYPE = 2048       # e.g. 4326 (WGS84)
GEOKEY_GEOG_CITATION = 2049
GEOKEY_PROJECTED_CSTYPE = 3072      # e.g. 32632 (WGS 84 / UTM zone 32N)
GEOKEY_PROJ_LINEAR_UNITS = 3076     # 9001 = Meter


@dataclass
class GeoTransform:
    """
    Standard 6-parameter affine geotransform:
      X = c + a * x + b * y
      Y = f + d * x + e * y
    Where (c, f) is origin, a is pixel width (dx), e is pixel height (dy, usually negative for north-up).
    """
    a: float  # pixel_width / ScaleX
    b: float  # row rotation / shearing (usually 0)
    c: float  # origin X (upper-left easting/lon)
    d: float  # column rotation / shearing (usually 0)
    e: float  # pixel_height (usually -ScaleY)
    f: float  # origin Y (upper-left northing/lat)

    def pixel_to_world(self, x: float, y: float) -> Tuple[float, float]:
        """Converts pixel coordinate (x, y) to world coordinate (X, Y)."""
        X = self.c + self.a * x + self.b * y
        Y = self.f + self.d * x + self.e * y
        return (round(X, 8), round(Y, 8))

    def world_to_pixel(self, X: float, Y: float) -> Tuple[float, float]:
        """Converts world coordinate (X, Y) back to pixel coordinate (x, y)."""
        det = self.a * self.e - self.b * self.d
        if abs(det) < 1e-15:
            raise ValueError("Degenerate affine transform determinant (cannot invert).")
        x = (self.e * (X - self.c) - self.b * (Y - self.f)) / det
        y = (-self.d * (X - self.c) + self.a * (Y - self.f)) / det
        return (round(x, 4), round(y, 4))

    def bbox_pixel_to_world(self, bbox_pixel: Union[List[float], Tuple[float, ...]], crs: str) -> Dict[str, Any]:
        """Converts bounding box [x1, y1, x2, y2] to geographic/projected world footprint."""
        x1, y1, x2, y2 = bbox_pixel
        # Calculate 4 corners in world coordinates
        c1 = self.pixel_to_world(x1, y1)
        c2 = self.pixel_to_world(x2, y1)
        c3 = self.pixel_to_world(x2, y2)
        c4 = self.pixel_to_world(x1, y2)

        xs = [c1[0], c2[0], c3[0], c4[0]]
        ys = [c1[1], c2[1], c3[1], c4[1]]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        return {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "crs": crs,
            "polygon_world": [c1, c2, c3, c4, c1],
        }

    def to_dict(self) -> Dict[str, float]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
            "e": self.e,
            "f": self.f,
        }


@dataclass
class GeoMetadata:
    """Geospatial metadata container extracted from raster headers."""
    geospatial_available: bool = False
    crs: Optional[str] = None
    crs_type: str = "unknown"  # "geographic" | "projected" | "unknown"
    crs_epsg: Optional[int] = None
    crs_units: str = "unknown"  # "degree" | "metre" | "unknown"
    transform: Optional[GeoTransform] = None
    resolution: Optional[Dict[str, Any]] = None  # {"x": float, "y": float, "unit": str}
    bounds_world: Optional[Dict[str, float]] = None
    dimensions_pixel: Optional[Dict[str, int]] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "geospatial_available": self.geospatial_available,
            "crs": self.crs,
            "crs_type": self.crs_type,
            "crs_epsg": self.crs_epsg,
            "crs_units": self.crs_units,
            "transform": self.transform.to_dict() if self.transform else None,
            "resolution": self.resolution,
            "bounds_world": self.bounds_world,
            "dimensions_pixel": self.dimensions_pixel,
            "reason": self.reason,
        }


class GeospatialEngine:
    """Core Geospatial Intelligence Engine for GeoTIFF extraction and coordinate transformations."""

    @classmethod
    def extract_metadata(cls, image_input: Union[Image.Image, str, bytes]) -> GeoMetadata:
        """
        Extracts genuine georeferencing metadata from a PIL Image, filepath, or byte stream.
        If no geospatial metadata exists (e.g. ordinary PNG/JPEG), returns a structured unavailable result.
        """
        tags: Dict[int, Any] = {}
        width, height = 0, 0

        # Case 1: Image path as string
        if isinstance(image_input, str):
            if HAS_TIFFFILE:
                try:
                    with tifffile.TiffFile(image_input) as tif:
                        page = tif.pages[0]
                        width, height = page.shape[1], page.shape[0] if len(page.shape) >= 2 else (0, 0)
                        for tag in page.tags.values():
                            tags[tag.code] = tag.value
                except Exception:
                    pass

            if not tags:
                try:
                    with Image.open(image_input) as img:
                        width, height = img.size
                        if hasattr(img, "tag_v2"):
                            tags = {k: v for k, v in img.tag_v2.items()}
                        elif hasattr(img, "tag"):
                            tags = {k: v for k, v in img.tag.items()}
                except Exception:
                    pass

        # Case 2: PIL Image instance
        elif isinstance(image_input, Image.Image):
            width, height = image_input.size
            if hasattr(image_input, "tag_v2"):
                tags = {k: v for k, v in image_input.tag_v2.items()}
            elif hasattr(image_input, "tag"):
                tags = {k: v for k, v in image_input.tag.items()}

        # Case 3: Raw Bytes
        elif isinstance(image_input, bytes):
            if HAS_TIFFFILE:
                try:
                    with tifffile.TiffFile(io.BytesIO(image_input)) as tif:
                        page = tif.pages[0]
                        width, height = page.shape[1], page.shape[0] if len(page.shape) >= 2 else (0, 0)
                        for tag in page.tags.values():
                            tags[tag.code] = tag.value
                except Exception:
                    pass

            if not tags:
                try:
                    with Image.open(io.BytesIO(image_input)) as img:
                        width, height = img.size
                        if hasattr(img, "tag_v2"):
                            tags = {k: v for k, v in img.tag_v2.items()}
                        elif hasattr(img, "tag"):
                            tags = {k: v for k, v in img.tag.items()}
                except Exception:
                    pass

        # Validate presence of standard GeoTIFF spatial tags
        has_pixel_scale = TAG_MODEL_PIXEL_SCALE in tags
        has_tiepoint = TAG_MODEL_TIEPOINT in tags
        has_transform_matrix = TAG_MODEL_TRANSFORMATION in tags

        if not (has_transform_matrix or (has_pixel_scale and has_tiepoint)):
            return GeoMetadata(
                geospatial_available=False,
                dimensions_pixel={"width": width, "height": height} if width and height else None,
                reason="No geospatial metadata available (standard ModelPixelScale/ModelTiepoint tags absent)",
            )

        # Parse Affine Geotransform
        transform: Optional[GeoTransform] = None
        scale_x, scale_y = 1.0, 1.0

        if has_pixel_scale and has_tiepoint:
            scale_tag = tags[TAG_MODEL_PIXEL_SCALE]
            tie_tag = tags[TAG_MODEL_TIEPOINT]

            scale_x = float(scale_tag[0])
            scale_y = float(scale_tag[1])

            # ModelTiepoint: (I, J, K, X, Y, Z)
            # Typically tiepoint 0 is at pixel (0, 0, 0) -> world (X0, Y0, Z0)
            i0, j0, _, x0, y0, _ = [float(v) for v in tie_tag[:6]]

            # North-up affine transform
            origin_x = x0 - i0 * scale_x
            origin_y = y0 + j0 * scale_y

            transform = GeoTransform(
                a=scale_x,
                b=0.0,
                c=origin_x,
                d=0.0,
                e=-scale_y,
                f=origin_y,
            )

        elif has_transform_matrix:
            mat = tags[TAG_MODEL_TRANSFORMATION]
            # 4x4 matrix flattened or nested
            if len(mat) == 16:
                a, b, c = float(mat[0]), float(mat[1]), float(mat[3])
                d, e, f = float(mat[4]), float(mat[5]), float(mat[7])
                scale_x, scale_y = abs(a), abs(e)
                transform = GeoTransform(a=a, b=b, c=c, d=d, e=e, f=f)

        if transform is None:
            return GeoMetadata(
                geospatial_available=False,
                dimensions_pixel={"width": width, "height": height},
                reason="Malformed georeferencing transform tags",
            )

        # Detect CRS and GeoKeys
        geo_keys = tags.get(TAG_GEO_KEY_DIRECTORY)
        crs_str, crs_type, crs_epsg, crs_units = cls._detect_crs_from_tags(geo_keys)

        # Compute full image bounds in world space
        bounds_world: Optional[Dict[str, float]] = None
        if width > 0 and height > 0:
            b_dict = transform.bbox_pixel_to_world([0, 0, width, height], crs_str)
            bounds_world = {
                "min_x": b_dict["min_x"],
                "min_y": b_dict["min_y"],
                "max_x": b_dict["max_x"],
                "max_y": b_dict["max_y"],
            }

        return GeoMetadata(
            geospatial_available=True,
            crs=crs_str,
            crs_type=crs_type,
            crs_epsg=crs_epsg,
            crs_units=crs_units,
            transform=transform,
            resolution={"x": round(scale_x, 6), "y": round(scale_y, 6), "unit": crs_units},
            bounds_world=bounds_world,
            dimensions_pixel={"width": width, "height": height},
            reason=None,
        )

    @classmethod
    def _detect_crs_from_tags(
        cls,
        geo_keys: Optional[Union[List[int], Tuple[int, ...]]]
    ) -> Tuple[str, str, Optional[int], str]:
        """Parses GeoKey directory to detect EPSG code, projection type, and units."""
        if not geo_keys or len(geo_keys) < 4:
            # Fallback when tiepoints exist but no GeoKeys: standard WGS84 assumption or unprojected
            return ("EPSG:4326", "geographic", 4326, "degree")

        # GeoKey directory format: [Header (4 ints), Key1 (4 ints), Key2 (4 ints), ...]
        num_keys = geo_keys[3]
        keys_dict: Dict[int, int] = {}
        idx = 4
        for _ in range(num_keys):
            if idx + 4 > len(geo_keys):
                break
            key_id, location, count, val_or_offset = geo_keys[idx:idx + 4]
            if location == 0:
                keys_dict[key_id] = val_or_offset
            idx += 4

        # 1. Projected CRS detection
        proj_code = keys_dict.get(GEOKEY_PROJECTED_CSTYPE)
        if proj_code and proj_code > 0 and proj_code != 32767:
            crs_str = f"EPSG:{proj_code}"
            crs_type = "projected"
            crs_epsg = proj_code
            crs_units = "metre"

            if HAS_PYPROJ:
                try:
                    c = pyproj.CRS.from_epsg(proj_code)
                    if c.is_geographic:
                        crs_type = "geographic"
                        crs_units = "degree"
                    else:
                        crs_type = "projected"
                        crs_units = "metre"
                except Exception:
                    pass
            return (crs_str, crs_type, crs_epsg, crs_units)

        # 2. Geographic CRS detection
        geog_code = keys_dict.get(GEOKEY_GEOGRAPHIC_TYPE)
        if geog_code and geog_code > 0 and geog_code != 32767:
            crs_str = f"EPSG:{geog_code}"
            crs_type = "geographic"
            crs_epsg = geog_code
            crs_units = "degree"
            return (crs_str, crs_type, crs_epsg, crs_units)

        # 3. Model Type detection
        model_type = keys_dict.get(GEOKEY_GT_MODEL_TYPE)
        if model_type == 1:
            return ("EPSG:3857", "projected", 3857, "metre")
        elif model_type == 2:
            return ("EPSG:4326", "geographic", 4326, "degree")

        return ("EPSG:4326", "geographic", 4326, "degree")

    @classmethod
    def calculate_ground_area(
        cls,
        area_pixels: float,
        geo_meta: GeoMetadata,
        center_lat: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates physical ground footprint area from pixel area and genuine raster resolution.
        """
        if not geo_meta.geospatial_available or not geo_meta.transform or not geo_meta.resolution:
            return {
                "area_pixels": area_pixels,
                "ground_area": None,
                "ground_area_unit": None,
                "geospatial_available": False,
            }

        res_x = geo_meta.resolution.get("x", 1.0)
        res_y = geo_meta.resolution.get("y", 1.0)

        # If projected CRS (e.g. UTM, WebMercator), units are in meters
        if geo_meta.crs_units in ("metre", "meter", "m"):
            ground_area_m2 = area_pixels * res_x * res_y
            return {
                "area_pixels": area_pixels,
                "ground_area": round(ground_area_m2, 2),
                "ground_area_unit": "m2",
                "geospatial_available": True,
            }

        # If geographic CRS (degrees), calculate metric area using latitude cosine scaling
        elif geo_meta.crs_units in ("degree", "deg"):
            if center_lat is None and geo_meta.bounds_world:
                center_lat = (geo_meta.bounds_world["min_y"] + geo_meta.bounds_world["max_y"]) / 2.0
            lat_rad = math.radians(center_lat if center_lat is not None else 0.0)
            # 1 deg lat ~ 111,320 m, 1 deg lon ~ 111,320 * cos(lat) m
            meters_per_deg_lat = 111320.0
            meters_per_deg_lon = 111320.0 * max(0.01, math.cos(lat_rad))
            ground_area_m2 = area_pixels * (res_x * meters_per_deg_lon) * (res_y * meters_per_deg_lat)
            return {
                "area_pixels": area_pixels,
                "ground_area": round(ground_area_m2, 2),
                "ground_area_unit": "m2",
                "geospatial_available": True,
            }

        return {
            "area_pixels": area_pixels,
            "ground_area": round(area_pixels * res_x * res_y, 4),
            "ground_area_unit": geo_meta.crs_units,
            "geospatial_available": True,
        }

    @classmethod
    def calculate_ground_distance(
        cls,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        geo_meta: GeoMetadata
    ) -> Dict[str, Any]:
        """
        Calculates ground distance between two world coordinates using genuine geodesy or projected metrics.
        """
        if not geo_meta.geospatial_available:
            return {
                "distance": None,
                "unit": None,
                "method": "unavailable",
            }

        # Projected CRS: Euclidean metric distance
        if geo_meta.crs_units in ("metre", "meter", "m"):
            dist_m = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            return {
                "distance": round(dist_m, 2),
                "unit": "meter",
                "method": "projected_euclidean",
            }

        # Geographic CRS: Haversine / Geodesic distance
        elif geo_meta.crs_units in ("degree", "deg"):
            lon1, lat1 = p1
            lon2, lat2 = p2
            if HAS_PYPROJ:
                try:
                    geod = pyproj.Geod(ellps="WGS84")
                    _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
                    return {
                        "distance": round(dist_m, 2),
                        "unit": "meter",
                        "method": "wgs84_geodesic",
                    }
                except Exception:
                    pass

            # Haversine formula fallback
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            delta_phi = math.radians(lat2 - lat1)
            delta_lambda = math.radians(lon2 - lon1)
            a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
            c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
            dist_m = 6371000.0 * c
            return {
                "distance": round(dist_m, 2),
                "unit": "meter",
                "method": "haversine_great_circle",
            }

        return {
            "distance": round(math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2), 4),
            "unit": geo_meta.crs_units,
            "method": "crs_native",
        }

    @classmethod
    def enrich_evidence_item(cls, item: Dict[str, Any], geo_meta: GeoMetadata) -> Dict[str, Any]:
        """
        Enriches a spatial evidence item (e.g. change_region or object_detection) with genuine world coordinates.
        Preserves original pixel and normalized coordinates.
        """
        enriched = dict(item)

        if not geo_meta.geospatial_available or not geo_meta.transform:
            enriched["geospatial_coordinates_available"] = False
            return enriched

        box = item.get("bbox_pixel") or item.get("box")
        if box and len(box) == 4:
            world_footprint = geo_meta.transform.bbox_pixel_to_world(box, geo_meta.crs or "EPSG:4326")
            enriched["bbox_world"] = world_footprint
            enriched["crs"] = geo_meta.crs
            enriched["geospatial_coordinates_available"] = True

            # Calculate physical ground area if area_pixels is present
            area_px = item.get("area_pixels")
            if area_px is not None:
                center_lat = (world_footprint["min_y"] + world_footprint["max_y"]) / 2.0 if geo_meta.crs_units == "degree" else None
                area_calc = cls.calculate_ground_area(float(area_px), geo_meta, center_lat)
                if area_calc.get("ground_area") is not None:
                    enriched["ground_area"] = area_calc["ground_area"]
                    enriched["ground_area_unit"] = area_calc["ground_area_unit"]

        return enriched
