"""
co_registration.py
------------------
True Geospatial Co-Registration, CRS Reprojection & GSD Harmonization Engine.

Provides mathematically rigorous, geospatially grounded raster alignment:
  1. RasterGeometry: Canonical container for CRS, affine transform, native GSD, bounds, nodata.
  2. CRS Validation: Strict matching, mismatch detection, and rejection of missing/invalid CRS.
  3. True Spatial Overlap: Common-CRS 4-corner bounding box projection and intersection calculation.
  4. Common Target Grid: Deterministic selection of target CRS, bounds, transform, and resolution.
  5. Geospatial Reprojection: rasterio.warp.reproject for continuous (bilinear) and categorical (nearest) rasters.
  6. Provenance & Telemetry: Explicit tracking of resampling, overlap ratio, valid/nodata pixels.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from PIL import Image

import rasterio
import rasterio.warp
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
import pyproj


class AlignmentStatus(str, Enum):
    ALIGNED = "ALIGNED"
    REPROJECTED = "REPROJECTED"
    RESAMPLED = "RESAMPLED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


class OverlapStatus(str, Enum):
    OVERLAP_CONFIRMED = "OVERLAP_CONFIRMED"
    NO_OVERLAP = "NO_OVERLAP"
    OVERLAP_UNAVAILABLE = "OVERLAP_UNAVAILABLE"


@dataclass
class RasterGeometry:
    """
    Canonical geospatial geometry contract for a raster input.
    All alignment and reprojection decisions are strictly derived from this metadata.
    """
    width: int
    height: int
    crs: Optional[str]
    bounds: Optional[Dict[str, float]]               # {"min_x": float, "min_y": float, "max_x": float, "max_y": float}
    transform: Optional[Tuple[float, ...]]           # Standard 6-element affine: (a, b, c, d, e, f)
    resolution_x: float
    resolution_y: float
    pixel_size: float
    units: str                                       # "metre", "meter", "degree", "deg", "unknown"
    native_gsd: float                                # Ground Sampling Distance in meters (approx for degree CRS)
    nodata: Optional[float] = None
    dtype: str = "float32"
    source: str = "unknown"
    geometry_provenance: str = "UNREFERENCED"        # "EXTRACTED_TIFF", "EXTRACTED_RASTERIO", "SUPPLIED_EXPLICIT", "UNREFERENCED"

    def is_valid_geospatial(self) -> bool:
        """Returns True only if CRS, transform, and bounds are present and valid."""
        return bool(
            self.crs
            and self.transform
            and len(self.transform) == 6
            and self.bounds
            and None not in (self.bounds.get("min_x"), self.bounds.get("min_y"), self.bounds.get("max_x"), self.bounds.get("max_y"))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "bounds": self.bounds,
            "transform": list(self.transform) if self.transform else None,
            "resolution_x": round(self.resolution_x, 6),
            "resolution_y": round(self.resolution_y, 6),
            "pixel_size": round(self.pixel_size, 6),
            "units": self.units,
            "native_gsd": round(self.native_gsd, 4),
            "nodata": self.nodata,
            "dtype": self.dtype,
            "source": self.source,
            "geometry_provenance": self.geometry_provenance,
        }

    @classmethod
    def from_image_or_metadata(
        cls,
        image_or_array: Union[Image.Image, np.ndarray],
        crs: Optional[str] = None,
        transform: Optional[Union[Tuple[float, ...], List[float], Affine]] = None,
        bounds: Optional[Union[Dict[str, float], Tuple[float, float, float, float], List[float]]] = None,
        resolution: Optional[Union[Dict[str, Any], Tuple[float, float], float]] = None,
        nodata: Optional[float] = None,
        source: str = "user_input",
    ) -> RasterGeometry:
        """
        Constructs a RasterGeometry instance from image/array and optional explicit/extracted parameters.
        Never fabricates CRS, transform, or bounds if genuinely absent.
        """
        # 1. Dimensions
        if isinstance(image_or_array, Image.Image):
            w, h = image_or_array.size
            dt = "uint8"
        elif isinstance(image_or_array, np.ndarray):
            h, w = image_or_array.shape[:2]
            dt = str(image_or_array.dtype)
        else:
            w, h = 0, 0
            dt = "unknown"

        # 2. Extract from GeoTIFF tags if Image.Image and parameters not fully supplied
        geo_meta = None
        if isinstance(image_or_array, Image.Image) and (not crs or not transform):
            from geospatial import GeospatialEngine
            geo_meta = GeospatialEngine.extract_metadata(image_or_array)
            if geo_meta.geospatial_available:
                crs = crs or geo_meta.crs
                if not transform and geo_meta.transform:
                    transform = (
                        geo_meta.transform.a, geo_meta.transform.b, geo_meta.transform.c,
                        geo_meta.transform.d, geo_meta.transform.e, geo_meta.transform.f,
                    )
                bounds = bounds or geo_meta.bounds_world

        # 3. Normalize transform
        t_tuple: Optional[Tuple[float, ...]] = None
        if transform is not None:
            if isinstance(transform, Affine):
                t_tuple = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
            elif isinstance(transform, (list, tuple)) and len(transform) >= 6:
                t_tuple = tuple(float(x) for x in transform[:6])

        # 4. Normalize bounds
        b_dict: Optional[Dict[str, float]] = None
        if isinstance(bounds, dict):
            min_x = bounds.get("min_x", bounds.get("west", bounds.get("left")))
            min_y = bounds.get("min_y", bounds.get("south", bounds.get("bottom")))
            max_x = bounds.get("max_x", bounds.get("east", bounds.get("right")))
            max_y = bounds.get("max_y", bounds.get("north", bounds.get("top")))
            if None not in (min_x, min_y, max_x, max_y):
                b_dict = {"min_x": float(min_x), "min_y": float(min_y), "max_x": float(max_x), "max_y": float(max_y)}
        elif isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            b_dict = {"min_x": float(bounds[0]), "min_y": float(bounds[1]), "max_x": float(bounds[2]), "max_y": float(bounds[3])}

        # If transform is present but bounds missing, compute exact bounds from corners
        if t_tuple is not None and b_dict is None and w > 0 and h > 0:
            a, b, c, d, e, f = t_tuple
            c1 = (c, f)
            c2 = (c + a * w, f + d * w)
            c3 = (c + a * w + b * h, f + d * w + e * h)
            c4 = (c + b * h, f + e * h)
            xs = [c1[0], c2[0], c3[0], c4[0]]
            ys = [c1[1], c2[1], c3[1], c4[1]]
            b_dict = {"min_x": min(xs), "min_y": min(ys), "max_x": max(xs), "max_y": max(ys)}

        # 5. Resolution & Units
        res_x, res_y = 1.0, 1.0
        units = "unknown"
        if t_tuple is not None:
            res_x = abs(t_tuple[0])
            res_y = abs(t_tuple[4])
        elif isinstance(resolution, dict):
            res_x = float(resolution.get("x", 1.0))
            res_y = float(resolution.get("y", 1.0))
            units = resolution.get("unit", "unknown")
        elif isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            res_x, res_y = float(resolution[0]), float(resolution[1])
        elif isinstance(resolution, (int, float)):
            res_x = res_y = float(resolution)

        # Infer units and GSD in meters
        native_gsd = max(res_x, res_y)
        if crs:
            crs_upper = str(crs).upper()
            if any(k in crs_upper for k in ("326", "327", "UTM", "3857")):
                units = "metre"
                native_gsd = max(res_x, res_y)
            elif "4326" in crs_upper or "CRS84" in crs_upper:
                units = "degree"
                # Approximate 1 degree ~ 111,320m at equator
                center_lat = 0.0
                if b_dict:
                    center_lat = (b_dict["min_y"] + b_dict["max_y"]) / 2.0
                m_lat = 111320.0
                m_lon = 111320.0 * max(0.01, math.cos(math.radians(center_lat)))
                native_gsd = max(res_x * m_lon, res_y * m_lat)

        provenance = "EXTRACTED_TIFF" if geo_meta and geo_meta.geospatial_available else (
            "SUPPLIED_EXPLICIT" if (crs and t_tuple and b_dict) else "UNREFERENCED"
        )

        return cls(
            width=w,
            height=h,
            crs=crs,
            bounds=b_dict,
            transform=t_tuple,
            resolution_x=res_x,
            resolution_y=res_y,
            pixel_size=math.sqrt(res_x * res_y),
            units=units,
            native_gsd=native_gsd,
            nodata=nodata,
            dtype=dt,
            source=source,
            geometry_provenance=provenance,
        )

    @classmethod
    def from_geotiff(cls, filepath: str) -> "RasterGeometry":
        with rasterio.open(filepath) as ds:
            w, h = ds.width, ds.height
            crs_str = str(ds.crs) if ds.crs else None
            t = ds.transform
            t_tuple = (t.a, t.b, t.c, t.d, t.e, t.f)
            b = ds.bounds
            b_dict = {"min_x": float(b.left), "min_y": float(b.bottom), "max_x": float(b.right), "max_y": float(b.top)}
            res_x, res_y = abs(t.a), abs(t.e)
            nodata = ds.nodata
            dtype = str(ds.dtypes[0]) if ds.dtypes else "unknown"

            units = "metre"
            native_gsd = max(res_x, res_y)
            if crs_str:
                crs_upper = crs_str.upper()
                if "4326" in crs_upper or "CRS84" in crs_upper:
                    units = "degree"
                    center_lat = (b_dict["min_y"] + b_dict["max_y"]) / 2.0
                    m_lat = 111320.0
                    m_lon = 111320.0 * max(0.01, math.cos(math.radians(center_lat)))
                    native_gsd = max(res_x * m_lon, res_y * m_lat)

            return cls(
                width=w,
                height=h,
                crs=crs_str,
                bounds=b_dict,
                transform=t_tuple,
                resolution_x=res_x,
                resolution_y=res_y,
                pixel_size=math.sqrt(res_x * res_y),
                units=units,
                native_gsd=native_gsd,
                nodata=nodata,
                dtype=dtype,
                source=filepath,
                geometry_provenance="GEOTIFF_FILE",
            )

    @classmethod
    def from_spectral_band(cls, band: Any) -> "RasterGeometry":
        h, w = band.data.shape[:2]
        crs_str = getattr(band, "crs", None)
        t = getattr(band, "transform", None)
        t_tuple = None
        if t is not None:
            if isinstance(t, Affine):
                t_tuple = (t.a, t.b, t.c, t.d, t.e, t.f)
            elif isinstance(t, (list, tuple)) and len(t) >= 6:
                t_tuple = tuple(float(x) for x in t[:6])
            elif isinstance(t, dict):
                t_tuple = (
                    float(t.get("a", 1.0)), float(t.get("b", 0.0)), float(t.get("c", 0.0)),
                    float(t.get("d", 0.0)), float(t.get("e", -1.0)), float(t.get("f", 0.0)),
                )

        b_dict = None
        if t_tuple:
            a, b, c, d, e, f = t_tuple
            min_x = c
            max_y = f
            max_x = c + a * w
            min_y = f + e * h
            b_dict = {
                "min_x": min(min_x, max_x),
                "min_y": min(min_y, max_y),
                "max_x": max(min_x, max_x),
                "max_y": max(min_y, max_y),
            }

        res = getattr(band, "native_resolution_m", 10.0)
        return cls(
            width=w,
            height=h,
            crs=crs_str,
            bounds=b_dict,
            transform=t_tuple,
            resolution_x=float(res),
            resolution_y=float(res),
            pixel_size=float(res),
            units="metre",
            native_gsd=float(res),
            nodata=getattr(band, "nodata", 0.0),
            dtype=str(band.data.dtype),
            source=getattr(band, "name", "spectral_band"),
            geometry_provenance="SENTINEL2_SPECTRAL_BAND",
        )



@dataclass
class TargetGrid:
    """
    Canonical definition of the common geospatial destination grid.
    """
    crs: str
    bounds: Dict[str, float]               # min_x, min_y, max_x, max_y in target CRS
    transform: Tuple[float, ...]           # 6-element affine transform in target CRS
    width: int
    height: int
    resolution_x: float
    resolution_y: float
    target_grid_reason: str


@dataclass
class CoRegistrationResult:
    """
    Explicit, auditable contract returned by any co-registration or reprojection operation.
    """
    status: AlignmentStatus
    overlap_status: OverlapStatus
    overlap_ratio: Optional[float]
    source_crs: Optional[str]
    target_crs: Optional[str]
    source_resolution: Optional[Dict[str, float]]
    target_resolution: Optional[Dict[str, float]]
    source_bounds: Optional[Dict[str, float]]
    target_bounds: Optional[Dict[str, float]]
    source_transform: Optional[Tuple[float, ...]]
    target_transform: Optional[Tuple[float, ...]]
    resampling_applied: bool
    resampling_method: str
    aligned_raster: Optional[np.ndarray] = None
    aligned_image: Optional[Image.Image] = None
    valid_pixel_count: Optional[int] = None
    nodata_pixel_count: Optional[int] = None
    error: Optional[str] = None
    geometry_provenance: str = "GEOSPATIAL_COREGISTRATION"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "overlap_status": self.overlap_status.value,
            "overlap_ratio": round(self.overlap_ratio, 4) if self.overlap_ratio is not None else None,
            "source_crs": self.source_crs,
            "target_crs": self.target_crs,
            "source_resolution": self.source_resolution,
            "target_resolution": self.target_resolution,
            "source_bounds": self.source_bounds,
            "target_bounds": self.target_bounds,
            "source_transform": list(self.source_transform) if self.source_transform else None,
            "target_transform": list(self.target_transform) if self.target_transform else None,
            "resampling_applied": self.resampling_applied,
            "resampling_method": self.resampling_method,
            "valid_pixel_count": self.valid_pixel_count,
            "nodata_pixel_count": self.nodata_pixel_count,
            "error": self.error,
            "geometry_provenance": self.geometry_provenance,
        }


class GeospatialCoRegistrationEngine:
    """
    Deterministic Remote Sensing Co-Registration, CRS Reprojection & GSD Harmonization Engine.
    """

    @staticmethod
    def validate_crs(crs: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Validates whether CRS string is parseable by pyproj / rasterio.
        Returns (is_valid, normalized_crs_or_error_msg).
        """
        if not crs or not str(crs).strip():
            return False, "CRS is missing."
        try:
            c = pyproj.CRS.from_user_input(crs)
            auth = c.to_authority()
            if auth:
                return True, f"{auth[0]}:{auth[1]}"
            return True, c.to_string()
        except Exception as e:
            return False, f"Invalid CRS '{crs}': {str(e)}"

    @staticmethod
    def transform_bounds(
        bounds: Dict[str, float],
        from_crs: str,
        to_crs: str
    ) -> Dict[str, float]:
        """
        Transforms 4-corner bounding box between CRSs using pyproj.
        Transforms all 4 corner coordinates (min_x, min_y), (min_x, max_y),
        (max_x, min_y), (max_x, max_y) and returns enclosing envelope.
        """
        c_from = from_crs.strip().upper()
        c_to = to_crs.strip().upper()
        if c_from == c_to:
            return dict(bounds)

        transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
        w, s = bounds["min_x"], bounds["min_y"]
        e, n = bounds["max_x"], bounds["max_y"]

        corners = [
            transformer.transform(w, s),
            transformer.transform(w, n),
            transformer.transform(e, s),
            transformer.transform(e, n),
        ]
        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]

        return {
            "min_x": min(xs),
            "min_y": min(ys),
            "max_x": max(xs),
            "max_y": max(ys),
        }

    @classmethod
    def calculate_spatial_overlap(
        cls,
        geom_a: RasterGeometry,
        geom_b: RasterGeometry,
        common_crs: Optional[str] = None
    ) -> Tuple[OverlapStatus, Optional[float], Optional[Dict[str, float]]]:
        """
        Calculates genuine geographic intersection between two raster geometries in a common CRS.
        Returns (overlap_status, overlap_ratio, intersection_bounds_in_common_crs).
        """
        if not geom_a.is_valid_geospatial() or not geom_b.is_valid_geospatial():
            return OverlapStatus.OVERLAP_UNAVAILABLE, None, None

        # Choose common CRS: prefer projected CRS of geom_a or geom_b
        if not common_crs:
            if "326" in (geom_a.crs or "") or "327" in (geom_a.crs or "") or "UTM" in (geom_a.crs or ""):
                common_crs = geom_a.crs
            elif "326" in (geom_b.crs or "") or "327" in (geom_b.crs or "") or "UTM" in (geom_b.crs or ""):
                common_crs = geom_b.crs
            else:
                common_crs = geom_a.crs

        assert common_crs is not None

        try:
            b_a = cls.transform_bounds(geom_a.bounds, geom_a.crs, common_crs)  # type: ignore
            b_b = cls.transform_bounds(geom_b.bounds, geom_b.crs, common_crs)  # type: ignore
        except Exception:
            return OverlapStatus.OVERLAP_UNAVAILABLE, None, None

        inter_min_x = max(b_a["min_x"], b_b["min_x"])
        inter_min_y = max(b_a["min_y"], b_b["min_y"])
        inter_max_x = min(b_a["max_x"], b_b["max_x"])
        inter_max_y = min(b_a["max_y"], b_b["max_y"])

        if inter_max_x <= inter_min_x or inter_max_y <= inter_min_y:
            return OverlapStatus.NO_OVERLAP, 0.0, None

        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        area_a = (b_a["max_x"] - b_a["min_x"]) * (b_a["max_y"] - b_a["min_y"])
        area_b = (b_b["max_x"] - b_b["min_x"]) * (b_b["max_y"] - b_b["min_y"])
        min_area = min(area_a, area_b)

        ratio = inter_area / min_area if min_area > 1e-12 else 0.0
        ratio = min(1.0, max(0.0, ratio))

        inter_bounds = {
            "min_x": inter_min_x,
            "min_y": inter_min_y,
            "max_x": inter_max_x,
            "max_y": inter_max_y,
        }
        return OverlapStatus.OVERLAP_CONFIRMED, ratio, inter_bounds

    @classmethod
    def establish_target_grid(
        cls,
        geom_a: RasterGeometry,
        geom_b: RasterGeometry,
        intersection_bounds: Optional[Dict[str, float]] = None,
        target_crs: Optional[str] = None,
    ) -> Tuple[Optional[TargetGrid], Optional[str]]:
        """
        Determines the canonical target CRS, bounding box, resolution, and affine transform.
        Deterministic policy:
          - Target resolution = finest ground resolution min(geom_a.native_gsd, geom_b.native_gsd).
          - Target CRS = shared projected CRS or primary projected CRS.
        """
        valid_a, err_a = cls.validate_crs(geom_a.crs)
        valid_b, err_b = cls.validate_crs(geom_b.crs)
        if not valid_a:
            return None, f"Input A has invalid CRS: {err_a}"
        if not valid_b:
            return None, f"Input B has invalid CRS: {err_b}"

        # 1. Target CRS Selection
        if target_crs is None:
            if geom_a.crs.strip().upper() == geom_b.crs.strip().upper():
                target_crs = geom_a.crs
                reason = "shared_projected_crs" if "326" in target_crs or "327" in target_crs or "UTM" in target_crs else "shared_crs"
            elif any(k in geom_a.crs for k in ("326", "327", "UTM", "3857")):
                target_crs = geom_a.crs
                reason = "reprojection_to_primary_projected_crs"
            elif any(k in geom_b.crs for k in ("326", "327", "UTM", "3857")):
                target_crs = geom_b.crs
                reason = "reprojection_to_secondary_projected_crs"
            else:
                target_crs = geom_a.crs
                reason = "reprojection_to_reference_crs"
        else:
            reason = "explicit_target_crs"

        # 2. Compute overlap in target CRS
        status, ratio, inter_b = cls.calculate_spatial_overlap(geom_a, geom_b, common_crs=target_crs)
        if status != OverlapStatus.OVERLAP_CONFIRMED or not inter_b:
            return None, "Inputs have no spatial geographic overlap in common CRS."

        # 3. Determine target resolution in target CRS units
        # If target CRS is projected (meters), resolution is in meters
        is_projected = any(k in target_crs for k in ("326", "327", "UTM", "3857", "metre", "meter"))

        # GSD harmonization: pick finest resolution
        finest_gsd_m = min(geom_a.native_gsd, geom_b.native_gsd)
        if is_projected:
            res_x = res_y = max(1.0, finest_gsd_m)
        else:
            # Geographic (degrees): convert meters to degrees
            center_lat = (inter_b["min_y"] + inter_b["max_y"]) / 2.0
            deg_per_m = 1.0 / (111320.0 * max(0.01, math.cos(math.radians(center_lat))))
            res_x = res_y = finest_gsd_m * deg_per_m

        w_span = inter_b["max_x"] - inter_b["min_x"]
        h_span = inter_b["max_y"] - inter_b["min_y"]

        width = max(1, int(round(w_span / res_x)))
        height = max(1, int(round(h_span / res_y)))

        # Construct target affine transform (North-up):
        # [res_x, 0.0, min_x, 0.0, -res_y, max_y]
        transform = (res_x, 0.0, inter_b["min_x"], 0.0, -res_y, inter_b["max_y"])

        grid = TargetGrid(
            crs=target_crs,
            bounds=inter_b,
            transform=transform,
            width=width,
            height=height,
            resolution_x=res_x,
            resolution_y=res_y,
            target_grid_reason=reason,
        )
        return grid, None

    @classmethod
    def reproject_raster(
        cls,
        raster: np.ndarray,
        source_geom: RasterGeometry,
        target_grid: TargetGrid,
        resampling: Union[Resampling, str] = Resampling.bilinear,
        resampling_method: Optional[Union[Resampling, str]] = None,
        dst_nodata: Optional[float] = None,
    ) -> CoRegistrationResult:
        """
        Geospatially reprojects/resamples a numpy raster array onto the canonical TargetGrid
        using rasterio.warp.reproject.
        Preserves genuine nodata semantics without fabricating numbers.
        """
        if resampling_method is not None:
            resampling = resampling_method

        if isinstance(resampling, str):
            resamp_str = resampling.lower()
            if resamp_str in ("nearest", "categorical"):
                resampling = Resampling.nearest
            elif resamp_str in ("cubic", "bicubic"):
                resampling = Resampling.cubic
            elif resamp_str == "lanczos":
                resampling = Resampling.lanczos
            else:
                resampling = Resampling.bilinear
        if not source_geom.is_valid_geospatial():
            return CoRegistrationResult(
                status=AlignmentStatus.UNAVAILABLE,
                overlap_status=OverlapStatus.OVERLAP_UNAVAILABLE,
                overlap_ratio=None,
                source_crs=source_geom.crs,
                target_crs=target_grid.crs,
                source_resolution={"x": source_geom.resolution_x, "y": source_geom.resolution_y},
                target_resolution={"x": target_grid.resolution_x, "y": target_grid.resolution_y},
                source_bounds=source_geom.bounds,
                target_bounds=target_grid.bounds,
                source_transform=source_geom.transform,
                target_transform=target_grid.transform,
                resampling_applied=False,
                resampling_method="none",
                error="Source raster lacks valid geospatial CRS/transform/bounds.",
            )

        src_crs = rasterio.crs.CRS.from_user_input(source_geom.crs)
        dst_crs = rasterio.crs.CRS.from_user_input(target_grid.crs)

        src_transform = Affine(*source_geom.transform)  # type: ignore
        dst_transform = Affine(*target_grid.transform)

        src_arr = np.asarray(raster)
        src_dtype = src_arr.dtype

        # Multi-band handling
        is_multi_band = (src_arr.ndim == 3)
        if is_multi_band:
            # If shape is [H, W, C], move channels to [C, H, W] for rasterio
            if src_arr.shape[2] in (1, 2, 3, 4):
                bands_arr = np.moveaxis(src_arr, -1, 0)
            else:
                bands_arr = src_arr
            num_bands = bands_arr.shape[0]
            dst_arr = np.zeros((num_bands, target_grid.height, target_grid.width), dtype=np.float32)
        else:
            bands_arr = src_arr
            num_bands = 1
            dst_arr = np.zeros((target_grid.height, target_grid.width), dtype=np.float32)

        src_nd = source_geom.nodata if source_geom.nodata is not None else np.nan
        dst_nd = dst_nodata if dst_nodata is not None else np.nan

        try:
            if is_multi_band:
                for b_idx in range(num_bands):
                    reproject(
                        source=bands_arr[b_idx].astype(np.float32),
                        destination=dst_arr[b_idx],
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=resampling,
                        src_nodata=src_nd,
                        dst_nodata=dst_nd,
                    )
                # Reshape back to [H, W, C]
                final_raster = np.moveaxis(dst_arr, 0, -1)
            else:
                reproject(
                    source=bands_arr.astype(np.float32),
                    destination=dst_arr,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    src_nodata=src_nd,
                    dst_nodata=dst_nd,
                )
                final_raster = dst_arr

        except Exception as e:
            return CoRegistrationResult(
                status=AlignmentStatus.FAILED,
                overlap_status=OverlapStatus.OVERLAP_UNAVAILABLE,
                overlap_ratio=None,
                source_crs=source_geom.crs,
                target_crs=target_grid.crs,
                source_resolution={"x": source_geom.resolution_x, "y": source_geom.resolution_y},
                target_resolution={"x": target_grid.resolution_x, "y": target_grid.resolution_y},
                source_bounds=source_geom.bounds,
                target_bounds=target_grid.bounds,
                source_transform=source_geom.transform,
                target_transform=target_grid.transform,
                resampling_applied=False,
                resampling_method=resampling.name,
                error=f"Reprojection execution failed: {str(e)}",
            )

        # Cast back to uint8 if input was uint8 and not NaN
        if src_dtype == np.uint8 and not np.isnan(dst_nd):
            final_raster_out = np.clip(np.nan_to_num(final_raster, nan=0.0), 0, 255).astype(np.uint8)
        else:
            final_raster_out = final_raster

        # Generate aligned PIL image if 2D or 3-channel
        aligned_pil: Optional[Image.Image] = None
        try:
            if final_raster_out.ndim == 2:
                # Normalize for preview
                norm = np.clip(np.nan_to_num(final_raster_out, nan=0.0), 0, 255).astype(np.uint8)
                aligned_pil = Image.fromarray(norm)
            elif final_raster_out.ndim == 3 and final_raster_out.shape[2] in (3, 4):
                norm = np.clip(np.nan_to_num(final_raster_out, nan=0.0), 0, 255).astype(np.uint8)
                aligned_pil = Image.fromarray(norm)
        except Exception:
            aligned_pil = None

        # Statistics
        if np.isnan(dst_nd):
            valid_px = int(np.sum(~np.isnan(final_raster)))
            nodata_px = int(np.sum(np.isnan(final_raster)))
        else:
            valid_px = int(np.sum(final_raster != dst_nd))
            nodata_px = int(np.sum(final_raster == dst_nd))

        crs_matched = (str(source_geom.crs).strip().upper() == str(target_grid.crs).strip().upper())
        res_matched = (
            abs(source_geom.resolution_x - target_grid.resolution_x) < 1e-4
            and abs(source_geom.resolution_y - target_grid.resolution_y) < 1e-4
        )
        shape_matched = (source_geom.width == target_grid.width and source_geom.height == target_grid.height)

        if crs_matched and res_matched and shape_matched and source_geom.transform == target_grid.transform:
            stat = AlignmentStatus.ALIGNED
            resamp_applied = False
        elif not crs_matched:
            stat = AlignmentStatus.REPROJECTED
            resamp_applied = True
        else:
            stat = AlignmentStatus.RESAMPLED
            resamp_applied = True

        return CoRegistrationResult(
            status=stat,
            overlap_status=OverlapStatus.OVERLAP_CONFIRMED,
            overlap_ratio=1.0,
            source_crs=source_geom.crs,
            target_crs=target_grid.crs,
            source_resolution={"x": source_geom.resolution_x, "y": source_geom.resolution_y},
            target_resolution={"x": target_grid.resolution_x, "y": target_grid.resolution_y},
            source_bounds=source_geom.bounds,
            target_bounds=target_grid.bounds,
            source_transform=source_geom.transform,
            target_transform=target_grid.transform,
            resampling_applied=resamp_applied,
            resampling_method=resampling.name,
            aligned_raster=final_raster_out,
            aligned_image=aligned_pil,
            valid_pixel_count=valid_px,
            nodata_pixel_count=nodata_px,
            error=None,
        )

    @classmethod
    def co_register_pair(
        cls,
        image_a: Union[Image.Image, np.ndarray],
        image_b: Union[Image.Image, np.ndarray],
        geom_a: Optional[RasterGeometry] = None,
        geom_b: Optional[RasterGeometry] = None,
        resampling_a: Resampling = Resampling.bilinear,
        resampling_b: Resampling = Resampling.bilinear,
    ) -> Tuple[CoRegistrationResult, CoRegistrationResult]:
        """
        Performs end-to-end co-registration between two raster inputs:
          1. Extracts or validates RasterGeometry.
          2. Validates CRSs (rejects missing/invalid).
          3. Confirms true geographic overlap.
          4. Establishes common TargetGrid.
          5. Reprojects/resamples both rasters onto the exact target grid.
        """
        if geom_a is None:
            geom_a = RasterGeometry.from_image_or_metadata(image_a, source="input_a")
        if geom_b is None:
            geom_b = RasterGeometry.from_image_or_metadata(image_b, source="input_b")

        # 1. Missing CRS check
        if not geom_a.crs or not geom_b.crs:
            err_msg = "Geospatial co-registration requires genuine CRS metadata on both inputs."
            failed_res = CoRegistrationResult(
                status=AlignmentStatus.UNAVAILABLE,
                overlap_status=OverlapStatus.OVERLAP_UNAVAILABLE,
                overlap_ratio=None,
                source_crs=geom_a.crs,
                target_crs=geom_b.crs,
                source_resolution={"x": geom_a.resolution_x, "y": geom_a.resolution_y},
                target_resolution={"x": geom_b.resolution_x, "y": geom_b.resolution_y},
                source_bounds=geom_a.bounds,
                target_bounds=geom_b.bounds,
                source_transform=geom_a.transform,
                target_transform=geom_b.transform,
                resampling_applied=False,
                resampling_method="none",
                error=err_msg,
            )
            return failed_res, failed_res

        # 2. Overlap Check
        overlap_stat, ratio, inter_b = cls.calculate_spatial_overlap(geom_a, geom_b)
        if overlap_stat == OverlapStatus.NO_OVERLAP:
            err_msg = "Rasters have no geographic spatial overlap."
            failed_res = CoRegistrationResult(
                status=AlignmentStatus.FAILED,
                overlap_status=OverlapStatus.NO_OVERLAP,
                overlap_ratio=0.0,
                source_crs=geom_a.crs,
                target_crs=geom_b.crs,
                source_resolution={"x": geom_a.resolution_x, "y": geom_a.resolution_y},
                target_resolution={"x": geom_b.resolution_x, "y": geom_b.resolution_y},
                source_bounds=geom_a.bounds,
                target_bounds=geom_b.bounds,
                source_transform=geom_a.transform,
                target_transform=geom_b.transform,
                resampling_applied=False,
                resampling_method="none",
                error=err_msg,
            )
            return failed_res, failed_res

        # 3. Establish Target Grid
        target_grid, grid_err = cls.establish_target_grid(geom_a, geom_b, intersection_bounds=inter_b)
        if not target_grid:
            err_msg = grid_err or "Failed to establish common target grid."
            failed_res = CoRegistrationResult(
                status=AlignmentStatus.FAILED,
                overlap_status=overlap_stat,
                overlap_ratio=ratio,
                source_crs=geom_a.crs,
                target_crs=None,
                source_resolution={"x": geom_a.resolution_x, "y": geom_a.resolution_y},
                target_resolution=None,
                source_bounds=geom_a.bounds,
                target_bounds=None,
                source_transform=geom_a.transform,
                target_transform=None,
                resampling_applied=False,
                resampling_method="none",
                error=err_msg,
            )
            return failed_res, failed_res

        # 4. Reproject Both Rasters
        arr_a = np.array(image_a) if isinstance(image_a, Image.Image) else image_a
        arr_b = np.array(image_b) if isinstance(image_b, Image.Image) else image_b

        res_a = cls.reproject_raster(arr_a, geom_a, target_grid, resampling=resampling_a)
        res_b = cls.reproject_raster(arr_b, geom_b, target_grid, resampling=resampling_b)

        res_a.overlap_ratio = ratio
        res_b.overlap_ratio = ratio

        return res_a, res_b
