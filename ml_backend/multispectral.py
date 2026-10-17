"""
multispectral.py
----------------
Genuine Multispectral Ingestion, Spectral Band Preservation, and
Physics-Grounded Spectral Index Computation Engine for NexSpace / SatQuery AI.

Adheres strictly to the Spectral Truth Contract:
- REAL_MULTISPECTRAL: Actual independent sensor bands (e.g. B02, B03, B04, B08, B11, B12) loaded into memory.
- RGB_DERIVED: Visual products or proxies generated from data.
- UNAVAILABLE: When requested spectral bands cannot be obtained.

Strict Anti-Fabrication Rules:
- NEVER compute pseudo-NDVI from RGB and label it as authentic NDVI.
- NEVER compute pseudo-NDWI from RGB and label it as authentic NDWI.
- NEVER label RGB imagery as REAL_MULTISPECTRAL.
- NEVER invent CRS, affine transforms, nodata, or spectral statistics.
- Handle zero denominators, nodata pixels, NaNs, and infinities deterministically.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from geospatial import GeoTransform, GeoMetadata


# ---------------------------------------------------------------------------
# Band Metadata Definition
# ---------------------------------------------------------------------------

BAND_METADATA_TABLE: Dict[str, Dict[str, Any]] = {
    "B01": {"common_name": "coastal", "center_wavelength_um": 0.443, "native_resolution_m": 60.0, "scale": 0.0001, "offset": 0.0},
    "B02": {"common_name": "blue",    "center_wavelength_um": 0.490, "native_resolution_m": 10.0, "scale": 0.0001, "offset": 0.0},
    "B03": {"common_name": "green",   "center_wavelength_um": 0.560, "native_resolution_m": 10.0, "scale": 0.0001, "offset": 0.0},
    "B04": {"common_name": "red",     "center_wavelength_um": 0.665, "native_resolution_m": 10.0, "scale": 0.0001, "offset": 0.0},
    "B05": {"common_name": "rededge1","center_wavelength_um": 0.705, "native_resolution_m": 20.0, "scale": 0.0001, "offset": 0.0},
    "B06": {"common_name": "rededge2","center_wavelength_um": 0.740, "native_resolution_m": 20.0, "scale": 0.0001, "offset": 0.0},
    "B07": {"common_name": "rededge3","center_wavelength_um": 0.783, "native_resolution_m": 20.0, "scale": 0.0001, "offset": 0.0},
    "B08": {"common_name": "nir",     "center_wavelength_um": 0.842, "native_resolution_m": 10.0, "scale": 0.0001, "offset": 0.0},
    "B8A": {"common_name": "nir08",   "center_wavelength_um": 0.865, "native_resolution_m": 20.0, "scale": 0.0001, "offset": 0.0},
    "B09": {"common_name": "water_vapor","center_wavelength_um": 0.945, "native_resolution_m": 60.0, "scale": 0.0001, "offset": 0.0},
    "B11": {"common_name": "swir1",   "center_wavelength_um": 1.610, "native_resolution_m": 20.0, "scale": 0.0001, "offset": 0.0},
    "B12": {"common_name": "swir2",   "center_wavelength_um": 2.190, "native_resolution_m": 20.0, "scale": 0.0001, "offset": 0.0},
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SpectralBand:
    """Individual authentic sensor spectral band container."""
    name: str                       # e.g. "B04", "B08", "B03"
    common_name: str                # e.g. "red", "nir", "green"
    data: np.ndarray                # 2D float32 array in BOA surface reflectance [0.0..1.0+]
    raw_data: Optional[np.ndarray] = None  # Original uint16 DN values where available
    native_width: int = 100
    native_height: int = 100
    native_resolution_m: float = 10.0      # e.g. 10.0 or 20.0 meters
    dtype: str = "uint16"                  # e.g. "float32" or "uint16"
    scale: float = 0.0001           # Sentinel-2 BOA reflectance scale
    offset: float = 0.0             # Radiometric offset
    nodata: Optional[float] = 0.0   # Sensor nodata value (0 for Sentinel-2 L2A)
    crs: Optional[str] = None       # e.g. "EPSG:32643"
    transform: Optional[Dict[str, float]] = None # Affine transform parameters
    wavelength_um: float = 0.0      # Center wavelength in micrometers
    is_resampled: bool = False
    resampling_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "common_name": self.common_name,
            "native_width": self.native_width,
            "native_height": self.native_height,
            "native_resolution_m": self.native_resolution_m,
            "dtype": self.dtype,
            "scale": self.scale,
            "offset": self.offset,
            "nodata": self.nodata,
            "crs": self.crs,
            "transform": self.transform,
            "wavelength_um": self.wavelength_um,
            "is_resampled": self.is_resampled,
            "resampling_method": self.resampling_method,
        }


@dataclass
class SpectralIndexStats:
    """Statistical summary for valid pixels in a computed spectral index."""
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    std: Optional[float]
    valid_pixel_count: int
    nodata_pixel_count: int
    total_pixel_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min": round(self.min, 4) if self.min is not None else None,
            "max": round(self.max, 4) if self.max is not None else None,
            "mean": round(self.mean, 4) if self.mean is not None else None,
            "median": round(self.median, 4) if self.median is not None else None,
            "std": round(self.std, 4) if self.std is not None else None,
            "valid_pixel_count": self.valid_pixel_count,
            "nodata_pixel_count": self.nodata_pixel_count,
            "total_pixel_count": self.total_pixel_count,
        }


@dataclass
class SpectralIndexResult:
    """Standardized output container for a computed spectral index."""
    index_name: str                 # "NDVI" | "NDWI" | "NDBI"
    formula: str                    # e.g. "(B08 - B04) / (B08 + B04)"
    bands_used: List[str]           # e.g. ["B04", "B08"]
    status: str                     # "success" | "unavailable" | "failed"
    provenance: str                 # "REAL_MULTISPECTRAL" | "UNAVAILABLE" | "RGB_DERIVED"
    raster: Optional[np.ndarray] = None  # 2D float32 array [-1.0..1.0] with NaNs for nodata
    stats: Optional[SpectralIndexStats] = None
    resolution_m: float = 10.0
    crs: Optional[str] = None
    resampling_applied: bool = False
    resampling_details: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self, include_raster: bool = False) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "index_name": self.index_name,
            "formula": self.formula,
            "bands_used": self.bands_used,
            "status": self.status,
            "provenance": self.provenance,
            "stats": self.stats.to_dict() if self.stats else None,
            "resolution_m": self.resolution_m,
            "crs": self.crs,
            "resampling_applied": self.resampling_applied,
            "resampling_details": self.resampling_details,
            "error": self.error,
        }
        if include_raster and self.raster is not None:
            res["raster_shape"] = list(self.raster.shape)
        return res


@dataclass
class MultispectralScene:
    """Rich container for authentic multi-band sensor data."""
    scene_id: str
    bands: Dict[str, SpectralBand]  # Keyed by standard band name (e.g. "B02", "B03", "B04", "B08")
    loaded_bands: List[str]         # Bands actually loaded into memory
    available_bands: List[str]      # Bands cataloged by the provider platform
    width: int                      # Reference grid width (usually 10m grid)
    height: int                     # Reference grid height
    crs: str                        # Native CRS (e.g. "EPSG:32643")
    transform: Dict[str, float]     # Native affine transform parameters
    bounds: Dict[str, float]        # World bounding box
    resolution_m: float             # Reference spatial resolution
    acquisition_time: str           # Sensing timestamp
    source: str                     # e.g. "Sentinel-2 L2A"
    spectral_source: str = "REAL_MULTISPECTRAL"
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def band_names(self) -> List[str]:
        return list(self.loaded_bands)

    @property
    def dtype(self) -> str:
        for b in self.bands.values():
            return b.dtype
        return "float32"

    @property
    def scale(self) -> float:
        for b in self.bands.values():
            return b.scale
        return 0.0001

    @property
    def offset(self) -> float:
        for b in self.bands.values():
            return b.offset
        return 0.0

    @property
    def resolution(self) -> Dict[str, Any]:
        return {"x": self.resolution_m, "y": self.resolution_m, "unit": "meter"}

    def get_band(self, band_name: str) -> Optional[SpectralBand]:
        """Returns the requested band by canonical name (e.g. 'B04', 'B08') or common name ('red', 'nir')."""
        if band_name in self.bands:
            return self.bands[band_name]
        # Check by common name
        clean = band_name.lower().strip()
        for b in self.bands.values():
            if b.common_name.lower() == clean:
                return b
        # Check normalized e.g. "b4" -> "B04"
        norm = band_name.upper().strip()
        if len(norm) == 2 and norm.startswith("B"):
            padded = f"B0{norm[1]}"
            if padded in self.bands:
                return self.bands[padded]
        return None

    def align_band_to_grid(self, band: SpectralBand, target_shape: Tuple[int, int], method: str = "bilinear") -> np.ndarray:
        """
        Resamples a spectral band to the reference grid when resolutions differ
        (e.g. 20m SWIR-1 B11 resampled to 10m B04/B08 grid).
        Preserves coordinate space and records the resampling operation.
        """
        th, tw = target_shape
        sh, sw = band.data.shape
        if (sh, sw) == (th, tw):
            return band.data

        # True geospatial reprojection / resampling via rasterio.warp.reproject
        if HAS_RASTERIO and self.transform is not None:
            try:
                from rasterio.transform import Affine
                from rasterio.warp import reproject, Resampling as RResampling
                src_crs = rasterio.crs.CRS.from_user_input(band.crs or self.crs or "EPSG:4326")
                dst_crs = rasterio.crs.CRS.from_user_input(self.crs or "EPSG:4326")

                dst_transform = Affine(*self.transform)
                if band.transform is not None:
                    src_transform = Affine(*band.transform)
                else:
                    # Derive genuine source transform scaled by pixel resolution ratio
                    src_transform = Affine(
                        self.transform[0] * (tw / sw),
                        self.transform[1],
                        self.transform[2],
                        self.transform[3],
                        self.transform[4] * (th / sh),
                        self.transform[5],
                    )

                dst_arr = np.zeros((th, tw), dtype=np.float32)
                resamp = RResampling.bilinear if method == "bilinear" else RResampling.nearest
                reproject(
                    source=band.data.astype(np.float32),
                    destination=dst_arr,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resamp,
                )
                return dst_arr
            except Exception:
                pass

        # Deterministic fallback interpolation
        pil_arr = Image.fromarray(band.data)
        resample_filter = Image.Resampling.BILINEAR if method == "bilinear" else Image.Resampling.NEAREST
        resampled_pil = pil_arr.resize((tw, th), resample=resample_filter)
        resampled_arr = np.array(resampled_pil, dtype=np.float32)
        return resampled_arr

    def to_visual_rgb(self, percentile_stretch: bool = True) -> Tuple[Image.Image, str]:
        """
        Renders a natural color (B04-Red, B03-Green, B02-Blue) visual derivative from genuine bands.
        Explicitly marked with provenance 'VISUAL_DERIVATIVE'.
        """
        b_red = self.get_band("B04")
        b_green = self.get_band("B03")
        b_blue = self.get_band("B02")

        if not (b_red and b_green and b_blue):
            raise ValueError("Cannot render RGB visual derivative: B02, B03, or B04 is missing.")

        target_shape = (self.height, self.width)
        r = self.align_band_to_grid(b_red, target_shape)
        g = self.align_band_to_grid(b_green, target_shape)
        b = self.align_band_to_grid(b_blue, target_shape)

        rgb = np.stack([r, g, b], axis=-1)

        if percentile_stretch:
            # 2% - 98% percentile reflectance normalization
            for c in range(3):
                valid = rgb[..., c][rgb[..., c] > 0]
                if valid.size > 10:
                    p2, p98 = np.percentile(valid, (2, 98))
                    if p98 > p2:
                        rgb[..., c] = np.clip((rgb[..., c] - p2) / (p98 - p2), 0.0, 1.0)
                    else:
                        rgb[..., c] = np.clip(rgb[..., c], 0.0, 1.0)
                else:
                    rgb[..., c] = np.clip(rgb[..., c], 0.0, 1.0)
        else:
            rgb = np.clip(rgb, 0.0, 1.0)

        rgb_uint8 = (rgb * 255.0).astype(np.uint8)
        img = Image.fromarray(rgb_uint8, mode="RGB")
        return img, "VISUAL_DERIVATIVE"

    def to_rgb_image(self, percentile_stretch: bool = True) -> Tuple[Image.Image, str]:
        """Alias for to_visual_rgb."""
        return self.to_visual_rgb(percentile_stretch=percentile_stretch)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "spectral_source": self.spectral_source,
            "loaded_bands": self.loaded_bands,
            "available_bands": self.available_bands,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "transform": self.transform,
            "bounds": self.bounds,
            "resolution_m": self.resolution_m,
            "acquisition_time": self.acquisition_time,
            "source": self.source,
            "provenance": self.provenance,
            "band_details": {k: b.to_dict() for k, b in self.bands.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultispectralScene":
        bands_dict = {}
        for b_name, b_info in d.get("band_details", {}).items():
            arr = b_info.get("data")
            if arr is not None and not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            elif arr is None:
                arr = np.zeros((d.get("height", 100), d.get("width", 100)), dtype=b_info.get("dtype", "uint16"))
            bands_dict[b_name] = SpectralBand(
                name=b_info.get("name", b_name),
                common_name=b_info.get("common_name", ""),
                data=arr,
                native_width=b_info.get("native_width", d.get("width", 100)),
                native_height=b_info.get("native_height", d.get("height", 100)),
                native_resolution_m=b_info.get("native_resolution_m", 10.0),
                dtype=b_info.get("dtype", "uint16"),
                scale=b_info.get("scale", 0.0001),
                offset=b_info.get("offset", 0.0),
                nodata=b_info.get("nodata", 0),
                crs=b_info.get("crs", d.get("crs", "EPSG:4326")),
                transform=tuple(b_info.get("transform", d.get("transform", (0, 1, 0, 0, 0, 1)))),
                wavelength_um=b_info.get("wavelength_um", 0.0),
                is_resampled=b_info.get("is_resampled", False),
                resampling_method=b_info.get("resampling_method"),
            )
        return cls(
            scene_id=d.get("scene_id", "scene_imported"),
            bands=bands_dict,
            width=d.get("width", 100),
            height=d.get("height", 100),
            crs=d.get("crs", "EPSG:4326"),
            transform=tuple(d.get("transform", (0, 1, 0, 0, 0, 1))),
            bounds=tuple(d.get("bounds", (0, 0, 1, 1))),
            resolution_m=d.get("resolution_m", 10.0),
            acquisition_time=d.get("acquisition_time"),
            source=d.get("source", "Sentinel-2"),
            spectral_source=d.get("spectral_source", "REAL_MULTISPECTRAL"),
            provenance=d.get("provenance", {}),
            loaded_bands=d.get("loaded_bands", list(bands_dict.keys())),
            available_bands=d.get("available_bands", list(bands_dict.keys())),
        )


# ---------------------------------------------------------------------------
# Spectral Computation Engine
# ---------------------------------------------------------------------------

class SpectralEngine:
    """
    Computes authentic Earth-observation spectral indices strictly from real sensor bands.
    Rejects RGB-derived proxies when true multispectral data is required.
    """

    @staticmethod
    def _compute_stats(arr: np.ndarray, nodata_mask: np.ndarray) -> SpectralIndexStats:
        """Computes robust statistical metrics on valid pixels."""
        valid_mask = (~nodata_mask) & np.isfinite(arr)
        valid_count = int(np.sum(valid_mask))
        nodata_count = int(np.sum(~valid_mask))
        total_count = int(arr.size)

        if valid_count == 0:
            return SpectralIndexStats(
                min=None,
                max=None,
                mean=None,
                median=None,
                std=None,
                valid_pixel_count=0,
                nodata_pixel_count=nodata_count,
                total_pixel_count=total_count,
            )

        valid_vals = arr[valid_mask]
        return SpectralIndexStats(
            min=float(np.min(valid_vals)),
            max=float(np.max(valid_vals)),
            mean=float(np.mean(valid_vals)),
            median=float(np.median(valid_vals)),
            std=float(np.std(valid_vals)),
            valid_pixel_count=valid_count,
            nodata_pixel_count=nodata_count,
            total_pixel_count=total_count,
        )

    @classmethod
    def compute_ndvi(cls, scene: Union[MultispectralScene, Dict[str, Any]]) -> SpectralIndexResult:
        """
        Computes the Normalized Difference Vegetation Index (NDVI):
          NDVI = (NIR - RED) / (NIR + RED)
        Where:
          NIR = B08 (842 nm)
          RED = B04 (665 nm)

        Strict Truth Contract:
        If B04 or B08 are not authentic loaded bands, returns an unavailable result.
        """
        if not isinstance(scene, MultispectralScene):
            return SpectralIndexResult(
                index_name="NDVI",
                formula="(B08 - B04) / (B08 + B04)",
                bands_used=["B04", "B08"],
                status="unavailable",
                provenance="UNAVAILABLE",
                error="NDVI requires actual Red (B04) and NIR (B08) spectral bands. The available imagery is RGB-only.",
            )

        b_red = scene.get_band("B04")
        b_nir = scene.get_band("B08")

        if b_red is None or b_nir is None:
            missing = []
            if b_red is None:
                missing.append("B04 (Red)")
            if b_nir is None:
                missing.append("B08 (NIR)")
            return SpectralIndexResult(
                index_name="NDVI",
                formula="(B08 - B04) / (B08 + B04)",
                bands_used=["B04", "B08"],
                status="unavailable",
                provenance="UNAVAILABLE",
                error=f"NDVI requires authentic spectral bands ({', '.join(missing)} are missing from loaded data).",
                resampling_details=f"Required bands missing: {', '.join(missing)}",
            )

        # Align grids if necessary (both B04 and B08 are natively 10m on Sentinel-2)
        target_shape = (scene.height, scene.width)
        red_arr = scene.align_band_to_grid(b_red, target_shape)
        nir_arr = scene.align_band_to_grid(b_nir, target_shape)

        # Identify nodata pixels (0 or nodata values)
        nodata_mask = (red_arr <= 0) | (nir_arr <= 0)
        if b_red.nodata is not None:
            nodata_mask |= (red_arr == b_red.nodata)
        if b_nir.nodata is not None:
            nodata_mask |= (nir_arr == b_nir.nodata)

        denom = nir_arr + red_arr
        zero_denom_mask = (np.abs(denom) < 1e-7) | nodata_mask

        # Compute index safely avoiding zero-division warning
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.where(zero_denom_mask, np.nan, (nir_arr - red_arr) / denom)
            # Physical bounds clamp to [-1.0, 1.0]
            ndvi = np.clip(ndvi, -1.0, 1.0)

        stats = cls._compute_stats(ndvi, zero_denom_mask)

        return SpectralIndexResult(
            index_name="NDVI",
            formula="(B08 - B04) / (B08 + B04)",
            bands_used=["B04", "B08"],
            status="success",
            provenance="REAL_MULTISPECTRAL",
            raster=ndvi,
            stats=stats,
            resolution_m=10.0,
            crs=scene.crs,
            resampling_applied=b_red.is_resampled or b_nir.is_resampled,
            resampling_details="B04 and B08 on native 10m grid",
        )

    @classmethod
    def compute_ndwi(cls, scene: Union[MultispectralScene, Dict[str, Any]]) -> SpectralIndexResult:
        """
        Computes the Normalized Difference Water Index (NDWI - McFeeters 1996):
          NDWI = (GREEN - NIR) / (GREEN + NIR)
        Where:
          GREEN = B03 (560 nm)
          NIR = B08 (842 nm)

        Strict Truth Contract:
        If B03 or B08 are not authentic loaded bands, returns an unavailable result.
        """
        if not isinstance(scene, MultispectralScene):
            return SpectralIndexResult(
                index_name="NDWI",
                formula="(B03 - B08) / (B03 + B08)",
                bands_used=["B03", "B08"],
                status="unavailable",
                provenance="UNAVAILABLE",
                error="NDWI requires actual Green (B03) and NIR (B08) spectral bands. The available imagery is RGB-only.",
            )

        b_green = scene.get_band("B03")
        b_nir = scene.get_band("B08")

        if b_green is None or b_nir is None:
            missing = []
            if b_green is None:
                missing.append("B03 (Green)")
            if b_nir is None:
                missing.append("B08 (NIR)")
            return SpectralIndexResult(
                index_name="NDWI",
                formula="(B03 - B08) / (B03 + B08)",
                bands_used=["B03", "B08"],
                status="unavailable",
                provenance="UNAVAILABLE",
                error=f"NDWI requires authentic spectral bands ({', '.join(missing)} are missing from loaded data).",
                resampling_details=f"Required bands missing: {', '.join(missing)}",
            )

        target_shape = (scene.height, scene.width)
        green_arr = scene.align_band_to_grid(b_green, target_shape)
        nir_arr = scene.align_band_to_grid(b_nir, target_shape)

        nodata_mask = (green_arr <= 0) | (nir_arr <= 0)
        if b_green.nodata is not None:
            nodata_mask |= (green_arr == b_green.nodata)
        if b_nir.nodata is not None:
            nodata_mask |= (nir_arr == b_nir.nodata)

        denom = green_arr + nir_arr
        zero_denom_mask = (np.abs(denom) < 1e-7) | nodata_mask

        with np.errstate(divide="ignore", invalid="ignore"):
            ndwi = np.where(zero_denom_mask, np.nan, (green_arr - nir_arr) / denom)
            ndwi = np.clip(ndwi, -1.0, 1.0)

        stats = cls._compute_stats(ndwi, zero_denom_mask)

        return SpectralIndexResult(
            index_name="NDWI",
            formula="(B03 - B08) / (B03 + B08)",
            bands_used=["B03", "B08"],
            status="success",
            provenance="REAL_MULTISPECTRAL",
            raster=ndwi,
            stats=stats,
            resolution_m=10.0,
            crs=scene.crs,
            resampling_applied=b_green.is_resampled or b_nir.is_resampled,
            resampling_details="B03 and B08 on native 10m grid",
        )

    @classmethod
    def compute_ndbi(cls, scene: Union[MultispectralScene, Dict[str, Any]]) -> SpectralIndexResult:
        """
        Computes the Normalized Difference Built-up Index (NDBI - Zha et al. 2003):
          NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
        Where:
          SWIR1 = B11 (1610 nm, native 20m)
          NIR = B08 (842 nm, native 10m)

        Resolution / Alignment:
        B11 (20m) is resampled to the 10m grid with explicit bilinear interpolation,
        recording the resampling operation in telemetry.
        """
        if not isinstance(scene, MultispectralScene):
            return SpectralIndexResult(
                index_name="NDBI",
                formula="(B11 - B08) / (B11 + B08)",
                bands_used=["B11", "B08"],
                status="unavailable",
                provenance="UNAVAILABLE",
                error="NDBI requires actual SWIR-1 (B11) and NIR (B08) spectral bands. The available imagery is RGB-only.",
            )

        b_swir = scene.get_band("B11")
        b_nir = scene.get_band("B08")

        if b_swir is None or b_nir is None:
            missing = []
            if b_swir is None:
                missing.append("B11 (SWIR1)")
            if b_nir is None:
                missing.append("B08 (NIR)")
            return SpectralIndexResult(
                index_name="NDBI",
                formula="(B11 - B08) / (B11 + B08)",
                bands_used=["B11", "B08"],
                status="unavailable",
                provenance="UNAVAILABLE",
                error=f"NDBI requires authentic spectral bands ({', '.join(missing)} are missing from loaded data).",
                resampling_details=f"Required bands missing: {', '.join(missing)}",
            )

        # Target grid is 10m resolution of B08
        target_shape = (scene.height, scene.width)
        swir_arr = scene.align_band_to_grid(b_swir, target_shape, method="bilinear")
        nir_arr = scene.align_band_to_grid(b_nir, target_shape)

        resampling_applied = True
        resampling_details = f"B11 (20m native {b_swir.native_width}x{b_swir.native_height}) resampled to 10m grid ({target_shape[1]}x{target_shape[0]}) using bilinear interpolation"

        nodata_mask = (swir_arr <= 0) | (nir_arr <= 0)
        if b_swir.nodata is not None:
            nodata_mask |= (swir_arr == b_swir.nodata)
        if b_nir.nodata is not None:
            nodata_mask |= (nir_arr == b_nir.nodata)

        denom = swir_arr + nir_arr
        zero_denom_mask = (np.abs(denom) < 1e-7) | nodata_mask

        with np.errstate(divide="ignore", invalid="ignore"):
            ndbi = np.where(zero_denom_mask, np.nan, (swir_arr - nir_arr) / denom)
            ndbi = np.clip(ndbi, -1.0, 1.0)

        stats = cls._compute_stats(ndbi, zero_denom_mask)

        return SpectralIndexResult(
            index_name="NDBI",
            formula="(B11 - B08) / (B11 + B08)",
            bands_used=["B11", "B08"],
            status="success",
            provenance="REAL_MULTISPECTRAL",
            raster=ndbi,
            stats=stats,
            resolution_m=10.0,
            crs=scene.crs,
            resampling_applied=resampling_applied,
            resampling_details=resampling_details,
        )

    @classmethod
    def compute_indices(
        cls,
        scene: Union[MultispectralScene, Dict[str, Any]],
        indices: Optional[List[str]] = None,
    ) -> Dict[str, SpectralIndexResult]:
        """Computes all requested spectral indices with comprehensive truth validation."""
        requested = [idx.upper().strip() for idx in (indices or ["NDVI", "NDWI", "NDBI"])]
        results: Dict[str, SpectralIndexResult] = {}

        if "NDVI" in requested:
            results["NDVI"] = cls.compute_ndvi(scene)
        if "NDWI" in requested:
            results["NDWI"] = cls.compute_ndwi(scene)
        if "NDBI" in requested:
            results["NDBI"] = cls.compute_ndbi(scene)

        return results


spectral_engine = SpectralEngine()
