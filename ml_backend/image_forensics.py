"""
image_forensics.py
------------------
Deterministic Image Forensics, Metadata Extraction, EXIF GPS Decoding & Quality Gating for NexSpace Phase 2.

Supports:
- JPEG, PNG, TIFF, GeoTIFF, WebP file forensics and magic-byte inspection
- GeoTIFF raster georeferencing (CRS, EPSG, Affine Transform, WGS84 bounding envelope) via geospatial.py
- Complete EXIF GPS extraction (GPSLatitude, GPSLongitude, GPSAltitude, GPSTimeStamp, GPSDOP, Ref hemispheres)
- Quality gating: Laplacian variance blur estimation, brightness extremes, featureless scene detection
- Anti-fabrication guarantees: preserves deterministic metadata provenance; returns UNKNOWN when evidence is absent
"""

from __future__ import annotations
import io
import os
import re
import sys
import math
import numpy as np
from PIL import Image, ExifTags
from typing import Dict, Any, Optional, Tuple, List, Union

# Ensure ml_backend is in path
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from geospatial import GeospatialEngine, GeoMetadata, GeoTransform


class ImageForensicsEngine:
    """Forensic inspector for uploaded user imagery."""

    MAGIC_SIGNATURES = [
        (b"\xFF\xD8\xFF", "image/jpeg", "JPEG"),
        (b"\x89PNG\r\n\x1a\n", "image/png", "PNG"),
        (b"II*\x00", "image/tiff", "TIFF (Little-Endian)"),
        (b"MM\x00*", "image/tiff", "TIFF (Big-Endian)"),
        (b"RIFF", "image/webp", "WebP"),
    ]

    @classmethod
    def detect_format_and_mime(cls, raw_bytes: bytes) -> Dict[str, Any]:
        """Inspects raw leading magic bytes to determine real file type."""
        if not raw_bytes or len(raw_bytes) < 8:
            return {"valid": False, "mime": "application/octet-stream", "format": "UNKNOWN", "error": "file_too_small"}

        for sig, mime, fmt in cls.MAGIC_SIGNATURES:
            if sig == b"RIFF":
                if raw_bytes[:4] == b"RIFF" and len(raw_bytes) >= 12 and raw_bytes[8:12] == b"WEBP":
                    return {"valid": True, "mime": mime, "format": fmt}
            elif raw_bytes.startswith(sig):
                return {"valid": True, "mime": mime, "format": fmt}

        return {"valid": False, "mime": "application/octet-stream", "format": "UNKNOWN", "error": "unrecognized_header"}

    @classmethod
    def _dms_to_decimal(cls, dms_value: Any, ref: Optional[str] = None) -> Optional[float]:
        """Converts Degrees/Minutes/Seconds rational tuples or floats to signed decimal degrees."""
        if dms_value is None:
            return None

        try:
            # Handle float or int direct values
            if isinstance(dms_value, (int, float)):
                dec = float(dms_value)
            elif isinstance(dms_value, (list, tuple)) and len(dms_value) >= 3:
                parts = []
                for p in dms_value[:3]:
                    if hasattr(p, "numerator") and hasattr(p, "denominator"):
                        val = float(p.numerator) / float(p.denominator) if p.denominator != 0 else 0.0
                    elif isinstance(p, (tuple, list)) and len(p) == 2:
                        val = float(p[0]) / float(p[1]) if p[1] != 0 else 0.0
                    else:
                        val = float(p)
                    parts.append(val)

                dec = parts[0] + (parts[1] / 60.0) + (parts[2] / 3600.0)
            else:
                return None

            if ref and isinstance(ref, str):
                ref_clean = ref.strip().upper()
                if ref_clean in ("S", "W"):
                    dec = -abs(dec)
                elif ref_clean in ("N", "E"):
                    dec = abs(dec)

            return round(dec, 8)
        except Exception:
            return None

    @classmethod
    def extract_exif_gps(cls, pil_img: Image.Image) -> Dict[str, Any]:
        """
        Extracts verified GPS tags from PIL Exif data.
        Never fabricates coordinates if tags are missing or malformed.
        """
        result = {
            "has_gps": False,
            "latitude": None,
            "longitude": None,
            "altitude_m": None,
            "timestamp_utc": None,
            "dop": None,
            "provenance": {},
            "warnings": [],
        }

        try:
            exif = pil_img.getexif()
            if not exif:
                return result

            # Find GPS Info IFD (Tag ID 34853)
            gps_info = None
            if hasattr(ExifTags, "IFD") and hasattr(ExifTags.IFD, "GPSInfo"):
                try:
                    gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo)
                except Exception:
                    pass

            if not gps_info:
                # Fallback to direct lookup
                gps_info = exif.get(34853)

            if not gps_info:
                return result

            # Parse tags via GPSTAGS dictionary
            parsed_tags: Dict[str, Any] = {}
            for k, v in gps_info.items():
                name = ExifTags.GPSTAGS.get(k, f"GPS_{k}")
                parsed_tags[name] = v

            lat_val = parsed_tags.get("GPSLatitude")
            lat_ref = parsed_tags.get("GPSLatitudeRef")
            lon_val = parsed_tags.get("GPSLongitude")
            lon_ref = parsed_tags.get("GPSLongitudeRef")

            lat = cls._dms_to_decimal(lat_val, lat_ref)
            lon = cls._dms_to_decimal(lon_val, lon_ref)

            if lat is not None and lon is not None:
                # Coordinate sanity validation
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    result["warnings"].append(f"Out of bounds GPS coordinates: {lat}, {lon}")
                    return result

                # Check for "Null Island" (0.0, 0.0) placeholder anomaly
                if abs(lat) < 1e-5 and abs(lon) < 1e-5:
                    result["warnings"].append("GPS coordinates are exactly (0,0) (Null Island placeholder)")

                result["has_gps"] = True
                result["latitude"] = lat
                result["longitude"] = lon

                # Altitude
                alt_val = parsed_tags.get("GPSAltitude")
                if alt_val is not None:
                    try:
                        if hasattr(alt_val, "numerator"):
                            alt = float(alt_val.numerator) / float(alt_val.denominator) if alt_val.denominator != 0 else 0.0
                        elif isinstance(alt_val, (tuple, list)) and len(alt_val) == 2:
                            alt = float(alt_val[0]) / float(alt_val[1]) if alt_val[1] != 0 else 0.0
                        else:
                            alt = float(alt_val)

                        alt_ref = parsed_tags.get("GPSAltitudeRef", 0)
                        if alt_ref == 1 or alt_ref == b"\x01":
                            alt = -alt
                        result["altitude_m"] = round(alt, 2)
                    except Exception:
                        pass

                # Timestamp
                date_str = parsed_tags.get("GPSDateStamp")
                time_val = parsed_tags.get("GPSTimeStamp")
                if date_str and time_val:
                    try:
                        h = int(time_val[0]) if isinstance(time_val[0], (int, float)) else int(time_val[0][0])
                        m = int(time_val[1]) if isinstance(time_val[1], (int, float)) else int(time_val[1][0])
                        s = int(time_val[2]) if isinstance(time_val[2], (int, float)) else int(time_val[2][0])
                        result["timestamp_utc"] = f"{date_str}T{h:02d}:{m:02d}:{s:02d}Z"
                    except Exception:
                        pass

                # Dilution of Precision
                dop_val = parsed_tags.get("GPSDOP")
                if dop_val is not None:
                    try:
                        result["dop"] = float(dop_val)
                    except Exception:
                        pass

                result["provenance"] = {
                    "source": "EXIF_GPS_IFD",
                    "latitude_ref": str(lat_ref),
                    "longitude_ref": str(lon_ref),
                    "tags_present": list(parsed_tags.keys()),
                }

        except Exception as e:
            result["warnings"].append(f"EXIF parsing error: {e}")

        return result

    @classmethod
    def evaluate_image_quality(cls, pil_img: Image.Image) -> Dict[str, Any]:
        """
        Assesses image quality, resolution, blur via Laplacian variance, and brightness extremes.
        """
        w, h = pil_img.size
        rgb = pil_img.convert("RGB")
        arr = np.array(rgb, dtype=np.float32)

        mean_brightness = float(np.mean(arr))
        std_brightness = float(np.std(arr))

        # Blur estimation via discrete Laplacian kernel approximation
        gray = np.array(pil_img.convert("L"), dtype=np.float32)
        if w >= 8 and h >= 8:
            # Simple 3x3 Laplacian filter
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            # Fast vectorized 2D convolution for variance calculation
            laplacian = (
                gray[1:-1, 2:] + gray[1:-1, :-2] + gray[2:, 1:-1] + gray[:-2, 1:-1] - 4.0 * gray[1:-1, 1:-1]
            )
            blur_variance = float(np.var(laplacian))
        else:
            blur_variance = 0.0

        is_blurry = blur_variance < 15.0 and std_brightness > 5.0
        is_extreme_dark = mean_brightness < 12.0
        is_extreme_bright = mean_brightness > 246.0
        is_featureless = std_brightness < 4.0
        is_low_res = w < 64 or h < 64

        issues = []
        if is_blurry:
            issues.append(f"Blurry image detected (Laplacian variance {blur_variance:.1f} < 15.0)")
        if is_extreme_dark:
            issues.append(f"Underexposed / extreme darkness (mean brightness {mean_brightness:.1f})")
        if is_extreme_bright:
            issues.append(f"Overexposed / extreme glare (mean brightness {mean_brightness:.1f})")
        if is_featureless:
            issues.append(f"Featureless scene (contrast stddev {std_brightness:.1f} < 4.0)")
        if is_low_res:
            issues.append(f"Low resolution image ({w}x{h} < 64x64)")

        quality_status = "REJECTED" if (is_featureless or is_low_res) else ("REVIEW" if issues else "ADEQUATE")

        return {
            "width": w,
            "height": h,
            "total_pixels": w * h,
            "mean_brightness": round(mean_brightness, 1),
            "std_brightness": round(std_brightness, 1),
            "blur_variance": round(blur_variance, 1),
            "is_blurry": is_blurry,
            "is_extreme_dark": is_extreme_dark,
            "is_extreme_bright": is_extreme_bright,
            "is_featureless": is_featureless,
            "is_low_res": is_low_res,
            "quality_status": quality_status,
            "quality_pass": quality_status != "REJECTED",
            "issues": issues,
        }

    @classmethod
    def inspect_file(cls, input_data: Union[bytes, str, Image.Image]) -> Dict[str, Any]:
        """
        Comprehensive forensic inspection of an input file (bytes, path, or PIL Image).
        Identifies GeoTIFF metadata, EXIF GPS, format characteristics, and image quality.
        """
        raw_bytes: Optional[bytes] = None
        img_path: Optional[str] = None
        pil_img: Optional[Image.Image] = None

        if isinstance(input_data, bytes):
            raw_bytes = input_data
            pil_img = Image.open(io.BytesIO(raw_bytes))
        elif isinstance(input_data, str) and os.path.exists(input_data):
            img_path = input_data
            with open(img_path, "rb") as f:
                raw_bytes = f.read()
            pil_img = Image.open(img_path)
        elif isinstance(input_data, Image.Image):
            pil_img = input_data
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            raw_bytes = buf.getvalue()
        else:
            raise ValueError("Input data must be raw bytes, existing file path, or PIL Image instance.")

        # 1. Header & Format Forensics
        format_info = cls.detect_format_and_mime(raw_bytes)

        # 2. Extract GeoTIFF / Georeferencing Metadata
        geo_meta: GeoMetadata
        if img_path:
            geo_meta = GeospatialEngine.extract_metadata(img_path)
        else:
            geo_meta = GeospatialEngine.extract_metadata(pil_img)

        # 3. Extract EXIF GPS Metadata
        exif_gps = cls.extract_exif_gps(pil_img)

        # 4. Quality Gating
        quality = cls.evaluate_image_quality(pil_img)

        # 5. Determine Deterministic Geolocation Classification
        deterministic_info: Dict[str, Any] = {
            "is_deterministic": False,
            "evidence_type": "NONE",
            "classification": "UNKNOWN",
            "coordinates": None,
            "bounds": None,
            "crs": None,
            "provenance": {},
        }

        # Priority 1: GeoTIFF Georeferenced Raster
        if geo_meta.geospatial_available and geo_meta.bounds_world:
            bw = geo_meta.bounds_world
            center_x = (bw["min_x"] + bw["max_x"]) / 2.0
            center_y = (bw["min_y"] + bw["max_y"]) / 2.0
            
            # Formulate WGS84 bbox [min_lon, min_lat, max_lon, max_lat]
            wgs84_bbox = [round(bw["min_x"], 6), round(bw["min_y"], 6), round(bw["max_x"], 6), round(bw["max_y"], 6)]

            deterministic_info = {
                "is_deterministic": True,
                "evidence_type": "GEOREFERENCED_RASTER",
                "classification": "EXACT",
                "coordinates": [round(center_y, 6), round(center_x, 6)],
                "bounds": wgs84_bbox,
                "crs": geo_meta.crs,
                "transform": geo_meta.transform.to_dict() if geo_meta.transform else None,
                "provenance": {
                    "source": "TIFF_TAG_GEOREFERENCING",
                    "crs_type": geo_meta.crs_type,
                    "resolution": geo_meta.resolution,
                    "dimensions": geo_meta.dimensions_pixel,
                },
            }

        # Priority 2: EXIF GPS Coordinates
        elif exif_gps["has_gps"]:
            lat = exif_gps["latitude"]
            lon = exif_gps["longitude"]
            # 2km default radius bounding box around GPS point
            delta_deg = 2.0 / 111.0
            bbox_gps = [
                round(lon - delta_deg, 6),
                round(lat - delta_deg, 6),
                round(lon + delta_deg, 6),
                round(lat + delta_deg, 6),
            ]

            deterministic_info = {
                "is_deterministic": True,
                "evidence_type": "EXIF_GPS",
                "classification": "GEOTAGGED",
                "coordinates": [lat, lon],
                "bounds": bbox_gps,
                "crs": "EPSG:4326",
                "altitude_m": exif_gps.get("altitude_m"),
                "provenance": exif_gps["provenance"],
            }

        return {
            "format": format_info["format"],
            "mime": format_info["mime"],
            "dimensions": {"width": quality["width"], "height": quality["height"]},
            "color_mode": pil_img.mode,
            "quality": quality,
            "geotiff_metadata": geo_meta.to_dict(),
            "exif_gps": exif_gps,
            "deterministic_geolocation": deterministic_info,
            "is_deterministic": deterministic_info["is_deterministic"],
            "classification": deterministic_info["classification"],
        }
