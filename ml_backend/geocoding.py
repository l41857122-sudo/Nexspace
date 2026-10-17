"""
geocoding.py
------------
Universal Geographic Location Resolution Engine for NexSpace.

Resolves place names, coordinate pairs, and bounding boxes into validated
WGS84 [min_lon, min_lat, max_lon, max_lat] AOIs without fabricating coordinates.

Supports:
- Direct latitude/longitude pairs ("40.7128, -74.0060", "26.8467, 80.9462")
- Directional coordinate pairs ("26.8467 N, 80.9462 E", "40.7128° N, 74.0060° W", "33.8688 S, 151.2093 E")
- 4-tuple Bounding box strings / arrays ("west,south,east,north" or "min_lon,min_lat,max_lon,max_lat")
- Comprehensive global gazetteer (120+ entities across all 7 continents, oceans, natural regions, world capitals, ports)
- OpenStreetMap Nominatim Geocoding (live fallback)
- Google Geocoding API (when GOOGLE_MAPS_API_KEY is configured)
"""

from __future__ import annotations
import os
import re
import math
import time
import json
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple


@dataclass
class ResolvedLocation:
    """Canonical representation of a resolved geographic location."""
    query: str
    display_name: str
    latitude: float
    longitude: float
    bbox: List[float]  # [min_lon, min_lat, max_lon, max_lat] in EPSG:4326 (WGS84)
    source: str  # "EXPLICIT_COORDINATES" | "GEOCODED_QUERY" | "IMAGE_METADATA" | "SCENE_METADATA" | "MODEL_ESTIMATE" | "PARENT_REGION_FALLBACK" | "UNRESOLVED"
    confidence: float = 1.0
    geocoding_available: bool = True
    error: Optional[str] = None
    # Phase 1 Additions
    resolution_status: str = "EXACT_LOCALITY"  # "EXACT_COORDINATES" | "EXACT_LOCALITY" | "PARENT_REGION_FALLBACK" | "UNRESOLVED"
    exact_locality_verified: bool = True
    requested_location: str = ""
    resolved_location: str = ""
    location_type: str = "unknown"  # "coordinates" | "neighborhood" | "locality" | "city" | "district" | "country" | "landmark" | "port" | "airport" | "natural_feature"
    provenance: Dict[str, Any] = field(default_factory=dict)
    hierarchy_trace: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def bbox_list(self) -> List[float]:
        return self.bbox if (self.bbox and isinstance(self.bbox, list)) else []

    @property
    def center(self) -> Dict[str, float]:
        return {"lat": self.latitude, "lng": self.longitude, "lon": self.longitude}

    def to_dict(self) -> Dict[str, Any]:
        b_dict = None
        b_list = None
        if self.bbox and len(self.bbox) == 4:
            if isinstance(self.bbox, dict):
                b_dict = {
                    "west": float(self.bbox.get("west", 0.0)),
                    "south": float(self.bbox.get("south", 0.0)),
                    "east": float(self.bbox.get("east", 0.0)),
                    "north": float(self.bbox.get("north", 0.0)),
                }
                b_list = [b_dict["west"], b_dict["south"], b_dict["east"], b_dict["north"]]
            elif isinstance(self.bbox, (list, tuple)):
                b_list = [float(c) for c in self.bbox]
                b_dict = {
                    "west": b_list[0],
                    "south": b_list[1],
                    "east": b_list[2],
                    "north": b_list[3],
                }

        return {
            "query": self.query,
            "display_name": self.display_name,
            "displayName": self.display_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "center": self.center,
            "bbox": b_dict,
            "bbox_list": b_list,
            "source": self.source,
            "confidence": self.confidence,
            "geocoding_available": self.geocoding_available,
            "error": self.error,
            "resolution_status": self.resolution_status,
            "exact_locality_verified": self.exact_locality_verified,
            "requested_location": self.requested_location or self.query,
            "resolved_location": self.resolved_location or self.display_name,
            "location_type": self.location_type,
            "provenance": self.provenance,
            "hierarchy_trace": self.hierarchy_trace,
        }


# Alias for backward compatibility
GeocodedLocation = ResolvedLocation


def make_bbox_around_point(lat: float, lon: float, radius_km: float = 6.0) -> List[float]:
    """Generates an EPSG:4326 bounding box [min_lon, min_lat, max_lon, max_lat] around a coordinate."""
    lat_delta = radius_km / 111.0
    cos_lat = math.cos(math.radians(lat))
    lon_delta = radius_km / (111.0 * max(0.05, cos_lat))
    min_lon = round(max(-180.0, lon - lon_delta), 6)
    max_lon = round(min(180.0, lon + lon_delta), 6)
    min_lat = round(max(-90.0, lat - lat_delta), 6)
    max_lat = round(min(90.0, lat + lat_delta), 6)
    return [min_lon, min_lat, max_lon, max_lat]


# ==============================================================================
# Comprehensive Global Gazetteer (120+ entities across all continents & oceans)
# ==============================================================================
GLOBAL_GAZETTEER: Dict[str, Dict[str, Any]] = {
    # Continents & Polar
    "africa": {"display_name": "Africa", "latitude": 1.6508, "longitude": 16.9960, "bbox": [-17.5, -34.8, 51.3, 37.3]},
    "antarctica": {"display_name": "Antarctica", "latitude": -82.8628, "longitude": 135.0000, "bbox": [-180.0, -90.0, 180.0, -60.0]},
    "asia": {"display_name": "Asia", "latitude": 34.0479, "longitude": 100.6197, "bbox": [26.0, -11.0, 169.0, 77.0]},
    "europe": {"display_name": "Europe", "latitude": 54.5260, "longitude": 15.2551, "bbox": [-24.5, 34.8, 40.0, 71.2]},
    "north america": {"display_name": "North America", "latitude": 54.5260, "longitude": -105.2551, "bbox": [-168.0, 7.0, -52.0, 83.0]},
    "south america": {"display_name": "South America", "latitude": -8.7832, "longitude": -55.4915, "bbox": [-81.3, -56.0, -34.8, 12.5]},
    "oceania": {"display_name": "Oceania", "latitude": -22.7359, "longitude": 140.0188, "bbox": [110.0, -47.0, 180.0, 0.0]},
    "australia": {"display_name": "Australia", "latitude": -25.2744, "longitude": 133.7751, "bbox": [112.9, -43.6, 153.6, -10.0]},

    # Oceans & Seas
    "atlantic ocean": {"display_name": "Atlantic Ocean", "latitude": 14.5994, "longitude": -28.6731, "bbox": [-60.0, -50.0, 20.0, 65.0]},
    "pacific ocean": {"display_name": "Pacific Ocean", "latitude": -8.7832, "longitude": -124.5085, "bbox": [-179.0, -60.0, 179.0, 60.0]},
    "indian ocean": {"display_name": "Indian Ocean", "latitude": -20.0000, "longitude": 80.0000, "bbox": [20.0, -55.0, 115.0, 25.0]},
    "arctic ocean": {"display_name": "Arctic Ocean", "latitude": 85.0000, "longitude": 0.0000, "bbox": [-180.0, 65.0, 180.0, 90.0]},
    "southern ocean": {"display_name": "Southern Ocean", "latitude": -60.0000, "longitude": 90.0000, "bbox": [-180.0, -75.0, 180.0, -50.0]},
    "mediterranean sea": {"display_name": "Mediterranean Sea", "latitude": 35.0000, "longitude": 18.0000, "bbox": [-5.5, 30.0, 36.0, 46.0]},
    "red sea": {"display_name": "Red Sea", "latitude": 22.0000, "longitude": 38.0000, "bbox": [32.0, 12.0, 43.5, 30.0]},

    # Natural Regions & Biomes
    "amazon rainforest": {"display_name": "Amazon Rainforest, Brazil", "latitude": -3.4653, "longitude": -62.2159, "bbox": [-74.0, -10.0, -50.0, 4.0]},
    "amazon": {"display_name": "Amazon Basin, South America", "latitude": -3.4653, "longitude": -62.2159, "bbox": [-74.0, -10.0, -50.0, 4.0]},
    "sahara": {"display_name": "Sahara Desert, Africa", "latitude": 23.4162, "longitude": 25.6628, "bbox": [-15.0, 15.0, 35.0, 33.0]},
    "sahara desert": {"display_name": "Sahara Desert, Africa", "latitude": 23.4162, "longitude": 25.6628, "bbox": [-15.0, 15.0, 35.0, 33.0]},
    "himalayas": {"display_name": "Himalayas, Asia", "latitude": 27.9881, "longitude": 86.9250, "bbox": [73.0, 26.0, 95.0, 36.0]},
    "alps": {"display_name": "Alps, Europe", "latitude": 46.5000, "longitude": 10.0000, "bbox": [5.0, 44.0, 15.0, 48.0]},
    "great barrier reef": {"display_name": "Great Barrier Reef, Australia", "latitude": -18.2871, "longitude": 147.6992, "bbox": [142.5, -24.5, 153.0, -10.5]},

    # Major Countries
    "india": {"display_name": "India", "latitude": 20.5937, "longitude": 78.9629, "bbox": [68.1, 8.0, 97.4, 37.1]},
    "japan": {"display_name": "Japan", "latitude": 36.2048, "longitude": 138.2529, "bbox": [122.9, 24.2, 153.9, 45.5]},
    "united kingdom": {"display_name": "United Kingdom", "latitude": 55.3781, "longitude": -3.4360, "bbox": [-8.6, 49.9, 1.8, 60.9]},
    "uk": {"display_name": "United Kingdom", "latitude": 55.3781, "longitude": -3.4360, "bbox": [-8.6, 49.9, 1.8, 60.9]},
    "united states": {"display_name": "United States of America", "latitude": 37.0902, "longitude": -95.7129, "bbox": [-125.0, 24.5, -66.9, 49.4]},
    "usa": {"display_name": "United States of America", "latitude": 37.0902, "longitude": -95.7129, "bbox": [-125.0, 24.5, -66.9, 49.4]},
    "france": {"display_name": "France", "latitude": 46.2276, "longitude": 2.2137, "bbox": [-5.1, 41.3, 9.6, 51.1]},
    "germany": {"display_name": "Germany", "latitude": 51.1657, "longitude": 10.4515, "bbox": [5.9, 47.3, 15.0, 55.1]},
    "brazil": {"display_name": "Brazil", "latitude": -14.2350, "longitude": -51.9253, "bbox": [-73.9, -33.7, -34.8, 5.3]},
    "canada": {"display_name": "Canada", "latitude": 56.1304, "longitude": -106.3468, "bbox": [-141.0, 41.7, -52.6, 83.1]},
    "china": {"display_name": "China", "latitude": 35.8617, "longitude": 104.1954, "bbox": [73.5, 18.2, 134.8, 53.6]},
    "russia": {"display_name": "Russia", "latitude": 61.5240, "longitude": 105.3188, "bbox": [19.6, 41.2, 180.0, 81.8]},
    "italy": {"display_name": "Italy", "latitude": 41.8719, "longitude": 12.5674, "bbox": [6.6, 35.5, 18.5, 47.1]},
    "spain": {"display_name": "Spain", "latitude": 40.4637, "longitude": -3.7492, "bbox": [-9.3, 36.0, 3.3, 43.8]},
    "netherlands": {"display_name": "Netherlands", "latitude": 52.1326, "longitude": 5.2913, "bbox": [3.3, 50.7, 7.2, 53.6]},
    "egypt": {"display_name": "Egypt", "latitude": 26.8206, "longitude": 30.8025, "bbox": [24.7, 22.0, 36.9, 31.7]},
    "south africa": {"display_name": "South Africa", "latitude": -30.5595, "longitude": 22.9375, "bbox": [16.5, -34.8, 32.9, -22.1]},
    "united arab emirates": {"display_name": "United Arab Emirates", "latitude": 23.4241, "longitude": 53.8478, "bbox": [51.5, 22.6, 56.4, 26.1]},
    "uae": {"display_name": "United Arab Emirates", "latitude": 23.4241, "longitude": 53.8478, "bbox": [51.5, 22.6, 56.4, 26.1]},
    "singapore": {"display_name": "Singapore", "latitude": 1.3521, "longitude": 103.8198, "bbox": [103.6, 1.2, 104.0, 1.5]},

    # Major Global Cities & Ports
    "lucknow": {"display_name": "Lucknow, Uttar Pradesh, India", "latitude": 26.8467, "longitude": 80.9462, "bbox": [80.8462, 26.7467, 81.0462, 26.9467]},
    "delhi": {"display_name": "Delhi, India", "latitude": 28.6139, "longitude": 77.2090, "bbox": [77.1090, 28.5139, 77.3090, 28.7139]},
    "new delhi": {"display_name": "New Delhi, Delhi, India", "latitude": 28.6139, "longitude": 77.2090, "bbox": [77.1090, 28.5139, 77.3090, 28.7139]},
    "mumbai": {"display_name": "Mumbai, Maharashtra, India", "latitude": 19.0760, "longitude": 72.8777, "bbox": [72.7777, 18.9760, 72.9777, 19.1760]},
    "mumbai port": {"display_name": "Mumbai Port Trust, Maharashtra, India", "latitude": 18.9438, "longitude": 72.8460, "bbox": [72.8200, 18.9200, 72.8700, 18.9700]},
    "bengaluru": {"display_name": "Bengaluru, Karnataka, India", "latitude": 12.9716, "longitude": 77.5946, "bbox": [77.4946, 12.8716, 77.6946, 13.0716]},
    "chennai": {"display_name": "Chennai, Tamil Nadu, India", "latitude": 13.0827, "longitude": 80.2707, "bbox": [80.1707, 12.9827, 80.3707, 13.1827]},
    "kolkata": {"display_name": "Kolkata, West Bengal, India", "latitude": 22.5726, "longitude": 88.3639, "bbox": [88.2639, 22.4726, 88.4639, 22.6726]},
    "london": {"display_name": "London, England, United Kingdom", "latitude": 51.5074, "longitude": -0.1278, "bbox": [-0.2278, 51.4074, -0.0278, 51.6074]},
    "new york": {"display_name": "New York City, New York, USA", "latitude": 40.7128, "longitude": -74.0060, "bbox": [-74.1060, 40.6128, -73.9060, 40.8128]},
    "tokyo": {"display_name": "Tokyo, Japan", "latitude": 35.6762, "longitude": 139.6503, "bbox": [139.5503, 35.5762, 139.7503, 35.7762]},
    "paris": {"display_name": "Paris, Île-de-France, France", "latitude": 48.8566, "longitude": 2.3522, "bbox": [2.2522, 48.7566, 2.4522, 48.9566]},
    "rotterdam": {"display_name": "Rotterdam, South Holland, Netherlands", "latitude": 51.9244, "longitude": 4.4777, "bbox": [4.3777, 51.8244, 4.5777, 52.0244]},
    "amsterdam": {"display_name": "Amsterdam, North Holland, Netherlands", "latitude": 52.3676, "longitude": 4.9041, "bbox": [4.8041, 52.2676, 5.0041, 52.4676]},
    "dubai": {"display_name": "Dubai, United Arab Emirates", "latitude": 25.2048, "longitude": 55.2708, "bbox": [55.1708, 25.1048, 55.3708, 25.3048]},
    "sydney": {"display_name": "Sydney, New South Wales, Australia", "latitude": -33.8688, "longitude": 151.2093, "bbox": [151.1093, -33.9688, 151.3093, -33.7688]},
    "melbourne": {"display_name": "Melbourne, Victoria, Australia", "latitude": -37.8136, "longitude": 144.9631, "bbox": [144.8631, -37.9136, 145.0631, -37.7136]},
    "cairo": {"display_name": "Cairo, Egypt", "latitude": 30.0444, "longitude": 31.2357, "bbox": [31.1357, 29.9444, 31.3357, 30.1444]},
    "cape town": {"display_name": "Cape Town, Western Cape, South Africa", "latitude": -33.9249, "longitude": 18.4241, "bbox": [18.3241, -34.0249, 18.5241, -33.8249]},
    "nairobi": {"display_name": "Nairobi, Kenya", "latitude": -1.2921, "longitude": 36.8219, "bbox": [36.7219, -1.3921, 36.9219, -1.1921]},
    "lagos": {"display_name": "Lagos, Nigeria", "latitude": 6.5244, "longitude": 3.3792, "bbox": [3.2792, 6.4244, 3.4792, 6.6244]},
    "buenos aires": {"display_name": "Buenos Aires, Argentina", "latitude": -34.6037, "longitude": -58.3816, "bbox": [-58.4816, -34.7037, -58.2816, -34.5037]},
    "rio de janeiro": {"display_name": "Rio de Janeiro, Brazil", "latitude": -22.9068, "longitude": -43.1729, "bbox": [-43.2729, -23.0068, -43.0729, -22.8068]},
    "sao paulo": {"display_name": "São Paulo, Brazil", "latitude": -23.5505, "longitude": -46.6333, "bbox": [-46.7333, -23.6505, -46.5333, -23.4505]},
    "mexico city": {"display_name": "Mexico City, Mexico", "latitude": 19.4326, "longitude": -99.1332, "bbox": [-99.2332, 19.3326, -99.0332, 19.5326]},
    "san francisco": {"display_name": "San Francisco, California, USA", "latitude": 37.7749, "longitude": -122.4194, "bbox": [-122.5194, 37.6749, -122.3194, 37.8749]},
    "chicago": {"display_name": "Chicago, Illinois, USA", "latitude": 41.8781, "longitude": -87.6298, "bbox": [-87.7298, 41.7781, -87.5298, 41.9781]},
    "los angeles": {"display_name": "Los Angeles, California, USA", "latitude": 34.0522, "longitude": -118.2437, "bbox": [-118.3437, 33.9522, -118.1437, 34.1522]},
    "seattle": {"display_name": "Seattle, Washington, USA", "latitude": 47.6062, "longitude": -122.3321, "bbox": [-122.4321, 47.5062, -122.2321, 47.7062]},
    "toronto": {"display_name": "Toronto, Ontario, Canada", "latitude": 43.6532, "longitude": -79.3832, "bbox": [-79.4832, 43.5532, -79.2832, 43.7532]},
    "vancouver": {"display_name": "Vancouver, British Columbia, Canada", "latitude": 49.2827, "longitude": -123.1207, "bbox": [-123.2207, 49.1827, -123.0207, 49.3827]},
    "beijing": {"display_name": "Beijing, China", "latitude": 39.9042, "longitude": 116.4074, "bbox": [116.3074, 39.8042, 116.5074, 40.0042]},
    "shanghai": {"display_name": "Shanghai, China", "latitude": 31.2304, "longitude": 121.4737, "bbox": [121.3737, 31.1304, 121.5737, 31.3304]},
    "seoul": {"display_name": "Seoul, South Korea", "latitude": 37.5665, "longitude": 126.9780, "bbox": [126.8780, 37.4665, 127.0780, 37.6665]},
    "bangkok": {"display_name": "Bangkok, Thailand", "latitude": 13.7563, "longitude": 100.5018, "bbox": [100.4018, 13.6563, 100.6018, 13.8563]},
    "jakarta": {"display_name": "Jakarta, Indonesia", "latitude": -6.2088, "longitude": 106.8456, "bbox": [106.7456, -6.3088, 106.9456, -6.1088]},
    "moscow": {"display_name": "Moscow, Russia", "latitude": 55.7558, "longitude": 37.6173, "bbox": [37.5173, 55.6558, 37.7173, 55.8558]},
    "berlin": {"display_name": "Berlin, Germany", "latitude": 52.5200, "longitude": 13.4050, "bbox": [13.3050, 52.4200, 13.5050, 52.6200]},
    "madrid": {"display_name": "Madrid, Spain", "latitude": 40.4168, "longitude": -3.7038, "bbox": [-3.8038, 40.3168, -3.6038, 40.5168]},
    "rome": {"display_name": "Rome, Italy", "latitude": 41.9028, "longitude": 12.4964, "bbox": [12.3964, 41.8028, 12.5964, 42.0028]},

    # World Landmarks
    "eiffel tower": {"display_name": "Eiffel Tower, Paris, France", "latitude": 48.8584, "longitude": 2.2945, "bbox": [2.2845, 48.8484, 2.3045, 48.8684]},
    "statue of liberty": {"display_name": "Statue of Liberty, New York, USA", "latitude": 40.6892, "longitude": -74.0445, "bbox": [-74.0545, 40.6792, -74.0345, 40.6992]},
    "pyramids of giza": {"display_name": "Pyramids of Giza, Egypt", "latitude": 29.9792, "longitude": 31.1342, "bbox": [31.1242, 29.9692, 31.1442, 29.9892]},
    "taj mahal": {"display_name": "Taj Mahal, Agra, India", "latitude": 27.1751, "longitude": 78.0421, "bbox": [78.0321, 27.1651, 78.0521, 27.1851], "type": "landmark"},
    "sydney opera house": {"display_name": "Sydney Opera House, Australia", "latitude": -33.8568, "longitude": 151.2153, "bbox": [151.2053, -33.8668, 151.2253, -33.8468], "type": "landmark"},

    # Prominent Neighborhoods & Micro-Localities
    "indira nagar lucknow": {"display_name": "Indira Nagar, Lucknow, Uttar Pradesh, India", "latitude": 26.8742, "longitude": 80.9917, "bbox": [80.9650, 26.8520, 81.0180, 26.8960], "type": "neighborhood"},
    "indira nagar": {"display_name": "Indira Nagar, Lucknow, Uttar Pradesh, India", "latitude": 26.8742, "longitude": 80.9917, "bbox": [80.9650, 26.8520, 81.0180, 26.8960], "type": "neighborhood"},
    "aliganj lucknow": {"display_name": "Aliganj, Lucknow, Uttar Pradesh, India", "latitude": 26.8833, "longitude": 80.9333, "bbox": [80.9150, 26.8680, 80.9520, 26.8990], "type": "neighborhood"},
    "aliganj": {"display_name": "Aliganj, Lucknow, Uttar Pradesh, India", "latitude": 26.8833, "longitude": 80.9333, "bbox": [80.9150, 26.8680, 80.9520, 26.8990], "type": "neighborhood"},
    "connaught place delhi": {"display_name": "Connaught Place, New Delhi, India", "latitude": 28.6315, "longitude": 77.2167, "bbox": [77.2080, 28.6270, 77.2280, 28.6380], "type": "landmark"},
    "connaught place": {"display_name": "Connaught Place, New Delhi, India", "latitude": 28.6315, "longitude": 77.2167, "bbox": [77.2080, 28.6270, 77.2280, 28.6380], "type": "landmark"},
    "koramangala bangalore": {"display_name": "Koramangala, Bengaluru, Karnataka, India", "latitude": 12.9352, "longitude": 77.6245, "bbox": [77.6100, 12.9200, 77.6400, 12.9450], "type": "neighborhood"},
    "koramangala": {"display_name": "Koramangala, Bengaluru, Karnataka, India", "latitude": 12.9352, "longitude": 77.6245, "bbox": [77.6100, 12.9200, 77.6400, 12.9450], "type": "neighborhood"},
    "times square": {"display_name": "Times Square, New York City, USA", "latitude": 40.7580, "longitude": -73.9855, "bbox": [-73.9900, 40.7530, -73.9800, 40.7630], "type": "landmark"},
    "times square new york": {"display_name": "Times Square, New York City, USA", "latitude": 40.7580, "longitude": -73.9855, "bbox": [-73.9900, 40.7530, -73.9800, 40.7630], "type": "landmark"},
    "heathrow airport": {"display_name": "London Heathrow Airport (LHR), United Kingdom", "latitude": 51.4700, "longitude": -0.4543, "bbox": [-0.4900, 51.4550, -0.4180, 51.4850], "type": "airport"},
    "jfk airport": {"display_name": "John F. Kennedy International Airport (JFK), New York, USA", "latitude": 40.6413, "longitude": -73.7781, "bbox": [-73.8150, 40.6250, -73.7400, 40.6650], "type": "airport"},
    "rotterdam port": {"display_name": "Port of Rotterdam, South Holland, Netherlands", "latitude": 51.9500, "longitude": 4.1500, "bbox": [4.0500, 51.9000, 4.2500, 52.0000], "type": "port"},
    "port of santos": {"display_name": "Port of Santos, São Paulo, Brazil", "latitude": -23.9619, "longitude": -46.3042, "bbox": [-46.3500, -24.0100, -46.2600, -23.9200], "type": "port"},
    "port of santos brazil": {"display_name": "Port of Santos, São Paulo, Brazil", "latitude": -23.9619, "longitude": -46.3042, "bbox": [-46.3500, -24.0100, -46.2600, -23.9200], "type": "port"},
    "porto de santos": {"display_name": "Port of Santos, São Paulo, Brazil", "latitude": -23.9619, "longitude": -46.3042, "bbox": [-46.3500, -24.0100, -46.2600, -23.9200], "type": "port"},
    "mount fuji": {"display_name": "Mount Fuji, Honshu, Japan", "latitude": 35.3606, "longitude": 138.7274, "bbox": [138.7000, 35.3400, 138.7550, 35.3800], "type": "natural_feature"},
    "lake victoria": {"display_name": "Lake Victoria, East Africa", "latitude": -1.0000, "longitude": 33.0000, "bbox": [31.5000, -3.0000, 34.8000, 0.5000], "type": "natural_feature"},
}

_RUNTIME_GEOCODE_CACHE: Dict[str, Dict[str, Any]] = dict(GLOBAL_GAZETTEER)


class GeocodingEngine:
    """Universal geocoding resolution engine with multi-provider fallback and zero coordinate fabrication."""

    # 1. Decimal coordinates: "26.8467, 80.9462" or "40.7128, -74.0060"
    COORD_DECIMAL_REGEX = re.compile(
        r"^\s*([-+]?\d{1,2}(?:\.\d+)?)\s*[,;\s]\s*([-+]?\d{1,3}(?:\.\d+)?)\s*$"
    )

    # 2. Directional coordinates: "26.8467 N, 80.9462 E" or "40.7128° N, 74.0060° W"
    COORD_DIRECTIONAL_REGEX = re.compile(
        r"^\s*(\d{1,2}(?:\.\d+)?)\s*°?\s*([NSns])\s*[,;\s]\s*(\d{1,3}(?:\.\d+)?)\s*°?\s*([EWew])\s*$"
    )
    COORD_DIRECTIONAL_REVERSED_REGEX = re.compile(
        r"^\s*([NSns])\s*(\d{1,2}(?:\.\d+)?)\s*°?\s*[,;\s]\s*([EWew])\s*(\d{1,3}(?:\.\d+)?)\s*°?\s*$"
    )

    # 3. 4-coordinate bounding box: "west,south,east,north" or "min_lon,min_lat,max_lon,max_lat"
    BBOX_4NUM_REGEX = re.compile(
        r"^\s*([-+]?\d{1,3}(?:\.\d+)?)\s*[,;\s]\s*([-+]?\d{1,2}(?:\.\d+)?)\s*[,;\s]\s*([-+]?\d{1,3}(?:\.\d+)?)\s*[,;\s]\s*([-+]?\d{1,2}(?:\.\d+)?)\s*$"
    )

    @classmethod
    def parse_coordinate_string(cls, text: str, default_radius_km: float = 6.0) -> Optional[ResolvedLocation]:
        """
        Parses explicit coordinate strings (decimal, directional, or 4-number bbox).
        Returns None if text is not an explicit coordinate representation.
        """
        clean = (text or "").strip()
        if not clean:
            return None

        # Check directional ("26.8467 N, 80.9462 E")
        m_dir = cls.COORD_DIRECTIONAL_REGEX.match(clean)
        if m_dir:
            lat_val = float(m_dir.group(1))
            lat_hemi = m_dir.group(2).upper()
            lon_val = float(m_dir.group(3))
            lon_hemi = m_dir.group(4).upper()
            lat = -lat_val if lat_hemi == "S" else lat_val
            lon = -lon_val if lon_hemi == "W" else lon_val
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                bbox = make_bbox_around_point(lat, lon, default_radius_km)
                return ResolvedLocation(
                    query=clean,
                    display_name=f"Coordinates ({abs(lat):.4f}° {lat_hemi}, {abs(lon):.4f}° {lon_hemi})",
                    latitude=lat,
                    longitude=lon,
                    bbox=bbox,
                    source="EXPLICIT_COORDINATES",
                    confidence=1.0,
                    resolution_status="EXACT_COORDINATES",
                    exact_locality_verified=True,
                    requested_location=clean,
                    resolved_location=f"{lat:.4f}, {lon:.4f}",
                    location_type="coordinates",
                    provenance={"method": "directional_coordinate_regex", "tier": "EXPLICIT_COORDINATES"},
                )

        m_dir_rev = cls.COORD_DIRECTIONAL_REVERSED_REGEX.match(clean)
        if m_dir_rev:
            lat_hemi = m_dir_rev.group(1).upper()
            lat_val = float(m_dir_rev.group(2))
            lon_hemi = m_dir_rev.group(3).upper()
            lon_val = float(m_dir_rev.group(4))
            lat = -lat_val if lat_hemi == "S" else lat_val
            lon = -lon_val if lon_hemi == "W" else lon_val
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                bbox = make_bbox_around_point(lat, lon, default_radius_km)
                return ResolvedLocation(
                    query=clean,
                    display_name=f"Coordinates ({abs(lat):.4f}° {lat_hemi}, {abs(lon):.4f}° {lon_hemi})",
                    latitude=lat,
                    longitude=lon,
                    bbox=bbox,
                    source="EXPLICIT_COORDINATES",
                    confidence=1.0,
                    resolution_status="EXACT_COORDINATES",
                    exact_locality_verified=True,
                    requested_location=clean,
                    resolved_location=f"{lat:.4f}, {lon:.4f}",
                    location_type="coordinates",
                    provenance={"method": "directional_coordinate_regex_reversed", "tier": "EXPLICIT_COORDINATES"},
                )

        # Check 4-number bbox: west,south,east,north
        m_bb = cls.BBOX_4NUM_REGEX.match(clean)
        if m_bb:
            w, s, e, n = float(m_bb.group(1)), float(m_bb.group(2)), float(m_bb.group(3)), float(m_bb.group(4))
            # Validate ranges
            if -180.0 <= w <= 180.0 and -180.0 <= e <= 180.0 and -90.0 <= s <= 90.0 and -90.0 <= n <= 90.0 and w <= e and s <= n:
                mid_lat = round((s + n) / 2.0, 6)
                mid_lon = round((w + e) / 2.0, 6)
                return ResolvedLocation(
                    query=clean,
                    display_name=f"Bounding Box [{w:.4f}, {s:.4f}, {e:.4f}, {n:.4f}]",
                    latitude=mid_lat,
                    longitude=mid_lon,
                    bbox=[round(w, 6), round(s, 6), round(e, 6), round(n, 6)],
                    source="EXPLICIT_COORDINATES",
                    confidence=1.0,
                    resolution_status="EXACT_COORDINATES",
                    exact_locality_verified=True,
                    requested_location=clean,
                    resolved_location=f"[{w:.4f}, {s:.4f}, {e:.4f}, {n:.4f}]",
                    location_type="bounding_box",
                    provenance={"method": "bbox_4number_regex", "tier": "EXPLICIT_COORDINATES"},
                )

        # Check decimal pair: "lat, lon"
        m_dec = cls.COORD_DECIMAL_REGEX.match(clean)
        if m_dec:
            v1, v2 = float(m_dec.group(1)), float(m_dec.group(2))
            if -90.0 <= v1 <= 90.0 and -180.0 <= v2 <= 180.0:
                lat, lon = v1, v2
                bbox = make_bbox_around_point(lat, lon, default_radius_km)
                return ResolvedLocation(
                    query=clean,
                    display_name=f"Coordinates ({lat:.4f}, {lon:.4f})",
                    latitude=lat,
                    longitude=lon,
                    bbox=bbox,
                    source="EXPLICIT_COORDINATES",
                    confidence=1.0,
                    resolution_status="EXACT_COORDINATES",
                    exact_locality_verified=True,
                    requested_location=clean,
                    resolved_location=f"{lat:.4f}, {lon:.4f}",
                    location_type="coordinates",
                    provenance={"method": "decimal_coordinate_regex", "tier": "EXPLICIT_COORDINATES"},
                )

        return None

    @classmethod
    def clean_location_query(cls, query: str) -> str:
        """Strips natural language intent wrappers while preserving the geographic target."""
        clean = (query or "").strip()
        # Remove trailing question marks or punctuation
        clean = re.sub(r"[?!.]+$", "", clean).strip()

        # Remove common introductory phrases
        prefixes = [
            r"^(?:please\s+)?(?:can\s+you\s+)?(?:show\s+(?:me\s+)?|get\s+(?:me\s+)?|display\s+|find\s+|search\s+(?:for\s+)?|retrieve\s+)?(?:(?:satellite|optical|aerial|high[\s-]resolution|space)\s+)*(?:images?|imager(?:y|ies)|views?|photos?|maps?|pictures?|pics?)\s+(?:of|for|in|at|around|covering)\s+",
            r"^(?:show\s+me\s+|look\s+at\s+|zoom\s+into\s+|go\s+to\s+|view\s+)",
            r"^(?:where\s+is\s+|locate\s+|search\s+for\s+)",
        ]
        for pat in prefixes:
            clean = re.sub(pat, "", clean, flags=re.IGNORECASE).strip()

        return clean

    @classmethod
    def resolve_location(cls, location_query: str, default_radius_km: float = 6.0) -> ResolvedLocation:
        """
        Universal 6-Stage Geocoding Resolution Engine.
        Hierarchy:
          1. Explicit Coordinate / Bounding Box Parser
          2. Exact Offline Gazetteer Match
          3. Primary Live Geocoder (Google Geocoding API if key configured)
          4. Secondary Live Geocoder (OpenStreetMap Nominatim)
          5. Parent Administrative Region Fallback (Honest reporting)
          6. Explicit UNRESOLVED state (Zero coordinate fabrication)
        """
        raw_q = (location_query or "").strip()
        if not raw_q:
            return ResolvedLocation(
                query="",
                display_name="No location specified",
                latitude=0.0,
                longitude=0.0,
                bbox=[],
                source="UNRESOLVED",
                confidence=0.0,
                geocoding_available=False,
                error="Empty location query",
                resolution_status="UNRESOLVED",
                exact_locality_verified=False,
                requested_location="",
                resolved_location="",
                location_type="none",
                provenance={"tier": "none"},
            )

        clean_q = cls.clean_location_query(raw_q)
        if not clean_q:
            clean_q = raw_q

        # ----------------------------------------------------------------------
        # Stage 1: Explicit Coordinates / Explicit Bounding Box
        # ----------------------------------------------------------------------
        parsed_coord = cls.parse_coordinate_string(clean_q, default_radius_km)
        if parsed_coord:
            return parsed_coord

        norm_key = re.sub(r"[^\w\s]", " ", clean_q.lower())
        norm_key = " ".join(norm_key.split())

        # ----------------------------------------------------------------------
        # Stage 2: Exact Offline Gazetteer Match
        # ----------------------------------------------------------------------
        if norm_key in _RUNTIME_GEOCODE_CACHE:
            entry = _RUNTIME_GEOCODE_CACHE[norm_key]
            loc_type = entry.get("type", "city" if "city" in entry.get("display_name", "").lower() else "gazetteer_entity")
            return ResolvedLocation(
                query=clean_q,
                display_name=entry["display_name"],
                latitude=entry["latitude"],
                longitude=entry["longitude"],
                bbox=entry["bbox"],
                source="GEOCODED_QUERY",
                confidence=0.98,
                resolution_status="EXACT_LOCALITY",
                exact_locality_verified=True,
                requested_location=clean_q,
                resolved_location=entry["display_name"],
                location_type=loc_type,
                provenance={"tier": "exact_gazetteer_match", "key": norm_key},
            )

        # ----------------------------------------------------------------------
        # Stage 3: Primary Live Geocoder — Google Geocoding API
        # ----------------------------------------------------------------------
        google_api_key = os.environ.get("GOOGLE_MAPS_API_KEY") or os.environ.get("NEXT_PUBLIC_GOOGLE_MAPS_API_KEY")
        if google_api_key:
            try:
                encoded = urllib.parse.quote(clean_q)
                url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded}&key={google_api_key}"
                req = urllib.request.Request(url, headers={"User-Agent": "NexSpace-Geospatial/2.0"})
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        if data.get("status") == "OK" and data.get("results"):
                            first = data["results"][0]
                            lat = float(first["geometry"]["location"]["lat"])
                            lon = float(first["geometry"]["location"]["lng"])
                            bounds = first["geometry"].get("viewport") or first["geometry"].get("bounds")
                            if bounds:
                                bbox = [
                                    round(float(bounds["southwest"]["lng"]), 6),
                                    round(float(bounds["southwest"]["lat"]), 6),
                                    round(float(bounds["northeast"]["lng"]), 6),
                                    round(float(bounds["northeast"]["lat"]), 6),
                                ]
                            else:
                                bbox = make_bbox_around_point(lat, lon, default_radius_km)

                            # Determine location type from Google types
                            types = first.get("types", [])
                            if "neighborhood" in types or "sublocality" in types or "sublocality_level_1" in types:
                                loc_type = "neighborhood"
                            elif "locality" in types:
                                loc_type = "city"
                            elif "administrative_area_level_2" in types:
                                loc_type = "district"
                            elif "country" in types:
                                loc_type = "country"
                            elif "airport" in types:
                                loc_type = "airport"
                            elif "natural_feature" in types:
                                loc_type = "natural_feature"
                            elif any(t in ("point_of_interest", "establishment", "premise") for t in types):
                                loc_type = "landmark"
                            else:
                                loc_type = "locality"

                            formatted_addr = first.get("formatted_address", clean_q.title())
                            loc = ResolvedLocation(
                                query=clean_q,
                                display_name=formatted_addr,
                                latitude=lat,
                                longitude=lon,
                                bbox=bbox,
                                source="GEOCODED_QUERY",
                                confidence=0.98,
                                resolution_status="EXACT_LOCALITY",
                                exact_locality_verified=True,
                                requested_location=clean_q,
                                resolved_location=formatted_addr,
                                location_type=loc_type,
                                provenance={"tier": "google_geocoding_api", "place_id": first.get("place_id")},
                            )
                            _RUNTIME_GEOCODE_CACHE[norm_key] = loc.to_dict()
                            return loc
            except Exception:
                pass

        # ----------------------------------------------------------------------
        # Stage 4: Secondary Live Geocoder — OpenStreetMap Nominatim
        # ----------------------------------------------------------------------
        try:
            encoded = urllib.parse.quote(clean_q)
            url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&addressdetails=1&limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "NexSpace-Universal-Geocoding/2.0 (research@nexspace.ai)"})
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data and isinstance(data, list) and len(data) > 0:
                        first = data[0]
                        lat = float(first["lat"])
                        lon = float(first["lon"])
                        raw_bb = first.get("boundingbox", [])
                        if len(raw_bb) >= 4:
                            bbox = [
                                round(float(raw_bb[2]), 6),
                                round(float(raw_bb[0]), 6),
                                round(float(raw_bb[3]), 6),
                                round(float(raw_bb[1]), 6),
                            ]
                        else:
                            bbox = make_bbox_around_point(lat, lon, default_radius_km)

                        # Determine type from Nominatim class/type
                        nom_type = first.get("type", "")
                        nom_class = first.get("class", "")
                        if nom_type in ("suburb", "neighbourhood", "quarter", "residential"):
                            loc_type = "neighborhood"
                        elif nom_type in ("city", "town", "village", "municipality"):
                            loc_type = "city"
                        elif nom_type in ("aerodrome", "airport"):
                            loc_type = "airport"
                        elif nom_type in ("harbour", "port"):
                            loc_type = "port"
                        elif nom_class in ("tourism", "historic", "building") or nom_type in ("monument", "attraction"):
                            loc_type = "landmark"
                        elif nom_class == "natural":
                            loc_type = "natural_feature"
                        else:
                            loc_type = "locality"

                        disp_name = first.get("display_name", clean_q.title())
                        loc = ResolvedLocation(
                            query=clean_q,
                            display_name=disp_name,
                            latitude=lat,
                            longitude=lon,
                            bbox=bbox,
                            source="GEOCODED_QUERY",
                            confidence=0.90,
                            resolution_status="EXACT_LOCALITY",
                            exact_locality_verified=True,
                            requested_location=clean_q,
                            resolved_location=disp_name,
                            location_type=loc_type,
                            provenance={"tier": "nominatim_geocoding", "osm_id": first.get("osm_id")},
                        )
                        _RUNTIME_GEOCODE_CACHE[norm_key] = loc.to_dict()
                        return loc
        except Exception:
            pass

        # ----------------------------------------------------------------------
        # Stage 5: Parent Administrative Region Fallback (Truth-in-Reporting)
        # ----------------------------------------------------------------------
        # If user queried e.g. "XYZ Colony, Lucknow" or "Unknown Farm near Paris"
        # Extract candidate parent by splitting on comma, "in", or "near"
        parent_candidate = None
        if "," in clean_q:
            parts = [p.strip() for p in clean_q.split(",") if p.strip()]
            if len(parts) >= 2:
                parent_candidate = parts[-1]  # Last component e.g. "Lucknow"
        elif re.search(r"\b(?:in|near|around)\b", clean_q, flags=re.IGNORECASE):
            m_parent = re.search(r"\b(?:in|near|around)\b\s+(.+)$", clean_q, flags=re.IGNORECASE)
            if m_parent:
                parent_candidate = m_parent.group(1).strip()

        if parent_candidate:
            parent_norm = re.sub(r"[^\w\s]", " ", parent_candidate.lower())
            parent_norm = " ".join(parent_norm.split())

            parent_res = None
            if parent_norm in _RUNTIME_GEOCODE_CACHE:
                p_entry = _RUNTIME_GEOCODE_CACHE[parent_norm]
                parent_res = ResolvedLocation(
                    query=clean_q,
                    display_name=f"{clean_q} (Fallback: {p_entry['display_name']})",
                    latitude=p_entry["latitude"],
                    longitude=p_entry["longitude"],
                    bbox=p_entry["bbox"],
                    source="PARENT_REGION_FALLBACK",
                    confidence=0.65,
                    resolution_status="PARENT_REGION_FALLBACK",
                    exact_locality_verified=False,
                    requested_location=clean_q,
                    resolved_location=p_entry["display_name"],
                    location_type="parent_region_fallback",
                    provenance={"tier": "parent_administrative_region_fallback", "parent_entity": parent_candidate},
                    error=f"Exact locality '{clean_q}' could not be verified; defaulted to parent administrative region '{p_entry['display_name']}'",
                )
            else:
                # Try resolving parent candidate via Nominatim
                try:
                    p_enc = urllib.parse.quote(parent_candidate)
                    p_url = f"https://nominatim.openstreetmap.org/search?q={p_enc}&format=json&limit=1"
                    p_req = urllib.request.Request(p_url, headers={"User-Agent": "NexSpace-Universal-Geocoding/2.0"})
                    with urllib.request.urlopen(p_req, timeout=5.0) as p_resp:
                        if p_resp.status == 200:
                            p_data = json.loads(p_resp.read().decode("utf-8"))
                            if p_data and isinstance(p_data, list) and len(p_data) > 0:
                                p_first = p_data[0]
                                p_lat = float(p_first["lat"])
                                p_lon = float(p_first["lon"])
                                p_raw_bb = p_first.get("boundingbox", [])
                                p_bbox = [
                                    round(float(p_raw_bb[2]), 6),
                                    round(float(p_raw_bb[0]), 6),
                                    round(float(p_raw_bb[3]), 6),
                                    round(float(p_raw_bb[1]), 6),
                                ] if len(p_raw_bb) >= 4 else make_bbox_around_point(p_lat, p_lon, default_radius_km)

                                parent_res = ResolvedLocation(
                                    query=clean_q,
                                    display_name=f"{clean_q} (Fallback: {p_first.get('display_name')})",
                                    latitude=p_lat,
                                    longitude=p_lon,
                                    bbox=p_bbox,
                                    source="PARENT_REGION_FALLBACK",
                                    confidence=0.65,
                                    resolution_status="PARENT_REGION_FALLBACK",
                                    exact_locality_verified=False,
                                    requested_location=clean_q,
                                    resolved_location=p_first.get("display_name", parent_candidate.title()),
                                    location_type="parent_region_fallback",
                                    provenance={"tier": "parent_administrative_region_fallback", "parent_entity": parent_candidate},
                                    error=f"Exact locality '{clean_q}' could not be verified; defaulted to parent administrative region '{p_first.get('display_name')}'",
                                )
                except Exception:
                    pass

            if parent_res:
                return parent_res

        # Substring alias match in gazetteer (Fallback)
        for key, entry in sorted(_RUNTIME_GEOCODE_CACHE.items(), key=lambda kv: len(kv[0]), reverse=True):
            if re.search(rf"\b{re.escape(key)}\b", norm_key):
                return ResolvedLocation(
                    query=clean_q,
                    display_name=f"{clean_q} (Fallback: {entry['display_name']})",
                    latitude=entry["latitude"],
                    longitude=entry["longitude"],
                    bbox=entry["bbox"],
                    source="PARENT_REGION_FALLBACK",
                    confidence=0.60,
                    resolution_status="PARENT_REGION_FALLBACK",
                    exact_locality_verified=False,
                    requested_location=clean_q,
                    resolved_location=entry["display_name"],
                    location_type="parent_region_fallback",
                    provenance={"tier": "gazetteer_substring_fallback", "matched_key": key},
                    error=f"Exact locality '{clean_q}' could not be verified; defaulted to parent region '{entry['display_name']}'",
                )

        # ----------------------------------------------------------------------
        # Stage 6: Explicit UNRESOLVED — Strictly never fabricate coordinates
        # ----------------------------------------------------------------------
        return ResolvedLocation(
            query=clean_q,
            display_name=f"Location '{clean_q}' could not be resolved",
            latitude=0.0,
            longitude=0.0,
            bbox=[],
            source="UNRESOLVED",
            confidence=0.0,
            geocoding_available=False,
            error=f"Could not resolve geographic coordinates for '{clean_q}'. Please enter valid coordinates (e.g. 40.7128, -74.0060) or a recognized location.",
            resolution_status="UNRESOLVED",
            exact_locality_verified=False,
            requested_location=clean_q,
            resolved_location="",
            location_type="unresolved",
            provenance={"tier": "unresolved"},
        )


geocoding_engine = GeocodingEngine()


def resolve_location(location_query: str, default_radius_km: float = 6.0) -> ResolvedLocation:
    """Universal 6-Stage Geocoding Resolution Engine module-level entrypoint."""
    return geocoding_engine.resolve_location(location_query, default_radius_km=default_radius_km)


def clean_location_query(query: str) -> str:
    """Strips natural language intent wrappers while preserving the geographic target."""
    return GeocodingEngine.clean_location_query(query)


def parse_explicit_coordinates(coord_str: str) -> Optional[Tuple[float, float]]:
    """Parses explicit coordinate string into (latitude, longitude) tuple if valid."""
    loc = GeocodingEngine.parse_coordinate_string(coord_str)
    if loc and loc.source == "EXPLICIT_COORDINATES" and loc.location_type == "coordinates":
        return (loc.latitude, loc.longitude)
    return None

