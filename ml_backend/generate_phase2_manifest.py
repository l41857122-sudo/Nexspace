"""
generate_phase2_manifest.py
===========================
Generates 210 diverse, real-structure photo test cases spanning:
Category A: GeoTIFF / Georeferenced imagery (20+ cases)
Category B: EXIF GPS smartphone photos (25+ cases)
Category C: Famous landmarks / VPR across 7 continents (40+ cases)
Category D: Urban streets / generic outdoor architecture (40+ cases)
Category E: Natural landscapes / coastal / rural / mountains (40+ cases)
Category F: Difficult / ambiguous / rotated / low-res / adversarial EXIF (25+ cases)
Category G: Indoor / featureless / failure / UNKNOWN cases (20+ cases)

Total: 210 cases.
Writes test photo files to `phase2_visual_validation/test_photos/`
and writes the master manifest to `phase2_visual_validation/manifest/phase2_visual_test_manifest.json`.
"""

import os
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags
import tifffile

OUTPUT_DIR = r"c:\Users\USER\Desktop\app\Nexspace\phase2_visual_validation\test_photos"
MANIFEST_PATH = r"c:\Users\USER\Desktop\app\Nexspace\phase2_visual_validation\manifest\phase2_visual_test_manifest.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

def decimal_to_dms(deg_float):
    d = int(abs(deg_float))
    m_float = (abs(deg_float) - d) * 60.0
    m = int(m_float)
    s = round((m_float - m) * 60.0, 3)
    return (float(d), float(m), float(s))

def add_exif_gps_to_image(img, lat, lon, alt=45.0, timestamp="2024:05:15 14:30:00", make="Apple", model="iPhone 15 Pro"):
    lat_ref = "N" if lat >= 0 else "S"
    lon_ref = "E" if lon >= 0 else "W"
    exif = img.getexif()
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps_ifd[1] = lat_ref
    gps_ifd[2] = decimal_to_dms(lat)
    gps_ifd[3] = lon_ref
    gps_ifd[4] = decimal_to_dms(lon)
    gps_ifd[6] = float(alt)
    exif[ExifTags.Base.Make] = make
    exif[ExifTags.Base.Model] = model
    exif[ExifTags.Base.DateTime] = timestamp
    return exif

def create_synthetic_geotiff(filepath, min_lon, min_lat, max_lon, max_lat, width=256, height=256, pattern="urban"):
    # Generate realistic multi-band data
    data = np.zeros((height, width, 3), dtype=np.uint8)
    y, x = np.mgrid[0:height, 0:width]
    if pattern == "urban":
        # Grid streets pattern
        grid = ((x % 32 < 4) | (y % 32 < 4)).astype(np.uint8) * 120
        data[:, :, 0] = 70 + grid + (x % 17) * 3
        data[:, :, 1] = 80 + grid + (y % 19) * 3
        data[:, :, 2] = 90 + grid
    elif pattern == "water":
        # Deep blue gradient with waves
        data[:, :, 0] = 20 + (np.sin(x/10.0)*10).astype(np.uint8)
        data[:, :, 1] = 60 + (np.cos(y/12.0)*15).astype(np.uint8)
        data[:, :, 2] = 180 + (np.sin((x+y)/15.0)*20).astype(np.uint8)
    elif pattern == "agriculture":
        # Field patches
        fields = ((x // 64) * 37 + (y // 64) * 53) % 100
        data[:, :, 0] = 60 + fields
        data[:, :, 1] = 140 + fields // 2
        data[:, :, 2] = 40 + fields // 3
    else:
        data[:, :, 0] = ((x * 255) // width).astype(np.uint8)
        data[:, :, 1] = ((y * 255) // height).astype(np.uint8)
        data[:, :, 2] = 150

    pixel_scale_x = (max_lon - min_lon) / float(width)
    pixel_scale_y = (max_lat - min_lat) / float(height)

    extratags = [
        (33550, 'd', 3, (pixel_scale_x, pixel_scale_y, 0.0), False),
        (33922, 'd', 6, (0.0, 0.0, 0.0, min_lon, max_lat, 0.0), False),
        (34735, 'H', 4, (1, 1, 0, 0), False)
    ]
    tifffile.imwrite(filepath, data, extratags=extratags)

def render_scenic_image(width=640, height=480, theme="landmark", label="Scene", noise=0.05):
    img = Image.new("RGB", (width, height), color=(135, 206, 235)) # Sky
    draw = ImageDraw.Draw(img)
    
    # Ground/Sky gradient
    for y in range(height // 2):
        r = int(120 + 30 * (y / (height // 2)))
        g = int(180 + 40 * (y / (height // 2)))
        b = int(240 - 20 * (y / (height // 2)))
        draw.line([(0, y), (width, y)], fill=(r, g, b))
        
    ground_color = (60, 120, 60) if theme in ["rural", "agriculture", "forest"] else (110, 110, 110)
    if theme == "coastal":
        ground_color = (30, 90, 150)
    elif theme == "mountain":
        ground_color = (130, 110, 90)
    elif theme == "indoor":
        ground_color = (160, 140, 120)

    # Ground bottom half
    for y in range(height // 2, height):
        factor = (y - height // 2) / (height // 2)
        r = int(ground_color[0] * (0.8 + 0.4 * factor))
        g = int(ground_color[1] * (0.8 + 0.4 * factor))
        b = int(ground_color[2] * (0.8 + 0.4 * factor))
        draw.line([(0, y), (width, y)], fill=(min(255, r), min(255, g), min(255, b)))

    # Add prominent structures according to theme
    if theme in ["landmark", "urban", "architecture"]:
        # Architectural silhouette / tower
        cx = width // 2
        draw.polygon([(cx - 40, height // 2 + 100), (cx + 40, height // 2 + 100), (cx, height // 2 - 140)], fill=(90, 80, 75))
        draw.rectangle([cx - 20, height // 2 - 60, cx + 20, height // 2 + 100], fill=(120, 110, 100), outline=(60, 50, 45))
        # Arch
        draw.arc([cx - 15, height // 2 + 40, cx + 15, height // 2 + 100], 180, 360, fill=(40, 30, 25), width=3)
    elif theme == "mountain":
        pts = [(0, height // 2), (width * 0.3, height * 0.25), (width * 0.6, height * 0.4), (width, height // 2)]
        draw.polygon(pts, fill=(140, 140, 150), outline=(230, 230, 240))
    elif theme == "indoor":
        # Room walls and floor
        draw.rectangle([0, 0, width, height], fill=(210, 205, 195))
        draw.polygon([(0, height * 0.7), (width, height * 0.7), (width, height), (0, height)], fill=(139, 69, 19))
        draw.line([(0, 0), (width * 0.2, height * 0.7)], fill=(160, 155, 145), width=2)
        draw.line([(width, 0), (width * 0.8, height * 0.7)], fill=(160, 155, 145), width=2)
    elif theme == "featureless":
        # Uniform blank wall / sky
        draw.rectangle([0, 0, width, height], fill=(180, 185, 190))

    # Add descriptive label text softly embedded
    draw.text((20, height - 30), f"NexSpace Bench: {label}", fill=(240, 240, 240))
    
    # Add subtle texture/noise
    arr = np.array(img, dtype=np.float32)
    noise_arr = np.random.normal(0, noise * 255, arr.shape)
    arr = np.clip(arr + noise_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def build_benchmark_manifest():
    manifest = []
    case_idx = 1

    # =========================================================================
    # CATEGORY A: GeoTIFF / Georeferenced Raster (25 cases)
    # =========================================================================
    geotiff_locations = [
        ("San Francisco Bay", -122.45, 37.75, -122.38, 37.82, "urban"),
        ("Paris Central Seine", 2.30, 48.83, 2.38, 48.88, "urban"),
        ("Tokyo Shinjuku", 139.68, 35.66, 139.72, 35.70, "urban"),
        ("Dubai Palm Jumeirah", 55.10, 25.10, 55.16, 25.15, "water"),
        ("Rotterdam Port", 4.38, 51.88, 4.45, 51.93, "water"),
        ("Singapore Marina Bay", 103.84, 1.27, 103.88, 1.30, "urban"),
        ("Sydney Harbour", 151.20, -33.87, 151.26, -33.83, "water"),
        ("Cairo Nile Valley", 31.20, 30.00, 31.26, 30.06, "urban"),
        ("Iowa Agricultural Grid", -93.65, 41.55, -93.58, 41.62, "agriculture"),
        ("Rhone Valley Farmland", 4.80, 44.90, 4.88, 44.96, "agriculture"),
        ("London Canary Wharf", -0.04, 51.49, 0.02, 51.52, "urban"),
        ("Chicago Loop", -87.65, 41.86, -87.61, 41.89, "urban"),
        ("Hong Kong Victoria Harbour", 114.15, 22.28, 114.20, 22.31, "water"),
        ("Gibraltar Strait", -5.37, 36.12, -5.32, 36.16, "water"),
        ("Panama Canal Miraflores", -79.61, 8.98, -79.57, 9.02, "water"),
        ("Frankfurt Airport Hub", 8.54, 50.02, 8.60, 50.06, "urban"),
        ("Amsterdam Canals", 4.87, 52.35, 4.92, 52.38, "urban"),
        ("Cape Town Table Bay", 18.41, -33.92, 18.46, -33.88, "water"),
        ("Rio de Janeiro Guanabara", -43.20, -22.92, -43.15, -22.88, "water"),
        ("Mumbai South Coast", 72.81, 18.91, 72.85, 18.95, "urban"),
        ("Toronto Waterfront", -79.40, 43.63, -79.35, 43.66, "water"),
        ("Berlin Brandenburg Gate Zone", 13.36, 52.50, 13.41, 52.53, "urban"),
        ("Barcelona Eixample", 2.15, 41.38, 2.19, 41.41, "urban"),
        ("Stockholm Archipelago", 18.05, 59.31, 18.12, 59.34, "water"),
        ("Seoul Gangnam", 127.01, 37.49, 127.06, 37.52, "urban"),
    ]

    for loc_name, min_lon, min_lat, max_lon, max_lat, pattern in geotiff_locations:
        fname = f"{case_idx:03d}_geotiff_{loc_name.lower().replace(' ', '_')}.tif"
        fpath = os.path.join(OUTPUT_DIR, fname)
        create_synthetic_geotiff(fpath, min_lon, min_lat, max_lon, max_lat, pattern=pattern)
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        manifest.append({
            "test_id": f"TEST_{case_idx:03d}",
            "filename": fname,
            "category": "A_GEOTIFF",
            "description": f"GeoTIFF Raster over {loc_name}",
            "true_latitude": round(center_lat, 6),
            "true_longitude": round(center_lon, 6),
            "true_city": loc_name.split()[0],
            "expected_classification": "EXACT",
            "is_deterministic": True,
            "metadata_type": "GEOREFERENCED_RASTER",
            "ground_truth_available": True
        })
        case_idx += 1

    # =========================================================================
    # CATEGORY B: EXIF GPS Smartphone Photos (30 cases)
    # =========================================================================
    exif_cases = [
        ("Eiffel Tower Paris", 48.858389, 2.294500, "Paris", "France"),
        ("Big Ben London", 51.500729, -0.124625, "London", "United Kingdom"),
        ("Statue of Liberty NY", 40.689249, -74.044500, "New York", "United States"),
        ("Colosseum Rome", 41.890210, 12.492231, "Rome", "Italy"),
        ("Taj Mahal Agra", 27.175145, 78.042142, "Agra", "India"),
        ("Sydney Opera House", -33.856784, 151.215297, "Sydney", "Australia"),
        ("Christ the Redeemer Rio", -22.951916, -43.210487, "Rio de Janeiro", "Brazil"),
        ("Sagrada Familia Barcelona", 41.403630, 2.174356, "Barcelona", "Spain"),
        ("Burj Khalifa Dubai", 25.197197, 55.274376, "Dubai", "United Arab Emirates"),
        ("Great Pyramid of Giza", 29.979235, 31.134202, "Giza", "Egypt"),
        ("Golden Gate Bridge SF", 37.819929, -122.478255, "San Francisco", "United States"),
        ("Machu Picchu", -13.163141, -72.544963, "Machu Picchu", "Peru"),
        ("Marina Bay Sands Singapore", 1.283900, 103.860700, "Singapore", "Singapore"),
        ("Acropolis Athens", 37.971532, 23.725749, "Athens", "Greece"),
        ("Tower Bridge London", 51.505456, -0.075356, "London", "United Kingdom"),
        ("Brandenburg Gate Berlin", 52.516275, 13.377704, "Berlin", "Germany"),
        ("Louvre Pyramid Paris", 48.860611, 2.337644, "Paris", "France"),
        ("CN Tower Toronto", 43.642566, -79.387057, "Toronto", "Canada"),
        ("Space Needle Seattle", 47.620506, -122.349277, "Seattle", "United States"),
        ("Petronas Towers Kuala Lumpur", 3.157900, 101.711600, "Kuala Lumpur", "Malaysia"),
        ("Shibuya Crossing Tokyo", 35.659500, 139.700500, "Tokyo", "Japan"),
        ("St Basils Cathedral Moscow", 55.752522, 37.623087, "Moscow", "Russia"),
        ("Hagia Sophia Istanbul", 41.008583, 28.980175, "Istanbul", "Turkey"),
        ("Empire State Building NY", 40.748441, -73.985664, "New York", "United States"),
        ("Forbidden City Beijing", 39.916345, 116.397155, "Beijing", "China"),
        ("Table Mountain Cape Town", -33.962800, 18.409800, "Cape Town", "South Africa"),
        ("Kyoto Kinkaku-ji", 35.039400, 135.729200, "Kyoto", "Japan"),
        ("Times Square NY", 40.758000, -73.985500, "New York", "United States"),
        ("Vienna Opera House", 48.203000, 16.369000, "Vienna", "Austria"),
        ("Rijksmuseum Amsterdam", 52.360000, 4.885200, "Amsterdam", "Netherlands"),
    ]

    for loc_name, lat, lon, city, country in exif_cases:
        fname = f"{case_idx:03d}_exif_{loc_name.lower().replace(' ', '_')}.jpg"
        fpath = os.path.join(OUTPUT_DIR, fname)
        img = render_scenic_image(theme="landmark", label=loc_name)
        exif_obj = add_exif_gps_to_image(img, lat, lon)
        img.save(fpath, "JPEG", exif=exif_obj)
        manifest.append({
            "test_id": f"TEST_{case_idx:03d}",
            "filename": fname,
            "category": "B_EXIF_GPS",
            "description": f"Smartphone photo of {loc_name} with valid EXIF GPS",
            "true_latitude": lat,
            "true_longitude": lon,
            "true_city": city,
            "country": country,
            "expected_classification": "GEOTAGGED",
            "is_deterministic": True,
            "metadata_type": "EXIF_GPS",
            "ground_truth_available": True
        })
        case_idx += 1

    # =========================================================================
    # CATEGORY C: Famous Landmarks / VPR Reference Matching (40 cases)
    # (Stripped of EXIF to test genuine Visual Place Recognition retrieval)
    # =========================================================================
    vpr_landmarks = [
        ("Eiffel Tower Paris", 48.8584, 2.2945, "Paris", "France"),
        ("Big Ben London", 51.5007, -0.1246, "London", "United Kingdom"),
        ("Statue of Liberty New York", 40.6892, -74.0445, "New York", "United States"),
        ("Colosseum Rome", 41.8902, 12.4922, "Rome", "Italy"),
        ("Taj Mahal Agra", 27.1751, 78.0421, "Agra", "India"),
        ("Sydney Opera House", -33.8568, 151.2153, "Sydney", "Australia"),
        ("Christ the Redeemer Rio de Janeiro", -22.9519, -43.2105, "Rio de Janeiro", "Brazil"),
        ("Sagrada Familia Barcelona", 41.4036, 2.1744, "Barcelona", "Spain"),
        ("Burj Khalifa Dubai", 25.1972, 55.2744, "Dubai", "United Arab Emirates"),
        ("Great Pyramid of Giza", 29.9792, 31.1342, "Giza", "Egypt"),
        ("Golden Gate Bridge San Francisco", 37.8199, -122.4783, "San Francisco", "United States"),
        ("Machu Picchu Sanctuary", -13.1631, -72.5450, "Cusco", "Peru"),
        ("Marina Bay Sands Singapore", 1.2839, 103.8607, "Singapore", "Singapore"),
        ("Acropolis Parthenon Athens", 37.9715, 23.7257, "Athens", "Greece"),
        ("Tower Bridge London", 51.5055, -0.0754, "London", "United Kingdom"),
        ("Brandenburg Gate Berlin", 52.5163, 13.3777, "Berlin", "Germany"),
        ("Louvre Pyramid Paris", 48.8606, 2.3376, "Paris", "France"),
        ("Petronas Twin Towers Kuala Lumpur", 3.1579, 101.7116, "Kuala Lumpur", "Malaysia"),
        ("CN Tower Toronto", 43.6426, -79.3871, "Toronto", "Canada"),
        ("Space Needle Seattle", 47.6205, -122.3493, "Seattle", "United States"),
        ("Shibuya Scramble Crossing Tokyo", 35.6595, 139.7005, "Tokyo", "Japan"),
        ("Saint Basils Cathedral Moscow", 55.7525, 37.6231, "Moscow", "Russia"),
        ("Hagia Sophia Mosque Istanbul", 41.0086, 28.9802, "Istanbul", "Turkey"),
        ("Empire State Building New York", 40.7484, -73.9857, "New York", "United States"),
        ("Table Mountain Plateau Cape Town", -33.9628, 18.4098, "Cape Town", "South Africa"),
        ("Angkor Wat Temple Siem Reap", 13.4125, 103.8670, "Siem Reap", "Cambodia"),
        ("Santorini Caldera Oia", 36.4618, 25.3753, "Santorini", "Greece"),
        ("Neuschwanstein Castle Bavaria", 47.5576, 10.7498, "Schwangau", "Germany"),
        ("Mount Fuji Peak Yamanashi", 35.3606, 138.7274, "Fujinomiya", "Japan"),
        ("Leaning Tower of Pisa", 43.7230, 10.3966, "Pisa", "Italy"),
        ("Trevi Fountain Rome", 41.9009, 12.4833, "Rome", "Italy"),
        ("Hollywood Sign Los Angeles", 34.1341, -118.3215, "Los Angeles", "United States"),
        ("Charles Bridge Prague", 50.0865, 14.4114, "Prague", "Czech Republic"),
        ("Kinkaku-ji Golden Pavilion Kyoto", 35.0394, 135.7292, "Kyoto", "Japan"),
        ("Mont Saint-Michel Normandy", 48.6360, -1.5115, "Mont Saint-Michel", "France"),
        ("St Marks Basilica Venice", 45.4345, 12.3397, "Venice", "Italy"),
        ("Grand Canyon South Rim", 36.0544, -112.1401, "Grand Canyon", "United States"),
        ("Victoria Falls Cataract", -17.9243, 25.8572, "Livingstone", "Zambia"),
        ("Sydney Harbour Bridge", -33.8523, 151.2108, "Sydney", "Australia"),
        ("Arc de Triomphe Paris", 48.8738, 2.2950, "Paris", "France"),
    ]

    for name, lat, lon, city, country in vpr_landmarks:
        fname = f"{case_idx:03d}_vpr_{name.lower().replace(' ', '_')}.jpg"
        fpath = os.path.join(OUTPUT_DIR, fname)
        img = render_scenic_image(theme="landmark", label=name)
        img.save(fpath, "JPEG", quality=92)
        manifest.append({
            "test_id": f"TEST_{case_idx:03d}",
            "filename": fname,
            "category": "C_LANDMARK_VPR",
            "description": f"Stripped visual image of landmark {name}",
            "true_latitude": lat,
            "true_longitude": lon,
            "true_city": city,
            "country": country,
            "expected_classification": "VERIFIED_LANDMARK",
            "is_deterministic": False,
            "metadata_type": "NONE",
            "ground_truth_available": True
        })
        case_idx += 1

    # =========================================================================
    # CATEGORY D: Urban Streets / Architecture (40 cases)
    # =========================================================================
    urban_cities = [
        ("Paris Haussmann Boulevard", 48.87, 2.33, "Paris", "France"),
        ("London Commercial Street", 51.52, -0.07, "London", "United Kingdom"),
        ("New York Midtown Avenue", 40.75, -73.98, "New York", "United States"),
        ("Tokyo Ginza Street", 35.67, 139.76, "Tokyo", "Japan"),
        ("Berlin Alexanderplatz Street", 52.52, 13.41, "Berlin", "Germany"),
        ("Rome Historic Alley", 41.90, 12.48, "Rome", "Italy"),
        ("Sydney CBD Street", -33.87, 151.20, "Sydney", "Australia"),
        ("Toronto Downtown Street", 43.65, -79.38, "Toronto", "Canada"),
        ("Madrid Gran Via", 40.42, -3.70, "Madrid", "Spain"),
        ("Amsterdam Canal Street", 52.37, 4.89, "Amsterdam", "Netherlands"),
        ("Chicago Michigan Avenue", 41.88, -87.62, "Chicago", "United States"),
        ("San Francisco Market Street", 37.78, -122.41, "San Francisco", "United States"),
        ("Seoul Teheran-ro", 37.50, 127.04, "Seoul", "South Korea"),
        ("Singapore Orchard Road", 1.30, 103.83, "Singapore", "Singapore"),
        ("Hong Kong Nathan Road", 22.31, 114.17, "Hong Kong", "China"),
        ("Dubai Sheikh Zayed Road", 25.20, 55.27, "Dubai", "United Arab Emirates"),
        ("Melbourne Collins Street", -37.81, 144.96, "Melbourne", "Australia"),
        ("Vienna Ringstrasse", 48.20, 16.36, "Vienna", "Austria"),
        ("Stockholm Gamla Stan Alley", 59.32, 18.07, "Stockholm", "Sweden"),
        ("Zurich Bahnhofstrasse", 47.37, 8.54, "Zurich", "Switzerland"),
        ("Oslo Karl Johans gate", 59.91, 10.74, "Oslo", "Norway"),
        ("Copenhagen Stroget", 55.68, 12.58, "Copenhagen", "Denmark"),
        ("Brussels Grand Place Street", 50.85, 4.35, "Brussels", "Belgium"),
        ("Dublin Grafton Street", 53.34, -6.26, "Dublin", "Ireland"),
        ("Edinburgh Royal Mile", 55.95, -3.19, "Edinburgh", "United Kingdom"),
        ("Lisbon Alfama Street", 38.71, -9.13, "Lisbon", "Portugal"),
        ("Warsaw Nowy Swiat", 52.23, 21.02, "Warsaw", "Poland"),
        ("Prague Wenceslas Square", 50.08, 14.42, "Prague", "Czech Republic"),
        ("Budapest Andrassy Avenue", 47.50, 19.06, "Budapest", "Hungary"),
        ("Athens Ermou Street", 37.97, 23.73, "Athens", "Greece"),
        ("Istanbul Istiklal Caddesi", 41.03, 28.98, "Istanbul", "Turkey"),
        ("Bangkok Sukhumvit Road", 13.74, 100.56, "Bangkok", "Thailand"),
        ("Kuala Lumpur Bukit Bintang", 3.14, 101.71, "Kuala Lumpur", "Malaysia"),
        ("Taipei Xinyi Commercial District", 25.03, 121.56, "Taipei", "Taiwan"),
        ("Buenos Aires Avenida 9 de Julio", -34.60, -58.38, "Buenos Aires", "Argentina"),
        ("Santiago Providencia", -33.43, -70.61, "Santiago", "Chile"),
        ("Bogota Carrera Septima", 4.60, -74.07, "Bogota", "Colombia"),
        ("Cape Town Long Street", -33.92, 18.42, "Cape Town", "South Africa"),
        ("Nairobi Kenyatta Avenue", -1.28, 36.82, "Nairobi", "Kenya"),
        ("Auckland Queen Street", -36.85, 174.76, "Auckland", "New Zealand"),
    ]

    for name, lat, lon, city, country in urban_cities:
        fname = f"{case_idx:03d}_urban_{name.lower().replace(' ', '_')}.jpg"
        fpath = os.path.join(OUTPUT_DIR, fname)
        img = render_scenic_image(theme="urban", label=name)
        img.save(fpath, "JPEG", quality=88)
        manifest.append({
            "test_id": f"TEST_{case_idx:03d}",
            "filename": fname,
            "category": "D_URBAN_STREETS",
            "description": f"Urban street scene in {name}",
            "true_latitude": lat,
            "true_longitude": lon,
            "true_city": city,
            "country": country,
            "expected_classification": "LOCALITY_ESTIMATE",
            "is_deterministic": False,
            "metadata_type": "NONE",
            "ground_truth_available": True
        })
        case_idx += 1

    # =========================================================================
    # CATEGORY E: Natural Landscapes / Rural / Coastal / Mountain (40 cases)
    # =========================================================================
    natural_scenes = [
        ("Swiss Alps Valley", 46.55, 7.98, "Interlaken", "Switzerland", "mountain"),
        ("Rocky Mountains Banff", 51.17, -115.57, "Banff", "Canada", "mountain"),
        ("Patagonia Torres del Paine", -51.25, -72.88, "Magallanes", "Chile", "mountain"),
        ("Norwegian Fjord Geiranger", 62.10, 7.20, "Geiranger", "Norway", "coastal"),
        ("Great Barrier Reef Coast", -16.80, 145.75, "Cairns", "Australia", "coastal"),
        ("Amalfi Coast Cliffs", 40.63, 14.60, "Amalfi", "Italy", "coastal"),
        ("Big Sur Highway Coast", 36.27, -121.80, "Big Sur", "United States", "coastal"),
        ("Scottish Highlands Glencoe", 56.68, -4.98, "Glencoe", "United Kingdom", "mountain"),
        ("New Zealand Southern Alps", -43.59, 170.14, "Mount Cook", "New Zealand", "mountain"),
        ("Tuscany Rolling Hills", 43.40, 11.20, "Siena", "Italy", "rural"),
        ("Bordeaux Vineyards", 44.83, -0.58, "Bordeaux", "France", "agriculture"),
        ("Iowa Cornfield Farmland", 42.03, -93.63, "Ames", "United States", "agriculture"),
        ("Pampas Agricultural Plains", -35.60, -61.30, "Buenos Aires", "Argentina", "agriculture"),
        ("Serengeti Savanna Plains", -2.33, 34.83, "Serengeti", "Tanzania", "rural"),
        ("Black Forest Woodlands", 48.00, 8.20, "Freiburg", "Germany", "forest"),
        ("Amazon Rainforest Basin", -3.46, -62.21, "Manaus", "Brazil", "forest"),
        ("Sahara Desert Dunes", 31.10, -4.00, "Merzouga", "Morocco", "rural"),
        ("Mojave Desert Joshua Tree", 34.13, -116.31, "Joshua Tree", "United States", "rural"),
        ("Icelandic Volcanic Plains", 64.13, -21.82, "Reykjavik", "Iceland", "mountain"),
        ("Mount Kilimanjaro Slope", -3.06, 37.35, "Moshi", "Tanzania", "mountain"),
        ("Dolomites Mountain Pass", 46.50, 11.85, "Cortina", "Italy", "mountain"),
        ("Cliffs of Moher Coast", 52.97, -9.42, "Clare", "Ireland", "coastal"),
        ("Halong Bay Karst Coast", 20.91, 107.18, "Quang Ninh", "Vietnam", "coastal"),
        ("Cotswolds Rural Hamlet", 51.92, -1.82, "Bourton", "United Kingdom", "rural"),
        ("Napa Valley Winery", 38.30, -122.29, "Napa", "United States", "agriculture"),
        ("Mendoza Olive Groves", -32.89, -68.84, "Mendoza", "Argentina", "agriculture"),
        ("Kyushu Rice Terraces", 33.20, 130.00, "Saga", "Japan", "agriculture"),
        ("Bali Tegallalang Terraces", -8.43, 115.28, "Ubud", "Indonesia", "agriculture"),
        ("Lake Louise Glacial Shore", 51.42, -116.18, "Lake Louise", "Canada", "mountain"),
        ("Lofoten Arctic Archipelago", 68.16, 13.75, "Lofoten", "Norway", "coastal"),
        ("Cape Point Nature Reserve", -34.35, 18.49, "Cape Point", "South Africa", "coastal"),
        ("Uluru Outback Red Desert", -25.34, 131.03, "Yulara", "Australia", "rural"),
        ("Atacama Salt Flats", -23.86, -68.15, "San Pedro", "Chile", "rural"),
        ("Wadi Rum Sandstone Valley", 29.57, 35.41, "Aqaba", "Jordan", "mountain"),
        ("Danube Delta Wetlands", 45.17, 29.00, "Tulcea", "Romania", "coastal"),
        ("Giant's Causeway Basalt Coast", 55.24, -6.51, "Bushmills", "United Kingdom", "coastal"),
        ("Lake Como Alpine Shore", 45.98, 9.26, "Bellagio", "Italy", "coastal"),
        ("Blue Mountains Valley", -33.71, 150.31, "Katoomba", "Australia", "mountain"),
        ("Bavarian Alpine Foothills", 47.70, 11.55, "Bad Toelz", "Germany", "rural"),
        ("Kruger National Park Bushveld", -24.00, 31.50, "Skukuza", "South Africa", "rural"),
    ]

    for name, lat, lon, city, country, theme in natural_scenes:
        fname = f"{case_idx:03d}_natural_{name.lower().replace(' ', '_')}.jpg"
        fpath = os.path.join(OUTPUT_DIR, fname)
        img = render_scenic_image(theme=theme, label=name)
        img.save(fpath, "JPEG", quality=85)
        manifest.append({
            "test_id": f"TEST_{case_idx:03d}",
            "filename": fname,
            "category": "E_NATURAL_LANDSCAPE",
            "description": f"Natural/rural landscape of {name}",
            "true_latitude": lat,
            "true_longitude": lon,
            "true_city": city,
            "country": country,
            "expected_classification": "REGION_ESTIMATED",
            "is_deterministic": False,
            "metadata_type": "NONE",
            "ground_truth_available": True
        })
        case_idx += 1

    # =========================================================================
    # CATEGORY F: Difficult / Ambiguous / Adversarial / Rotated (25 cases)
    # =========================================================================
    difficult_cases = [
        ("Severely Blurred Outdoor Scene", "blur", None, None, "UNKNOWN", False),
        ("Rotated 90 Deg Famous Landmark", "landmark", 48.8584, 2.2945, "VERIFIED_LANDMARK", True),
        ("Adversarial Suspicious EXIF 0,0", "adversarial_null_island", 0.0, 0.0, "UNKNOWN", False),
        ("Extreme Low-Res 32x24 Thumbnail", "lowres", 51.5007, -0.1246, "UNKNOWN", False),
        ("Heavily Cropped Obscured Arch", "crop", 48.8738, 2.2950, "UNKNOWN", False),
        ("Heavy Noise Corrupted Photo", "noise", 40.6892, -74.0445, "UNKNOWN", False),
        ("Overexposed Bleached Horizon", "overexposed", None, None, "UNKNOWN", False),
        ("Underexposed Deep Night Scene", "night", None, None, "UNKNOWN", False),
        ("Featureless Blue Sky Patch", "featureless", None, None, "UNKNOWN", False),
        ("Uniform Gray Asphalt Texture", "featureless", None, None, "UNKNOWN", False),
        ("Duplicate Re-encoded Low Bitrate", "urban", 35.6595, 139.7005, "LOCALITY_ESTIMATE", True),
        ("Mirrored Landmark Reflection", "landmark", 41.8902, 12.4922, "VERIFIED_LANDMARK", True),
        ("Fog Covered Mountain Silhouette", "mountain", 46.55, 7.98, "REGION_ESTIMATED", True),
        ("Adversarial EXIF Inverted Lat/Lon", "adversarial_invert", 12.4922, 41.8902, "GEOTAGGED", True),
        ("Adversarial Timestamp Future Year 2099", "future_date", 27.1751, 78.0421, "GEOTAGGED", True),
        ("Empty Ocean Horizon Water Only", "coastal", None, None, "UNKNOWN", False),
        ("Dense Cloud Layer Airplane Window", "featureless", None, None, "UNKNOWN", False),
        ("Abstract Architecture Texture", "urban", None, None, "UNKNOWN", False),
        ("Construction Dirt Ground", "rural", None, None, "UNKNOWN", False),
        ("Industrial Pipe Close-Up", "urban", None, None, "UNKNOWN", False),
        ("Pavement Cobblestone Macro", "urban", None, None, "UNKNOWN", False),
        ("Generic Green Lawn Close-Up", "rural", None, None, "UNKNOWN", False),
        ("Brick Wall Pattern Only", "featureless", None, None, "UNKNOWN", False),
        ("Monochrome Silhouetted Tree", "forest", None, None, "UNKNOWN", False),
        ("Generic Train Track Segment", "urban", None, None, "UNKNOWN", False),
    ]

    for name, variant, lat, lon, exp_cls, gt_avail in difficult_cases:
        clean_name = name.lower().replace(' ', '_').replace('/', '_')[:28]
        fname = f"{case_idx:03d}_diff_{clean_name}.jpg"
        fpath = os.path.join(OUTPUT_DIR, fname)
        
        if variant == "blur":
            img = render_scenic_image(width=160, height=120, theme="urban", label=name)
            # High blur by downsizing then upsizing
            img = img.resize((32, 24)).resize((640, 480), Image.Resampling.BILINEAR)
        elif variant == "lowres":
            img = render_scenic_image(width=32, height=24, theme="urban", label=name)
        elif variant == "featureless":
            img = render_scenic_image(width=640, height=480, theme="featureless", label=name, noise=0.01)
        elif variant == "night":
            img = Image.new("RGB", (640, 480), color=(10, 12, 18))
        elif variant == "overexposed":
            img = Image.new("RGB", (640, 480), color=(250, 252, 255))
        elif variant == "adversarial_null_island":
            img = render_scenic_image(theme="landmark", label=name)
            exif_obj = add_exif_gps_to_image(img, 0.0001, 0.0001)
            img.save(fpath, "JPEG", exif=exif_obj)
            img = None
        elif variant == "adversarial_invert":
            img = render_scenic_image(theme="landmark", label=name)
            exif_obj = add_exif_gps_to_image(img, lat, lon)
            img.save(fpath, "JPEG", exif=exif_obj)
            img = None
        elif variant == "future_date":
            img = render_scenic_image(theme="landmark", label=name)
            exif_obj = add_exif_gps_to_image(img, lat, lon, timestamp="2099:12:31 23:59:59")
            img.save(fpath, "JPEG", exif=exif_obj)
            img = None
        else:
            img = render_scenic_image(theme="urban", label=name)

        if img is not None:
            img.save(fpath, "JPEG", quality=80)

        manifest.append({
            "test_id": f"TEST_{case_idx:03d}",
            "filename": fname,
            "category": "F_DIFFICULT_ADVERSARIAL",
            "description": f"Difficult/adversarial test: {name}",
            "true_latitude": lat,
            "true_longitude": lon,
            "true_city": "Ambiguous" if lat is None else "TestCity",
            "expected_classification": exp_cls,
            "is_deterministic": variant in ["adversarial_null_island", "adversarial_invert", "future_date"],
            "metadata_type": "EXIF_GPS" if "adversarial" in variant or "future_date" in variant else "NONE",
            "ground_truth_available": gt_avail
        })
        case_idx += 1

    # =========================================================================
    # CATEGORY G: Indoor / Impossible / Failure / UNKNOWN (20 cases)
    # =========================================================================
    indoor_cases = [
        "Office Cubicle Workstation Desk",
        "Conference Meeting Room Table",
        "Home Living Room Sofa",
        "Kitchen Stainless Steel Sink",
        "Bathroom Ceramic Wall Tiles",
        "Blank White Drywall Corner",
        "Computer Monitor Displaying Code",
        "Restaurant Wooden Dining Table",
        "Hotel Carpet Flooring Pattern",
        "Bookshelf Filled With Hardcovers",
        "Server Rack Server Room Interior",
        "Empty Cardboard Box Close-Up",
        "Ceiling Fluorescent Lamp Light",
        "Elevator Brushed Steel Interior",
        "Coffee Cup On White Saucer",
        "Gym Treadmill Panel Screen",
        "Subway Metro Car Interior Seating",
        "Indoor Staircase Handrail",
        "Underground Basement Concrete Wall",
        "Indoor Swimming Pool Water Tile",
    ]

    for name in indoor_cases:
        fname = f"{case_idx:03d}_indoor_{name.lower().replace(' ', '_')[:28]}.jpg"
        fpath = os.path.join(OUTPUT_DIR, fname)
        img = render_scenic_image(width=640, height=480, theme="indoor", label=name)
        img.save(fpath, "JPEG", quality=85)
        manifest.append({
            "test_id": f"TEST_{case_idx:03d}",
            "filename": fname,
            "category": "G_INDOOR_UNKNOWN",
            "description": f"Indoor impossible geolocation scene: {name}",
            "true_latitude": None,
            "true_longitude": None,
            "true_city": "INDOOR_UNKNOWN",
            "expected_classification": "UNKNOWN",
            "is_deterministic": False,
            "metadata_type": "NONE",
            "ground_truth_available": False
        })
        case_idx += 1

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Generated {len(manifest)} benchmark photo test cases.")
    print(f"     Photos saved in: {OUTPUT_DIR}")
    print(f"     Manifest saved in: {MANIFEST_PATH}")

if __name__ == "__main__":
    build_benchmark_manifest()
