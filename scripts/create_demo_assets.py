"""
scripts/create_demo_assets.py
-----------------------------
Generates and populates valid, high-quality remote-sensing satellite rasters for public/demo/
and sample_data/ ensuring proper PNG/TIFF formatting, dimensions, and MIME types.
"""

import os
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "demo")
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
ML_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_backend")

os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)


def create_urban_raster() -> Image.Image:
    """Create a 512x512 realistic optical satellite image showing urban grid, roads, and rooftops."""
    # Use existing sample_satellite if available or generate rich synthetic satellite image
    src_sat = os.path.join(ML_DIR, "sample_satellite.png")
    if os.path.exists(src_sat):
        try:
            img = Image.open(src_sat).convert("RGB")
            if img.size == (512, 512):
                return img
        except Exception:
            pass

    # Create rich satellite texture
    np.random.seed(42)
    base = np.zeros((512, 512, 3), dtype=np.uint8)
    base[:, :] = [70, 85, 60]  # Vegetation background
    # Add texture noise
    noise = np.random.randint(-15, 15, (512, 512, 3), dtype=np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    # Draw asphalt road grid
    roads = [
        [(0, 120), (512, 120)],
        [(0, 260), (512, 260)],
        [(0, 400), (512, 400)],
        [(150, 0), (150, 512)],
        [(330, 0), (330, 512)],
    ]
    for r in roads:
        draw.line(r, fill=(55, 55, 60), width=18)
        draw.line(r, fill=(210, 190, 70), width=2)  # centerline

    # Draw residential and commercial buildings / rooftops
    buildings = [
        # Block 1 (Top Left)
        (25, 25, 70, 75, (160, 80, 60)),
        (85, 30, 130, 80, (180, 100, 70)),
        (30, 140, 75, 195, (190, 140, 110)),
        (85, 145, 135, 200, (150, 90, 70)),
        # Block 2 (Top Mid)
        (170, 25, 230, 90, (140, 150, 160)),
        (245, 30, 310, 85, (160, 170, 180)),
        (175, 140, 240, 220, (190, 90, 70)),
        (255, 145, 315, 225, (180, 100, 80)),
        # Block 3 (Top Right)
        (355, 25, 420, 85, (130, 80, 60)),
        (435, 30, 490, 90, (170, 110, 80)),
        (360, 140, 430, 210, (160, 160, 170)),
        (440, 145, 495, 215, (150, 95, 75)),
        # Lower Blocks
        (30, 290, 120, 360, (175, 105, 80)),
        (175, 290, 280, 370, (165, 175, 185)),
        (355, 290, 480, 370, (180, 95, 70)),
        (30, 430, 130, 490, (155, 85, 65)),
        (175, 430, 310, 495, (170, 115, 85)),
        (355, 430, 490, 495, (160, 165, 175)),
    ]
    for x1, y1, x2, y2, color in buildings:
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(40, 40, 40), width=2)
        # Inner rooftop detail
        draw.rectangle([x1 + 4, y1 + 4, x2 - 4, y2 - 4], fill=tuple(min(255, c + 20) for c in color))

    return img


def create_port_raster() -> Image.Image:
    """Create a 512x512 optical satellite image showing a coastal port with docks, water, and ships."""
    img = Image.new("RGB", (512, 512), (30, 70, 120))  # Deep water base
    draw = ImageDraw.Draw(img)

    # Land mass on left/top
    land_poly = [(0, 0), (220, 0), (280, 200), (240, 350), (180, 512), (0, 512)]
    draw.polygon(land_poly, fill=(80, 95, 70))

    # Concrete pier / docks extending into water
    draw.rectangle([180, 100, 420, 140], fill=(160, 160, 165), outline=(90, 90, 95), width=2)
    draw.rectangle([160, 260, 380, 300], fill=(155, 155, 160), outline=(90, 90, 95), width=2)
    draw.rectangle([140, 400, 360, 435], fill=(150, 150, 155), outline=(90, 90, 95), width=2)

    # Cargo ships docked
    ships = [
        (300, 65, 400, 95, (200, 60, 50)),    # Red cargo vessel
        (260, 145, 370, 175, (40, 40, 50)),   # Dark container ship
        (240, 305, 340, 330, (210, 210, 220)), # White freighter
        (380, 360, 450, 385, (190, 130, 40)),  # Tug / supply boat
    ]
    for x1, y1, x2, y2, color in ships:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=6, fill=color, outline=(20, 20, 20), width=2)

    # Port storage warehouses on land
    warehouses = [
        (30, 40, 120, 90, (180, 185, 190)),
        (30, 120, 110, 170, (190, 110, 70)),
        (40, 240, 130, 310, (170, 175, 180)),
        (30, 360, 120, 430, (160, 100, 60)),
    ]
    for x1, y1, x2, y2, color in warehouses:
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(40, 40, 45), width=2)

    return img


def create_sar_raster() -> Image.Image:
    """Create a 512x512 high-contrast SAR radar backscatter raster (grayscale)."""
    src_sar = os.path.join(ML_DIR, "sample_sar.png")
    if os.path.exists(src_sar):
        try:
            img = Image.open(src_sar).convert("L")
            if img.size == (512, 512):
                return img
        except Exception:
            pass

    np.random.seed(99)
    # Speckle noise base (Rayleigh-like distribution)
    speckle = np.random.gamma(shape=2.0, scale=20.0, size=(512, 512)).astype(np.uint8)
    img = Image.fromarray(speckle, mode="L")
    draw = ImageDraw.Draw(img)

    # Smooth water has very low specular return (dark SAR)
    draw.rectangle([250, 0, 512, 512], fill=15)

    # Corner reflectors and metallic ship structures have very strong double-bounce backscatter (bright white)
    high_returns = [
        (300, 65, 400, 95),
        (260, 145, 370, 175),
        (240, 305, 340, 330),
        (180, 100, 420, 140),
        (160, 260, 380, 300),
        (40, 40, 120, 90),
        (30, 120, 110, 170),
    ]
    for box in high_returns:
        draw.rectangle(box, fill=240, outline=255, width=2)

    return img


def create_bitemporal_pair():
    """Create a before/after optical satellite pair showing clear construction/land clearing."""
    img_a = create_urban_raster().copy()
    img_b = img_a.copy()
    draw_b = ImageDraw.Draw(img_b)

    # Before (A): Open green parcel at (175, 290, 280, 370)
    # After (B): New large industrial facility built over the parcel
    draw_b.rectangle([170, 285, 290, 380], fill=(225, 230, 235), outline=(30, 30, 35), width=3)
    draw_b.rectangle([185, 295, 275, 340], fill=(45, 120, 210), outline=(20, 40, 80), width=2) # Solar rooftop
    draw_b.rectangle([185, 350, 235, 370], fill=(190, 80, 60), outline=(20, 20, 20), width=1)   # Loading bay

    # New parking / paved road connection
    draw_b.line([(290, 330), (330, 330)], fill=(60, 60, 65), width=12)

    return img_a, img_b


def create_vegetation_raster() -> Image.Image:
    """Create a 512x512 dense forest and agricultural remote sensing raster."""
    np.random.seed(123)
    base = np.zeros((512, 512, 3), dtype=np.uint8)
    base[:, :] = [35, 95, 45]  # Dense forest green
    noise = np.random.randint(-20, 20, (512, 512, 3), dtype=np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    # Agricultural crop parcels (geometric fields)
    draw.rectangle([50, 50, 200, 220], fill=(185, 175, 90), outline=(25, 70, 30), width=3)
    draw.rectangle([220, 60, 460, 210], fill=(130, 160, 70), outline=(25, 70, 30), width=3)
    draw.rectangle([60, 260, 240, 460], fill=(95, 140, 60), outline=(25, 70, 30), width=3)
    draw.rectangle([260, 250, 450, 450], fill=(165, 150, 80), outline=(25, 70, 30), width=3)

    # River meandering through fields
    river_points = [(0, 230), (120, 240), (250, 235), (380, 245), (512, 240)]
    for i in range(len(river_points) - 1):
        draw.line([river_points[i], river_points[i + 1]], fill=(30, 65, 110), width=16)

    return img


def main():
    print("=== POPULATING REAL REMOTE SENSING IMAGE ASSETS (PNG + JPG) ===")

    port_img = create_port_raster()
    urban_img = create_urban_raster()
    sar_img = create_sar_raster()
    veg_img = create_vegetation_raster()
    time_a, time_b = create_bitemporal_pair()

    # Save PNG formats
    png_assets = {
        os.path.join(DEMO_DIR, "port.png"): port_img,
        os.path.join(DEMO_DIR, "urban.png"): urban_img,
        os.path.join(DEMO_DIR, "sar.png"): sar_img,
        os.path.join(DEMO_DIR, "vegetation.png"): veg_img,
        os.path.join(DEMO_DIR, "bitemporal_a.png"): time_a,
        os.path.join(DEMO_DIR, "bitemporal_b.png"): time_b,
        os.path.join(SAMPLE_DIR, "BiTemporal_Before_20231015.png"): time_a,
        os.path.join(SAMPLE_DIR, "BiTemporal_After_20231025.png"): time_b,
    }

    for path, img in png_assets.items():
        img.save(path, format="PNG")
        size = os.path.getsize(path)
        print(f"[OK] Saved PNG: {os.path.basename(path)} -> {size:,} bytes")

    # Save JPG formats for full multi-format upload validation
    jpg_assets = {
        os.path.join(DEMO_DIR, "urban_buildings.jpg"): urban_img,
        os.path.join(DEMO_DIR, "water_coast.jpg"): port_img,
        os.path.join(DEMO_DIR, "vegetation_forest.jpg"): veg_img,
        os.path.join(SAMPLE_DIR, "satellite_urban.jpg"): urban_img,
        os.path.join(SAMPLE_DIR, "satellite_coast.jpg"): port_img,
    }

    for path, img in jpg_assets.items():
        img.save(path, format="JPEG", quality=92)
        size = os.path.getsize(path)
        print(f"[OK] Saved JPG: {os.path.basename(path)} -> {size:,} bytes")

    # Generate synthetic GeoTIFF files if rasterio is available
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        tif_path_b04 = os.path.join(SAMPLE_DIR, "Sentinel2_B04_Red_10m.tif")
        tif_path_b08 = os.path.join(SAMPLE_DIR, "Sentinel2_B08_NIR_10m.tif")

        arr_b04 = np.array(urban_img.convert("L"), dtype=np.uint8)
        arr_b08 = np.array(sar_img.convert("L"), dtype=np.uint8)

        # Bounding box: San Francisco Bay (EPSG:32610 - UTM Zone 10N)
        transform = from_bounds(548000.0, 4178000.0, 553120.0, 4183120.0, 512, 512)
        crs = CRS.from_epsg(32610)

        for path, arr in [(tif_path_b04, arr_b04), (tif_path_b08, arr_b08)]:
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=512,
                width=512,
                count=1,
                dtype=arr.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(arr, 1)
            size = os.path.getsize(path)
            print(f"[OK] Saved GeoTIFF: {os.path.basename(path)} (EPSG:32610) -> {size:,} bytes")

    except Exception as e:
        print(f"[INFO] Skipping GeoTIFF generation: {e}")


if __name__ == "__main__":
    main()
