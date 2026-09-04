import os
from PIL import Image, ImageDraw

def generate_satellite_sample(filepath="sample_satellite.png", size=(512, 512)):
    # Create aerial scene canvas
    img = Image.new("RGB", size, (46, 139, 87)) # Base green terrain / vegetation
    draw = ImageDraw.Draw(img)

    # 1. Coastal Water Body (Left side / bay)
    water_points = [(0, 0), (160, 0), (140, 200), (180, 350), (130, 512), (0, 512)]
    draw.polygon(water_points, fill=(30, 144, 255))

    # 2. Sandy shoreline / beach buffer
    draw.line([(160, 0), (140, 200), (180, 350), (130, 512)], fill=(238, 214, 175), width=6)

    # 3. Main Asphalt Highway and intersecting roads
    draw.line([(0, 260), (512, 260)], fill=(60, 60, 60), width=16) # East-West highway
    draw.line([(0, 260), (512, 260)], fill=(255, 215, 0), width=2) # Center line
    draw.line([(320, 0), (320, 512)], fill=(70, 70, 70), width=12) # North-South road

    # 4. Urban Infrastructure & Building Clusters (Residential + Commercial rooftops)
    buildings = [
        # Top-Right Quadrant
        ([360, 40, 440, 100], (178, 34, 34)),   # Red tile roof
        ([450, 50, 490, 110], (165, 42, 42)),   # Brown warehouse
        ([350, 130, 420, 190], (128, 128, 128)), # Gray concrete commercial building
        ([430, 140, 480, 200], (205, 92, 92)),  # Terra cotta roof
        # Bottom-Right Quadrant
        ([350, 300, 430, 360], (139, 69, 19)),  # Dark wood roof
        ([440, 310, 490, 370], (178, 34, 34)),  # Red brick building
        ([360, 400, 460, 480], (112, 128, 144)), # Slate roof
    ]
    for bbox, col in buildings:
        draw.rectangle(bbox, fill=col, outline=(30, 30, 30), width=2)

    # 5. Agricultural parcel / Crop fields (Bottom-Left)
    draw.rectangle([190, 300, 300, 480], fill=(154, 205, 50), outline=(85, 107, 47), width=3)
    # Field crop lines
    for y in range(310, 480, 15):
        draw.line([(195, y), (295, y)], fill=(107, 142, 35), width=2)

    img.save(filepath, format="PNG")
    print(f"Saved satellite aerial sample to {filepath} ({size[0]}x{size[1]})")
    return filepath

if __name__ == "__main__":
    generate_satellite_sample()
