import sys
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load optical sample
opt = Image.open("sample_satellite.png")
w, h = opt.size
opt_arr = np.array(opt)

# Derive realistic SAR backscatter simulation based on remote sensing radar physics:
# 1. Base speckle noise (Rayleigh/Gamma distributed)
np.random.seed(42)
speckle = np.random.gamma(shape=4.0, scale=0.25, size=(h, w))

sar_base = np.zeros((h, w), dtype=np.float32)

# Classify regions from optical RGB:
# Blue/Water: R < 70, G < 140, B > 120 -> low backscatter (dark specular reflection)
is_water = (opt_arr[:, :, 2] > 120) & (opt_arr[:, :, 0] < 80)
# Gray/Roads/Asphalt: low color saturation, R ~ G ~ B
is_road = (np.abs(opt_arr[:, :, 0].astype(int) - opt_arr[:, :, 1].astype(int)) < 15) & (opt_arr[:, :, 0] < 120) & ~is_water
# Red/Roofs/Buildings: R > 150 & G < 100
is_building = (opt_arr[:, :, 0] > 140) & (opt_arr[:, :, 1] < 100)
# Vegetation: G > 100 & G > R
is_vegetation = (opt_arr[:, :, 1] > 100) & (opt_arr[:, :, 1] > opt_arr[:, :, 0]) & ~is_water

# Assign radar backscatter values:
sar_base[is_water] = 20.0       # Water: -22 dB equivalent
sar_base[is_road] = 45.0        # Smooth road: -16 dB equivalent
sar_base[is_vegetation] = 95.0  # Volume scatter vegetation: -10 dB equivalent
sar_base[is_building] = 220.0   # Double bounce corner reflector: +2 dB equivalent

# Fill unassigned pixels
unassigned = sar_base == 0.0
sar_base[unassigned] = 80.0

# Multiply by multiplicative radar speckle noise
sar_intensity = np.clip(sar_base * speckle, 0.0, 255.0).astype(np.uint8)

sar_img = Image.fromarray(sar_intensity, mode="L")
sar_img.save("sample_sar.png")
print("Saved realistic paired sample_sar.png (512x512)!")
