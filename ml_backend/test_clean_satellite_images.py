import os
import sys
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from tools import GroundingTool

gt = GroundingTool()

print("\n=======================================================")
print("  CLEAN UNANNOTATED IMAGE 1: satellite_urban.jpg")
print("=======================================================")
urban_path = os.path.join(os.path.dirname(_dir), "sample_data", "satellite_urban.jpg")
if os.path.exists(urban_path):
    img_urban = Image.open(urban_path)
    w, h = img_urban.size
    print(f"Loaded satellite_urban.jpg ({w}x{h})")
    res_urban = gt.execute({"image": img_urban, "target_phrase": "buildings"})
    print(f"Status: {res_urban.status}, Detections: {len(res_urban.evidence)}")
    for i, ev in enumerate(res_urban.evidence):
        print(f"  #{i+1:02d}: label='{ev.get('label')}', score={ev.get('score')}, box={ev.get('box')}")

print("\n=======================================================")
print("  CLEAN UNANNOTATED IMAGE 2: satellite_coast.jpg")
print("=======================================================")
coast_path = os.path.join(os.path.dirname(_dir), "sample_data", "satellite_coast.jpg")
if os.path.exists(coast_path):
    img_coast = Image.open(coast_path)
    w, h = img_coast.size
    print(f"Loaded satellite_coast.jpg ({w}x{h})")
    res_coast = gt.execute({"image": img_coast, "target_phrase": "water"})
    print(f"Status: {res_coast.status}, Detections: {len(res_coast.evidence)}")
    for i, ev in enumerate(res_coast.evidence):
        print(f"  #{i+1:02d}: label='{ev.get('label')}', score={ev.get('score')}, box={ev.get('box')}")
