import os
import sys
from PIL import Image
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from grounding_adapters import GroundingDINOAdapter

img_path = os.path.join(os.path.dirname(_dir), "sample_data", "nexspace_test_image_001.jpg")
img = Image.open(img_path)
w, h = img.size

dino = GroundingDINOAdapter()
dino.load()
res = dino.infer(img, "buildings", box_threshold=0.20)
dets = res.get("detections", [])
print(f"Total DINO detections for 'buildings': {len(dets)}")
for i, d in enumerate(dets):
    print(f"  #{i+1}: score={d['score']:.4f}, box={d['box']}, label={d['label']}")
