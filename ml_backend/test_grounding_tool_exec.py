import os
import sys
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from tools import GroundingTool

img_path = os.path.join(os.path.dirname(_dir), "sample_data", "nexspace_test_image_001.jpg")
img = Image.open(img_path)

gt = GroundingTool()
res = gt.execute({"image": img, "target_phrase": "Where are the buildings?"})
print(f"GroundingTool status: {res.status}")
print(f"Evidence count: {len(res.evidence)}")
for i, ev in enumerate(res.evidence[:10]):
    print(f"  #{i+1:02d}: label='{ev.get('label')}', score={ev.get('score')}, box={ev.get('box')}")
