import os
import sys
from PIL import Image

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from grounding_adapters import GroundingDINOAdapter, RSAdaptedVisionGroundingAdapter
from tiling import tiled_inference_engine

img_path = os.path.join(os.path.dirname(_dir), "sample_data", "nexspace_test_image_001.jpg")
img = Image.open(img_path)
w, h = img.size
print(f"Loaded image {img_path}: size={w}x{h}")

print("\n--- 1. Grounding DINO on Full Image ---")
dino = GroundingDINOAdapter()
dino.load()
res_dino = dino.infer(img, "buildings", box_threshold=0.25)
dets_dino = res_dino.get("detections", [])
print(f"DINO direct detections count: {len(dets_dino)}")
for i, d in enumerate(dets_dino[:10]):
    print(f"  #{i+1}: score={d.get('score'):.4f}, box={d.get('box')}, label={d.get('label')}")

print("\n--- 2. Grounding DINO Tiled Inference ---")
tiled_dino = tiled_inference_engine.run_tiled_grounding(
    image=img,
    target_phrase="buildings",
    grounding_runtime=dino,
    tile_size=1024,
    overlap=0.20,
    iou_threshold=0.45,
    box_threshold=0.20,
)
dets_tiled_dino = tiled_dino.get("detections", [])
print(f"DINO tiled detections count: {len(dets_tiled_dino)}")
for i, d in enumerate(dets_tiled_dino[:10]):
    print(f"  #{i+1}: score={d.get('score'):.4f}, box={d.get('box')}, label={d.get('label')}")

print("\n--- 3. RS Adapted Adapter on Full Image ---")
rs = RSAdaptedVisionGroundingAdapter()
rs.load()
res_rs = rs.infer(img, "buildings", box_threshold=0.25)
dets_rs = res_rs.get("detections", [])
print(f"RS direct detections count: {len(dets_rs)}")
for i, d in enumerate(dets_rs[:10]):
    print(f"  #{i+1}: score={d.get('score')}, box={d.get('box')}, label={d.get('label')}")

print("\n--- 4. RS Adapted Adapter Tiled Inference ---")
tiled_rs = tiled_inference_engine.run_tiled_grounding(
    image=img,
    target_phrase="buildings",
    grounding_runtime=rs,
    tile_size=1024,
    overlap=0.20,
    iou_threshold=0.45,
    box_threshold=0.20,
)
dets_tiled_rs = tiled_rs.get("detections", [])
print(f"RS tiled detections count: {len(dets_tiled_rs)}")
for i, d in enumerate(dets_tiled_rs[:10]):
    print(f"  #{i+1}: score={d.get('score')}, box={d.get('box')}, label={d.get('label')}")
