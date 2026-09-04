import sys
import time
from PIL import Image, ImageDraw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
print(f"Loading {model_id}...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

img = Image.new("RGB", (300, 300), color=(34, 139, 34))
d = ImageDraw.Draw(img)
d.rectangle([50, 50, 150, 150], fill=(220, 20, 60)) # Red building / box
d.rectangle([180, 180, 260, 260], fill=(70, 130, 180)) # Blue roof building

text = "building. roof."
inputs = processor(images=img, text=text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.25,
    text_threshold=0.25,
    target_sizes=[img.size[::-1]],
)

print("\n--- RESULTS FROM REAL GROUNDING DINO ---")
print("Scores:", results[0]["scores"].tolist())
print("Boxes (pixel coordinates [x1, y1, x2, y2]):", results[0]["boxes"].tolist())
print("Labels:", results[0]["labels"])
