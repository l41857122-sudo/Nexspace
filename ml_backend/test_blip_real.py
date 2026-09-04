import sys, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print('Testing BlipProcessor + BlipForConditionalGeneration...')
t0 = time.perf_counter()
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
t_load = (time.perf_counter() - t0) * 1000.0
print(f'BLIP model loaded successfully in {t_load:.2f}ms!')

img = Image.new('RGB', (100, 100), color=(34, 139, 34))
t1 = time.perf_counter()
inputs = processor(img, return_tensors='pt')
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True).strip()
t_inf = (time.perf_counter() - t1) * 1000.0
print(f'GENUINE REAL INFERENCE CAPTION: "{caption}"')
print(f'Inference time: {t_inf:.2f}ms')
