import urllib.request
import json
import base64
import os

img_path = os.path.join("sample_data", "nexspace_test_image_001.jpg")
with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
data_url = f"data:image/jpeg;base64,{b64}"

req_data = {
    "query": "Where are the buildings?",
    "image_data": data_url,
    "session_id": "test_bbox_session"
}
req = urllib.request.Request(
    "http://localhost:8000/api/query",
    data=json.dumps(req_data).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req)
res_json = json.loads(resp.read().decode("utf-8"))
grounding = res_json.get("grounding", {})
dets = grounding.get("detections", [])
print(f"Total live API detections: {len(dets)}")
for i, d in enumerate(dets):
    score = d.get("score")
    score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
    print(f"  #{i+1:02d}: label='{d.get('label')}', score={score_str}, box={d.get('box')}")
