import os
from PIL import Image, ImageDraw, ImageFont

art_dir = r"C:\Users\USER\.gemini\antigravity-ide\brain\fecdc26f-de45-4892-b1b2-81ebb8c1f182"
local_dir = r"C:\Users\USER\Desktop\app\Nexspace\ml_backend\visual_proof_artifacts"

stages = [
    ("1. QUERY TERMINAL", os.path.join(art_dir, "01_query_multitask.png")),
    ("2. SCAN RESULTS (11 BUILDINGS + SCENE)", os.path.join(art_dir, "02_scan_results_multitask.png")),
    ("3. EVIDENCE VIEWER (CANONICAL NODES)", os.path.join(art_dir, "03_evidence_viewer_multitask.png")),
    ("4. EXECUTION TRACE (DECOMPOSED MULTI-TASK)", os.path.join(art_dir, "04_execution_trace_multitask.png")),
    ("5. REPORT GENERATOR (ACTIVE MULTI-TASK REPORT)", os.path.join(art_dir, "05_report_multitask.png")),
    ("6. DASHBOARD & AUDIT HISTORY", os.path.join(art_dir, "06_dashboard_multitask.png")),
]

cols = 2
rows = 3
card_w = 1200
card_h = 750
pad = 30
header_h = 110

total_w = cols * card_w + (cols + 1) * pad
total_h = rows * card_h + (rows + 1) * pad + header_h

board = Image.new("RGB", (total_w, total_h), (10, 18, 30))
draw = ImageDraw.Draw(board)

title_text = "NEXSPACE — REAL USER END-TO-END CANONICAL INVESTIGATION PROOF"
sub_text = 'Exact Query: "Describe this image and locate the buildings"  |  Multi-Task: Scene Caption + 11 Candidate Buildings  |  Live UI Verification'

try:
    font_title = ImageFont.truetype("arial.ttf", 34)
    font_sub = ImageFont.truetype("arial.ttf", 20)
    font_label = ImageFont.truetype("arial.ttf", 22)
except Exception:
    font_title = ImageFont.load_default()
    font_sub = font_title
    font_label = font_title

draw.text((pad + 10, 25), title_text, fill=(6, 182, 212), font=font_title)
draw.text((pad + 10, 70), sub_text, fill=(148, 163, 184), font=font_sub)

for idx, (label, img_path) in enumerate(stages):
    c = idx % cols
    r = idx // cols
    x = pad + c * (card_w + pad)
    y = header_h + pad + r * (card_h + pad)

    draw.rectangle([x, y, x + card_w, y + card_h], fill=(13, 24, 38), outline=(30, 41, 59), width=2)
    draw.rectangle([x, y, x + card_w, y + 45], fill=(15, 30, 48))
    draw.text((x + 15, y + 10), label, fill=(56, 189, 248), font=font_label)

    if os.path.exists(img_path):
        img = Image.open(img_path)
        avail_w = card_w - 20
        avail_h = card_h - 65
        img.thumbnail((avail_w, avail_h), Image.Resampling.LANCZOS)
        ix = x + 10 + (avail_w - img.width) // 2
        iy = y + 55 + (avail_h - img.height) // 2
        board.paste(img, (ix, iy))

out_path = os.path.join(art_dir, "NEXSPACE_REAL_USER_END_TO_END_PROOF.png")
board.save(out_path, quality=95)
local_out = os.path.join(local_dir, "NEXSPACE_REAL_USER_END_TO_END_PROOF.png")
board.save(local_out, quality=95)
print("SUCCESS: NEXSPACE_REAL_USER_END_TO_END_PROOF.png saved at:", out_path)

