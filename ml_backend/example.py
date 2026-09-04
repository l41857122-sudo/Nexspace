"""
example.py
----------
Runnable script demonstrating the upgraded GeoVLMController across 4 primary scenarios:
  1. Open-ended optical query
  2. Low-confidence counting query
  3. Joint Optical + SAR multi-modal fusion
  4. Bi-temporal change analysis pair
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image, ImageDraw
from orchestrator import GeoVLMController


def create_dummy_image(color=(100, 150, 200), size=(224, 224), draw_pattern=False) -> Image.Image:
    img = Image.new("RGB", size, color)
    if draw_pattern:
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill=(250, 80, 80))
        draw.ellipse([80, 80, 120, 120], fill=(255, 255, 255))
    return img


def main():
    controller = GeoVLMController()

    img_optical = create_dummy_image((120, 180, 120))
    img_sar = create_dummy_image((80, 80, 80))
    img_before = create_dummy_image((100, 100, 100))
    img_after = create_dummy_image((100, 100, 100), draw_pattern=True)

    print("==================================================================")
    print("SCENARIO 1: Open-ended Query")
    print("==================================================================")
    res1 = controller.handle_request(
        query="What is visible in this image? Is there a river?",
        optical_image=img_optical
    )
    print("Task Type:", res1["task_type"])
    print("Confidence:", res1["confidence"])
    print("Selected Tools:", res1["selected_tools"])
    print("\nSynthesized Response:\n", res1["response_text"])

    print("\n==================================================================")
    print("SCENARIO 2: Counting Query")
    print("==================================================================")
    res2 = controller.handle_request(
        query="How many residential buildings are in this area?",
        optical_image=img_optical
    )
    print("Task Type:", res2["task_type"])
    print("Confidence:", res2["confidence"])
    print("Requires Count Warning:", res2["routing_decision"]["requires_count_warning"])
    print("\nSynthesized Response:\n", res2["response_text"])

    print("\n==================================================================")
    print("SCENARIO 3: Joint Optical + SAR Fusion")
    print("==================================================================")
    res3 = controller.handle_request(
        query="Describe this scene using both sensors",
        optical_image=img_optical,
        sar_image=img_sar
    )
    print("Task Type:", res3["task_type"])
    print("Confidence:", res3["confidence"])
    print("Selected Tools:", res3["selected_tools"])
    print("\nSynthesized Response:\n", res3["response_text"])

    print("\n==================================================================")
    print("SCENARIO 4: Bi-temporal Change Analysis")
    print("==================================================================")
    res4 = controller.handle_request(
        query="What changed between these dates?",
        change_image_a=img_before,
        change_image_b=img_after
    )
    print("Task Type:", res4["task_type"])
    print("Confidence:", res4["confidence"])
    print("Change Summary:", res4["change_analysis"]["summary"])
    print("\nSynthesized Response:\n", res4["response_text"])


if __name__ == "__main__":
    main()
