"""
test_group1_cases.py
--------------------
Evaluates the exact Group 1 test cases on NexSpace Test Image #001:
  A. "Is there a body of water?" -> Expected: VQA only
  B. "Are there bridges?" -> Expected: VQA only
  C. "Is vegetation present?" -> Expected: VQA only
  D. "How many boats are visible?" -> Expected: Grounding + counting path
  E. "Where is the bridge?" -> Expected: Grounding/spatial path
  F. "Describe the scene." -> Expected: Caption/scene path
"""

import os
import sys
import time
import json
from PIL import Image

# Set stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from orchestrator import GeoVLMController
from router import IntentClassifier
from synthesis import _pluralize_phrase

TEST_IMAGE_PATH = os.path.join(current_dir, "..", "sample_data", "nexspace_test_image_001.jpg")


def test_pluralization():
    print("======================================================================")
    print("PLURALIZATION TESTS")
    print("======================================================================")
    cases = [
        ("body of water", "bodies of water"),
        ("piece of land", "pieces of land"),
        ("patch of vegetation", "patches of vegetation"),
        ("building", "buildings"),
        ("bridge", "bridges"),
        ("boat", "boats"),
        ("ship", "ships"),
        ("road", "roads"),
        ("highway", "highways"),
        ("factory", "factories"),
        ("city", "cities"),
        ("vegetation", "vegetation"),
        ("water", "water"),
        ("farmland", "farmland"),
        ("structures", "structures"),
    ]
    for inp, expected in cases:
        actual = _pluralize_phrase(inp)
        match = (actual == expected)
        print(f"  '{inp}' -> '{actual}' | Expected: '{expected}' | {'✓ PASS' if match else '✗ FAIL'}")
        assert match, f"Mismatch: {inp} -> {actual} != {expected}"
    print("All pluralization tests passed!\n")


def test_group1_cases():
    print("======================================================================")
    print("GROUP 1 EXACT TEST CASES (ON NEXSPACE TEST IMAGE #001)")
    print("======================================================================")

    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"Test image not found at {TEST_IMAGE_PATH}")

    img = Image.open(TEST_IMAGE_PATH).convert("RGB")
    controller = GeoVLMController()

    test_queries = [
        ("A", "Is there a body of water?", ["VQA"]),
        ("B", "Are there bridges?", ["VQA"]),
        ("C", "Is vegetation present?", ["VQA"]),
        ("D", "How many boats are visible?", ["Grounding"]),
        ("E", "Where is the bridge?", ["Grounding"]),
        ("F", "Describe the scene.", ["Optical_Caption"]),
    ]

    results = []

    for case_id, query, expected_tools in test_queries:
        print(f"\n--- CASE {case_id}: \"{query}\" ---")
        t0 = time.perf_counter()
        res = controller.handle_request(query=query, optical_image=img)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        plan = res.get("investigation_report", {}).get("plan", {})
        selected_tools = plan.get("selected_tools", [])
        tool_results = res.get("results", [])

        # Find models used
        models_used = []
        for r in tool_results:
            meta = r.get("model_metadata", {})
            src = meta.get("model_id") or meta.get("method") or r.get("confidence_source") or r.get("tool_name")
            models_used.append(f"{r.get('tool_name')}: {src}")

        models_str = ", ".join(models_used) if models_used else "None"
        answer = res.get("response_text", "").strip()
        conf = res.get("confidence")
        conf_type = res.get("confidence_type")
        conf_source = res.get("confidence_source")
        provenance = f"{conf} ({conf_type}, source: {conf_source})"

        print(f"  • Selected Tools:     {selected_tools}")
        print(f"  • Expected Tools:     {expected_tools}")
        print(f"  • Actual Model Used:  {models_str}")
        print(f"  • Latency:            {elapsed_ms:.2f} ms")
        print(f"  • Confidence Prov.:   {provenance}")
        print(f"  • Answer:\n    {answer}")

        # Assert tool selection
        assert selected_tools == expected_tools, f"Tool mismatch for '{query}': {selected_tools} != {expected_tools}"

        results.append({
            "case_id": case_id,
            "query": query,
            "selected_tools": selected_tools,
            "expected_tools": expected_tools,
            "models_used": models_used,
            "latency_ms": elapsed_ms,
            "confidence": conf,
            "confidence_type": conf_type,
            "confidence_source": conf_source,
            "answer": answer,
        })

    # Save to scratch
    out_file = os.path.join(current_dir, "group1_verification_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    test_pluralization()
    test_group1_cases()
