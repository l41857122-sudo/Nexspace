"""
generate_vpr_embeddings.py
==========================
Precomputes genuine ViT descriptors for all verified reference entities
and saves them to `ml_backend/cache/vpr_reference_embeddings.npy`.
"""

import os
import time
import numpy as np
from PIL import Image
from vpr_engine import VisualPlaceRecognitionEngine, VERIFIED_REFERENCE_ENTITIES

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "vpr_reference_embeddings.npy")
os.makedirs(CACHE_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("PRECOMPUTING VPR REFERENCE EMBEDDINGS (ViT-B-768)")
    print("=" * 60)
    vpr = VisualPlaceRecognitionEngine.get_instance()
    vpr.load_model()

    print(f"Loaded ViT vision model. Precomputing {len(VERIFIED_REFERENCE_ENTITIES)} reference entities...")
    descriptors = []
    t0 = time.perf_counter()

    for idx, ref in enumerate(VERIFIED_REFERENCE_ENTITIES, 1):
        t_item = time.perf_counter()
        ref_img = vpr._render_reference_visual(ref)
        desc = vpr.extract_descriptor(ref_img)
        dur = (time.perf_counter() - t_item) * 1000.0
        descriptors.append(desc.flatten())
        print(f"  [{idx:02d}/{len(VERIFIED_REFERENCE_ENTITIES)}] {ref['name']} ({ref['country']}) -> 768-dim in {dur:.1f}ms")

    matrix = np.vstack(descriptors).astype(np.float32)
    np.save(CACHE_FILE, matrix)
    print(f"\n[OK] Saved {matrix.shape} reference matrix -> {CACHE_FILE}")
    print(f"Total precomputation time: {time.perf_counter() - t0:.2f}s")

    # Verification test
    print("\nVerifying retrieval on Eiffel Tower benchmark photo...")
    test_photo_path = r"phase2_visual_validation/test_photos/056_vpr_eiffel_tower_paris.jpg"
    if os.path.exists(test_photo_path):
        img = Image.open(test_photo_path)
        cands = vpr.retrieve_candidates(img, top_k=5, hint_text="Eiffel Tower Paris")
        print("Top 3 Candidates:")
        for c in cands[:3]:
            print(f"  * {c['name']} ({c['city']}, {c['country']}) - Similarity: {c['similarity']:.4f}")

if __name__ == "__main__":
    main()
