"""
Real-model smoke test. Run this ONCE after `python download_models.py`:

    python smoke_test.py

It loads the actual vision model, asks questions about the sample photos (including the food-safety
cases), prints every answer with its timing, and checks that the safety layer reacted correctly.
The first question is slower (model warm-up).
"""

import os
import sys
import time

from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from multimodal.multimodal_vlm import PixelTalkMultimodalVLM  # noqa: E402

# (image, questions, expected safety alert for the LAST question or None)
PLAN = [
    ("golden_retriever_park.jpg", ["What breed is the dog?", "What is next to it on the grass?"], None),
    ("street_crosswalk_city.jpg", ["What color is the taxi?", "Is anyone carrying a backpack?"], None),
    ("red_apple_kitchen.jpg", ["Is it safe to eat?"], "FOOD_ADVISORY"),
    ("moldy_bread_plate.jpg", ["Is it safe to eat?"], "HAZARD_BLOCKED"),
    ("wild_mushroom_forest.jpg", ["Is it safe to eat?"], "HAZARD_BLOCKED"),
    ("laptop_desk.jpg", ["Is it safe to eat?"], "HAZARD_BLOCKED"),
]


def main() -> int:
    print("Loading the vision model (first run may take a few minutes)...")
    t0 = time.time()
    vlm = PixelTalkMultimodalVLM()
    if not vlm.is_ready:
        print("\nFAILED to load a vision model:")
        for note in vlm.load_notes:
            print("  -", note)
        return 1
    print(f"Loaded: {vlm.engine_label}  ({time.time() - t0:.0f}s)")
    for note in vlm.load_notes:
        print("  NOTE:", note)
    print()

    problems = 0
    for fname, questions, expected_alert in PLAN:
        vlm.set_image(Image.open(os.path.join(ROOT, "sample_images", fname)), image_id=fname)
        print(f"--- {fname}")
        for i, q in enumerate(questions):
            started, first = time.time(), None
            pieces = []
            for chunk in vlm.answer_stream(q, temperature=0.0):
                if first is None:
                    first = time.time() - started
                pieces.append(chunk)
            res = vlm.last_result or {}
            answer = (res.get("answer") or "".join(pieces)).replace("\n", " ")
            print(f"  Q: {q}\n  A: {answer}\n     (first words after {first or 0:.1f}s, total {res.get('latency_s')}s, "
                  f"alert={res.get('safety_alert')})")
            if not answer.strip() or "I hit an error" in answer or "isn't available" in answer:
                print("  !! no usable answer")
                problems += 1
            if expected_alert and i == len(questions) - 1 and res.get("safety_alert") != expected_alert:
                print(f"  !! expected safety alert {expected_alert}, got {res.get('safety_alert')}")
                problems += 1
    print("\n" + ("All checks passed." if not problems else f"{problems} problem(s) found - see '!!' lines above."))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
