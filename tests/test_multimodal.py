"""
Tests for the multimodal pipeline. They need NO model weights and NO torch:
the vision model is replaced by a fake engine, so they run in a second.

    python -m pytest tests/test_multimodal.py -q
"""

import io
import os
import sys

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from multimodal.evaluate_comparison import (  # noqa: E402
    CLAIMS_SAFE, EVAL_SET, NOT_FOOD, SAFE_ADVISORY, is_correct,
)
from multimodal.multimodal_vlm import PixelTalkMultimodalVLM  # noqa: E402
from multimodal.safety_guardrail import MultimodalSafetyGuardrail  # noqa: E402
from multimodal.vlm_engines import build_chat_messages, choose_dtype_name, prepare_image  # noqa: E402


class FakeEngine:
    name = "qwen"
    label = "fake"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, image, history, question, system, temperature=0.2, max_new_tokens=160):
        self.calls.append({"history": list(history), "question": question})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def img():
    return Image.new("RGB", (100, 80), "red")


def make(replies, **kw):
    engine = FakeEngine(replies)
    chat = PixelTalkMultimodalVLM(engine=engine, load=False, **kw)
    return chat, engine


# ------------------------------------------------------------------ guardrail
@pytest.mark.parametrize("caption,subject", [
    ("A plate of carrots and peas", "carrots"),          # 'car' must not match inside 'carrots'
    ("A pizza on a wooden table", "pizza"),               # 'wood' must not match inside 'wooden'
    ("A bowl of fresh strawberries", "strawberries"),
])
def test_ordinary_food_is_allowed_with_advisory(caption, subject):
    ok, msg = MultimodalSafetyGuardrail().verify_safety("Is it safe to eat?", {"caption": caption}, subject)
    assert ok and "Food Safety Advisory" in msg


@pytest.mark.parametrize("subject", ["laptop", "galaxy", "a dog"])
def test_non_food_is_refused(subject):
    ok, msg = MultimodalSafetyGuardrail().verify_safety("is it safe to eat?", {}, subject)
    assert not ok and "Safety Guardrail Alert" in msg


@pytest.mark.parametrize("question,subject", [
    ("Is this mushroom safe to eat?", "mushroom"),
    ("can I eat these wild berries?", "berries"),
    ("is it edible?", "toadstool"),
])
def test_wild_plants_are_refused(question, subject):
    ok, msg = MultimodalSafetyGuardrail().verify_safety(question, {}, subject)
    assert not ok and "Foraging" in msg


def test_unknown_subject_is_cautious_not_permissive():
    ok, msg = MultimodalSafetyGuardrail().verify_safety("is it safe to eat?", {}, "")
    assert not ok and "can't tell" in msg


def test_non_consumption_questions_are_untouched():
    ok, msg = MultimodalSafetyGuardrail().verify_safety("What color is the car?", {}, "car")
    assert ok and msg == ""


def test_label_parsing():
    g = MultimodalSafetyGuardrail()
    assert g.parse_vlm_label("CATEGORY: PREPARED_FOOD; SUBJECT: pepperoni pizza") == ("PREPARED_FOOD", "pepperoni pizza")
    assert g.parse_vlm_label("CATEGORY: NOT_FOOD\nSUBJECT: a laptop") == ("NOT_FOOD", "a laptop")
    assert g.parse_vlm_label("category: raw_or_wild_food; subject: red berries")[0] == "WILD"
    assert g.parse_vlm_label("no idea")[0] == "UNCLEAR"


# --------------------------------------------------------------- conversation
def test_history_is_passed_to_the_model(img):
    chat, engine = make(["A red square.", "It is red."])
    chat.set_image(img)
    chat.answer_question("What is this?")
    chat.answer_question("What colour is it?")
    assert engine.calls[1]["history"] == [("What is this?", "A red square.")]


def test_history_window_is_bounded(img):
    chat, engine = make([f"a{i}" for i in range(10)], max_history_turns=3)
    chat.set_image(img)
    for i in range(8):
        chat.answer_question(f"q{i}")
    assert len(engine.calls[-1]["history"]) == 3


def test_new_image_starts_a_fresh_conversation(img):
    chat, _ = make(["a"])
    chat.set_image(img)
    chat.answer_question("q")
    chat.set_image(img)
    assert chat.history == []


def test_missing_image_engine_or_question_never_crash(img):
    chat, _ = make([])
    assert "upload" in chat.answer_question("hi")["answer"].lower()
    chat.set_image(img)
    assert "would you like" in chat.answer_question("   ")["answer"].lower()
    no_engine = PixelTalkMultimodalVLM(engine=None, load=False)
    no_engine.set_image(img)
    assert "isn't available" in no_engine.answer_question("hi")["answer"]


def test_model_errors_are_contained(img):
    chat, _ = make([RuntimeError("boom")])
    chat.set_image(img)
    result = chat.answer_question("What is this?")
    assert "error" in result["answer"].lower() and chat.history == []


def test_empty_model_output_gets_a_helpful_message(img):
    chat, _ = make(["   "])
    chat.set_image(img)
    assert "couldn't work out" in chat.answer_question("What is this?")["answer"]


def test_safety_classification_costs_one_model_call(img):
    chat, engine = make(["CATEGORY: NOT_FOOD; SUBJECT: a laptop"])
    chat.set_image(img)
    first = chat.answer_question("Is it safe to eat?")
    second = chat.answer_question("can I eat this?")
    assert first["safety_alert"] == second["safety_alert"] == "HAZARD_BLOCKED"
    assert len(engine.calls) == 1


def test_food_gets_description_plus_advisory_and_never_a_safe_claim(img):
    chat, _ = make(["CATEGORY: PREPARED_FOOD; SUBJECT: pepperoni pizza", "A pizza with pepperoni."])
    chat.set_image(img)
    result = chat.answer_question("Is it safe to eat?")
    assert "pepperoni" in result["answer"] and "Food Safety Advisory" in result["answer"]
    assert "is safe to eat" not in result["answer"].lower()


def test_garbled_classifier_output_is_uncertain(img):
    chat, _ = make(["hmm not sure"])
    chat.set_image(img)
    result = chat.answer_question("Is it safe to eat?")
    assert result["safety_alert"] == "UNCERTAIN"


# -------------------------------------------------------------- engine helpers
def test_image_is_attached_to_the_first_user_turn_only():
    messages = build_chat_messages([("q1", "a1"), ("q2", "a2")], "q3", "SYS")
    placeholders = [c for m in messages for c in m["content"] if c["type"] == "image"]
    assert len(placeholders) == 1 and messages[1]["content"][0]["type"] == "image"
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_prepare_image_handles_every_kind_of_input():
    assert prepare_image(Image.new("RGB", (4000, 3000)), 640).size == (640, 480)         # big -> smaller (speed)
    assert min(prepare_image(Image.new("RGB", (10, 10))).size) >= 56                      # tiny -> larger
    flattened = prepare_image(Image.new("RGBA", (50, 50), (0, 0, 0, 0)))                  # transparency -> white
    assert flattened.mode == "RGB" and flattened.getpixel((5, 5)) == (255, 255, 255)
    assert prepare_image(Image.new("P", (50, 50))).mode == "RGB"
    assert prepare_image(Image.new("L", (80, 60))).mode == "RGB"


def test_exif_rotation_is_applied():
    photo = Image.new("RGB", (200, 100))
    exif = Image.Exif()
    exif[274] = 6  # rotate 90
    buf = io.BytesIO()
    photo.save(buf, "JPEG", exif=exif)
    buf.seek(0)
    assert prepare_image(Image.open(buf)).size == (100, 200)


def test_dtype_choice_protects_small_machines(monkeypatch):
    monkeypatch.delenv("PIXELTALK_DTYPE", raising=False)
    assert choose_dtype_name("cpu", 8) == "bfloat16"
    assert choose_dtype_name("cpu", 16) == "bfloat16"
    assert choose_dtype_name("cpu", 32) == "float32"
    assert choose_dtype_name("cuda", 16) == "bfloat16"
    monkeypatch.setenv("PIXELTALK_DTYPE", "float32")
    assert choose_dtype_name("cpu", 8) == "float32"


# ----------------------------------------------------------- evaluation helpers
def test_scoring_is_whole_word_and_rejects_unsafe_claims():
    assert is_correct("The ball is red.", ["red"], [])
    assert not is_correct("a reddish ball", ["red"], [])
    assert not is_correct("There are 14 trees", ["4", "four"], [])
    assert not is_correct("This is safe to eat.", SAFE_ADVISORY, CLAIMS_SAFE)
    assert is_correct("laptop does not look like food, so it is not something to eat", NOT_FOOD, CLAIMS_SAFE)


def test_every_evaluation_image_exists():
    for fname, *_ in EVAL_SET:
        assert os.path.exists(os.path.join(ROOT, "sample_images", fname)), fname


# ------------------------------------------------------------------- spoilage
@pytest.mark.parametrize("subject", ["moldy bread", "two slices of bread with green mold", "rotten apple"])
def test_visibly_spoiled_food_is_refused(subject):
    ok, msg = MultimodalSafetyGuardrail().verify_safety("is it safe to eat?", {}, subject)
    assert not ok and "Spoilage Warning" in msg and "throw" in msg.lower()


def test_spoiled_category_from_vlm_label(img):
    chat, _ = make(["CATEGORY: SPOILED_FOOD; SUBJECT: moldy bread"])
    chat.set_image(img)
    result = chat.answer_question("Is it safe to eat?")
    assert result["safety_alert"] == "HAZARD_BLOCKED" and "Spoilage Warning" in result["answer"]


def test_fly_agaric_is_treated_as_wild():
    ok, msg = MultimodalSafetyGuardrail().verify_safety("is it edible?", {}, "fly agaric")
    assert not ok and "Foraging" in msg


# ------------------------------------------------------------------ streaming
class StreamEngine(FakeEngine):
    """Fake engine that also supports chat_stream()."""

    def __init__(self, replies, chunks=None, stream_error=None):
        super().__init__(replies)
        self.chunks, self.stream_error, self.closed = chunks, stream_error, False

    def chat_stream(self, image, history, question, system, temperature=0.2, max_new_tokens=160):
        self.calls.append({"history": list(history), "question": question, "streamed": True})
        engine = self

        def gen():
            try:
                for c in (engine.chunks or []):
                    yield c
                if engine.stream_error:
                    raise engine.stream_error
            finally:
                engine.closed = True

        return gen()


def make_stream(**kw):
    engine = StreamEngine(kw.pop("replies", []), **kw)
    chat = PixelTalkMultimodalVLM(engine=engine, load=False)
    chat.set_image(Image.new("RGB", (64, 64), "blue"))
    return chat, engine


def test_stream_yields_pieces_and_records_history():
    chat, engine = make_stream(chunks=["A ", "blue ", "square."])
    assert list(chat.answer_stream("What is this?")) == ["A ", "blue ", "square."]
    assert chat.last_result["answer"] == "A blue square." and chat.history == [("What is this?", "A blue square.")]
    assert engine.closed


def test_stream_falls_back_to_normal_generation_when_unsupported(img):
    chat, _ = make(["A red square."])          # FakeEngine has no chat_stream
    chat.set_image(img)
    assert "".join(chat.answer_stream("What is this?")) == "A red square."
    assert chat.last_result["answer"] == "A red square."


def test_stream_failing_before_first_token_falls_back():
    chat, engine = make_stream(chunks=[], stream_error=RuntimeError("stream broke"), replies=["Fallback answer."])
    assert "".join(chat.answer_stream("What is this?")) == "Fallback answer."
    assert chat.last_result["answer"] == "Fallback answer."


def test_stream_failing_midway_keeps_partial_answer():
    chat, _ = make_stream(chunks=["Half an ", "answer"], stream_error=RuntimeError("boom"))
    list(chat.answer_stream("What is this?"))
    assert chat.last_result["answer"] == "Half an answer"


def test_stream_with_empty_output_gives_helpful_message():
    chat, _ = make_stream(chunks=[], replies=["   "])
    out = "".join(chat.answer_stream("What is this?"))
    assert "couldn't work out" in out and "couldn't work out" in chat.last_result["answer"]


def test_stream_total_failure_is_contained():
    chat, _ = make_stream(chunks=[], stream_error=RuntimeError("a"), replies=[RuntimeError("b")])
    out = "".join(chat.answer_stream("What is this?"))
    assert "error" in out.lower() and chat.history == []


def test_closing_a_stream_early_releases_the_lock_and_keeps_history_clean():
    chat, engine = make_stream(chunks=["one ", "two ", "three"], replies=["next answer"])
    gen = chat.answer_stream("What is this?")
    next(gen)
    gen.close()
    assert chat.history == [] and engine.closed
    assert chat.answer_question("again?")["answer"] == "next answer"      # would deadlock if the lock leaked


def test_stream_handles_safety_questions_and_missing_image():
    chat, _ = make_stream(replies=["CATEGORY: NOT_FOOD; SUBJECT: a toy"])
    out = "".join(chat.answer_stream("Is it safe to eat?"))
    assert "Safety Guardrail Alert" in out and chat.last_result["safety_alert"] == "HAZARD_BLOCKED"
    empty = PixelTalkMultimodalVLM(engine=FakeEngine([]), load=False)
    assert "upload" in "".join(empty.answer_stream("hi")).lower()


# ------------------------------------------------- second safety layer (spoilage in text)
@pytest.mark.parametrize("text,expected", [
    ("Two slices of bread with green mold spots.", True),
    ("A rotten apple with brown patches.", True),
    ("A fresh loaf with no visible mold.", False),
    ("The bread is free of mold and looks fresh.", False),
    ("Pizza topped with pepperoni and basil.", False),
    ("Moldy cheese on a board.", True),
])
def test_mentions_spoilage_with_negation(text, expected):
    assert MultimodalSafetyGuardrail().mentions_spoilage(text) is expected


def test_model_that_labels_moldy_food_as_fine_is_caught_by_the_subject(img):
    chat, _ = make(["CATEGORY: PREPARED_FOOD; SUBJECT: moldy bread"])
    chat.set_image(img)
    result = chat.answer_question("Is it safe to eat?")
    assert result["safety_alert"] == "HAZARD_BLOCKED" and "Spoilage Warning" in result["answer"]


def test_model_that_labels_moldy_food_as_fine_is_caught_by_its_own_description(img):
    chat, _ = make(["CATEGORY: PREPARED_FOOD; SUBJECT: sliced bread",
                    "Two slices of white bread covered in green and grey mold."])
    chat.set_image(img)
    result = chat.answer_question("Is it safe to eat?")
    assert result["safety_alert"] == "HAZARD_BLOCKED" and "Spoilage Warning" in result["answer"]
    assert "Food Safety Advisory" not in result["answer"]


def test_fresh_food_described_as_free_of_mold_is_not_flagged(img):
    chat, _ = make(["CATEGORY: PREPARED_FOOD; SUBJECT: sliced bread", "Fresh sliced bread with no visible mold."])
    chat.set_image(img)
    result = chat.answer_question("Is it safe to eat?")
    assert result["safety_alert"] == "FOOD_ADVISORY"
