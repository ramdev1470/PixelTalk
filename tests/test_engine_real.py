"""
Integration tests that run the REAL engine code (real transformers + torch) on tiny random-weight models.
They need `torch`, `transformers`, `torchvision` and `tokenizers`; they are skipped automatically if any is
missing. Answers are gibberish by design (random weights): what is verified is that loading, the chat
template, the image processor, generation, streaming, cancelling and the fallback chain really work.
"""

import os
import sys
import threading

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")

from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from multimodal.multimodal_vlm import PixelTalkMultimodalVLM  # noqa: E402
from multimodal.vlm_engines import BlipVQAEngine, QwenVLEngine, load_engine  # noqa: E402
from tests.tiny_blip import build_tiny_blip_vqa  # noqa: E402
from tests.tiny_qwen import build_tiny_qwen  # noqa: E402

SAMPLE = os.path.join(ROOT, "sample_images", "golden_retriever_park.jpg")


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory):
    return build_tiny_qwen(str(tmp_path_factory.mktemp("tiny-qwen")))


@pytest.fixture(scope="module")
def tiny_blip(tmp_path_factory):
    return build_tiny_blip_vqa(str(tmp_path_factory.mktemp("tiny-blip")))


@pytest.fixture(scope="module")
def qwen(tiny_qwen):
    return QwenVLEngine(model_id=tiny_qwen, device="cpu")


@pytest.fixture
def photo():
    return Image.open(SAMPLE)


def test_qwen_engine_loads_and_answers(qwen, photo):
    reply = qwen.chat(photo, [], "What is in the picture?", "system", temperature=0.0, max_new_tokens=6)
    assert isinstance(reply, str)


def test_qwen_engine_accepts_history_and_sampling(qwen, photo):
    reply = qwen.chat(photo, [("What is it?", "A dog.")], "What colour?", "system", temperature=0.8, max_new_tokens=6)
    assert isinstance(reply, str)


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_qwen_engine_handles_unusual_image_modes(qwen, mode):
    reply = qwen.chat(Image.new(mode, (90, 60)), [], "What is this?", "system", temperature=0.0, max_new_tokens=3)
    assert isinstance(reply, str)


def test_streaming_matches_non_streaming(qwen, photo):
    whole = qwen.chat(photo, [], "Describe this.", "system", temperature=0.0, max_new_tokens=8)
    pieces = list(qwen.chat_stream(photo, [], "Describe this.", "system", temperature=0.0, max_new_tokens=8))
    assert len(pieces) >= 1 and "".join(pieces).strip() == whole


def test_closing_a_stream_stops_background_generation(qwen, photo):
    gen = qwen.chat_stream(photo, [], "Describe this.", "system", temperature=0.0, max_new_tokens=300)
    next(gen)
    gen.close()
    alive = [t for t in threading.enumerate() if t is not threading.main_thread() and "tqdm" not in t.name]
    assert not alive


def test_full_pipeline_on_the_real_engine(qwen, photo):
    chat = PixelTalkMultimodalVLM(engine=qwen, load=False)
    chat.set_image(photo)
    assert "error" not in chat.answer_question("What is this?", temperature=0.0, max_new_tokens=6)["answer"].lower()
    list(chat.answer_stream("And the colour?", temperature=0.0, max_new_tokens=6))
    assert chat.last_result["turn_index"] == 2
    # A model that can't be parsed must lead to a cautious answer, never to "safe".
    risky = chat.answer_question("Is it safe to eat?", temperature=0.0)
    assert "is safe to eat" not in risky["answer"].lower()


def test_blip_fallback_engine_works(tiny_blip, photo):
    engine = BlipVQAEngine(model_id=tiny_blip, device="cpu")
    assert isinstance(engine.chat(photo, [], "what is the main subject in this image?", "s", max_new_tokens=5), str)


def test_loader_prefers_the_chat_model(tiny_qwen, monkeypatch):
    monkeypatch.setenv("PIXELTALK_VLM", tiny_qwen)
    monkeypatch.setenv("PIXELTALK_OFFLINE", "1")
    engine, notes = load_engine()
    assert engine is not None and engine.name == "qwen" and notes == []


def test_loader_falls_back_and_explains_why(tiny_blip, monkeypatch, tmp_path):
    monkeypatch.setenv("PIXELTALK_VLM", str(tmp_path / "does-not-exist"))
    monkeypatch.setenv("PIXELTALK_FALLBACK_VLM", tiny_blip)
    monkeypatch.setenv("PIXELTALK_OFFLINE", "1")
    engine, notes = load_engine()
    assert engine is not None and engine.name == "blip-vqa"
    assert any("Chat VLM unavailable" in n for n in notes) and any("fallback" in n.lower() for n in notes)


def test_loader_reports_total_failure_without_crashing(monkeypatch, tmp_path):
    monkeypatch.setenv("PIXELTALK_VLM", str(tmp_path / "nope"))
    monkeypatch.setenv("PIXELTALK_FALLBACK_VLM", str(tmp_path / "nope2"))
    monkeypatch.setenv("PIXELTALK_OFFLINE", "1")
    engine, notes = load_engine()
    assert engine is None and len(notes) == 2
    chat = PixelTalkMultimodalVLM(engine=None, load=False)
    chat.set_image(Image.new("RGB", (50, 50)))
    assert "isn't available" in chat.answer_question("hi")["answer"]
