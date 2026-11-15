"""
Drives the real Streamlit app (streamlit.testing) with the real engine code on a tiny random-weight model.
Skipped automatically when streamlit / torch / torchvision / tokenizers are not installed.
"""

import os
import sys

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.tiny_qwen import build_tiny_qwen  # noqa: E402


@pytest.fixture(scope="module")
def app_env(tmp_path_factory):
    path = build_tiny_qwen(str(tmp_path_factory.mktemp("ui-qwen")))
    saved = {k: os.environ.get(k) for k in ("PIXELTALK_VLM", "PIXELTALK_OFFLINE", "HF_HUB_OFFLINE")}
    os.environ.update({"PIXELTALK_VLM": path, "PIXELTALK_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})
    yield path
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def app(app_env):
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    return at


def ask(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception
    return at


def test_app_starts_and_shows_the_loaded_model(app):
    assert any("cpu" in s.value for s in app.sidebar.success)
    assert app.session_state["active_image"] is not None


def test_question_followup_and_history(app):
    ask(app, "What is the main subject?")
    msgs = app.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "bot"] and msgs[1]["content"].strip()
    ask(app, "And what colour is it?")
    assert app.session_state["messages"][-1]["data"]["turn_index"] == 2


def test_user_text_is_escaped_in_the_chat(app):
    ask(app, "<script>alert(1)</script>")
    bubbles = [m.value for m in app.markdown if "chat-bubble" in m.value]
    assert bubbles and not any("<script>" in b for b in bubbles)


def test_food_question_on_a_model_that_cannot_classify_is_cautious(app):
    ask(app, "Is it safe to eat?")
    last = app.session_state["messages"][-1]
    assert "is safe to eat" not in last["content"].lower()
    assert last["data"]["safety_alert"] in ("UNCERTAIN", "HAZARD_BLOCKED", "FOOD_ADVISORY")


def test_every_sample_image_loads_and_resets_the_chat(app):
    labels = list(app.selectbox[0].options)
    assert len(labels) >= 13
    for label in labels:
        app.selectbox[0].set_value(label).run()
        assert not app.exception, (label, app.exception)
        assert app.session_state["messages"] == []
        assert app.session_state["active_image"] is not None


def test_new_chat_button_clears_the_conversation_but_keeps_the_image(app):
    ask(app, "hello")
    assert len(app.session_state["messages"]) == 2
    [b for b in app.sidebar.button if "New chat" in b.label][0].click().run()
    assert app.session_state["messages"] == []
    assert app.session_state["active_image"] is not None
    assert app.session_state["vlm"].history == []


def test_detail_slider_changes_the_image_size_sent_to_the_model(app):
    engine = app.session_state["vlm"].engine
    app.sidebar.select_slider[0].set_value("Fast").run()
    assert not app.exception and engine.max_side == 448
    app.sidebar.select_slider[0].set_value("Detailed").run()
    assert engine.max_side == 784
    app.sidebar.select_slider[0].set_value("Balanced").run()
    assert engine.max_side == 560


def test_baseline_and_comparator_modes_run_on_the_real_baseline(app):
    app.sidebar.radio[0].set_value("📋 Baseline (SIH1604: Caption + Seq2Seq)").run()
    assert not app.exception, app.exception
    assert app.session_state["baseline_ready"] is True
    ask(app, "is it safe to eat?")
    assert app.session_state["messages"][-1]["content"].strip()
    app.sidebar.radio[0].set_value("⚖️ Side-by-Side Comparator").run()
    ask(app, "what is this?")
    assert len(app.session_state["comparison_history"]) == 1
