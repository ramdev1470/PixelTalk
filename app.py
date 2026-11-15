"""
PixelTalk - Conversational Image Recognition Chatbot (Streamlit web app)

Pipelines
  1. Multimodal VLM (default): a chat vision-language model that looks at the real
     pixels on every turn, with multi-turn memory and a food/hazard safety layer.
  2. Baseline (SIH1604): BLIP caption -> text-only dialogue model. Kept for comparison.
  3. Side-by-side comparator.

Run:  streamlit run app.py
"""

import hashlib
import html
import json
import os
import sys
import tempfile
from PIL import Image, UnidentifiedImageError

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "pixel_matplotlib")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
for p in [_parent_dir, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from PixelTalk.models.dialogue_manager import DialogueManager
    from PixelTalk.models.vision_captioner import VisionCaptioner
    from PixelTalk.multimodal.multimodal_vlm import PixelTalkMultimodalVLM
    from PixelTalk.multimodal.vlm_engines import load_engine, prepare_image
except ImportError:
    from models.dialogue_manager import DialogueManager
    from models.vision_captioner import VisionCaptioner
    from multimodal.multimodal_vlm import PixelTalkMultimodalVLM
    from multimodal.vlm_engines import load_engine, prepare_image

st.set_page_config(
    page_title="PixelTalk - Conversational Image Chatbot",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E88E5 0%, #7B1FA2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #757575;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .badge-bar {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .badge {
        background-color: #f0f4f8;
        color: #1565C0;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #bbdefb;
    }
    .badge-vlm {
        background-color: #EDE7F6;
        color: #512DA8;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #D1C4E9;
    }
    .caption-card {
        background: #f8fafc;
        border-left: 4px solid #1E88E5;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.98rem;
        line-height: 1.5;
        color: #1a202c;
    }
    .meta-box {
        background: #F1F8E9;
        border-left: 4px solid #43A047;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #2E7D32;
    }
    .vlm-box {
        background: #F3E5F5;
        border-left: 4px solid #8E24AA;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #4A148C;
    }
    .tag-chip {
        display: inline-block;
        background: #E3F2FD;
        color: #0D47A1;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 0.82rem;
        margin: 3px;
        font-weight: 600;
    }
    .chat-bubble-user {
        background: #E3F2FD;
        border-radius: 14px 14px 2px 14px;
        padding: 10px 14px;
        margin: 8px 0;
        max-width: 85%;
        margin-left: auto;
        color: #0d47a1;
        font-weight: 500;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .chat-bubble-bot {
        background: #F5F5F5;
        border-radius: 14px 14px 14px 2px;
        padding: 10px 14px;
        margin: 8px 0;
        max-width: 85%;
        margin-right: auto;
        color: #212121;
        border-left: 3px solid #7B1FA2;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .comparison-card {
        background: #FAFAFA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)
# ----------------- CACHED RESOURCES (loaded once per server, not per click) -----------------
@st.cache_resource(show_spinner="Loading the vision-language model (the first start can take a few minutes)...")
def load_vlm_engine():
    return load_engine()


@st.cache_resource(show_spinner="Loading the baseline pipeline...")
def load_baseline_pipeline():
    ckpt_path = os.path.join(_current_dir, "checkpoints", "seq2seq_attention.pt")
    vocab_path = os.path.join(_current_dir, "checkpoints", "vocab.json")
    captioner = VisionCaptioner(use_hf=True)
    baseline = DialogueManager(checkpoint_path=ckpt_path, vocab_path=vocab_path, default_backend="flan_t5")
    return captioner, baseline


engine, engine_notes = load_vlm_engine()

# ----------------- PER-SESSION STATE -----------------
_defaults = {
    "messages": [], "comparison_history": [], "active_image": None, "image_key": None,
    "image_filename": "", "baseline_ready": False, "image_caption": "",
    "image_attributes": [], "scene_metadata": {}, "image_features": {},
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if "vlm" not in st.session_state:
    # The heavy model is shared; each browser session gets its OWN conversation state.
    st.session_state.vlm = PixelTalkMultimodalVLM(engine=engine, load=False)
vlm = st.session_state.vlm

SAMPLE_IMAGES = {
    "🐕 Golden Retriever in a Park (photo)": "golden_retriever_park.jpg",
    "🍞 Moldy Bread (food safety)": "moldy_bread_plate.jpg",
    "🍎 Red Apple (food)": "red_apple_kitchen.jpg",
    "🚦 City Crosswalk (street scene)": "street_crosswalk_city.jpg",
    "🍄 Red Mushroom (foraging safety)": "wild_mushroom_forest.jpg",
    "✨ Milky Way Galaxy (astronomy)": "galaxy_stars.jpg",
    "☄️ Asteroid (space)": "asteroid_clean.jpg",
    "📷 Woman with Camera (photo)": "woman_camera.jpg",
    "🍕 Pizza (drawing)": "pizza.jpg",
    "🐶 Dog and Ball (drawing)": "dog_park.jpg",
    "🚗 Red Car (drawing)": "vintage_car.jpg",
    "💻 Laptop Workspace (drawing)": "laptop_desk.jpg",
    "🏔️ Mountain Lake (drawing)": "mountain_lake.jpg",
}
SAMPLE_DIR = os.path.join(_current_dir, "sample_images")


def esc(text) -> str:
    """Escape model/user text before putting it in HTML (prevents broken layout / injection)."""
    return html.escape(str(text or "")).replace("\n", "<br>")


SAFETY_BADGES = {
    "HAZARD_BLOCKED": "🛡️ Safety check: not safe to eat",
    "FOOD_ADVISORY": "🛡️ Safety check: advisory added",
    "UNCERTAIN": "🛡️ Safety check: can't tell",
}


def bot_bubble(text: str, data=None, streaming: bool = False) -> str:
    data = data or {}
    foot = []
    if data.get("safety_alert") in SAFETY_BADGES:
        foot.append(SAFETY_BADGES[data["safety_alert"]])
    if data.get("latency_s"):
        foot.append(f"⏱ {data['latency_s']}s")
    foot_html = f"<br><small>{' · '.join(foot)}</small>" if foot else ""
    cursor = " ▌" if streaming else ""
    return f'<div class="chat-bubble-bot">🤖 <strong>PixelTalk:</strong> {esc(text)}{cursor}{foot_html}</div>'


def build_transcript() -> str:
    lines = []
    for m in st.session_state.messages:
        lines.append(("You: " if m["role"] == "user" else "PixelTalk: ") + m["content"])
    for item in st.session_state.comparison_history:
        lines += [f"You: {item['question']}", f"Baseline: {item['baseline_ans']}", f"VLM: {item['vlm_ans']}"]
    if not lines:
        return ""
    header = f"PixelTalk chat about: {st.session_state.image_filename or 'image'}\n" + "=" * 40
    return header + "\n\n" + "\n\n".join(lines) + "\n"


def load_user_image(source) -> Image.Image:
    img = Image.open(source)
    img.load()
    return prepare_image(img, max_side=1600)  # EXIF-rotate, flatten alpha, RGB, bound memory


def new_conversation():
    st.session_state.messages = []
    st.session_state.comparison_history = []
    img = st.session_state.active_image
    if img is not None:
        vlm.set_image(img, image_id=st.session_state.image_key or "", metadata=st.session_state.scene_metadata,
                      caption=st.session_state.image_caption)
        if st.session_state.baseline_ready:
            _, baseline_dm = load_baseline_pipeline()
            baseline_dm.set_image_context(st.session_state.image_caption, st.session_state.image_attributes,
                                          st.session_state.scene_metadata)


def ensure_baseline() -> bool:
    """Caption the image for the baseline pipeline, only when that pipeline is actually used."""
    if st.session_state.baseline_ready or st.session_state.active_image is None:
        return st.session_state.active_image is not None
    try:
        captioner, baseline_dm = load_baseline_pipeline()
    except Exception as exc:  # noqa: BLE001
        st.error(f"The baseline pipeline could not be loaded: {exc}")
        return False
    img = st.session_state.active_image
    with st.spinner("Captioning the image for the baseline pipeline..."):
        caption, attrs, metadata = captioner.generate_caption(img)
        feats = captioner.analyze_image_features(img)
    st.session_state.image_caption, st.session_state.image_attributes = caption, attrs
    st.session_state.scene_metadata, st.session_state.image_features = metadata, feats
    baseline_dm.set_image_context(caption, attrs, metadata)
    st.session_state.baseline_ready = True
    return True


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### 🏛️ Pipeline")
    arch_mode = st.radio(
        "Select active pipeline:",
        [
            "⚡ Multimodal VLM (sees the pixels every turn)",
            "📋 Baseline (SIH1604: Caption + Seq2Seq)",
            "⚖️ Side-by-Side Comparator",
        ],
        index=0,
    )
    use_vlm = "Multimodal" in arch_mode or "Comparator" in arch_mode
    use_baseline = "Baseline" in arch_mode or "Comparator" in arch_mode

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.1,
                            help="Keep this LOW for factual answers about images. 0 = most deterministic.")
    detail = st.select_slider("Image detail (speed vs. accuracy)", options=["Fast", "Balanced", "Detailed"],
                              value="Balanced",
                              help="Larger images read small text and details better but answer more slowly on CPU.")
    max_history = st.slider("Conversation memory (turns)", 1, 8, 6)
    vlm.max_history_turns = max_history
    if engine is not None and hasattr(engine, "max_side"):
        _cpu = getattr(getattr(engine, "device", None), "type", "cpu") != "cuda"
        _sizes = {"Fast": 448, "Balanced": 560, "Detailed": 784} if _cpu else {"Fast": 640, "Balanced": 1024, "Detailed": 1280}
        engine.max_side = _sizes[detail]

    st.markdown("---")
    st.markdown("### 🧠 Vision model")
    if engine is None:
        st.error("No vision model could be loaded.")
    else:
        st.success(engine.label)
    for note in engine_notes:
        st.warning(note)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 New chat", width="stretch"):
            new_conversation()
            st.rerun()
    with c2:
        if st.button("⚡ Reload models", width="stretch"):
            st.cache_resource.clear()
            st.session_state.pop("vlm", None)
            st.session_state.image_key = None
            st.session_state.baseline_ready = False
            st.rerun()

    st.download_button("⬇️ Download chat", data=build_transcript(), file_name="pixeltalk_chat.txt",
                       mime="text/plain", width="stretch", disabled=not build_transcript())

# ----------------- HEADER -----------------
st.markdown('<div class="main-title">PixelTalk: Conversational Image Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask questions about any image and get grounded answers</div>', unsafe_allow_html=True)
st.markdown("""
<div class="badge-bar">
    <span class="badge-vlm">👁️ Sees the real pixels every turn</span>
    <span class="badge-vlm">🧠 Multi-turn memory</span>
    <span class="badge-vlm">🛡️ Food &amp; hazard safety layer</span>
    <span class="badge">📋 Baseline kept for comparison</span>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.25], gap="large")

# ----------------- LEFT: IMAGE -----------------
with col_left:
    st.markdown("### 📷 Image")
    image_mode = st.radio("Image source:", ["Sample image", "Upload your own"], horizontal=True)

    image_to_process, image_filename, image_key = None, "", None

    if image_mode == "Upload your own":
        uploaded = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg", "webp", "bmp", "gif"])
        if uploaded is not None:
            try:
                data = uploaded.getvalue()
                image_key = "upload:" + hashlib.md5(data).hexdigest()
                if image_key != st.session_state.image_key:
                    image_to_process = load_user_image(uploaded)
                else:
                    image_to_process = st.session_state.active_image
                image_filename = uploaded.name
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("That file couldn't be read as an image. Please try a PNG, JPG, WebP, BMP or GIF.")
    else:
        sample_label = st.selectbox("Sample gallery:", list(SAMPLE_IMAGES.keys()))
        sample_path = os.path.join(SAMPLE_DIR, SAMPLE_IMAGES[sample_label])
        if os.path.exists(sample_path):
            image_key = "sample:" + SAMPLE_IMAGES[sample_label]
            if image_key != st.session_state.image_key:
                image_to_process = load_user_image(sample_path)
            else:
                image_to_process = st.session_state.active_image
            image_filename = os.path.basename(sample_path)
        else:
            st.warning(f"Sample file missing: {SAMPLE_IMAGES[sample_label]}")

    if image_to_process is not None:
        if image_key != st.session_state.image_key:  # a different image -> fresh conversation
            st.session_state.active_image = image_to_process
            st.session_state.image_key = image_key
            st.session_state.image_filename = image_filename
            st.session_state.baseline_ready = False
            st.session_state.image_caption, st.session_state.image_attributes = "", []
            st.session_state.scene_metadata, st.session_state.image_features = {}, {}
            st.session_state.messages, st.session_state.comparison_history = [], []
            vlm.set_image(image_to_process, image_id=image_key)

        if use_baseline:
            ensure_baseline()

        st.image(image_to_process, caption=image_filename, width="stretch")

        if st.session_state.baseline_ready and use_baseline:
            st.markdown(f'<div class="caption-card"><strong>Baseline caption:</strong> {esc(st.session_state.image_caption)}</div>',
                        unsafe_allow_html=True)
            meta = st.session_state.scene_metadata
            parts = []
            if meta.get("setting"):
                parts.append(f"📍 <strong>Setting:</strong> {esc(meta['setting'])}")
            if meta.get("background"):
                parts.append(f"🌲 <strong>Background:</strong> {esc(meta['background'])}")
            if meta.get("attire"):
                parts.append(f"🧥 <strong>Attire / objects:</strong> {esc(meta['attire'])}")
            if parts:
                st.markdown(f'<div class="meta-box">{"<br>".join(parts)}</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Choose a sample or upload an image to begin.")

# ----------------- RIGHT: CONVERSATION -----------------
with col_right:
    st.markdown("### 💬 Conversation")

    if use_vlm and not vlm.is_ready:
        st.error("The vision model isn't loaded, so I can't answer yet. Run `python download_models.py`, then click "
                 "**Reload models** in the sidebar.")

    st.markdown("**Suggested questions:**")
    quick_questions = [
        "What is the main subject?",
        "Describe this image in detail.",
        "What colors stand out?",
        "What is in the background?",
        "Is there any text in the image?",
        "Is it safe to eat?",
    ]
    quick_cols = st.columns(3)
    selected_quick_q = None
    for idx, q_text in enumerate(quick_questions):
        with quick_cols[idx % 3]:
            if st.button(q_text, key=f"quick_{idx}", width="stretch"):
                selected_quick_q = q_text

    user_query = st.chat_input("Ask a question about the image...")
    prompt_to_run = selected_quick_q or user_query

    def run_vlm(prompt):
        return vlm.answer_question(prompt, temperature=temperature)

    def run_baseline(prompt):
        _, baseline_dm = load_baseline_pipeline()
        return baseline_dm.answer_question(prompt, max_history_turns=max_history)

    # ---- Comparator ----
    if "Comparator" in arch_mode:
        st.markdown("#### ⚖️ Baseline vs. Multimodal VLM")
        box = st.container(height=380)
        with box:
            if not st.session_state.comparison_history:
                st.markdown(
                    '<div style="text-align:center;color:#9E9E9E;padding:40px 10px;"><h4>⚖️ Comparator</h4>'
                    "<p>Ask a question to run both pipelines on the same image.</p></div>",
                    unsafe_allow_html=True,
                )
            for item in st.session_state.comparison_history:
                st.markdown(f"**👤 Question:** {item['question']}")
                cb, cv = st.columns(2)
                with cb:
                    st.markdown(
                        '<div class="comparison-card" style="border-left:3px solid #1976D2;">'
                        f'<strong style="color:#1976D2;">📋 Baseline (caption only):</strong><br>{esc(item["baseline_ans"])}</div>',
                        unsafe_allow_html=True)
                with cv:
                    st.markdown(
                        '<div class="comparison-card" style="border-left:3px solid #7B1FA2;">'
                        f'<strong style="color:#7B1FA2;">👁️ Multimodal VLM:</strong><br>{esc(item["vlm_ans"])}'
                        f'<br><small>⏱ {item.get("latency")}s</small></div>',
                        unsafe_allow_html=True)
                st.markdown("---")

        if prompt_to_run:
            if st.session_state.active_image is None:
                st.warning("Please choose or upload an image first.")
            elif ensure_baseline():
                with st.spinner("Running both pipelines..."):
                    base_res = run_baseline(prompt_to_run)
                    vlm_res = run_vlm(prompt_to_run)
                st.session_state.comparison_history.append({
                    "question": prompt_to_run, "baseline_ans": base_res["answer"],
                    "vlm_ans": vlm_res["answer"], "latency": vlm_res.get("latency_s"),
                })
                st.rerun()

    # ---- Single pipeline chat ----
    else:
        box = st.container(height=420)
        with box:
            if not st.session_state.messages:
                name = "Multimodal VLM" if use_vlm else "Baseline (caption + seq2seq)"
                st.markdown(
                    f'<div style="text-align:center;color:#9E9E9E;padding:40px 10px;"><h4>Active pipeline: {name}</h4>'
                    "<p>Ask about objects, colors, counts, text, actions, background, or safety.</p></div>",
                    unsafe_allow_html=True)
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-bubble-user">👤 {esc(msg["content"])}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(bot_bubble(msg["content"], msg.get("data")), unsafe_allow_html=True)

        if prompt_to_run:
            if st.session_state.active_image is None:
                st.warning("Please choose or upload an image first.")
            elif use_baseline and not ensure_baseline():
                pass
            else:
                st.session_state.messages.append({"role": "user", "content": prompt_to_run})
                with box:
                    st.markdown(f'<div class="chat-bubble-user">👤 {esc(prompt_to_run)}</div>', unsafe_allow_html=True)
                    if use_vlm:
                        slot = st.empty()
                        slot.markdown(bot_bubble("Looking at the image... (on a CPU the first words can take a while)",
                                                 streaming=True), unsafe_allow_html=True)
                        shown = ""
                        for chunk in vlm.answer_stream(prompt_to_run, temperature=temperature):
                            shown += chunk
                            slot.markdown(bot_bubble(shown, streaming=True), unsafe_allow_html=True)
                        res = vlm.last_result or {"answer": shown.strip()}
                    else:
                        with st.spinner("Thinking..."):
                            res = run_baseline(prompt_to_run)
                st.session_state.messages.append({"role": "bot", "content": res["answer"], "data": res})
                st.rerun()

# ----------------- REPORT -----------------
with st.expander("📊 How it works & measured results", expanded=False):
    st.markdown("""
**Baseline (SIH1604).** A captioning model turns the image into one sentence, and a text-only dialogue model answers
from that sentence. Anything the caption missed (colors, counts, small objects, text) is lost.

**Multimodal VLM.** A chat vision-language model receives the image itself with every question and the conversation
history, so follow-ups like "what color is it?" are resolved by the model. If something isn't visible it is told to
say so instead of guessing.

**Safety layer.** For "can I eat/drink this?" questions the image is classified first. Non-food and wild mushrooms,
berries and plants are refused; ordinary food is described together with an advisory. A photo can never prove food is
safe, so the app never says "yes, it is safe".

**Honest limits.** No vision model is always right. It can still be wrong on tiny details, crowded scenes, counting
and unusual objects. Treat answers as a helpful reading of the image, not a guarantee.
""")
    results_path = os.path.join(_current_dir, "results", "eval_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as fh:
                res = json.load(fh)
            st.markdown(f"**Measured on {res.get('n_questions')} hand-labelled questions "
                        f"({res.get('generated_at', '')[:10]}, engine: {res.get('engine', 'n/a')}).**")
            cols = st.columns(2)
            for col, key, title in ((cols[0], "baseline", "Baseline"), (cols[1], "vlm", "Multimodal VLM")):
                if res.get(key):
                    col.metric(f"{title} accuracy", f"{res[key]['accuracy'] * 100:.1f}%",
                               help=f"Average latency {res[key]['avg_latency_s']}s per question")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read results file: {exc}")
    else:
        st.info("No benchmark has been run yet. Run `python -m multimodal.evaluate_comparison` to measure both "
                "pipelines on your own machine; the numbers will appear here.")
