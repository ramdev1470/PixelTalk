"""
Vision-language engines for PixelTalk.

Two interchangeable back-ends sharing one tiny interface:

    engine.chat(image, history, question, system, temperature, max_new_tokens) -> str

* QwenVLEngine    - a modern instruction-tuned chat VLM (default: Qwen2-VL-2B-Instruct).
                    Sees the real pixels on every turn, understands conversation
                    history, answers in full sentences.
* BlipVQAEngine   - the old BLIP-VQA model. Only used as an emergency fallback when
                    the Qwen model cannot be loaded. Gives short answers.

Heavy imports (torch / transformers) happen lazily so the pure-Python parts of the
project (guardrail, history logic, tests) work without them.
"""

import os
from typing import List, Optional, Tuple

from PIL import Image, ImageOps

DEFAULT_VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
STREAM_TIMEOUT_S = 900  # max wait for the next token (first token can be slow on CPU)
FALLBACK_VQA_MODEL = "Salesforce/blip-vqa-base"


# --------------------------------------------------------------------------- utils
def resolve_cached_hf_path(model_id: str) -> str:
    """Return the local snapshot dir of a cached HF model, or the id itself."""
    if os.path.isdir(model_id):
        return model_id
    org_repo = model_id.replace("/", "--")
    hub_root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub"
    )
    hub_dir = os.path.join(hub_root, f"models--{org_repo}", "snapshots")
    if os.path.isdir(hub_dir):
        snaps = [os.path.join(hub_dir, s) for s in sorted(os.listdir(hub_dir))]
        snaps = [s for s in snaps if os.path.isdir(s)]
        with_cfg = [s for s in snaps if os.path.exists(os.path.join(s, "config.json"))]
        if with_cfg:
            return with_cfg[0]
        if snaps:
            return snaps[0]
    return model_id


def prepare_image(image: Image.Image, max_side: int = 896, min_side: int = 56) -> Image.Image:
    """
    Make any user image safe and fast for the model:
      * apply EXIF rotation (phone photos are often sideways)
      * flatten transparency onto white (PNG/GIF/WebP with alpha)
      * convert to RGB
      * shrink so the longest side <= max_side  (this is the main latency control)
      * enlarge tiny images so the vision tower gets at least `min_side` pixels
    """
    img = ImageOps.exif_transpose(image)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")

    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    w, h = img.size
    if min(w, h) < min_side:
        scale = min_side / float(min(w, h))
        img = img.resize((max(min_side, round(w * scale)), max(min_side, round(h * scale))), Image.LANCZOS)
    return img


def default_max_side(device_type: str) -> int:
    env = os.environ.get("PIXELTALK_MAX_SIDE")
    if env and env.isdigit():
        return int(env)
    return 1024 if device_type == "cuda" else 560


def total_ram_gb() -> Optional[float]:
    """Total physical RAM in GB (Windows / Linux / macOS), or None if unknown."""
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / 1024 ** 3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except Exception:  # noqa: BLE001
        return None


def choose_dtype_name(device_type: str, ram_gb: Optional[float], bf16_cuda: bool = True) -> str:
    """
    Pick a weight precision that fits in memory.
    A 2B-parameter model needs ~9 GB in float32 but only ~4.5 GB in bfloat16.
    Override with the environment variable PIXELTALK_DTYPE=float32|bfloat16|float16.
    """
    forced = os.environ.get("PIXELTALK_DTYPE", "").strip().lower()
    if forced in ("float32", "bfloat16", "float16"):
        return forced
    if device_type == "cuda":
        return "bfloat16" if bf16_cuda else "float16"
    if ram_gb is not None and ram_gb < 24:
        return "bfloat16"
    return "float32"


def build_chat_messages(history: List[Tuple[str, str]], question: str, system: str) -> list:
    """
    Build a chat-template conversation. The image placeholder is attached to the
    FIRST user message only; later turns refer back to it, exactly like a normal
    multimodal chat.
    """
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]}]
    first = True
    for q, a in list(history) + [(question, None)]:
        content = []
        if first:
            content.append({"type": "image"})
            first = False
        content.append({"type": "text", "text": q})
        messages.append({"role": "user", "content": content})
        if a is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
    return messages


def _load_with_dtype(loader, path, dtype, local_only):
    """transformers v5 uses `dtype=`, older versions `torch_dtype=`; try both."""
    last = None
    for key in ("dtype", "torch_dtype"):
        try:
            return loader(path, local_files_only=local_only, **{key: dtype})
        except Exception as exc:  # noqa: BLE001 - we re-raise the last one
            last = exc
    raise last


# ------------------------------------------------------------------------ Qwen VLM
class QwenVLEngine:
    name = "qwen"

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as AutoVLM

        self.torch = torch
        self.model_id = model_id or os.environ.get("PIXELTALK_VLM", DEFAULT_VLM_MODEL)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        bf16_ok = torch.cuda.is_bf16_supported() if self.device.type == "cuda" else True
        dtype_name = choose_dtype_name(self.device.type, total_ram_gb(), bf16_ok)
        dtype = getattr(torch, dtype_name)
        self.dtype_name = dtype_name
        self.max_side = default_max_side(self.device.type)

        path = resolve_cached_hf_path(self.model_id)
        offline_only = os.environ.get("PIXELTALK_OFFLINE") == "1"

        def load_all(local_only):
            proc = AutoProcessor.from_pretrained(path, local_files_only=local_only)
            mdl = _load_with_dtype(AutoVLM.from_pretrained, path, dtype, local_only)
            return proc, mdl

        try:
            self.processor, self.model = load_all(True)
        except Exception:
            if offline_only:
                raise
            self.processor, self.model = load_all(False)

        self.model.to(self.device)
        self.model.eval()
        self.label = f"{self.model_id} ({self.device.type}, {self.dtype_name})"

    # ------------------------------------------------------------------ generation
    def _prepare_inputs(self, image, history, question, system):
        img = prepare_image(image, max_side=self.max_side)
        messages = build_chat_messages(history, question, system)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[img], padding=True, return_tensors="pt")
        return inputs.to(self.device)

    @staticmethod
    def _gen_kwargs(temperature, max_new_tokens):
        sample = temperature is not None and temperature > 0.05
        kwargs = {"max_new_tokens": int(max_new_tokens), "do_sample": bool(sample), "repetition_penalty": 1.05}
        if sample:
            kwargs.update({"temperature": float(temperature), "top_p": 0.9})
        return kwargs

    def chat(self, image, history, question, system, temperature=0.2, max_new_tokens=160) -> str:
        torch = self.torch
        inputs = self._prepare_inputs(image, history, question, system)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self._gen_kwargs(temperature, max_new_tokens))
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        reply = self.processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return reply.strip()

    def chat_stream(self, image, history, question, system, temperature=0.2, max_new_tokens=160):
        """
        Generator yielding text pieces while the model writes. Generation runs in a background
        thread; closing the generator (e.g. the user navigates away) stops it at the next token.
        """
        import threading

        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        torch = self.torch
        inputs = self._prepare_inputs(image, history, question, system)
        kwargs = self._gen_kwargs(temperature, max_new_tokens)
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=STREAM_TIMEOUT_S,
        )
        cancel = threading.Event()
        errors = []

        class _Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return torch.full((input_ids.shape[0],), cancel.is_set(), dtype=torch.bool, device=input_ids.device)

        def _run():
            try:
                with torch.inference_mode():
                    self.model.generate(
                        **inputs, streamer=streamer, stopping_criteria=StoppingCriteriaList([_Cancel()]), **kwargs,
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                streamer.end()  # unblock the consumer

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        try:
            for piece in streamer:
                yield piece
            if errors:
                raise errors[0]
        finally:
            cancel.set()
            worker.join(timeout=60)


# ----------------------------------------------------------------- BLIP VQA fallback
class BlipVQAEngine:
    name = "blip-vqa"

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        import torch
        from transformers import BlipForQuestionAnswering, BlipProcessor

        self.torch = torch
        self.model_id = model_id or os.environ.get("PIXELTALK_FALLBACK_VLM", FALLBACK_VQA_MODEL)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = resolve_cached_hf_path(self.model_id)
        try:
            self.processor = BlipProcessor.from_pretrained(path, local_files_only=True)
            self.model = BlipForQuestionAnswering.from_pretrained(path, local_files_only=True)
        except Exception:
            if os.environ.get("PIXELTALK_OFFLINE") == "1":
                raise
            self.processor = BlipProcessor.from_pretrained(path)
            self.model = BlipForQuestionAnswering.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()
        self.max_side = default_max_side(self.device.type)
        self.label = f"{self.model_id} ({self.device.type}) [fallback: short answers only]"

    def chat(self, image, history, question, system, temperature=0.0, max_new_tokens=30) -> str:
        torch = self.torch
        img = prepare_image(image, max_side=self.max_side)
        inputs = self.processor(images=img, text=question, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=min(int(max_new_tokens), 30))
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------- loader
def explain_load_error(exc: Exception) -> str:
    """Turn a loader exception into one actionable sentence for the UI."""
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    hints = []
    if "torchvision" in low:
        hints.append("Fix: run `pip install torchvision` (needed by the Qwen2-VL image processor).")
    if "no module named" in low and "torchvision" not in low:
        hints.append("Fix: run `pip install -r requirements.txt`.")
    if any(k in low for k in ("connecterror", "name or service", "localentrynotfound", "offline", "maxretry", "getaddrinfo")):
        hints.append("The model files are not downloaded yet: run `python download_models.py` while online.")
    if any(k in low for k in ("out of memory", "not enough memory", "defaultcpuallocator", "paging file", "[enforce fail")):
        hints.append("Not enough RAM: close other programs, or set PIXELTALK_DTYPE=bfloat16 and try again.")
    if "does not recognize this architecture" in low or "unrecognized configuration" in low:
        hints.append("Your `transformers` is too old for this model: run `pip install -U transformers`.")
    return text[:300] + (" " + " ".join(hints) if hints else "")


def load_engine(preferred: str = "auto", model_id: Optional[str] = None, device: Optional[str] = None):
    """
    Returns (engine_or_None, notes). Tries the chat VLM first and only falls back to
    BLIP-VQA if that fails; `notes` explains what happened so the UI can show it.
    """
    notes = []
    if preferred in ("auto", "qwen"):
        try:
            return QwenVLEngine(model_id=model_id, device=device), notes
        except Exception as exc:  # noqa: BLE001
            notes.append("Chat VLM unavailable. " + explain_load_error(exc))
            if preferred == "qwen":
                return None, notes
    try:
        engine = BlipVQAEngine(device=device)
        notes.append("Running on the BLIP-VQA fallback: answers will be short and less capable.")
        return engine, notes
    except Exception as exc:  # noqa: BLE001
        notes.append("BLIP-VQA fallback unavailable. " + explain_load_error(exc))
        return None, notes
