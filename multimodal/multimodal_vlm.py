"""
PixelTalk multimodal question answering.

Orchestrates: image -> (safety check when the user asks about eating/drinking)
-> chat VLM -> grounded, conversational answer, with proper multi-turn memory.

Conversation state lives in the PixelTalkMultimodalVLM object, the heavy model
lives in a shareable engine. Create ONE engine per process and ONE
PixelTalkMultimodalVLM per user session:

    engine, notes = load_engine()
    chat = PixelTalkMultimodalVLM(engine=engine)
    chat.set_image(pil_image)
    chat.answer_question("What is in the picture?")
"""

import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from PIL import Image

try:
    from .safety_guardrail import (
        MultimodalSafetyGuardrail, NOT_FOOD, SPOILED, UNCLEAR, WILD,
    )
    from .vlm_engines import load_engine, resolve_cached_hf_path  # noqa: F401  (re-exported)
except ImportError:  # executed as a plain script / from tests with sys.path tweaks
    from multimodal.safety_guardrail import (
        MultimodalSafetyGuardrail, NOT_FOOD, SPOILED, UNCLEAR, WILD,
    )
    from multimodal.vlm_engines import load_engine, resolve_cached_hf_path  # noqa: F401

SYSTEM_PROMPT = (
    "You are PixelTalk, a careful visual assistant chatting with a user about ONE image. "
    "Rules: "
    "1) Answer only from what is actually visible in the image and from the conversation so far. "
    "2) If something is not visible, too small, blurry or ambiguous, say you can't tell instead of guessing. "
    "3) Never invent text, brands, names, counts or details. When counting, count carefully. "
    "4) You may read text that is visible in the image. "
    "5) Do not identify real people from their faces; describe them instead. "
    "6) Reply in the language the user writes in. "
    "7) Be natural and concise: usually 1-3 sentences, longer only if the user asks for detail. "
    "8) For follow-ups like 'it', 'that', 'they', use the conversation to work out what is meant."
)

CLASSIFY_PROMPT = (
    "Look at the main subject of this image and classify it. Reply in exactly this one-line format:\n"
    "CATEGORY: <PREPARED_FOOD or SPOILED_FOOD or RAW_OR_WILD_FOOD or NOT_FOOD or UNCLEAR>; SUBJECT: <2-5 word name of the main subject>\n"
    "PREPARED_FOOD = ordinary food or drink that looks fresh (cooked, packaged, or normal shop-bought produce). "
    "SPOILED_FOOD = food with visible mold, fuzzy spots, rot or decay. "
    "RAW_OR_WILD_FOOD = mushrooms, wild berries, foraged plants. "
    "NOT_FOOD = anything that is not food (objects, animals, people, scenery, space). "
    "UNCLEAR = you cannot tell."
)

DESCRIBE_FOOD_PROMPT = (
    "Describe the food or drink in this image in one or two sentences: what it is, which "
    "ingredients or toppings are visible, and anything that looks unusual (discoloration, burnt "
    "or raw parts). Do not say whether it is safe to eat."
)

# One model instance is shared by all sessions; generation must not run concurrently.
_GENERATION_LOCK = threading.Lock()


class PixelTalkMultimodalVLM:
    def __init__(
        self,
        model_name: Optional[str] = None,
        lora_weights_path: Optional[str] = None,  # kept for backwards compatibility (unused)
        device: Optional[str] = None,
        engine: Any = None,
        preferred: str = "auto",
        max_history_turns: int = 6,
        load: bool = True,
    ):
        self.safety_guardrail = MultimodalSafetyGuardrail()
        self.max_history_turns = max_history_turns
        self.engine = engine
        self.load_notes: List[str] = []
        if self.engine is None and load:
            self.engine, self.load_notes = load_engine(preferred=preferred, model_id=model_name, device=device)
        self.reset()

    # ----------------------------------------------------------------- state
    @property
    def is_ready(self) -> bool:
        return self.engine is not None

    @property
    def engine_label(self) -> str:
        return getattr(self.engine, "label", "no model loaded")

    def reset(self):
        self.active_image: Optional[Image.Image] = None
        self.active_image_id = ""
        self.image_metadata: Dict[str, Any] = {}
        self.image_caption = ""
        self.history: List[Tuple[str, str]] = []
        self._classification: Optional[Tuple[str, str]] = None
        self.last_result: Optional[Dict[str, Any]] = None

    def set_image(self, image: Image.Image, image_id: str = "", metadata: Optional[Dict[str, Any]] = None, caption: str = ""):
        """Load a new image. Starts a fresh conversation."""
        self.reset()
        self.active_image = image
        self.active_image_id = image_id
        self.image_metadata = metadata or {}
        self.image_caption = caption or self.image_metadata.get("caption", "")

    # --------------------------------------------------------------- helpers
    def _recent_history(self) -> List[Tuple[str, str]]:
        if self.max_history_turns <= 0:
            return []
        return self.history[-self.max_history_turns:]

    def _ask(self, question: str, history=None, temperature=0.2, max_new_tokens=160) -> str:
        with _GENERATION_LOCK:
            return self.engine.chat(
                self.active_image, history if history is not None else self._recent_history(),
                question, SYSTEM_PROMPT, temperature=temperature, max_new_tokens=max_new_tokens,
            )

    def _classify_subject(self) -> Tuple[str, str]:
        """(category, subject) of the main object. Cached per image: one model call at most."""
        if self._classification is not None:
            return self._classification
        guard = self.safety_guardrail
        if getattr(self.engine, "name", "") == "blip-vqa":
            subject = self._ask("what is the main subject in this image?", history=[], temperature=0.0, max_new_tokens=15)
            result = (guard.classify_text(subject), subject.strip().lower())
        else:
            raw = self._ask(CLASSIFY_PROMPT, history=[], temperature=0.0, max_new_tokens=40)
            category, subject = guard.parse_vlm_label(raw)
            if category == UNCLEAR and subject:
                category = guard.classify_text(subject)
            result = (category, subject)
        self._classification = result
        return result

    def _result(self, answer, raw="", safety_alert=None, started=None, **extra) -> Dict[str, Any]:
        out = {
            "answer": answer,
            "raw_vlm_output": raw,
            "pixel_grounded": self.active_image is not None and self.is_ready,
            "safety_alert": safety_alert,
            "turn_index": len(self.history),
            "engine": self.engine_label,
            "latency_s": round(time.time() - started, 2) if started else None,
        }
        out.update(extra)
        return out

    # ----------------------------------------------------------------- main
    def _precheck(self, q: str, started: float) -> Optional[Dict[str, Any]]:
        """Return a ready-made result if the request can't go to the model, else None."""
        if not self.is_ready:
            notes = " ".join(self.load_notes) or "No vision model could be loaded."
            return self._result(
                "The vision model isn't available right now. " + notes + " Run `python download_models.py` and restart.",
                started=started, safety_alert=None,
            )
        if self.active_image is None:
            return self._result("Please upload or select an image first, then ask me about it.", started=started)
        if not q:
            return self._result("What would you like to know about the image?", started=started)
        return None

    def answer_stream(self, question: str, max_new_tokens: int = 160, temperature: float = 0.2) -> Iterator[str]:
        """
        Like answer_question, but yields the answer in pieces as the model writes it, so the UI
        can show text immediately instead of a long blank wait. When the generator is exhausted
        the full result dict is available as `self.last_result`.
        Falls back to ordinary generation if streaming is unavailable or fails.
        """
        started = time.time()
        q = (question or "").strip()
        self.last_result = None

        early = self._precheck(q, started)
        if early is None and self.safety_guardrail.is_consumption_query(q):
            early = self._answer_consumption_safe(q, started, max_new_tokens, temperature)
        if early is not None:
            self.last_result = early
            yield early["answer"]
            return

        pieces: List[str] = []
        error: Optional[Exception] = None
        text = ""
        _GENERATION_LOCK.acquire()
        stream = None
        try:
            try:
                if hasattr(self.engine, "chat_stream"):
                    stream = self.engine.chat_stream(
                        self.active_image, self._recent_history(), q, SYSTEM_PROMPT,
                        temperature=temperature, max_new_tokens=max_new_tokens,
                    )
            except Exception:  # noqa: BLE001 - streaming unavailable: use normal generation below
                stream = None

            if stream is not None:
                try:
                    for chunk in stream:
                        if chunk:
                            pieces.append(chunk)
                            yield chunk
                except GeneratorExit:
                    raise
                except Exception as exc:  # noqa: BLE001
                    error = exc

            if pieces:
                text = "".join(pieces)
            else:
                # Streaming unavailable, failed before the first token, or produced nothing:
                # fall back to ordinary generation.
                try:
                    text = self.engine.chat(
                        self.active_image, self._recent_history(), q, SYSTEM_PROMPT,
                        temperature=temperature, max_new_tokens=max_new_tokens,
                    )
                    error = None
                except Exception as exc2:  # noqa: BLE001
                    error = exc2
                if error is not None:
                    self.last_result = self._result(
                        f"Sorry, I hit an error while looking at the image ({type(error).__name__}). Please try again.",
                        started=started, error=str(error),
                    )
                    yield self.last_result["answer"]
                    return
                yield self._clean(text)  # shown once; the user sees the whole answer at this point
        finally:
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()  # stops background generation if the consumer walked away
                except Exception:  # noqa: BLE001
                    pass
            _GENERATION_LOCK.release()

        answer = self._clean(text)
        self.history.append((q, answer))
        self.last_result = self._result(answer, raw=answer, started=started)

    def _answer_consumption_safe(self, q, started, max_new_tokens, temperature) -> Dict[str, Any]:
        try:
            return self._answer_consumption(q, started, max_new_tokens, temperature)
        except Exception as exc:  # noqa: BLE001
            return self._result(
                f"Sorry, I hit an error while looking at the image ({type(exc).__name__}). Please try again.",
                started=started, error=str(exc),
            )

    def answer_question(self, question: str, max_new_tokens: int = 160, temperature: float = 0.2) -> Dict[str, Any]:
        started = time.time()
        q = (question or "").strip()

        early = self._precheck(q, started)
        if early is not None:
            return early

        try:
            if self.safety_guardrail.is_consumption_query(q):
                return self._answer_consumption(q, started, max_new_tokens, temperature)

            answer = self._ask(q, temperature=temperature, max_new_tokens=max_new_tokens)
            answer = self._clean(answer)
            self.history.append((q, answer))
            return self._result(answer, raw=answer, started=started)
        except Exception as exc:  # noqa: BLE001 - never crash the UI
            return self._result(
                f"Sorry, I hit an error while looking at the image ({type(exc).__name__}). Please try again.",
                started=started, error=str(exc),
            )

    def _answer_consumption(self, q, started, max_new_tokens, temperature) -> Dict[str, Any]:
        category, subject = self._classify_subject()
        permitted, message, decided = self.safety_guardrail.assess(
            q, self.image_metadata, subject, vlm_category=category,
        )

        if not permitted:
            self.history.append((q, message))
            alert = "HAZARD_BLOCKED" if decided in (WILD, NOT_FOOD, SPOILED) else "UNCERTAIN"
            return self._result(message, raw=subject, safety_alert=alert, started=started)

        # Ordinary food: describe it, then ALWAYS add the advisory. Never say "safe".
        if getattr(self.engine, "name", "") == "blip-vqa":
            description = f"This looks like {subject}." if subject else "This looks like food."
        else:
            description = self._clean(self._ask(DESCRIBE_FOOD_PROMPT, history=[], temperature=0.0, max_new_tokens=120))
        # Second layer: the model may have labelled the image "fine" yet describe mold or rot.
        if self.safety_guardrail.mentions_spoilage(description):
            _, warning = self.safety_guardrail.decide(SPOILED, subject)
            answer = f"{description}\n\n{warning}"
            self.history.append((q, answer))
            return self._result(answer, raw=description, safety_alert="HAZARD_BLOCKED", started=started)

        answer = f"{description}\n\n{message}"
        self.history.append((q, answer))
        return self._result(answer, raw=description, safety_alert="FOOD_ADVISORY", started=started)

    @staticmethod
    def _clean(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "I couldn't work out an answer from the image. Could you rephrase or ask about something more specific?"
        return text
