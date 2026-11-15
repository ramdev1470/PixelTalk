"""
Safety guardrail for PixelTalk.

Handles "can I eat / drink / swallow this?" style questions responsibly.
A photo can never prove that food is safe (bacteria, spoilage, allergens and
cooking temperature are invisible), so the policy is:

  * NOT_FOOD      -> refuse clearly: "this is not food".
  * SPOILED       -> refuse: visible mold / rot means "do not eat", full stop.
  * WILD          -> refuse: wild mushrooms / berries / plants can't be told
                     apart from deadly look-alikes by a photo.
  * PREPARED_FOOD -> allowed to discuss, ALWAYS with an advisory. Never "yes, it
                     is safe".
  * UNCLEAR       -> be cautious, ask for a clearer photo.

All matching is whole-word (so "carrots" is not "car", "wooden" is not "wood").
"""

import re
from typing import Any, Dict, Optional, Tuple

PREPARED_FOOD = "PREPARED_FOOD"
SPOILED = "SPOILED"
WILD = "WILD"
NOT_FOOD = "NOT_FOOD"
UNCLEAR = "UNCLEAR"
VALID_CATEGORIES = (PREPARED_FOOD, SPOILED, WILD, NOT_FOOD, UNCLEAR)


def _singular(word: str) -> str:
    """Very small singulariser: berries->berry, tomatoes->tomato, carrots->carrot."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("oes", "ses", "xes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str):
    """Lower-cased whole-word tokens (plus singular forms)."""
    words = re.findall(r"[a-z]+", (text or "").lower())
    out = set(words)
    out.update(_singular(w) for w in words)
    return out


class MultimodalSafetyGuardrail:
    def __init__(self):
        self.edible_words = {
            "pizza", "burger", "salad", "fruit", "apple", "banana", "orange", "grape",
            "strawberry", "blueberry", "raspberry", "mango", "peach", "pear", "lemon",
            "bread", "toast", "soup", "cheese", "sushi", "cake", "sandwich", "pancake",
            "waffle", "vegetable", "carrot", "potato", "tomato", "onion", "broccoli",
            "corn", "pea", "bean", "pasta", "noodle", "ramen", "rice", "curry", "stew",
            "steak", "chicken", "meat", "fish", "egg", "pie", "pastry", "cookie",
            "biscuit", "chocolate", "dessert", "snack", "taco", "burrito", "fries",
            "food", "meal", "dish", "dinner", "lunch", "breakfast", "cereal", "yogurt",
            "icecream", "donut", "doughnut", "muffin", "bagel", "pepperoni", "sausage",
            "coffee", "tea", "juice", "drink", "beverage", "smoothie",
        }
        self.wild_words = {
            "mushroom", "fungus", "fungi", "toadstool", "berry", "foraged", "wild",
            "amanita", "agaric", "puffball", "chanterelle", "morel",
        }
        self.spoiled_words = {
            "mold", "mould", "moldy", "mouldy", "rotten", "rotting", "rot", "spoiled",
            "spoilt", "decayed", "decaying", "slimy", "putrid", "expired",
        }
        self.non_food_words = {
            "laptop", "computer", "phone", "smartphone", "camera", "battery", "wire",
            "cable", "screen", "keyboard", "electronics", "chemical", "detergent",
            "cleaner", "soap", "bleach", "medicine", "pill", "tablet", "car", "vehicle",
            "engine", "truck", "bike", "bicycle", "metal", "plastic", "glass", "tool",
            "galaxy", "star", "asteroid", "planet", "space", "mountain", "lake", "tree",
            "building", "road", "street", "person", "woman", "man", "dog", "cat",
            "desk", "table", "chair", "toy", "book", "notebook",
        }
        self.consumption_patterns = [
            r"\bsafe\s+to\s+(eat|drink|consume|swallow)\b",
            r"\b(can|could|may|should)\s+(i|we|you|one)\s+(eat|drink|consume|swallow)\b",
            r"\bis\s+(it|this|that)\s+(edible|poisonous|toxic|drinkable|eatable)\b",
            r"\b(edible|poisonous|toxic|drinkable|eatable)\b",
            r"\bsafe\s+for\s+(consumption|eating)\b",
        ]

    # ------------------------------------------------------------------ intent
    def is_consumption_query(self, question: str) -> bool:
        q = (question or "").lower()
        return any(re.search(p, q) for p in self.consumption_patterns)

    # ------------------------------------------------------------ classification
    def classify_text(self, *texts: str) -> str:
        """
        Classify from free text (subject name / caption / question) using whole
        words. Order matters: wild first (highest risk), then food, then non-food.
        Returns one of VALID_CATEGORIES.
        """
        for text in texts:
            if tokenize(text) & self.wild_words:
                return WILD
        for text in texts:
            toks = tokenize(text)
            non_food_only = bool(toks & self.non_food_words) and not (toks & self.edible_words)
            if toks & self.spoiled_words and not non_food_only:
                return SPOILED  # visible mold/rot on something that is (or may be) food
            if toks & self.edible_words:
                return PREPARED_FOOD
            if toks & self.non_food_words:
                return NOT_FOOD
        return UNCLEAR

    def mentions_spoilage(self, text: str) -> bool:
        """
        True if the text says something is moldy / rotten / spoiled. Handles simple negation so that
        "no visible mold" or "free of rot" does not trigger a false alarm.
        """
        low = (text or "").lower()
        for match in re.finditer(r"[a-z]+", low):
            if match.group(0) in self.spoiled_words or _singular(match.group(0)) in self.spoiled_words:
                before = low[max(0, match.start() - 40):match.start()]
                if re.search(r"\b(no|not|without|free of|free from|none|nor|any sign of)\b[^.;]*$", before):
                    continue
                return True
        return False

    @staticmethod
    def parse_vlm_label(raw: str) -> Tuple[str, str]:
        """
        Parse a model reply of the form 'CATEGORY: X; SUBJECT: y'.
        Returns (category, subject). Category is UNCLEAR if it can't be parsed.
        """
        text = (raw or "").strip()
        cat = UNCLEAR
        m = re.search(r"CATEGORY\s*[:=]\s*([A-Za-z_ ]+)", text, re.IGNORECASE)
        label_source = m.group(1) if m else text
        norm = re.sub(r"[^A-Za-z]+", "_", label_source.upper())
        # Check NOT_FOOD before PREPARED_FOOD-style substrings ("NOT_FOOD" contains "FOOD").
        for c in (NOT_FOOD, SPOILED, WILD, PREPARED_FOOD, UNCLEAR):
            if c in norm:
                cat = c
                break
        if cat == UNCLEAR and "RAW" in norm:
            cat = WILD
        if cat == UNCLEAR and re.search(r"MOLD|MOULD|ROT|DECAY", norm):
            cat = SPOILED
        subject = ""
        s = re.search(r"SUBJECT\s*[:=]\s*(.+)", text, re.IGNORECASE)
        if s:
            subject = re.split(r"[\n;]", s.group(1))[0].strip().strip(".").lower()
        return cat, subject

    # ----------------------------------------------------------------- policy

    def decide(self, category: str, subject: str = "") -> Tuple[bool, str]:
        """Pure policy: map a category to (is_permitted, message)."""
        label = (subject or "").strip() or "this item"

        if category == WILD:
            return False, (
                "⚠️ High-Risk Foraging Warning: this looks like a wild mushroom, berry or plant. "
                "Deadly look-alikes (for example the death cap, Amanita phalloides) cannot be "
                "told apart from edible species in a photo. Please do not eat anything foraged "
                "based on an AI answer; ask a local expert in person."
            )
        if category == SPOILED:
            return False, (
                f"⚠️ Spoilage Warning: {label} shows visible signs of mold or rot, so please do not "
                "eat it. Mold can spread through soft foods such as bread far beyond what you can "
                "see, and cutting off the visible part does not make it safe. Throw the whole item away."
            )
        if category == NOT_FOOD:
            return False, (
                f"⚠️ Safety Guardrail Alert: {label} does not look like food, so it is not "
                "something to eat or drink. Swallowing non-food items can cause poisoning, "
                "choking or blockage."
            )
        if category == PREPARED_FOOD:
            return True, (
                "ℹ️ Food Safety Advisory: a photo can't show bacteria, spoilage, how well "
                "something was cooked, or allergens. Check freshness, storage and ingredients "
                "yourself before eating."
            )
        return False, (
            "ℹ️ I can't tell from this image whether it is food, and I won't guess about "
            "eating something. A clearer, closer photo would help."
        )

    # ------------------------------------------------------------ public API
    def assess(
        self,
        question: str,
        image_metadata: Optional[Dict[str, Any]] = None,
        vlm_detected_subject: str = "",
        vlm_category: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Returns (is_permitted, message, final_category).
        Non-consumption questions are always permitted: (True, "", None).
        If `vlm_category` (from the vision model) is given it takes priority over text matching,
        except that wild-plant risk and visible spoilage are always checked first.
        """
        if not self.is_consumption_query(question):
            return True, "", None

        meta = image_metadata or {}
        subject = (vlm_detected_subject or "").strip()

        def done(category):
            permitted, message = self.decide(category, subject)
            return permitted, message, category

        if vlm_category == WILD or self.classify_text(subject, question) == WILD:
            return done(WILD)
        if vlm_category in (PREPARED_FOOD, None, UNCLEAR) and self.mentions_spoilage(subject):
            return done(SPOILED)  # e.g. the model labelled it fine but named it "moldy bread"
        if vlm_category in (PREPARED_FOOD, SPOILED, NOT_FOOD, UNCLEAR):
            return done(vlm_category)

        meta_cat = str(meta.get("category", "")).lower()
        if meta_cat in {"technology", "vehicles", "electronics", "hardware"}:
            return done(NOT_FOOD)
        if meta_cat == "food":
            return done(PREPARED_FOOD)

        # What the model says the main object is beats the caption.
        category = self.classify_text(subject)
        if category == UNCLEAR:
            category = self.classify_text(meta.get("caption", ""))
        return done(category)

    def verify_safety(
        self,
        question: str,
        image_metadata: Optional[Dict[str, Any]] = None,
        vlm_detected_subject: str = "",
        vlm_category: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Backwards-compatible wrapper around assess(): returns (is_permitted, message)."""
        permitted, message, _ = self.assess(question, image_metadata, vlm_detected_subject, vlm_category)
        return permitted, message
