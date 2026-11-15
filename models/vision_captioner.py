"""
Vision Front-End Module for PixelTalk
Provides image captioning and multi-aspect scene feature extraction using Salesforce BLIP.
Extracts:
1. Primary descriptive caption
2. Environment & location context
3. Background scenery elements
4. Subject attire, objects, and visual attribute tags
"""

import os
import re
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageStat
import numpy as np

def resolve_cached_hf_path(model_id: str) -> str:
    """Resolves local snapshot directory for HuggingFace model if cached, avoiding network delays."""
    if os.path.isdir(model_id):
        return model_id
    org_repo = model_id.replace("/", "--")
    hub_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{org_repo}/snapshots")
    if os.path.exists(hub_dir):
        valid_snapshots = []
        for s in os.listdir(hub_dir):
            full_s = os.path.join(hub_dir, s)
            if os.path.isdir(full_s) and (
                os.path.exists(os.path.join(full_s, "config.json")) or 
                os.path.exists(os.path.join(full_s, "preprocessor_config.json"))
            ):
                valid_snapshots.append(full_s)
        if valid_snapshots:
            return valid_snapshots[0]
        snapshots = [os.path.join(hub_dir, s) for s in os.listdir(hub_dir) if os.path.isdir(os.path.join(hub_dir, s))]
        if snapshots:
            return snapshots[0]
    return model_id

class VisionCaptioner:
    """
    Vision front-end providing comprehensive image captioning and multi-aspect scene probing.
    """
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", use_hf: bool = True):
        self.model_name = resolve_cached_hf_path(model_name)
        self.use_hf = use_hf
        self.hf_processor = None
        self.hf_model = None

        if self.use_hf:
            self._load_hf_model()

    def _load_hf_model(self):
        """Loads Hugging Face BLIP vision-language model from cache."""
        try:
            import warnings
            warnings.filterwarnings("ignore")
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print(f"Loading HuggingFace vision model: {self.model_name}...")
            self.hf_processor = BlipProcessor.from_pretrained(self.model_name, local_files_only=True)
            self.hf_model = BlipForConditionalGeneration.from_pretrained(self.model_name, local_files_only=True)
            self.hf_model.eval()
            print("HuggingFace vision model loaded successfully.")
        except Exception as e:
            print(f"HuggingFace vision model unavailable ({e}). Using feature analyzer.")
            self.hf_processor = None
            self.hf_model = None

    def analyze_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extracts color palette, tone, orientation, and brightness."""
        rgb_img = image.convert("RGB")
        stat = ImageStat.Stat(rgb_img)
        avg_r, avg_g, avg_b = stat.mean[:3]
        brightness = (avg_r * 0.299 + avg_g * 0.587 + avg_b * 0.114)

        color_tones = []
        if avg_g > 105 and avg_g > avg_r and avg_g > avg_b:
            color_tones.append("green / forest nature")
        elif avg_b > 120 and avg_b > avg_r:
            color_tones.append("blue / cool outdoor")
        elif avg_r > 130 and avg_r > avg_b:
            color_tones.append("warm / amber red")
        elif brightness > 180:
            color_tones.append("bright daylight")
        elif brightness < 70:
            color_tones.append("dark / shadow")
        else:
            color_tones.append("balanced natural")

        small_img = rgb_img.resize((5, 5))
        pixels = list(small_img.getdata())
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in pixels[:6]]

        width, height = rgb_img.size
        aspect_ratio = round(width / max(height, 1), 2)
        orientation = "landscape" if aspect_ratio > 1.1 else ("portrait" if aspect_ratio < 0.9 else "square")

        return {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "orientation": orientation,
            "brightness": round(brightness, 1),
            "dominant_tone": color_tones[0] if color_tones else "balanced",
            "palette": hex_colors
        }

    def probe_aspect(self, image: Image.Image, prefix: str) -> str:
        """Probes specific visual aspects with conditional generation."""
        if self.hf_model is None or self.hf_processor is None:
            return ""
        try:
            import torch
            inputs = self.hf_processor(images=image, text=prefix, return_tensors="pt")
            with torch.no_grad():
                out = self.hf_model.generate(**inputs, max_new_tokens=25, repetition_penalty=1.4)
            ans = self.hf_processor.decode(out[0], skip_special_tokens=True).strip()
            # Remove prefix
            if ans.lower().startswith(prefix.lower()):
                ans = ans[len(prefix):].strip()
            # Deduplicate consecutive tokens
            clean_toks = []
            for tok in ans.split():
                if not clean_toks or tok.lower() != clean_toks[-1].lower():
                    clean_toks.append(tok)
            return " ".join(clean_toks)
        except Exception:
            return ""

    @staticmethod
    def _strip_article(text: str) -> str:
        """Remove a leading article/preposition as WORDS (str.lstrip strips characters, not words)."""
        return re.sub(r"^(?:in\s+)?(?:a|an|the)\s+", "", text.strip(), flags=re.IGNORECASE).strip()

    @staticmethod
    def _dedupe_words(text: str) -> str:
        out = []
        for tok in text.split():
            if not out or tok.lower() != out[-1].lower():
                out.append(tok)
        return " ".join(out)

    def generate_caption(self, image: Image.Image, filename: str = "") -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Generates the caption and scene breakdown used by the *baseline* pipeline.

        Everything comes from the model or from measured pixel statistics. Nothing is
        guessed from the file name and nothing is invented: if a probe fails the field
        stays empty and the baseline answers "I can't tell" for it.

        Returns (primary_caption, attributes, metadata).
        """
        rgb_img = image.convert("RGB")
        feats = self.analyze_image_features(rgb_img)

        primary_caption = ""
        setting_text = ""
        background_text = ""
        attire_text = ""

        if self.hf_model is not None and self.hf_processor is not None:
            try:
                import torch
                inputs = self.hf_processor(images=rgb_img, return_tensors="pt")
                with torch.no_grad():
                    out = self.hf_model.generate(
                        **inputs, max_new_tokens=40, repetition_penalty=1.5, no_repeat_ngram_size=2
                    )
                raw_cap = self._dedupe_words(self.hf_processor.decode(out[0], skip_special_tokens=True).strip())
                if raw_cap:
                    primary_caption = raw_cap[0].upper() + raw_cap[1:]
                    if not primary_caption.endswith("."):
                        primary_caption += "."

                setting_text = self._strip_article(self.probe_aspect(rgb_img, "this is a photo taken in"))
                background_text = self.probe_aspect(rgb_img, "in the background there are").strip()
                attire_text = self.probe_aspect(rgb_img, "a photo of a person wearing").strip()
            except Exception as e:
                print(f"BLIP caption warning: {e}")

        if not primary_caption:
            primary_caption = (
                f"An image ({feats['orientation']}, {feats['dominant_tone']} tones). "
                "No captioning model is loaded, so its content cannot be described."
            )

        # Only mention attire when a person is actually part of the caption.
        if not re.search(r"\b(man|woman|person|people|boy|girl|child|men|women|lady|guy|player|rider)\b", primary_caption.lower()):
            attire_text = ""

        stop_words = {"a", "an", "the", "in", "on", "at", "of", "with", "and", "is", "are", "her", "his",
                      "their", "its", "front", "this", "photo", "image", "tones", "model", "loaded",
                      "captioning", "content", "cannot", "described"}
        all_text = f"{primary_caption} {setting_text} {background_text} {attire_text}".lower()
        words = [re.sub(r"[^a-zA-Z]", "", w) for w in all_text.split()]
        seen, unique_attrs = set(), []
        for w in words:
            if len(w) > 3 and w not in stop_words and w not in seen:
                seen.add(w)
                unique_attrs.append(w)

        metadata = {
            "setting": setting_text,
            "background": background_text,
            "attire": attire_text,
            "dominant_tone": feats["dominant_tone"],
        }
        return primary_caption, unique_attrs[:6], metadata
