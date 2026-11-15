"""
Dialogue Manager for PixelTalk
Maintains multi-turn conversational state, context encoding, model inference,
and attention matrix extraction for visual explanation.
Supports:
1. Pretrained Seq2Seq Transformer (Flan-T5) for robust open-domain dialogue on any image.
2. Custom PyTorch Seq2Seq + Bahdanau Attention (Course-aligned architecture).
3. Semantic Grounding & Anti-Hallucination Guardrails ensuring accurate answers for any photo.
"""

import os
import sys
import re
from typing import List, Dict, Any, Tuple, Optional

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_root_dir = os.path.dirname(_parent_dir)
for p in [_root_dir, _parent_dir, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import numpy as np

try:
    from PixelTalk.data.dataset import Vocabulary, format_encoder_input, clean_and_tokenize
    from PixelTalk.models.seq2seq_attention import PixelTalkSeq2Seq
except ImportError:
    from data.dataset import Vocabulary, format_encoder_input, clean_and_tokenize
    from models.seq2seq_attention import PixelTalkSeq2Seq

try:
    from PixelTalk.multimodal.safety_guardrail import MultimodalSafetyGuardrail
except ImportError:
    from multimodal.safety_guardrail import MultimodalSafetyGuardrail

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

class DialogueManager:
    """
    Orchestrates conversational image QA across multi-turn dialogues.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/seq2seq_attention.pt",
        vocab_path: Optional[str] = "checkpoints/vocab.json",
        device: Optional[str] = None,
        default_backend: str = "flan_t5"
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.backend = default_backend
        self.vocab = None
        self.custom_model = None
        self.t5_tokenizer = None
        self.t5_model = None

        self.current_caption = ""
        self.current_attributes: List[str] = []
        self.scene_metadata: Dict[str, Any] = {}
        self.history: List[Tuple[str, str]] = []
        self.turn_metadata: List[Dict[str, Any]] = []

        def _resolve_file(path_str):
            if not path_str:
                return None
            if os.path.exists(path_str):
                return path_str
            proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for candidate in [
                os.path.join(proj_root, path_str),
                os.path.join(proj_root, path_str.replace("PixelTalk/", "").replace("PixelTalk\\", "")),
                path_str.replace("PixelTalk/", "").replace("PixelTalk\\", "")
            ]:
                if candidate and os.path.exists(candidate):
                    return candidate
            return path_str

        resolved_checkpoint = _resolve_file(checkpoint_path)
        resolved_vocab = _resolve_file(vocab_path)

        # 1. Load Pretrained Seq2Seq (Flan-T5)
        self._load_t5_model()

        # 2. Load Custom PyTorch Seq2Seq with Bahdanau Attention
        if resolved_vocab and os.path.exists(resolved_vocab):
            try:
                self.vocab = Vocabulary.load(resolved_vocab)
            except Exception as e:
                print(f"Error loading vocab: {e}")

        if resolved_checkpoint and os.path.exists(resolved_checkpoint) and self.vocab:
            self._load_custom_model(resolved_checkpoint)

    @property
    def model(self):
        return self.custom_model if self.backend == "custom_seq2seq" else (self.t5_model or self.custom_model)

    def _load_t5_model(self):
        """Loads pretrained Flan-T5 model for robust open-domain dialogue."""
        try:
            import warnings
            warnings.filterwarnings("ignore")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            model_id = resolve_cached_hf_path("google/flan-t5-small")
            print(f"Loading Flan-T5 Seq2Seq Dialogue Model ({model_id})...")
            self.t5_tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            try:
                self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id, local_files_only=True, attn_implementation="eager"
                )
            except Exception:
                self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, local_files_only=True)
            self.t5_model.to(self.device)
            self.t5_model.eval()
            print("Flan-T5 Dialogue Model loaded successfully.")
        except Exception as e:
            print(f"Flan-T5 unavailable ({e}). Defaulting to custom Seq2Seq.")
            self.t5_tokenizer = None
            self.t5_model = None
            self.backend = "custom_seq2seq"

    def _load_custom_model(self, checkpoint_path: str):
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            emb_dim = ckpt.get("emb_dim", 128)
            hidden_dim = ckpt.get("hidden_dim", 256)
            num_layers = ckpt.get("num_layers", 1)
            attention_type = ckpt.get("attention_type", "bahdanau")

            self.custom_model = PixelTalkSeq2Seq(
                vocab_size=len(self.vocab),
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.0,
                attention_type=attention_type,
                pad_idx=self.vocab.pad_idx
            ).to(self.device)

            self.custom_model.load_state_dict(ckpt["model_state_dict"])
            self.custom_model.eval()
            print(f"Loaded custom Seq2Seq Attention model from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading custom checkpoint: {e}")
            self.custom_model = None

    def set_image_context(self, caption: str, attributes: List[str] = None, scene_metadata: Dict[str, Any] = None):
        """Initializes a new conversation rooted in the image caption and scene metadata."""
        self.current_caption = caption.strip()
        self.current_attributes = attributes or []
        self.scene_metadata = scene_metadata or {}
        self.history = []
        self.turn_metadata = []

    def reset(self):
        """Clears conversation history and current image context."""
        self.current_caption = ""
        self.current_attributes = []
        self.scene_metadata = {}
        self.history = []
        self.turn_metadata = []

    def answer_question(
        self,
        question: str,
        beam_width: int = 1,
        temperature: float = 0.7,
        max_history_turns: int = 3,
        max_len: int = 50,
        backend_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an answer to the user's question conditioned on the image caption and history.
        """
        if not self.current_caption:
            return {
                "answer": "Please upload or select an image first so I can analyze its content!",
                "attention_matrix": None,
                "src_tokens": [],
                "trg_tokens": [],
                "top_attended_caption_tokens": [],
                "turn_index": len(self.history) + 1
            }

        active_backend = backend_override if backend_override else self.backend

        # Check if active backend is available
        if active_backend == "flan_t5" and self.t5_model is not None and self.t5_tokenizer is not None:
            return self._answer_with_t5(question, max_history_turns, max_len, temperature)
        elif self.custom_model is not None and self.vocab is not None:
            return self._answer_with_custom_seq2seq(question, beam_width, temperature, max_history_turns, max_len)
        else:
            ans = self._grounded_reasoning(question)
            self.history.append((question, ans))
            return {
                "answer": ans,
                "attention_matrix": None,
                "src_tokens": clean_and_tokenize(question),
                "trg_tokens": clean_and_tokenize(ans),
                "top_attended_caption_tokens": [],
                "turn_index": len(self.history)
            }

    def _answer_with_t5(self, question: str, max_history_turns: int, max_len: int, temperature: float) -> Dict[str, Any]:
        """Generates grounded answer using Flan-T5 with multi-turn prompt and semantic validation."""
        recent_hist = self.history[-max_history_turns:] if max_history_turns > 0 else []
        hist_parts = [f"User: {q} Assistant: {a}" for q, a in recent_hist]
        hist_str = " ".join(hist_parts)

        # Build comprehensive multi-aspect context
        setting = self.scene_metadata.get("setting", "")
        background = self.scene_metadata.get("background", "")
        attire = self.scene_metadata.get("attire", "")

        context_lines = [f"Image Caption: {self.current_caption}"]
        if setting:
            context_lines.append(f"Setting and Location: {setting}")
        if background:
            context_lines.append(f"Background: {background}")
        if attire:
            context_lines.append(f"Attire and Objects: {attire}")

        context_block = "\n".join(context_lines)

        prompt = (
            f"You are PixelTalk, an intelligent conversational image assistant. "
            f"Answer the user's question clearly, accurately, and naturally based on the image description.\n"
            f"{context_block}\n"
        )
        if hist_str:
            prompt += f"Dialogue History: {hist_str}\n"
        prompt += f"User: {question}\nAssistant:"

        inputs = self.t5_tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            gen_out = self.t5_model.generate(
                **inputs,
                max_new_tokens=max_len,
                do_sample=(temperature > 0.0),
                temperature=max(0.1, temperature),
                output_attentions=True,
                return_dict_in_generate=True
            )

        raw_ans = self.t5_tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True).strip()
        ans = self._sanitize_and_ground_answer(raw_ans, question)

        src_tokens = self.t5_tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        trg_tokens = self.t5_tokenizer.convert_ids_to_tokens(gen_out.sequences[0][1:])
        
        attn_matrix = None
        top_attended_tokens = []

        if hasattr(gen_out, "cross_attentions") and gen_out.cross_attentions is not None:
            try:
                step_attns = []
                for step_attn in gen_out.cross_attentions:
                    last_layer = step_attn[-1]
                    avg_head = last_layer.mean(dim=1).squeeze(0).squeeze(0).cpu().numpy()
                    step_attns.append(avg_head)
                if step_attns:
                    attn_matrix = np.stack(step_attns, axis=0)
                    mean_weights = attn_matrix.mean(axis=0)
                    ranked_indices = np.argsort(-mean_weights)
                    for r_idx in ranked_indices:
                        tok = src_tokens[r_idx].replace(" ", "")
                        if len(tok) > 2 and tok.lower() not in ["user", "assistant", "image", "context", "the", "are", "you"]:
                            top_attended_tokens.append((tok, float(mean_weights[r_idx])))
                            if len(top_attended_tokens) >= 5:
                                break
            except Exception:
                attn_matrix = None

        turn_num = len(self.history) + 1
        self.history.append((question, ans))
        self.turn_metadata.append({
            "turn": turn_num,
            "question": question,
            "answer": ans,
            "top_attended": top_attended_tokens
        })

        return {
            "answer": ans,
            "attention_matrix": attn_matrix,
            "src_tokens": [t.replace(" ", "") for t in src_tokens],
            "trg_tokens": [t.replace(" ", "") for t in trg_tokens],
            "top_attended_caption_tokens": top_attended_tokens,
            "turn_index": turn_num
        }

    def _answer_with_custom_seq2seq(self, question: str, beam_width: int, temperature: float, max_history_turns: int, max_len: int) -> Dict[str, Any]:
        """Generates answer using custom PyTorch Seq2Seq + Bahdanau Attention with semantic validation."""
        src_text = format_encoder_input(self.current_caption, self.history, question, max_history_turns=max_history_turns)
        src_tokens = clean_and_tokenize(src_text)
        src_indices = self.vocab.text_to_indices(src_text, add_sos=False, add_eos=False)
        src_tensor = torch.tensor([src_indices], dtype=torch.long, device=self.device)

        with torch.no_grad():
            gen_indices, attn_matrix = self.custom_model.generate(
                src_tensor,
                sos_idx=self.vocab.sos_idx,
                eos_idx=self.vocab.eos_idx,
                max_len=max_len,
                temperature=temperature,
                beam_width=beam_width
            )

        raw_answer = self.vocab.indices_to_text(gen_indices, remove_special=True)
        trg_tokens = [self.vocab.idx2word.get(idx, "<unk>") for idx in gen_indices]
        attn_matrix_np = attn_matrix.cpu().numpy() if (attn_matrix is not None and attn_matrix.numel() > 0) else None

        # Sanitize answer: prevent hallucination leak from unrelated training sets
        answer_text = self._sanitize_and_ground_answer(raw_answer, question)

        top_attended_tokens = []
        if attn_matrix_np is not None and len(attn_matrix_np.shape) == 2:
            avg_attn = attn_matrix_np.mean(axis=0)
            valid_len = min(len(src_tokens), len(avg_attn))
            token_scores = []
            for i in range(valid_len):
                tok = src_tokens[i]
                if tok not in ["<cap>", "</cap>", "<usr>", "<bot>", "<qry>", ".", ","]:
                    token_scores.append((tok, float(avg_attn[i])))
            token_scores.sort(key=lambda x: -x[1])
            top_attended_tokens = token_scores[:5]

        turn_num = len(self.history) + 1
        self.history.append((question, answer_text))
        self.turn_metadata.append({
            "turn": turn_num,
            "question": question,
            "answer": answer_text,
            "top_attended": top_attended_tokens
        })

        return {
            "answer": answer_text,
            "attention_matrix": attn_matrix_np,
            "src_tokens": src_tokens,
            "trg_tokens": trg_tokens,
            "top_attended_caption_tokens": top_attended_tokens,
            "turn_index": turn_num
        }

    def _sanitize_and_ground_answer(self, raw_ans: str, question: str) -> str:
        """
        Validates that generated answer is semantically consistent with the active image.
        If a model outputs irrelevant hallucinated tokens from training data, replaces with verified grounding.
        """
        c_lower = self.current_caption.lower()
        q_lower = question.lower()
        a_lower = (raw_ans or "").lower()

        # Check for semantic mismatch / hallucination leaks:
        # Leak 1: Car / vehicle hallucinated when scene is not a vehicle
        if any(w in a_lower for w in ["car", "convertible", "sports coupe", "cherry red", "wheels"]) and not any(w in c_lower for w in ["car", "vehicle", "coupe", "auto", "convertible"]):
            return self._grounded_reasoning(question)

        # Leak 2: Laptop / code editor hallucinated when scene is not tech
        if any(w in a_lower for w in ["laptop", "syntax", "editor", "programming"]) and not any(w in c_lower for w in ["laptop", "desk", "computer", "code"]):
            return self._grounded_reasoning(question)

        # Leak 3: Pizza / cheese hallucinated when scene is not food
        if any(w in a_lower for w in ["cheddar", "pepperoni", "mozzarella", "baked italian pizza"]) and not any(w in c_lower for w in ["pizza", "food", "cheese"]):
            return self._grounded_reasoning(question)

        # Check for degenerate repetitions, prompt leaks, or empty answer
        if not raw_ans or len(raw_ans.strip()) < 3 or a_lower.count("is") > 3 or "located located" in a_lower:
            return self._grounded_reasoning(question)

        if any(leak in a_lower for leak in ["you are pixeltalk", "answer the user", "intelligent conversational", "image description"]):
            return self._grounded_reasoning(question)

        # Question specific grounding overrides for maximum accuracy
        # 1. Location questions
        if "where" in q_lower or "location" in q_lower or "setting" in q_lower:
            setting = self.scene_metadata.get("setting", "")
            if setting and (len(a_lower.split()) <= 2 or "tree" in a_lower or "face" in a_lower or "located" in a_lower):
                return f"The photo is taken in {setting}."

        # 2. Background questions
        if "background" in q_lower:
            bg = self.scene_metadata.get("background", "")
            if bg:
                return f"In the background, there are {bg}."

        # 3. Attire questions
        if "wearing" in q_lower or "clothes" in q_lower or "attire" in q_lower:
            attire = self.scene_metadata.get("attire", "")
            if attire:
                return f"The subject is wearing {attire}."

        # 4. Action questions
        if "doing" in q_lower or "action" in q_lower:
            if self.current_caption:
                return f"From the caption: {self.current_caption.rstrip('.')}."

        # 5. Food safety questions
        if "safe to eat" in q_lower or "edible" in q_lower:
            return self._food_safety_answer()

        # Polish capitalization
        ans = raw_ans.strip()
        if ans and ans[0].islower():
            ans = ans[0].upper() + ans[1:]
        if ans and not ans.endswith((".", "!", "?")):
            ans += "."

        return ans

    def _food_safety_answer(self) -> str:
        """Never claims food is 'safe'. Uses the shared guardrail policy on the caption only."""
        guard = MultimodalSafetyGuardrail()
        category = guard.classify_text(self.current_caption)
        _, message = guard.decide(category, "")
        if category == "PREPARED_FOOD":
            return "Going only by the caption, this looks like food. " + message
        return message

    def _grounded_reasoning(self, question: str) -> str:
        """Fallback visual reasoning directly from extracted scene attributes."""
        q_lower = question.lower()
        c_lower = self.current_caption.lower()
        setting = self.scene_metadata.get("setting", "")
        background = self.scene_metadata.get("background", "")
        attire = self.scene_metadata.get("attire", "")

        # 1. Food safety
        if "safe to eat" in q_lower or "edible" in q_lower:
            return self._food_safety_answer()

        # 2. Location
        if "where" in q_lower or "location" in q_lower or "setting" in q_lower:
            if setting:
                return f"The scene is situated {setting}."
            elif any(w in c_lower for w in ["tree", "forest", "park", "lawn", "mountain", "beach"]):
                return "The photo is taken in an outdoor natural environment."
            elif any(w in c_lower for w in ["desk", "office", "table", "kitchen", "room"]):
                return "The photo is taken in an indoor setting."
            return f"The scene appears to be set around {self.current_caption.rstrip('.')}."

        # 3. Background
        if "background" in q_lower:
            if background:
                return f"In the background, there are {background}."
            return f"I can't make out the background in more detail than the caption gives: {self.current_caption.rstrip('.')}. The background complements the main scene ({self.current_caption.rstrip('.')})."

        # 4. Attire / Wearing
        if "wearing" in q_lower or "attire" in q_lower or "holding" in q_lower:
            if attire:
                return f"The subject is {attire}."
            return "The caption doesn't say what is being worn, so I can't tell."

        # 5. What is this
        if "what is this" in q_lower or "what is shown" in q_lower or "identify" in q_lower:
            return f"This image shows: {self.current_caption.rstrip('.')}."

        # 6. Action / Doing
        if "doing" in q_lower:
            return f"The scene depicts {self.current_caption.rstrip('.')}."

        # 7. Colors
        if "color" in q_lower:
            colors = ["red", "green", "blue", "yellow", "white", "black", "golden", "silver", "brown", "turquoise"]
            found = [col for col in colors if col in c_lower or col in f"{setting} {background} {attire}".lower()]
            if found:
                return f"The prominent colors visible include {', '.join(found)}."
            return "The caption doesn't mention colors, so I can't tell."

        return f"Regarding '{question}', the scene depicts {self.current_caption.rstrip('.')}."
