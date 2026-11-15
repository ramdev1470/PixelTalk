"""
Dataset and Vocabulary Pipeline for PixelTalk Seq2Seq Dialogue Manager
Handles tokenization, multi-turn history formatting, vocabulary building, and PyTorch DataLoaders.
"""

import json
import re
import os
import torch
from torch.utils.data import Dataset, DataLoader

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
CAP_START = "<cap>"
CAP_END = "</cap>"
USR_START = "<usr>"
BOT_START = "<bot>"
QRY_START = "<qry>"

SPECIAL_TOKENS = [
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    CAP_START, CAP_END, USR_START, BOT_START, QRY_START
]

def clean_and_tokenize(text: str) -> list[str]:
    """Tokenizes text preserving special tokens and standard punctuation."""
    if not text:
        return []
    # Protect special tokens like <cap>, </cap>, <usr>, <bot>, <qry>
    for sp in SPECIAL_TOKENS:
        text = text.replace(sp, f" {sp} ")
    # Lowercase everything except special tokens
    tokens = []
    raw_tokens = text.split()
    for tok in raw_tokens:
        if tok in SPECIAL_TOKENS:
            tokens.append(tok)
        else:
            # Separate punctuation
            sub_tokens = re.findall(r"[\w']+|[.,!?;:]", tok.lower())
            tokens.extend(sub_tokens)
    return tokens


class Vocabulary:
    """Manages mapping between words and integer token IDs."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = {}
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        # Register special tokens
        for sp in SPECIAL_TOKENS:
            self._add_word(sp)

    def _add_word(self, word: str) -> int:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.word_counts[word] = 1
            return idx
        else:
            self.word_counts[word] += 1
            return self.word2idx[word]

    def add_sentence(self, sentence: str):
        tokens = clean_and_tokenize(sentence)
        for tok in tokens:
            self._add_word(tok)

    def build_from_texts(self, texts: list[str], min_freq: int = 1):
        # Count words first
        counts = {}
        for txt in texts:
            tokens = clean_and_tokenize(txt)
            for tok in tokens:
                if tok in SPECIAL_TOKENS:
                    continue
                counts[tok] = counts.get(tok, 0) + 1

        for tok, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            if cnt >= min_freq:
                self._add_word(tok)

    def text_to_indices(self, text: str, add_sos: bool = False, add_eos: bool = False) -> list[int]:
        tokens = clean_and_tokenize(text)
        indices = []
        if add_sos:
            indices.append(self.sos_idx)
        for tok in tokens:
            indices.append(self.word2idx.get(tok, self.unk_idx))
        if add_eos:
            indices.append(self.eos_idx)
        return indices

    def indices_to_text(self, indices: list[int], remove_special: bool = True) -> str:
        words = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx == self.eos_idx:
                break
            w = self.idx2word.get(idx, UNK_TOKEN)
            if remove_special and w in SPECIAL_TOKENS:
                continue
            words.append(w)
        
        # Format punctuation neatly
        out = " ".join(words)
        out = re.sub(r'\s+([.,!?;:])', r'\1', out)
        if out and out[0].islower():
            out = out[0].upper() + out[1:]
        return out

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data = {
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "word_counts": self.word_counts
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: str):
        vocab = cls()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        vocab.word_counts = data.get("word_counts", {})
        vocab.pad_idx = vocab.word2idx[PAD_TOKEN]
        vocab.sos_idx = vocab.word2idx[SOS_TOKEN]
        vocab.eos_idx = vocab.word2idx[EOS_TOKEN]
        vocab.unk_idx = vocab.word2idx[UNK_TOKEN]
        return vocab

    def __len__(self):
        return len(self.word2idx)


def format_encoder_input(caption: str, history: list[tuple[str, str]], question: str, max_history_turns: int = 3) -> str:
    """
    Constructs the structured conversation-history encoded input for the encoder:
    <cap> [caption] </cap> <usr> [turn1_q] <bot> [turn1_a] ... <qry> [current_q]
    """
    parts = [CAP_START, caption.strip(), CAP_END]
    
    # Use sliding window over recent history turns
    recent_history = history[-max_history_turns:] if max_history_turns > 0 else []
    for q, a in recent_history:
        parts.extend([USR_START, q.strip(), BOT_START, a.strip()])
        
    parts.extend([QRY_START, question.strip()])
    return " ".join(parts)


class DialogueDataset(Dataset):
    """
    Multi-turn Conversational Dialogue Dataset grounded on image captions.
    """
    def __init__(self, data_json_path: str, vocab: Vocabulary = None, max_src_len: int = 128, max_trg_len: int = 40):
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len
        self.samples = []

        with open(data_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        all_texts_for_vocab = []

        for scenario in raw_data["scenarios"]:
            caption = scenario["caption"]
            dialogue = scenario["dialogue"]
            history = []
            all_texts_for_vocab.append(caption)

            for item in dialogue:
                if isinstance(item, (list, tuple)):
                    q, a = item[0], item[1]
                else:
                    q, a = item["question"], item["answer"]
                
                src_text = format_encoder_input(caption, history, q)
                trg_text = a
                all_texts_for_vocab.append(q)
                all_texts_for_vocab.append(a)

                self.samples.append({
                    "src_text": src_text,
                    "trg_text": trg_text,
                    "caption": caption,
                    "question": q,
                    "answer": a,
                    "category": scenario.get("category", "general")
                })

                history.append((q, a))

        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_from_texts(all_texts_for_vocab, min_freq=1)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        src_indices = self.vocab.text_to_indices(sample["src_text"], add_sos=False, add_eos=False)[:self.max_src_len]
        trg_indices = self.vocab.text_to_indices(sample["trg_text"], add_sos=True, add_eos=True)[:self.max_trg_len]

        return {
            "src": torch.tensor(src_indices, dtype=torch.long),
            "trg": torch.tensor(trg_indices, dtype=torch.long),
            "src_text": sample["src_text"],
            "trg_text": sample["trg_text"],
            "question": sample["question"],
            "answer": sample["answer"]
        }


def collate_dialogue_batch(batch, pad_idx=0):
    """Pads source and target sequences to form a uniform batch tensor."""
    srcs = [b["src"] for b in batch]
    trgs = [b["trg"] for b in batch]
    src_texts = [b["src_text"] for b in batch]
    trg_texts = [b["trg_text"] for b in batch]

    src_lengths = torch.tensor([len(s) for s in srcs], dtype=torch.long)
    trg_lengths = torch.tensor([len(t) for t in trgs], dtype=torch.long)

    max_src = max(len(s) for s in srcs)
    max_trg = max(len(t) for t in trgs)

    src_padded = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
    trg_padded = torch.full((len(batch), max_trg), pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, trgs)):
        src_padded[i, :len(s)] = s
        trg_padded[i, :len(t)] = t

    return {
        "src": src_padded,
        "trg": trg_padded,
        "src_lengths": src_lengths,
        "trg_lengths": trg_lengths,
        "src_texts": src_texts,
        "trg_texts": trg_texts
    }
