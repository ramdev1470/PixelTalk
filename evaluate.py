"""
Evaluation and Benchmarking Pipeline for PixelTalk
Calculates:
1. BLEU-1, BLEU-2, BLEU-3, BLEU-4 Scores
2. Perplexity (PPL) and Cross-Entropy Loss
3. Turn-by-turn Context Retention
4. Attention Distribution Analysis (Caption vs History vs Query)
"""

import os
import sys
import json
import math
from collections import Counter
from typing import List, Dict, Any, Tuple

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
for p in [_parent_dir, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from PixelTalk.data.dataset import DialogueDataset, collate_dialogue_batch, Vocabulary, clean_and_tokenize
    from PixelTalk.models.seq2seq_attention import PixelTalkSeq2Seq
    from PixelTalk.models.dialogue_manager import DialogueManager
except ImportError:
    from data.dataset import DialogueDataset, collate_dialogue_batch, Vocabulary, clean_and_tokenize
    from models.seq2seq_attention import PixelTalkSeq2Seq
    from models.dialogue_manager import DialogueManager

def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Returns counter of n-grams."""
    return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])

def sentence_bleu(reference: List[str], candidate: List[str], max_n: int = 4) -> Dict[str, float]:
    """
    Computes smoothed BLEU-1, 2, 3, 4 scores.
    """
    if not candidate or not reference:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}

    ref_len = len(reference)
    cand_len = len(candidate)

    # Brevity penalty
    bp = 1.0 if cand_len > ref_len else math.exp(1 - (ref_len / max(cand_len, 1)))

    scores = {}
    p_n = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(reference, n)
        cand_ngrams = get_ngrams(candidate, n)

        clipped_count = sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in cand_ngrams.items())
        total_cand = max(len(candidate) - n + 1, 1)
        precision = (clipped_count + 1e-7) / total_cand # smoothing
        p_n.append(precision)
        
        # Cumulative geometric mean
        geom_mean = math.exp(sum(math.log(p) for p in p_n) / n)
        scores[f"bleu_{n}"] = round(bp * geom_mean * 100.0, 2)

    return scores

def resolve_benchmark_path(p: str, default_rel: str) -> str:
    if p and os.path.exists(p):
        return p
    for cand in [
        default_rel,
        os.path.join(_current_dir, default_rel),
        os.path.join(_parent_dir, default_rel),
        os.path.join(_current_dir, default_rel.replace("PixelTalk/", "")),
        default_rel.replace("PixelTalk/", ""),
    ]:
        if cand and os.path.exists(cand):
            return cand
    return p or default_rel

def evaluate_benchmarks(
    checkpoint_path: str = "checkpoints/seq2seq_attention.pt",
    vocab_path: str = "checkpoints/vocab.json",
    data_path: str = "data/coco_dialogue_data.json"
):
    print("=" * 65)
    print("      PixelTalk Dialogue Manager Evaluation Benchmark")
    print("=" * 65)

    checkpoint_path = resolve_benchmark_path(checkpoint_path, "checkpoints/seq2seq_attention.pt")
    vocab_path = resolve_benchmark_path(vocab_path, "checkpoints/vocab.json")
    data_path = resolve_benchmark_path(data_path, "data/coco_dialogue_data.json")

    if not os.path.exists(checkpoint_path) or not os.path.exists(vocab_path):
        print(f"[!] Checkpoint or vocab missing: {checkpoint_path}, {vocab_path}")
        return

    dm = DialogueManager(checkpoint_path=checkpoint_path, vocab_path=vocab_path)
    if dm.model is None:
        print("[!] Model could not be loaded for evaluation.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data["scenarios"]
    print(f"Loaded {len(scenarios)} test scenarios from {data_path}")

    # Track metrics
    bleu_records = {f"bleu_{i}": [] for i in range(1, 5)}
    turn_accuracies = {} # turn_idx -> list of BLEU-1
    attn_caption_shares = []
    attn_query_shares = []
    sample_dialogues = []

    for s_idx, sc in enumerate(scenarios):
        caption = sc["caption"]
        dm.set_image_context(caption, sc.get("attributes", []))
        
        for t_idx, item in enumerate(sc["dialogue"]):
            q = item[0] if isinstance(item, (list, tuple)) else item["question"]
            ref_a = item[1] if isinstance(item, (list, tuple)) else item["answer"]

            res = dm.answer_question(q, beam_width=2, temperature=0.7)
            gen_a = res["answer"]
            attn_matrix = res["attention_matrix"]
            src_tokens = res["src_tokens"]

            # Compute BLEU
            ref_toks = clean_and_tokenize(ref_a)
            cand_toks = clean_and_tokenize(gen_a)
            b_scores = sentence_bleu(ref_toks, cand_toks)
            for k, v in b_scores.items():
                bleu_records[k].append(v)

            # Turn-level tracking
            turn_no = t_idx + 1
            if turn_no not in turn_accuracies:
                turn_accuracies[turn_no] = []
            turn_accuracies[turn_no].append(b_scores["bleu_1"])

            # Analyze attention allocation
            if attn_matrix is not None and len(attn_matrix.shape) == 2 and len(src_tokens) > 0:
                mean_attn = attn_matrix.mean(axis=0)
                # Find caption tokens vs query tokens
                cap_weight = 0.0
                qry_weight = 0.0
                in_cap = False
                in_qry = False
                for idx, tok in enumerate(src_tokens[:len(mean_attn)]):
                    if tok == "<cap>":
                        in_cap = True; continue
                    elif tok == "</cap>":
                        in_cap = False; continue
                    elif tok == "<qry>":
                        in_qry = True; continue

                    if in_cap:
                        cap_weight += float(mean_attn[idx])
                    elif in_qry:
                        qry_weight += float(mean_attn[idx])

                tot = cap_weight + qry_weight + 1e-6
                attn_caption_shares.append(cap_weight / tot)
                attn_query_shares.append(qry_weight / tot)

            # Collect diverse qualitative samples
            if len(sample_dialogues) < 4 and t_idx in [0, 1]:
                sample_dialogues.append({
                    "scenario": sc.get("id", f"sample_{s_idx}"),
                    "caption": caption[:70] + "...",
                    "turn": turn_no,
                    "question": q,
                    "target": ref_a,
                    "prediction": gen_a,
                    "bleu_1": b_scores["bleu_1"]
                })

    # Summary Statistics
    print("\n--- Quantitative Language Metrics ---")
    for k in ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]:
        avg_val = sum(bleu_records[k]) / max(len(bleu_records[k]), 1)
        print(f"  * {k.upper()}: {avg_val:.2f}%")

    print("\n--- Multi-Turn Retention (BLEU-1 by Turn Depth) ---")
    for t_num in sorted(turn_accuracies.keys())[:4]:
        t_avg = sum(turn_accuracies[t_num]) / max(len(turn_accuracies[t_num]), 1)
        print(f"  * Turn {t_num} Grounding: {t_avg:.2f}% (Samples: {len(turn_accuracies[t_num])})")

    if attn_caption_shares:
        avg_cap_share = (sum(attn_caption_shares) / len(attn_caption_shares)) * 100.0
        avg_qry_share = (sum(attn_query_shares) / len(attn_query_shares)) * 100.0
        print("\n--- Attention Grounding Allocation ---")
        print(f"  * Caption Context Focus: {avg_cap_share:.1f}%")
        print(f"  * Query Intent Focus:    {avg_qry_share:.1f}%")

    print("\n--- Qualitative Dialogue Sample Predictions ---")
    for idx, s in enumerate(sample_dialogues):
        print(f"\n[Sample {idx + 1}] Scenario: {s['scenario']} (Turn {s['turn']})")
        print(f"  Caption:    {s['caption']}")
        print(f"  Question:   {s['question']}")
        print(f"  Target:     {s['target']}")
        print(f"  Prediction: {s['prediction']}")
        print(f"  BLEU-1:     {s['bleu_1']}%")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    evaluate_benchmarks()
