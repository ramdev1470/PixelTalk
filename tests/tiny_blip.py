"""Tiny random-weight BLIP-VQA (real classes, toy sizes) for testing the fallback engine offline."""

import os


def build_tiny_blip_vqa(out_dir: str, seed: int = 0) -> str:
    import torch
    from transformers import (
        BertTokenizer, BlipConfig, BlipForQuestionAnswering, BlipImageProcessor, BlipProcessor,
    )

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[DEC]", "[ENC]", "what", "is", "the", "color", "this", "a",
             "dog", "car", "red", "blue", "yes", "no", "?", "in", "picture", "main", "subject", "image"]
    vocab_path = os.path.join(out_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(words) + "\n")
    tokenizer = BertTokenizer(vocab_path, do_lower_case=True)
    n = len(words)

    config = BlipConfig(
        text_config=dict(vocab_size=n, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=2, encoder_hidden_size=32, max_position_embeddings=64,
                         bos_token_id=words.index("[DEC]"), eos_token_id=words.index("[SEP]"),
                         pad_token_id=0, sep_token_id=words.index("[SEP]")),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                           image_size=64, patch_size=16),
    )
    model = BlipForQuestionAnswering(config)
    processor = BlipProcessor(image_processor=BlipImageProcessor(size={"height": 64, "width": 64}), tokenizer=tokenizer)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    return out_dir
