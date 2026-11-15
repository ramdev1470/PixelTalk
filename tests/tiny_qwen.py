"""
Builds a TINY random-weight Qwen2-VL (real architecture + real processor classes, toy sizes) so the
real engine code can be tested end to end in a few seconds, with no download and no GPU.

The model produces gibberish (its weights are random), but every code path that matters is
exercised for real: chat template, image processor, vision tower, generate(), streaming, decoding.

    from tests.tiny_qwen import build_tiny_qwen
    path = build_tiny_qwen("/tmp/tiny-qwen")
"""

import os

CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}"
    "{% endfor %}<|im_end|>\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
           "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"]

CORPUS = [
    "what is in the picture a dog a car a red ball on the grass yes no maybe",
    "the image shows a golden retriever sitting in a park with a yellow tennis ball",
    "CATEGORY PREPARED_FOOD SPOILED_FOOD NOT_FOOD UNCLEAR SUBJECT bread pizza mushroom",
    "describe this image in detail colors background text people streets",
] * 20


def build_tiny_qwen(out_dir: str, seed: int = 0) -> str:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (
        PreTrainedTokenizerFast, Qwen2VLConfig, Qwen2VLForConditionalGeneration,
        Qwen2VLImageProcessor, Qwen2VLProcessor, Qwen2VLVideoProcessor,
    )

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        additional_special_tokens=SPECIAL[1:],
    )
    tid = tokenizer.convert_tokens_to_ids

    config = Qwen2VLConfig(
        text_config=dict(
            vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3]},
            bos_token_id=tid("<|endoftext|>"), eos_token_id=tid("<|im_end|>"), pad_token_id=tid("<|endoftext|>"),
        ),
        vision_config=dict(depth=2, embed_dim=32, hidden_size=64, num_heads=2, mlp_ratio=2),
        image_token_id=tid("<|image_pad|>"), video_token_id=tid("<|video_pad|>"),
        vision_start_token_id=tid("<|vision_start|>"), vision_end_token_id=tid("<|vision_end|>"),
    )
    model = Qwen2VLForConditionalGeneration(config)
    model.generation_config.eos_token_id = tid("<|im_end|>")
    model.generation_config.pad_token_id = tid("<|endoftext|>")

    processor = Qwen2VLProcessor(
        image_processor=Qwen2VLImageProcessor(), tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(), chat_template=CHAT_TEMPLATE,
    )
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    return out_dir
