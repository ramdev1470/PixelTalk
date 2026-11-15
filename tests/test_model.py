"""
Unit Tests for Seq2Seq Attention Model
Verifies encoder, decoder, attention mechanisms (Bahdanau & Luong), and generation.
"""

import unittest
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_root_dir = os.path.dirname(_parent_dir)
for p in [_root_dir, _parent_dir, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

try:
    from PixelTalk.models.seq2seq_attention import (
        BahdanauAttention,
        LuongAttention,
        Seq2SeqEncoder,
        Seq2SeqDecoder,
        PixelTalkSeq2Seq
    )
except ImportError:
    from models.seq2seq_attention import (
        BahdanauAttention,
        LuongAttention,
        Seq2SeqEncoder,
        Seq2SeqDecoder,
        PixelTalkSeq2Seq
    )


class TestSeq2SeqAttention(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 50
        self.emb_dim = 32
        self.hidden_dim = 64
        self.batch_size = 4
        self.src_len = 10
        self.trg_len = 8
        self.pad_idx = 0

    def test_bahdanau_attention_shapes_and_sums(self):
        attn = BahdanauAttention(enc_dim=self.hidden_dim * 2, dec_dim=self.hidden_dim, attn_dim=self.hidden_dim)
        dec_hidden = torch.randn(self.batch_size, self.hidden_dim)
        enc_outputs = torch.randn(self.batch_size, self.src_len, self.hidden_dim * 2)

        # Create mask with some padded tokens
        mask = torch.ones(self.batch_size, self.src_len, dtype=torch.bool)
        mask[:, -3:] = False # last 3 tokens are padding

        context, weights = attn(dec_hidden, enc_outputs, mask=mask)

        # Context vector should match encoder output dimension
        self.assertEqual(context.shape, (self.batch_size, self.hidden_dim * 2))
        # Attention weights should match (batch_size, src_len)
        self.assertEqual(weights.shape, (self.batch_size, self.src_len))
        # Weights should sum to 1.0 across source length
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
        # Padded elements should receive zero attention
        self.assertTrue(torch.all(weights[:, -3:] < 1e-6))

    def test_luong_attention_shapes(self):
        attn = LuongAttention(enc_dim=self.hidden_dim * 2, dec_dim=self.hidden_dim)
        dec_hidden = torch.randn(self.batch_size, self.hidden_dim)
        enc_outputs = torch.randn(self.batch_size, self.src_len, self.hidden_dim * 2)

        context, weights = attn(dec_hidden, enc_outputs)
        self.assertEqual(context.shape, (self.batch_size, self.hidden_dim * 2))
        self.assertEqual(weights.shape, (self.batch_size, self.src_len))
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_encoder_forward(self):
        encoder = Seq2SeqEncoder(self.vocab_size, self.emb_dim, self.hidden_dim, pad_idx=self.pad_idx)
        src = torch.randint(1, self.vocab_size, (self.batch_size, self.src_len))
        enc_outputs, dec_init_hidden = encoder(src)

        self.assertEqual(enc_outputs.shape, (self.batch_size, self.src_len, self.hidden_dim * 2))
        self.assertEqual(dec_init_hidden.shape, (1, self.batch_size, self.hidden_dim))

    def test_full_seq2seq_forward(self):
        model = PixelTalkSeq2Seq(
            vocab_size=self.vocab_size,
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            pad_idx=self.pad_idx
        )
        src = torch.randint(1, self.vocab_size, (self.batch_size, self.src_len))
        trg = torch.randint(1, self.vocab_size, (self.batch_size, self.trg_len))

        logits, attentions = model(src, trg, teacher_forcing_ratio=0.5)
        # Logits should be [batch_size, trg_len - 1, vocab_size]
        self.assertEqual(logits.shape, (self.batch_size, self.trg_len - 1, self.vocab_size))
        # Attentions should be [batch_size, trg_len - 1, src_len]
        self.assertEqual(attentions.shape, (self.batch_size, self.trg_len - 1, self.src_len))

    def test_generation_greedy_and_beam(self):
        model = PixelTalkSeq2Seq(
            vocab_size=self.vocab_size,
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            pad_idx=self.pad_idx
        )
        src = torch.randint(1, self.vocab_size, (1, self.src_len))

        # Greedy generation
        tokens_greedy, attn_greedy = model.generate(src, max_len=15, beam_width=1)
        self.assertIsInstance(tokens_greedy, list)
        if len(tokens_greedy) > 0:
            self.assertEqual(attn_greedy.shape, (len(tokens_greedy), self.src_len))

        # Beam search generation
        tokens_beam, attn_beam = model.generate(src, max_len=15, beam_width=3)
        self.assertIsInstance(tokens_beam, list)
        if len(tokens_beam) > 0:
            self.assertEqual(attn_beam.shape, (len(tokens_beam), self.src_len))

if __name__ == "__main__":
    unittest.main()
