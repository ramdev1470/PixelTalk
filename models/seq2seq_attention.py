"""
PyTorch Seq2Seq Architecture with Bahdanau & Luong Attention for Conversational Dialogue
Implements:
1. Bidirectional Encoder with Word Embeddings & Padding Masking
2. Bahdanau (Additive) and Luong (Dot/General) Attention Mechanisms
3. Attentive GRU Decoder with Context Concatenation
4. Greedy and Beam Search Inference Decoders with Attention Weight Extraction
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism:
    score(s_{t-1}, h_i) = v_a^T tanh(W_a s_{t-1} + U_a h_i)
    """
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.U_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            dec_hidden: [batch_size, dec_dim] (current decoder state)
            enc_outputs: [batch_size, src_len, enc_dim]
            mask: [batch_size, src_len] (True for valid tokens, False for padding)
        Returns:
            context: [batch_size, enc_dim]
            attn_weights: [batch_size, src_len]
        """
        # dec_proj: [batch_size, 1, attn_dim]
        dec_proj = self.W_dec(dec_hidden).unsqueeze(1)
        # enc_proj: [batch_size, src_len, attn_dim]
        enc_proj = self.U_enc(enc_outputs)

        # energy: [batch_size, src_len, 1] -> [batch_size, src_len]
        energy = self.v_a(torch.tanh(dec_proj + enc_proj)).squeeze(-1)

        if mask is not None:
            # Mask out padding tokens with a large negative number
            energy = energy.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(energy, dim=-1) # [batch_size, src_len]
        # context: [batch_size, 1, src_len] x [batch_size, src_len, enc_dim] -> [batch_size, enc_dim]
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)

        return context, attn_weights


class LuongAttention(nn.Module):
    """
    Luong (General / Multiplicative) Attention Mechanism:
    score(s_t, h_i) = s_t^T W_a h_i
    """
    def __init__(self, enc_dim: int, dec_dim: int):
        super().__init__()
        self.W = nn.Linear(enc_dim, dec_dim, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        # enc_proj: [batch_size, src_len, dec_dim]
        enc_proj = self.W(enc_outputs)
        # dec_hidden: [batch_size, dec_dim, 1]
        dec_hidden_t = dec_hidden.unsqueeze(-1)
        # scores: [batch_size, src_len]
        scores = torch.bmm(enc_proj, dec_hidden_t).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn_weights


class Seq2SeqEncoder(nn.Module):
    """
    Bidirectional GRU Encoder.
    Encodes image caption, conversation history, and user question.
    """
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # Bridge to project concatenated forward & backward hidden states to decoder hidden dimension
        self.fc_bridge = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor = None):
        """
        src: [batch_size, src_len]
        Returns:
            enc_outputs: [batch_size, src_len, hidden_dim * 2]
            dec_init_hidden: [num_layers, batch_size, hidden_dim]
        """
        # embedded: [batch_size, src_len, emb_dim]
        embedded = self.dropout(self.embedding(src))

        enc_outputs, hidden = self.gru(embedded)
        # hidden shape: [num_layers * 2, batch_size, hidden_dim]
        # Separate forward and backward states for the top layer
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        cat_hidden = torch.cat([forward_hidden, backward_hidden], dim=1) # [batch_size, hidden_dim * 2]
        dec_init_hidden = torch.tanh(self.fc_bridge(cat_hidden)).unsqueeze(0) # [1, batch_size, hidden_dim]

        if self.num_layers > 1:
            dec_init_hidden = dec_init_hidden.repeat(self.num_layers, 1, 1)

        return enc_outputs, dec_init_hidden


class Seq2SeqDecoder(nn.Module):
    """
    Attentive GRU Decoder.
    At each step, combines embedding of previous token with the attention context vector.
    """
    def __init__(self, vocab_size: int, emb_dim: int, enc_dim: int, dec_dim: int, num_layers: int = 1, dropout: float = 0.2, attention_type: str = "bahdanau", pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        if attention_type.lower() == "luong":
            self.attention = LuongAttention(enc_dim, dec_dim)
        else:
            self.attention = BahdanauAttention(enc_dim, dec_dim, dec_dim)

        # GRU input: [embedded token + context vector]
        self.gru = nn.GRU(
            emb_dim + enc_dim,
            dec_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection combines: dec_hidden + context + embedded
        self.fc_out = nn.Linear(dec_dim + enc_dim + emb_dim, vocab_size)

    def forward_step(self, trg_token: torch.Tensor, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Single decoding step.
        trg_token: [batch_size]
        dec_hidden: [num_layers, batch_size, dec_dim]
        enc_outputs: [batch_size, src_len, enc_dim]
        """
        # trg_token -> [batch_size, 1]
        trg_token = trg_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg_token)) # [batch_size, 1, emb_dim]

        # Use top layer hidden state for attention
        top_hidden = dec_hidden[-1] # [batch_size, dec_dim]
        context, attn_weights = self.attention(top_hidden, enc_outputs, mask=mask) # [batch_size, enc_dim], [batch_size, src_len]

        # Concatenate embedded token and context vector for GRU input
        gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=2) # [batch_size, 1, emb_dim + enc_dim]

        gru_out, new_hidden = self.gru(gru_input, dec_hidden)
        # gru_out: [batch_size, 1, dec_dim]

        # Project to vocabulary logits
        combined = torch.cat([gru_out.squeeze(1), context, embedded.squeeze(1)], dim=1) # [batch_size, dec_dim + enc_dim + emb_dim]
        logits = self.fc_out(combined) # [batch_size, vocab_size]

        return logits, new_hidden, attn_weights


class PixelTalkSeq2Seq(nn.Module):
    """
    Complete Conversational Image Recognition Seq2Seq Model with Attention.
    """
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 256, num_layers: int = 1, dropout: float = 0.2, attention_type: str = "bahdanau", pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.encoder = Seq2SeqEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx
        )
        enc_dim = hidden_dim * 2

        self.decoder = Seq2SeqDecoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            enc_dim=enc_dim,
            dec_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_type=attention_type,
            pad_idx=pad_idx
        )

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """True for non-padding elements, False for padding."""
        return src != self.pad_idx

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        """
        src: [batch_size, src_len]
        trg: [batch_size, trg_len] (starts with <sos>, ends with <eos>)
        Returns:
            outputs: [batch_size, trg_len - 1, vocab_size]
            attentions: [batch_size, trg_len - 1, src_len]
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        src_len = src.size(1)

        mask = self.create_mask(src)
        enc_outputs, dec_hidden = self.encoder(src)

        outputs = torch.zeros(batch_size, trg_len - 1, self.vocab_size, device=src.device)
        attentions = torch.zeros(batch_size, trg_len - 1, src_len, device=src.device)

        # First token fed to decoder is <sos>
        current_input = trg[:, 0]

        for t in range(1, trg_len):
            logits, dec_hidden, attn_weights = self.decoder.forward_step(
                current_input, dec_hidden, enc_outputs, mask=mask
            )
            outputs[:, t - 1] = logits
            attentions[:, t - 1] = attn_weights

            # Teacher forcing decision
            use_teacher_forcing = (random.random() < teacher_forcing_ratio)
            top1 = logits.argmax(1)
            current_input = trg[:, t] if use_teacher_forcing else top1

        return outputs, attentions

    @torch.no_grad()
    def generate(self, src: torch.Tensor, sos_idx: int = 1, eos_idx: int = 2, max_len: int = 35, temperature: float = 1.0, beam_width: int = 1):
        """
        Generates response using Greedy Search or Beam Search.
        src: [1, src_len]
        Returns:
            generated_token_ids: list of ints
            attention_matrix: [trg_generated_len, src_len] numpy/torch tensor
        """
        self.eval()
        device = src.device
        mask = self.create_mask(src)
        enc_outputs, dec_hidden = self.encoder(src)

        if beam_width <= 1:
            # Greedy / Temperature decoding
            current_input = torch.tensor([sos_idx], dtype=torch.long, device=device)
            generated_tokens = []
            attentions_list = []

            for _ in range(max_len):
                logits, dec_hidden, attn_weights = self.decoder.forward_step(
                    current_input, dec_hidden, enc_outputs, mask=mask
                )
                attentions_list.append(attn_weights.squeeze(0).cpu())

                if temperature > 0.0 and temperature != 1.0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = logits.argmax(dim=-1).item()

                if next_token == eos_idx:
                    break

                generated_tokens.append(next_token)
                current_input = torch.tensor([next_token], dtype=torch.long, device=device)

            if generated_tokens and attentions_list:
                attn_matrix = torch.stack(attentions_list[:len(generated_tokens)], dim=0)
            else:
                attn_matrix = torch.empty(0)
            return generated_tokens, attn_matrix
        else:
            # Beam search
            return self._beam_search(enc_outputs, dec_hidden, mask, sos_idx, eos_idx, max_len, beam_width)

    def _beam_search(self, enc_outputs, dec_hidden, mask, sos_idx, eos_idx, max_len, beam_width):
        device = enc_outputs.device
        # Beam tuple: (score, tokens_list, dec_hidden, attentions_list)
        beams = [(0.0, [sos_idx], dec_hidden, [])]

        for _ in range(max_len):
            all_candidates = []
            all_completed = True

            for score, tokens, hidden, attns in beams:
                last_token = tokens[-1]
                if last_token == eos_idx and len(tokens) > 1:
                    all_candidates.append((score, tokens, hidden, attns))
                    continue

                all_completed = False
                inp = torch.tensor([last_token], dtype=torch.long, device=device)
                logits, new_hidden, attn_weights = self.decoder.forward_step(
                    inp, hidden, enc_outputs, mask=mask
                )
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0) # [vocab_size]

                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                for log_prob, idx in zip(topk_log_probs, topk_indices):
                    new_score = score + log_prob.item()
                    new_tokens = tokens + [idx.item()]
                    new_attns = attns + [attn_weights.squeeze(0).cpu()]
                    all_candidates.append((new_score, new_tokens, new_hidden, new_attns))

            if all_completed:
                break

            # Rank by length-normalized score
            beams = sorted(all_candidates, key=lambda x: x[0] / (len(x[1]) ** 0.7), reverse=True)[:beam_width]

        best_score, best_tokens, _, best_attns = beams[0]
        # Remove <sos> and stop at <eos>
        res_tokens = []
        for t in best_tokens[1:]:
            if t == eos_idx:
                break
            res_tokens.append(t)

        attn_matrix = torch.stack(best_attns[:len(res_tokens)], dim=0) if best_attns and res_tokens else torch.empty(0)
        return res_tokens, attn_matrix
