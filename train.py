"""
Training Pipeline for PixelTalk Seq2Seq + Attention Dialogue Manager
Trains the model on MS-COCO grounded multi-turn dialogue pairs, tracking loss,
validation metrics, and saving the best model checkpoint.
"""

import os
import sys
import argparse
import time
import math

# Ensure project root and PixelTalk dir are in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
for p in [_parent_dir, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

try:
    from PixelTalk.data.dataset import DialogueDataset, collate_dialogue_batch, Vocabulary
    from PixelTalk.models.seq2seq_attention import PixelTalkSeq2Seq
except ImportError:
    from data.dataset import DialogueDataset, collate_dialogue_batch, Vocabulary
    from models.seq2seq_attention import PixelTalkSeq2Seq


def calculate_token_accuracy(predictions: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> float:
    """Computes non-padded token accuracy."""
    pred_tokens = predictions.argmax(dim=-1)
    mask = targets != pad_idx
    correct = (pred_tokens == targets) & mask
    total = mask.sum().item()
    return (correct.sum().item() / max(total, 1)) * 100.0

def train_epoch(model, dataloader, optimizer, criterion, clip_norm, teacher_forcing_ratio, device, pad_idx):
    model.train()
    epoch_loss = 0.0
    total_acc = 0.0

    for batch in dataloader:
        src = batch["src"].to(device)
        trg = batch["trg"].to(device) # [batch_size, trg_len]

        optimizer.zero_grad()
        # outputs: [batch_size, trg_len - 1, vocab_size]
        outputs, _ = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

        # Target shifted by 1 (ignoring <sos>)
        target_seq = trg[:, 1:]

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            target_seq.reshape(-1)
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        epoch_loss += loss.item()
        total_acc += calculate_token_accuracy(outputs, target_seq, pad_idx)

    return epoch_loss / len(dataloader), total_acc / len(dataloader)

def evaluate(model, dataloader, criterion, device, pad_idx):
    model.eval()
    epoch_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            trg = batch["trg"].to(device)

            outputs, _ = model(src, trg, teacher_forcing_ratio=0.0) # No teacher forcing during eval
            target_seq = trg[:, 1:]

            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                target_seq.reshape(-1)
            )

            epoch_loss += loss.item()
            total_acc += calculate_token_accuracy(outputs, target_seq, pad_idx)

    return epoch_loss / len(dataloader), total_acc / len(dataloader)

def run_training(args):
    # Robust path resolution
    if not os.path.exists(args.data_path):
        alt = args.data_path.replace("PixelTalk/", "").replace("PixelTalk\\", "")
        if os.path.exists(alt):
            args.data_path = alt
    if args.checkpoint_dir.startswith("PixelTalk/") or args.checkpoint_dir.startswith("PixelTalk\\"):
        alt_ckpt = args.checkpoint_dir.replace("PixelTalk/", "").replace("PixelTalk\\", "")
        args.checkpoint_dir = alt_ckpt

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[*] Training on device: {device}", flush=True)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Dataset & Vocab
    print("[*] Loading dataset...", flush=True)
    full_dataset = DialogueDataset(args.data_path)
    vocab = full_dataset.vocab
    vocab_path = os.path.join(args.checkpoint_dir, "vocab.json")
    vocab.save(vocab_path)
    print(f"[*] Saved vocabulary ({len(vocab)} words) to {vocab_path}", flush=True)

    # Train / Val Split (85% / 15%)
    val_size = max(int(len(full_dataset) * args.val_split), 1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"[*] Split: {train_size} training samples, {val_size} validation samples.", flush=True)

    pad_idx = vocab.pad_idx
    collate_fn = lambda b: collate_dialogue_batch(b, pad_idx=pad_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 2. Model Initialization
    model = PixelTalkSeq2Seq(
        vocab_size=len(vocab),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        attention_type=args.attention_type,
        pad_idx=pad_idx
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] PixelTalk Seq2Seq initialized ({args.attention_type.capitalize()} Attention). Trainable Parameters: {total_params:,}", flush=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_val_loss = float("inf")
    checkpoint_file = os.path.join(args.checkpoint_dir, "seq2seq_attention.pt")

    print("\n--- Starting Training Loop ---", flush=True)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # Slowly decay teacher forcing from 0.7 down to 0.3
        tf_ratio = max(0.3, 0.7 - (0.4 * (epoch / args.epochs)))

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            clip_norm=args.clip_norm, teacher_forcing_ratio=tf_ratio,
            device=device, pad_idx=pad_idx
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device=device, pad_idx=pad_idx)
        scheduler.step(val_loss)

        epoch_time = time.time() - t0
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "emb_dim": args.emb_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "attention_type": args.attention_type,
                "vocab_size": len(vocab)
            }, checkpoint_file)
            saved_indicator = " [* Saved Best]"
        else:
            saved_indicator = ""

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.2f}s) | "
                  f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:5.1f}) Acc: {train_acc:5.1f}% | "
                  f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:5.1f}) Acc: {val_acc:5.1f}%{saved_indicator}", flush=True)

    total_time = time.time() - start_time
    print(f"\n[*] Training completed in {total_time:.2f}s! Best Validation Loss: {best_val_loss:.4f}", flush=True)
    print(f"[*] Checkpoint saved at: {checkpoint_file}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PixelTalk Seq2Seq with Attention")
    parser.add_argument("--data_path", type=str, default="PixelTalk/data/coco_dialogue_data.json")
    parser.add_argument("--checkpoint_dir", type=str, default="PixelTalk/checkpoints")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--attention_type", type=str, default="bahdanau", choices=["bahdanau", "luong"])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=2)
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()
    run_training(args)
