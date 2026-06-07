"""
finetune_example.py — Adapt AQ1 to new quantum hardware in ~50 lines.

This shows how to take the pretrained AQ1 checkpoint and finetune it on
syndrome data from your own QPU. Works for any d=3 rotated surface code
with 8 stabilizers.

Usage:
    python3 finetune_example.py \
        --pretrained checkpoints/lr3e4/best_model.pt \
        --data path/to/your_hardware_data.h5 \
        --output checkpoints/my_hardware_finetuned.pt
"""
import sys, argparse, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
import random

from model import AQ1Decoder
from train import SyndromeDataset, make_grouped_loader, make_lr_lambda

def finetune(pretrained_path, data_path, output_path,
             peak_lr=5e-5, epochs=20, batch_size=512,
             freeze_embedding=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load pretrained model
    checkpoint = torch.load(pretrained_path, map_location=device)
    model = AQ1Decoder().to(device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"Loaded pretrained model (val_loss={checkpoint['val_loss']:.4f})")

    # Optionally freeze embedding layer to preserve general features
    if freeze_embedding:
        for p in model.embedding.parameters():
            p.requires_grad = False
        print("Embedding layer frozen")

    # Load hardware data
    train_ds = SyndromeDataset(data_path, split='train', val_split=0.1)
    val_ds   = SyndromeDataset(data_path, split='val',   val_split=0.1)
    train_loader = make_grouped_loader(train_ds, batch_size, shuffle=True)
    val_loader   = make_grouped_loader(val_ds,   batch_size, shuffle=False)
    print(f"Finetuning on {len(train_ds):,} real shots")

    # Lower LR than pretraining — we're adapting, not learning from scratch
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=peak_lr
    )
    total_steps = epochs * len(train_loader)
    warmup_steps = min(200, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(peak_lr, warmup_steps, total_steps)
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        for meas, events, stab_ids, labels in train_loader:
            meas, events, stab_ids, labels = (
                meas.to(device), events.to(device),
                stab_ids.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            logits = model(meas, events, stab_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(labels)
            correct += ((torch.sigmoid(logits) > 0.5) == labels).sum().item()
            total += len(labels)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for meas, events, stab_ids, labels in val_loader:
                meas, events, stab_ids, labels = (
                    meas.to(device), events.to(device),
                    stab_ids.to(device), labels.to(device)
                )
                logits = model(meas, events, stab_ids)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_correct += ((torch.sigmoid(logits) > 0.5) == labels).sum().item()
                val_total += len(labels)

        tl = train_loss / total
        vl = val_loss / val_total
        va = val_correct / val_total
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"train loss {tl:.4f} | val loss {vl:.4f} acc {va:.4f} "
              f"LER {1-va:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_loss': vl,
                'val_ler': 1 - va,
                'finetuned_on': data_path,
            }, output_path)
            print(f"  --> saved best model (val_LER={1-va:.4f})")

    print(f"\nFinetuning complete. Best val LER: {1-val_correct/val_total:.4f}")
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrained', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--freeze_embedding', action='store_true')
    args = ap.parse_args()
    finetune(args.pretrained, args.data, args.output,
             args.lr, args.epochs, args.freeze_embedding)
