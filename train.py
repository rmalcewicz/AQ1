import os
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import AQ1Decoder

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--data',     default='data/d3_zbasis_test.h5')
_parser.add_argument('--lr',       type=float, default=3e-4)
_parser.add_argument('--epochs',   type=int,   default=50)
_parser.add_argument('--run_name', default='default')
_parser.add_argument('--save-dir', default='checkpoints')
_parser.add_argument('--log-path', default='logs/train_log.txt')
_args, _ = _parser.parse_known_args()
DATA_PATH = _args.data
LR        = _args.lr
EPOCHS    = _args.epochs
RUN_NAME  = _args.run_name
SAVE_DIR  = _args.save_dir
LOG_PATH  = _args.log_path
N_STABILIZERS = 8
D_MODEL       = 128
N_HEADS       = 4
N_TRANSFORMER_LAYERS = 4
DROPOUT       = 0.1
BATCH_SIZE    = 1024
# EPOCHS set by argparse above
# LR set by argparse above
WARMUP_STEPS  = 2000
GRAD_CLIP     = 1.0
VAL_SPLIT     = 0.1
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class SyndromeDataset(Dataset):
    def __init__(self, h5_path, split='train', val_split=0.1):
        self.samples = []
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                syndromes  = f[key]['syndromes'][:]   # (shots, detectors)
                observables = f[key]['observables'][:] # (shots, 1)
                n = len(syndromes)
                split_idx = int(n * (1 - val_split))
                if split == 'train':
                    s = syndromes[:split_idx]
                    o = observables[:split_idx]
                else:
                    s = syndromes[split_idx:]
                    o = observables[split_idx:]
                self.samples.append((s, o))
        # Process each config separately (different round counts = different shapes)
        self.syn_list   = []
        self.evt_list   = []
        self.label_list = []
        for syn, obs in self.samples:
            shots, detectors = syn.shape
            rounds = detectors // N_STABILIZERS
            s = syn[:, :rounds * N_STABILIZERS].reshape(shots, rounds, N_STABILIZERS)
            # Compute detection events
            e = np.zeros_like(s)
            e[:, 0, :] = s[:, 0, :]
            e[:, 1:, :] = s[:, 1:, :] ^ s[:, :-1, :]
            self.syn_list.append(torch.tensor(s, dtype=torch.long))
            self.evt_list.append(torch.tensor(e, dtype=torch.long))
            self.label_list.append(torch.tensor(obs[:, 0], dtype=torch.float32))
        self.stab_ids = torch.arange(N_STABILIZERS)
        # Build flat index grouped by round count so batches have same shape
        from collections import defaultdict
        by_rounds = defaultdict(list)
        for i, syn in enumerate(self.syn_list):
            rounds = syn.shape[1]
            for j in range(len(self.label_list[i])):
                by_rounds[rounds].append((i, j))
        self.index = []
        for rounds in sorted(by_rounds.keys()):
            self.index.extend(by_rounds[rounds])
        print(f'[{split}] {len(self.index):,} samples across {len(self.samples)} configs')
    def __len__(self):
        return len(self.index)
    def __getitem__(self, idx):
        ci, si = self.index[idx]
        return (self.syn_list[ci][si],
                self.evt_list[ci][si],
                self.stab_ids,
                self.label_list[ci][si])

# ─────────────────────────────────────────────
# TRAINING UTILS
# ─────────────────────────────────────────────
import math

def make_lr_lambda(peak_lr, warmup_steps, total_steps, min_lr=1e-5):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return (min_lr/peak_lr) + (1 - min_lr/peak_lr) * cos
    return fn

def log(msg, path=LOG_PATH):
    print(msg)
    with open(path, 'a') as f:
        f.write(msg + '\n')

# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')
    # Data
    train_ds = SyndromeDataset(DATA_PATH, split='train')
    val_ds   = SyndromeDataset(DATA_PATH, split='val')
    from collections import defaultdict
    import random

    def make_grouped_loader(ds, batch_size, shuffle):
        by_rounds = defaultdict(list)
        for idx, (ci, si) in enumerate(ds.index):
            r = ds.syn_list[ci].shape[1]
            by_rounds[r].append(idx)
        batches = []
        for r, idxs in by_rounds.items():
            if shuffle:
                random.shuffle(idxs)
            for i in range(0, len(idxs), batch_size):
                batches.append(idxs[i:i+batch_size])
        if shuffle:
            random.shuffle(batches)
        return DataLoader(ds, batch_sampler=batches,
                          num_workers=4, pin_memory=True)

    train_loader = make_grouped_loader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader   = make_grouped_loader(val_ds,   BATCH_SIZE, shuffle=False)

    # Model
    model = AQ1Decoder(
        n_stabilizers=N_STABILIZERS,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_transformer_layers=N_TRANSFORMER_LAYERS,
        dropout=DROPOUT
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'Parameters: {n_params:,}')
    # Loss + optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # compute total steps for cosine decay
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = min(1000, total_steps // 10)
    lr_lambda = make_lr_lambda(LR, warmup_steps, total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_loss = float('inf')
    step = 0
    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()
        for meas, events, stab_ids, labels in train_loader:
            meas      = meas.to(device)
            events    = events.to(device)
            stab_ids  = stab_ids.to(device)
            labels    = labels.to(device)
            optimizer.zero_grad()
            logits = model(meas, events, stab_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            step += 1
            train_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)
        train_loss /= train_total
        train_acc   = train_correct / train_total
        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for meas, events, stab_ids, labels in val_loader:
                meas     = meas.to(device)
                events   = events.to(device)
                stab_ids = stab_ids.to(device)
                labels   = labels.to(device)
                logits = model(meas, events, stab_ids)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)
        val_loss /= val_total
        val_acc   = val_correct / val_total
        elapsed   = time.time() - t0
        log(f'Epoch {epoch+1:3d}/{EPOCHS} | '
            f'train loss {train_loss:.4f} acc {train_acc:.4f} | '
            f'val loss {val_loss:.4f} acc {val_acc:.4f} | '
            f'lr {scheduler.get_last_lr()[0]:.2e} | '
            f'{elapsed:.1f}s')
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = f'{SAVE_DIR}/best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, path)
            log(f'  --> saved best model (val_loss={val_loss:.4f})')
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            path = f'{SAVE_DIR}/checkpoint_epoch{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
            }, path)
    log('Training complete.')
if __name__ == '__main__':
    main()
