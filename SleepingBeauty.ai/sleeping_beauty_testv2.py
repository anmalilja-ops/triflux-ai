#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XY Dual-Axis MLP with Positional IDs, Controlled Random Row Sampling,
and Adaptive Sleeping Beauty REM Consolidation.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.datasets import fashion_mnist as mnist
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
EPOCHS               = 25000
BATCH_SIZE           = 1024
NUM_CLASSES          = 10
PASSES_PER_EPOCH     = 2     

# Positional Encoding & Randomness Controls
POS_EMBED_DIM        = 4     
ROW_RANDOM_RATE      = 0.01  

# Each axis stream hidden layers
STREAM_IN     = 28 + POS_EMBED_DIM
STREAM_LAYERS = [256, 128]       
STREAM_OUT    = 128          
MERGER_LAYERS = [1024, 512, 256, 128]       

USE_BATCHNORM = True
DTYPE         = torch.float32

# Adaptive dropout
DROPOUT_SCALE     = 0.01
DROPOUT_MAX       = 0.9    
DROPOUT_EXPONENT  = -1.2   
DROPOUT_SPRING    = 4
UPDATE_EVERY      = 8

# Learning rate
LR_START    = 0.002
LR_EXPONENT = 1.15

# Early Stopping & Sleep Mode
PATIENCE = 5000  
SLEEP_PATIENCE = 200  # Stall for 200 epochs -> go to sleep
SLEEP_DURATION = 50   # Dream for 50 epochs

# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# --------------------------------------------------------------------------- #
# Helper: build a simple MLP block
# --------------------------------------------------------------------------- #
def make_mlp(in_dim: int, hidden: list, out_dim: int,
             dropout: float, use_bn: bool) -> nn.Sequential:
    modules = []
    curr = in_dim
    for h in hidden:
        modules.append(nn.Linear(curr, h, dtype=DTYPE))
        if use_bn:
            modules.append(nn.BatchNorm1d(h, dtype=DTYPE))
        modules.append(nn.SiLU())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        curr = h
    modules.append(nn.Linear(curr, out_dim, dtype=DTYPE))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Fast GPU Augmentation Helper
# --------------------------------------------------------------------------- #
def fast_gpu_augment(x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    img = x.view(B, 28, 28)

    flip_h = torch.rand(B, 1, 1, device=x.device) > 0.5
    img = torch.where(flip_h, torch.flip(img, dims=[2]), img)

    dx = torch.randint(-1, 2, (1,)).item()
    dy = torch.randint(-1, 2, (1,)).item()
    img = torch.roll(img, shifts=(dy, dx), dims=(1, 2))

    return img.view(B, 784)

# --------------------------------------------------------------------------- #
# Controlled Random Row Perturbation
# --------------------------------------------------------------------------- #
def apply_controlled_random_rows(img: torch.Tensor, rate: float) -> torch.Tensor:
    if rate <= 0.0:
        return img
    B = img.shape[0]
    mask = torch.rand(B, 28, device=img.device) < rate
    if mask.any():
        rand_idx = torch.randint(0, 28, (B, 28), device=img.device)
        batch_idx = torch.arange(B, device=img.device).unsqueeze(1).expand(B, 28)
        perturbed = img[batch_idx, rand_idx]
        img = torch.where(mask.unsqueeze(2), perturbed, img)
    return img

# --------------------------------------------------------------------------- #
# XY Dual-Axis Model with Positional Encodings
# --------------------------------------------------------------------------- #
class XYDualAxisNetWithPos(nn.Module):
    def __init__(self, dropout: float, pos_dim: int = POS_EMBED_DIM):
        super().__init__()
        self.row_pos_embed = nn.Embedding(28, pos_dim)
        self.col_pos_embed = nn.Embedding(28, pos_dim)
        self.y_row_encoder = make_mlp(STREAM_IN, STREAM_LAYERS, STREAM_OUT,
                                       dropout, USE_BATCHNORM)
        self.x_col_encoder = make_mlp(STREAM_IN, STREAM_LAYERS, STREAM_OUT,
                                       dropout, USE_BATCHNORM)
        merger_in_dim = (28 * STREAM_OUT) * 2 
        self.merger = make_mlp(merger_in_dim, MERGER_LAYERS, NUM_CLASSES,
                                dropout, USE_BATCHNORM)
        self._dropout = dropout
        self.pos_dim  = pos_dim
        self.register_buffer("indices_28", torch.arange(28, dtype=torch.long))

    def forward(self, x: torch.Tensor, perturb_rate: float = 0.0) -> torch.Tensor:
        B = x.shape[0]
        img = x.view(B, 28, 28)

        if self.training and perturb_rate > 0:
            img = apply_controlled_random_rows(img, perturb_rate)

        row_pos = self.row_pos_embed(self.indices_28).unsqueeze(0).expand(B, 28, self.pos_dim)
        col_pos = self.col_pos_embed(self.indices_28).unsqueeze(0).expand(B, 28, self.pos_dim)

        rows_with_pos = torch.cat([img, row_pos], dim=2)
        rows_flat     = rows_with_pos.reshape(B * 28, STREAM_IN)
        y_encoded     = self.y_row_encoder(rows_flat)
        y_structural_map = y_encoded.view(B, -1)          

        cols = img.permute(0, 2, 1)                       
        cols_with_pos = torch.cat([cols, col_pos], dim=2)
        cols_flat     = cols_with_pos.reshape(B * 28, STREAM_IN)
        x_encoded     = self.x_col_encoder(cols_flat)
        x_structural_map = x_encoded.view(B, -1)          

        combined = torch.cat([y_structural_map, x_structural_map], dim=1) 
        return self.merger(combined)                       

    def set_dropout(self, p: float):
        self._dropout = p
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print(f"\nUsing device: {DEVICE}\n")

    # ── Load Fashion-MNIST ──────────────────────────────────────────────────
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test  = X_test.reshape(X_test.shape[0],  -1).astype(np.float32)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_tr = torch.tensor(X_train, dtype=DTYPE, device=DEVICE)
    y_tr = torch.tensor(y_train.astype(np.int64), device=DEVICE)
    X_te = torch.tensor(X_test,  dtype=DTYPE, device=DEVICE)
    y_te = torch.tensor(y_test.astype(np.int64), device=DEVICE)

    N_TRAIN = X_tr.shape[0]

    print(f"Dataset             : Fashion-MNIST")
    print(f"Positional ID Dim   : {POS_EMBED_DIM} features per row/col")
    print(f"Random Row Swap Rate: {ROW_RANDOM_RATE * 100:.1f}% (1 in {int(1/ROW_RANDOM_RATE)} rows)")
    print(f"Images Seen / Epoch : {N_TRAIN * PASSES_PER_EPOCH:,}\n")

    # ── Model + optimizer ───────────────────────────────────────────────────
    model     = XYDualAxisNetWithPos(DROPOUT_SCALE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_START, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}\n")

    # ── Adaptive helpers ────────────────────────────────────────────────────
    def adjust_lr(test_acc: float) -> float:
        new_lr = LR_START * (1.0 - test_acc) ** LR_EXPONENT
        for g in optimizer.param_groups:
            g["lr"] = new_lr
        return new_lr

    def compute_dropout(train_acc: float) -> float:
        return min(DROPOUT_SCALE * (1.0 - train_acc) ** DROPOUT_EXPONENT, DROPOUT_MAX)

    # ── Training utilities ──────────────────────────────────────────────────
    def train_one_epoch() -> float:
        model.train()
        correct, total = 0, 0

        for p_idx in range(PASSES_PER_EPOCH):
            perm   = torch.randperm(N_TRAIN, device=DEVICE)
            X_shuf = X_tr[perm]
            y_shuf = y_tr[perm]

            if p_idx == 1:
                X_shuf = fast_gpu_augment(X_shuf)

            for i in range(0, N_TRAIN, BATCH_SIZE):
                xb = X_shuf[i : i + BATCH_SIZE]
                yb = y_shuf[i : i + BATCH_SIZE]

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb, perturb_rate=ROW_RANDOM_RATE)
                loss   = F.cross_entropy(logits, yb)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                correct += (logits.argmax(1) == yb).sum().item()
                total   += yb.size(0)

        return correct / total

    def evaluate() -> float:
        model.eval()
        with torch.no_grad():
            logits = model(X_te, perturb_rate=0.0)
            return (logits.argmax(1) == y_te).float().mean().item()

    # ── Training loop ───────────────────────────────────────────────────────
    print("=" * 80)
    print("TRAINING WITH POSITIONAL ID ENCODING + ROW SAMPLING + SLEEP MODE")
    print("=" * 80)

    best_test_acc     = 0.0
    current_dropout   = DROPOUT_SCALE
    epochs_no_improve = 0
    
    # Sleep State Variables
    is_sleeping = False
    sleep_counter = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # 💤 TRIGGER SLEEPING BEAUTY ADAPTIVELY 💤
        if not is_sleeping and epochs_no_improve >= SLEEP_PATIENCE:
            is_sleeping = True
            sleep_counter = SLEEP_DURATION
            for g in optimizer.param_groups:
                g["lr"] = 0.00005  # Drop LR for dream phase
            print(f"\n  [💤 ENTERING REM SLEEP] Epoch {epoch} | Stalled for {SLEEP_PATIENCE} epochs. Dreaming to consolidate weights...")

        # If currently sleeping, run dream batch instead of real data
        if is_sleeping:
            model.train()
            for _ in range(5):  # 5 mini-dream batches per epoch
                X_noise = torch.randn(BATCH_SIZE, 784, device=DEVICE) * 0.05
                optimizer.zero_grad(set_to_none=True)
                mock_out = model(X_noise, perturb_rate=0.03)
                loss = F.cross_entropy(mock_out, torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE))
                loss.backward()
                optimizer.step()
            
            sleep_counter -= 1
            if sleep_counter <= 0:
                is_sleeping = False
                epochs_no_improve = 0
                print(f"  [☀️ WAKING UP] Epoch {epoch} | Resuming normal training with refreshed weights.\n")
            
            elapsed = time.time() - t0
            print(f"{epoch:5d} | [💤 Dreaming] consolidation phase... | time: {elapsed:.2f}s")
            continue

        # Normal Waking Training Phase
        train_acc = train_one_epoch()
        test_acc  = evaluate()
        gap       = (train_acc - test_acc) * 100.0
        lr        = adjust_lr(test_acc)

        if epoch % UPDATE_EVERY == 0:
            target          = compute_dropout(train_acc)
            current_dropout += (target - current_dropout) / DROPOUT_SPRING
            model.set_dropout(current_dropout)

        # Early Stopping Check
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_xy_gen10_pos_embed.pt')
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0

        print(f"{epoch:5d} | "
              f"train: {train_acc*100:.3f}% | "
              f"test : {test_acc*100:.3f}% (best: {best_test_acc*100:.3f}%) | "
              f"gap: {gap:.3f}% | "
              f"LR: {lr:.6f} | Drop: {current_dropout:.3f} | "
              f"Stalled: {epochs_no_improve} | "
              f"time: {elapsed:.2f}s")

        if epochs_no_improve >= PATIENCE:
            print(f"\n[!] Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.")
            break

    print("\n" + "=" * 80)
    print(f"Final Best Test Accuracy : {best_test_acc*100:.3f}%")
    print("=" * 80)