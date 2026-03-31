#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XY Dual-Axis MLP trained on full MNIST (70,000 samples, 28x28 = 784 features).

Architecture:
  - Y-stream: reads each image as 28 rows (top→bottom), each row = 28 features
  - X-stream: reads each image as 28 columns (left→right), each col = 28 features
  - Both streams output embeddings that are stacked into a "3D" combined tensor
  - A merger network fuses both streams → 10 class output

Adaptive systems (from gen5):
  LR      = LR_START * (1 - test_acc)  ** LR_EXPONENT
  Dropout = DROPOUT_SCALE * (1 - train_acc) ** DROPOUT_EXPONENT (capped at DROPOUT_MAX)
  Smoothing: current = current + (target - current) / DROPOUT_SPRING
  Update frequency: every UPDATE_EVERY epochs

Gen 9 — XY dual-axis 3D-stacked siamese-style network.
Author:  <your name>
Date:    2026-03-30
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
EPOCHS      = 1000
BATCH_SIZE  = 1024
NUM_CLASSES = 10

# Each axis stream hidden layers
STREAM_LAYERS = [256,128]      # X stream and Y stream are identical size
STREAM_OUT    = 128             # output embedding size per stream

# Merger network (takes concatenated X+Y embeddings)
MERGER_LAYERS = [256,128]      # fuses the two 128-dim streams

USE_BATCHNORM = True
DTYPE         = torch.float32

# Adaptive dropout
DROPOUT_SCALE    = 0.18
DROPOUT_MAX      = 0.9
DROPOUT_EXPONENT = -0.26
DROPOUT_SPRING   = 3
UPDATE_EVERY     = 2

# Learning rate
LR_START    = 0.0323
LR_EXPONENT = 1.4

# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        modules.append(nn.ReLU())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        curr = h
    modules.append(nn.Linear(curr, out_dim, dtype=DTYPE))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# XY Dual-Axis Model
# --------------------------------------------------------------------------- #
class XYDualAxisNet(nn.Module):
    """
    Reads MNIST images as both row-slices (Y axis) and column-slices (X axis).
    Each axis gets its own stream MLP. Outputs are concatenated (3D stack)
    and fused by a merger network.

    Input shape: (batch, 784)  →  reshaped to (batch, 28, 28) internally
    """

    def __init__(self, dropout: float):
        super().__init__()

        # Y-stream: processes 28 rows, each row has 28 pixels
        # We encode each row independently then pool across rows
        self.y_row_encoder = make_mlp(28, STREAM_LAYERS, STREAM_OUT,
                                       dropout, USE_BATCHNORM)

        # X-stream: processes 28 columns, each col has 28 pixels
        self.x_col_encoder = make_mlp(28, STREAM_LAYERS, STREAM_OUT,
                                       dropout, USE_BATCHNORM)

        # Merger: fuses concatenated [y_out, x_out] → class logits
        # Input dim = STREAM_OUT * 2 (Y and X stacked = "3D combined")
        self.merger = make_mlp(STREAM_OUT * 2, MERGER_LAYERS, NUM_CLASSES,
                                dropout, USE_BATCHNORM)

        self._dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Reshape flat 784 → 28x28
        img = x.view(B, 28, 28)                          # (B, 28, 28)

        # ── Y stream: encode each of the 28 rows ──────────────────────────
        # rows shape: (B, 28, 28) → process all 28 rows at once
        rows = img                                        # (B, 28, 28)
        rows_flat = rows.reshape(B * 28, 28)              # (B*28, 28)
        y_encoded = self.y_row_encoder(rows_flat)         # (B*28, STREAM_OUT)
        y_encoded = y_encoded.view(B, 28, STREAM_OUT)     # (B, 28, STREAM_OUT)
        y_pooled  = y_encoded.mean(dim=1)                 # (B, STREAM_OUT)  ← pool rows

        # ── X stream: encode each of the 28 columns ───────────────────────
        cols = img.permute(0, 2, 1)                       # (B, 28, 28) cols now first
        cols_flat = cols.reshape(B * 28, 28)              # (B*28, 28)
        x_encoded = self.x_col_encoder(cols_flat)         # (B*28, STREAM_OUT)
        x_encoded = x_encoded.view(B, 28, STREAM_OUT)     # (B, 28, STREAM_OUT)
        x_pooled  = x_encoded.mean(dim=1)                 # (B, STREAM_OUT)  ← pool cols

        # ── 3D stack: concatenate X and Y embeddings ──────────────────────
        # This is the "3D combined block" — X slice + Y slice stacked
        combined = torch.cat([y_pooled, x_pooled], dim=1) # (B, STREAM_OUT*2)

        # ── Merger: fuse and classify ──────────────────────────────────────
        return self.merger(combined)                       # (B, NUM_CLASSES)

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

    # ── Load MNIST ──────────────────────────────────────────────────────────
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test  = X_test.reshape(X_test.shape[0],  -1).astype(np.float32)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_tr = torch.tensor(X_train, dtype=DTYPE,          device=DEVICE)
    y_tr = torch.tensor(y_train.astype(np.int64),      device=DEVICE)
    X_te = torch.tensor(X_test,  dtype=DTYPE,          device=DEVICE)
    y_te = torch.tensor(y_test.astype(np.int64),       device=DEVICE)

    N_TRAIN = X_tr.shape[0]

    print(f"Dataset : MNIST")
    print(f"Train   : {N_TRAIN:,} samples")
    print(f"Test    : {X_te.shape[0]:,} samples\n")

    print(f"Architecture:")
    print(f"  Y-stream (rows)  : 28 → {STREAM_LAYERS} → {STREAM_OUT}  (mean pool over 28 rows)")
    print(f"  X-stream (cols)  : 28 → {STREAM_LAYERS} → {STREAM_OUT}  (mean pool over 28 cols)")
    print(f"  3D stack         : concat [{STREAM_OUT} + {STREAM_OUT}] = {STREAM_OUT*2} dim")
    print(f"  Merger           : {STREAM_OUT*2} → {MERGER_LAYERS} → {NUM_CLASSES}\n")

    print(f"Dropout : {DROPOUT_SCALE} * (1 - train_acc) ^ {DROPOUT_EXPONENT}  capped at {DROPOUT_MAX}")
    print(f"Spring  : {DROPOUT_SPRING} epochs of inertia")
    print(f"Update  : every {UPDATE_EVERY} epochs")
    print(f"LR      : {LR_START} * (1 - test_acc) ^ {LR_EXPONENT}\n")

    # Dropout curve preview
    print("Dropout curve preview:")
    for acc in [0.85, 0.90, 0.95, 0.98, 0.99, 0.999]:
        d = min(DROPOUT_SCALE * (1.0 - acc) ** DROPOUT_EXPONENT, DROPOUT_MAX)
        print(f"  train {acc*100:.1f}%  →  dropout {d:.4f}")
    print()

    # ── Model + optimizer ───────────────────────────────────────────────────
    model     = XYDualAxisNet(DROPOUT_SCALE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)

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
        perm       = torch.randperm(N_TRAIN, device=DEVICE)
        X_shuf     = X_tr[perm]
        y_shuf     = y_tr[perm]

        for i in range(0, N_TRAIN, BATCH_SIZE):
            xb = X_shuf[i : i + BATCH_SIZE]
            yb = y_shuf[i : i + BATCH_SIZE]

            optimizer.zero_grad()
            logits = model(xb)
            loss   = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            correct += (logits.argmax(1) == yb).sum().item()
            total   += yb.size(0)

        return correct / total

    def evaluate() -> float:
        model.eval()
        with torch.no_grad():
            logits = model(X_te)
            return (logits.argmax(1) == y_te).float().mean().item()

    # ── Training loop ───────────────────────────────────────────────────────
    print("Training XY Dual-Axis Net with adaptive LR + spring dropout...\n")

    best_test_acc   = 0.0
    current_dropout = DROPOUT_SCALE

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_acc = train_one_epoch()
        test_acc  = evaluate()
        lr        = adjust_lr(test_acc)

        if epoch % UPDATE_EVERY == 0:
            target          = compute_dropout(train_acc)
            current_dropout += (target - current_dropout) / DROPOUT_SPRING
            model.set_dropout(current_dropout)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_xy_model.pt')

        print(f"{epoch:4d} | train: {train_acc*100:.3f}% | "
              f"test : {test_acc*100:.3f}% | LR: {lr:.8f} | "
              f"Dropout: {current_dropout:.3f} | time: {time.time()-t0:.3f}s")

    print("\n--- Summary ---")
    print(f"Best test accuracy : {best_test_acc*100:.2f}%")
    print(f"Y-stream layers    : 28 → {STREAM_LAYERS} → {STREAM_OUT}")
    print(f"X-stream layers    : 28 → {STREAM_LAYERS} → {STREAM_OUT}")
    print(f"Merger layers      : {STREAM_OUT*2} → {MERGER_LAYERS} → {NUM_CLASSES}")
    print(f"Total parameters   : {total_params:,}")
    print(f"Dropout scale      : {DROPOUT_SCALE}")
    print(f"Dropout max        : {DROPOUT_MAX}")
    print(f"Dropout exponent   : {DROPOUT_EXPONENT}")
    print(f"Dropout spring     : {DROPOUT_SPRING}")
    print(f"Update every       : {UPDATE_EVERY} epochs")
    print(f"BatchNorm          : {USE_BATCHNORM}")
    print(f"LR exponent        : {LR_EXPONENT}\n")
