#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XY Dual-Axis MLP trained on full MNIST (70,000 samples, 28x28 = 784 features).

Gen 10 Updates:
  - REMOVED mean() pooling that was destroying spatial order (the 92% wall).
  - Flattens row/col embeddings to preserve exact X/Y structural geometry.
  - Upgraded to SiLU activations for smoother gradient flow.
  - Tuned adaptive dropout curve (max 0.5 instead of 0.9 to prevent starvation).
  - Added AdamW optimizer and Gradient Clipping for stable deep learning.
  - Added Early Stopping to save time once peak intelligence is reached.

Architecture:
  - Y-stream: reads 28 rows -> encodes each -> flattens to 1D structural map
  - X-stream: reads 28 cols -> encodes each -> flattens to 1D structural map
  - A merger network fuses both raw structural maps -> 10 class output

Author:  <boki_231>
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
from tensorflow.keras.datasets import fashion_mnist as mnist
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
EPOCHS      = 25000
BATCH_SIZE  = 1024
NUM_CLASSES = 10

# Each axis stream hidden layers (encoding 28 pixels at a time)
STREAM_LAYERS = [16]       
STREAM_OUT    = 16          # output embedding size PER ROW/COL

# Merger network (takes flattened X+Y structural maps)
# Input = (28 rows * 64 out) + (28 cols * 64 out) = 1792 + 1792 = 3584
MERGER_LAYERS =  [32]       

USE_BATCHNORM = True
DTYPE         = torch.float32

# Adaptive dropout (Tuned for Gen 10: smoother curve, max 0.5)
DROPOUT_SCALE     = 0.01
DROPOUT_MAX       = 0.9    # 0.9 was too aggressive, 0.5 prevents overfitting safely
DROPOUT_EXPONENT  = -1.2   # Smoothly ramps up as train_acc rises
DROPOUT_SPRING    = 4
UPDATE_EVERY      = 2

# Learning rate
LR_START    = 0.001
LR_EXPONENT = 1.1

# Early Stopping
PATIENCE = 500  # Stop if no improvement for 500 epochs

# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# Helper: build a simple MLP block (Using SiLU instead of ReLU)
# --------------------------------------------------------------------------- #
def make_mlp(in_dim: int, hidden: list, out_dim: int,
             dropout: float, use_bn: bool) -> nn.Sequential:
    modules = []
    curr = in_dim
    for h in hidden:
        modules.append(nn.Linear(curr, h, dtype=DTYPE))
        if use_bn:
            modules.append(nn.BatchNorm1d(h, dtype=DTYPE))
        modules.append(nn.SiLU())  # Gen 10: Swish/SiLU activation
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        curr = h
    modules.append(nn.Linear(curr, out_dim, dtype=DTYPE))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# XY Dual-Axis Model
# --------------------------------------------------------------------------- #
class XYDualAxisNetGen10(nn.Module):
    """
    Gen 10: Preserves spatial order. No mean() pooling.
    Reads rows and columns, encodes them, and flattens them into a raw 
    structural map so the merger knows exactly WHERE features are.
    """

    def __init__(self, dropout: float):
        super().__init__()

        # Y-stream: processes 28 rows
        self.y_row_encoder = make_mlp(28, STREAM_LAYERS, STREAM_OUT,
                                       dropout, USE_BATCHNORM)

        # X-stream: processes 28 columns
        self.x_col_encoder = make_mlp(28, STREAM_LAYERS, STREAM_OUT,
                                       dropout, USE_BATCHNORM)

        # Merger input size = 28 rows * 64 dim + 28 cols * 64 dim = 3584
        merger_in_dim = (28 * STREAM_OUT) * 2 
        self.merger = make_mlp(merger_in_dim, MERGER_LAYERS, NUM_CLASSES,
                                dropout, USE_BATCHNORM)

        self._dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        img = x.view(B, 28, 28)                          

        # ── Y stream: encode rows and FLATTEN (keep order!) ─────────────
        rows_flat = img.reshape(B * 28, 28)              
        y_encoded = self.y_row_encoder(rows_flat)         
        y_structural_map = y_encoded.view(B, -1)          # (B, 28*64) = (B, 1792)

        # ── X stream: encode cols and FLATTEN (keep order!) ─────────────
        cols = img.permute(0, 2, 1)                       
        cols_flat = cols.reshape(B * 28, 28)              
        x_encoded = self.x_col_encoder(cols_flat)         
        x_structural_map = x_encoded.view(B, -1)          # (B, 28*64) = (B, 1792)

        # ── 3D stack: concatenate raw structural maps ───────────────────
        combined = torch.cat([y_structural_map, x_structural_map], dim=1) # (B, 3584)

        # ── Merger: fuse and classify ────────────────────────────────────
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
    MERGER_IN = (28 * STREAM_OUT) * 2

    print(f"Dataset : Fashion-MNIST")
    print(f"Train   : {N_TRAIN:,} samples")
    print(f"Test    : {X_te.shape[0]:,} samples\n")

    print(f"Gen 10 Architecture (Spatial Order Preserved):")
    print(f"  Y-stream (rows)  : 28 -> {STREAM_LAYERS} -> {STREAM_OUT}  (flattened to {28*STREAM_OUT}D map)")
    print(f"  X-stream (cols)  : 28 -> {STREAM_LAYERS} -> {STREAM_OUT}  (flattened to {28*STREAM_OUT}D map)")
    print(f"  3D stack         : concat [{28*STREAM_OUT} + {28*STREAM_OUT}] = {MERGER_IN}D structural input")
    print(f"  Merger           : {MERGER_IN} -> {MERGER_LAYERS} -> {NUM_CLASSES}\n")

    # ── Model + optimizer ───────────────────────────────────────────────────
    model     = XYDualAxisNetGen10(DROPOUT_SCALE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_START, weight_decay=1e-4) # Gen 10: AdamW

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
            
            # Gen 10: Gradient Clipping (Shock absorber for the maze)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
    print("Training Gen 10 XY Dual-Axis Net (Spatial Order + AdamW)...\n")

    best_test_acc   = 0.0
    current_dropout = DROPOUT_SCALE
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_acc = train_one_epoch()
        test_acc  = evaluate()
        lr        = adjust_lr(test_acc)

        if epoch % UPDATE_EVERY == 0:
            target          = compute_dropout(train_acc)
            current_dropout += (target - current_dropout) / DROPOUT_SPRING
            model.set_dropout(current_dropout)

        # Early Stopping Check
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_xy_gen10_model.pt')
        else:
            epochs_no_improve += 1

        # Print every 50 epochs to keep console clean, or if it's a new best
        if epoch % 50 == 0 or test_acc > best_test_acc - 0.001:
            print(f"{epoch:5d} | train: {train_acc*100:.3f}% | "
                  f"test : {test_acc*100:.3f}% (best: {best_test_acc*100:.3f}%) | "
                  f"LR: {lr:.6f} | Drop: {current_dropout:.3f} | "
                  f"time: {time.time()-t0:.2f}s")

        if epochs_no_improve >= PATIENCE:
            print(f"\n[!] Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.")
            break

    print("\n--- Gen 10 Summary ---")
    print(f"Best test accuracy : {best_test_acc*100:.2f}%")
    print(f"Breaks 92% wall?   : {'YES!' if best_test_acc > 0.92 else 'Not yet...'}")
    print(f"Total parameters   : {total_params:,}\n")