#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tiny MLP demo on MNIST dataset using PyTorch (CUDA if available)
with adaptive learning‑rate schedule.

Author:  <your name>
Date:    2026‑03‑29
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
EPOCHS      = 20000
BATCH_SIZE  = 256
NUM_CLASSES = 10

LAYERS          = [ 784,392,196,98,10]
DROPOUT         = 0.17
USE_BATCHNORM   = True
DTYPE           = torch.float32

LR_START    = 0.0013
LR_EXPONENT = 1.132

# --------------------------------------------------------------------------- #
# Device setup
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}\n")

# --------------------------------------------------------------------------- #
# Load data (MNIST)
# --------------------------------------------------------------------------- #
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_raw = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
y_raw = np.concatenate([y_train, y_test], axis=0).astype(np.int64)

# Flatten 28x28 → 784
X_raw = X_raw.reshape(X_raw.shape[0], -1)

# Standardise features
scaler = StandardScaler()
X_std = scaler.fit_transform(X_raw)

# Convert to tensors
X_tensor = torch.tensor(X_std, dtype=DTYPE, device=DEVICE)
y_tensor = torch.tensor(y_raw, dtype=torch.long, device=DEVICE)

INPUT_SIZE = X_tensor.shape[1]
N_TOTAL    = X_tensor.shape[0]

print(f"Dataset: MNIST ({N_TOTAL} samples, {INPUT_SIZE} features)")
print(f"Train / Test split: {X_train.shape[0]} / {X_test.shape[0]}\n")

# --------------------------------------------------------------------------- #
# Train‑test split (original MNIST split)
# --------------------------------------------------------------------------- #
train_idx = torch.arange(0, X_train.shape[0], device=DEVICE)
test_idx  = torch.arange(X_train.shape[0], N_TOTAL, device=DEVICE)

X_tr = X_tensor[train_idx]
y_tr = y_tensor[train_idx]
X_te = X_tensor[test_idx]
y_te = y_tensor[test_idx]

N_TRAIN = len(X_tr)

# --------------------------------------------------------------------------- #
# Model definition
# --------------------------------------------------------------------------- #
class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, layers: list[int], n_out: int,
                 dropout: float, use_bn: bool):
        super().__init__()
        modules = []
        in_f = input_dim

        for width in layers:
            modules.append(nn.Linear(in_f, width, dtype=DTYPE))
            if use_bn:
                modules.append(nn.BatchNorm1d(width, dtype=DTYPE))
            modules.append(nn.ReLU())
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))
            in_f = width

        modules.append(nn.Linear(in_f, n_out, dtype=DTYPE))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = TinyMLP(INPUT_SIZE, LAYERS, NUM_CLASSES,
                DROPOUT, USE_BATCHNORM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)

# --------------------------------------------------------------------------- #
# Adaptive learning‑rate helper
# --------------------------------------------------------------------------- #
def adjust_lr(current_acc: float) -> float:
    new_lr = LR_START * (1.0 - current_acc) ** LR_EXPONENT
    for g in optimizer.param_groups:
        g["lr"] = new_lr
    return new_lr

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def train_one_epoch() -> float:
    model.train()
    correct, total = 0, 0

    perm = torch.randperm(N_TRAIN, device=DEVICE)
    X_shuffled = X_tr[perm]
    y_shuffled = y_tr[perm]

    for i in range(0, N_TRAIN, BATCH_SIZE):
        xb = X_shuffled[i : i + BATCH_SIZE]
        yb = y_shuffled[i : i + BATCH_SIZE]

        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()

        correct += (logits.argmax(1) == yb).sum().item()
        total   += yb.size(0)

    return correct / total


def evaluate() -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        acc = (logits.argmax(1) == y_te).float().mean().item()
    return acc

# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #
print("Training with adaptive LR...\n")
best_test_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    t_start = time.time()

    train_acc = train_one_epoch()
    test_acc  = evaluate()
    lr        = adjust_lr(test_acc)

    if test_acc > best_test_acc:
        best_test_acc = test_acc

    print(f"{epoch:4d} | "
          f"train: {train_acc * 100:.2f}% | "
          f"test : {test_acc * 100:.2f}% | "
          f"LR   : {lr:.6f} | "
          f"time : {time.time() - t_start:.2f}s")

print("\n--- Summary ---")
print(f"Best test accuracy: {best_test_acc * 100:.2f}%")
print(f"Model layers:      {LAYERS}")
print(f"Dropout:           {DROPOUT}")
print(f"BatchNorm:         {USE_BATCHNORM}")
print(f"LR exponent:       {LR_EXPONENT}\n")