#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLP trained on full MNIST (70 000 samples, 28x28 = 784 features).
Adaptive LR based on test accuracy, adaptive dropout based on train
accuracy only — independent of test.
Fast in-memory tensor loading — no DataLoader overhead.

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

# Funnel architecture — input (784) is separate, not in this list
LAYERS        = [392, 196, 98]
DROPOUT       = 0.08
USE_BATCHNORM = True
DTYPE         = torch.float32

# Adaptive dropout settings
DROPOUT_MAX      = 0.7
DROPOUT_EXPONENT = 2.31

# Learning‑rate schedule
LR_START    = 0.0013
LR_EXPONENT = 1.132

# --------------------------------------------------------------------------- #
# Device setup
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}\n")

# --------------------------------------------------------------------------- #
# Load MNIST — fast in-memory style
# --------------------------------------------------------------------------- #
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten 28x28 → 784
X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
X_test  = X_test.reshape(X_test.shape[0],  -1).astype(np.float32)

# Standardise
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)       # use train stats only

# Load everything onto device once — no per-batch copies
X_tr = torch.tensor(X_train, dtype=DTYPE,       device=DEVICE)
y_tr = torch.tensor(y_train.astype(np.int64),   device=DEVICE)
X_te = torch.tensor(X_test,  dtype=DTYPE,       device=DEVICE)
y_te = torch.tensor(y_test.astype(np.int64),    device=DEVICE)

INPUT_SIZE = X_tr.shape[1]
N_TRAIN    = X_tr.shape[0]

print(f"Dataset : MNIST")
print(f"Train   : {N_TRAIN:,} samples")
print(f"Test    : {X_te.shape[0]:,} samples\n")

# --------------------------------------------------------------------------- #
# Model definition
# --------------------------------------------------------------------------- #
class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, layers: list, n_out: int,
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

    def set_dropout(self, new_dropout: float):
        for module in self.net:
            if isinstance(module, nn.Dropout):
                module.p = new_dropout

model = TinyMLP(INPUT_SIZE, LAYERS, NUM_CLASSES,
                DROPOUT, USE_BATCHNORM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)

# --------------------------------------------------------------------------- #
# Adaptive helpers
# --------------------------------------------------------------------------- #
def adjust_lr(current_acc: float) -> float:
    new_lr = LR_START * (1.0 - current_acc) ** LR_EXPONENT
    for g in optimizer.param_groups:
        g["lr"] = new_lr
    return new_lr

def compute_dropout(train_acc: float) -> float:
    """Dropout rises with train accuracy — independent of test."""
    return min(DROPOUT * (1.0 + train_acc ** DROPOUT_EXPONENT), DROPOUT_MAX)

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def train_one_epoch() -> float:
    model.train()
    correct, total = 0, 0

    perm      = torch.randperm(N_TRAIN, device=DEVICE)
    X_shuffled = X_tr[perm]
    y_shuffled = y_tr[perm]

    for i in range(0, N_TRAIN, BATCH_SIZE):
        xb = X_shuffled[i : i + BATCH_SIZE]
        yb = y_shuffled[i : i + BATCH_SIZE]

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
        acc = (logits.argmax(1) == y_te).float().mean().item()
    return acc

# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #
print("Training with adaptive LR and train-driven adaptive dropout...\n")
best_test_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    t_start = time.time()

    train_acc       = train_one_epoch()
    test_acc        = evaluate()
    lr              = adjust_lr(test_acc)
    dropout_current = compute_dropout(train_acc)
    model.set_dropout(dropout_current)

    if test_acc > best_test_acc:
        best_test_acc = test_acc

    print(f"{epoch:4d} | train: {train_acc*100:.2f}% | "
          f"test : {test_acc*100:.2f}% | LR: {lr:.6f} | "
          f"Dropout: {dropout_current:.3f} | time: {time.time()-t_start:.2f}s")

print("\n--- Summary ---")
print(f"Best test accuracy : {best_test_acc*100:.2f}%")
print(f"Model layers       : {LAYERS}")
print(f"Base dropout       : {DROPOUT}")
print(f"Dropout exponent   : {DROPOUT_EXPONENT}")
print(f"BatchNorm          : {USE_BATCHNORM}")
print(f"LR exponent        : {LR_EXPONENT}\n")