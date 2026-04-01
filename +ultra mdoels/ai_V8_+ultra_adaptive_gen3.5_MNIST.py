#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLP trained on full MNIST (70 000 samples, 28x28 = 784 features).
Adaptive LR based on test accuracy, adaptive dropout mirroring
the LR formula but driven by train accuracy only.

  LR      = LR_START    * (1 - test_acc)  ** LR_EXPONENT
  Dropout = DROPOUT_MAX * (1 - train_acc) ** DROPOUT_EXPONENT

  Higher exponent = more aggressive curve (same logic as LR).

Gen 3.5 — mirror-LR dropout formula.

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
EPOCHS      = 40000
BATCH_SIZE  = 256
NUM_CLASSES = 10

# Funnel architecture — input (784) is separate, not in this list
LAYERS        = [784,392,196,98]
USE_BATCHNORM = True
DTYPE         = torch.float32

# Adaptive dropout
DROPOUT_MAX      = 0.21   # scales down from here as train acc drops
DROPOUT_EXPONENT = -0.08  # start same as LR — tune from here

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

X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
X_test  = X_test.reshape(X_test.shape[0],  -1).astype(np.float32)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_tr = torch.tensor(X_train, dtype=DTYPE,     device=DEVICE)
y_tr = torch.tensor(y_train.astype(np.int64), device=DEVICE)
X_te = torch.tensor(X_test,  dtype=DTYPE,     device=DEVICE)
y_te = torch.tensor(y_test.astype(np.int64),  device=DEVICE)

INPUT_SIZE = X_tr.shape[1]
N_TRAIN    = X_tr.shape[0]

print(f"Dataset : MNIST")
print(f"Train   : {N_TRAIN:,} samples")
print(f"Test    : {X_te.shape[0]:,} samples\n")
print(f"Dropout : {DROPOUT_MAX} * (1 - train_acc) ^ {DROPOUT_EXPONENT}")
print(f"LR      : {LR_START} * (1 - test_acc)  ^ {LR_EXPONENT}\n")

# preview the curve at key accuracies
print("Dropout curve preview:")
for acc in [0.85, 0.90, 0.95, 0.98, 0.99, 0.999]:
    d = DROPOUT_MAX * (1.0 - acc) ** DROPOUT_EXPONENT
    print(f"  train {acc*100:.1f}%  →  dropout {d:.4f}")
print()

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
                DROPOUT_MAX * (1.0 - 0.85) ** DROPOUT_EXPONENT,  # init at ~85% acc
                USE_BATCHNORM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)

# --------------------------------------------------------------------------- #
# Adaptive helpers
# --------------------------------------------------------------------------- #
def adjust_lr(current_acc: float) -> float:
    """LR shrinks as test accuracy rises."""
    new_lr = LR_START * (1.0 - current_acc) ** LR_EXPONENT
    for g in optimizer.param_groups:
        g["lr"] = new_lr
    return new_lr

def compute_dropout(train_acc: float) -> float:
    """
    Mirror of LR formula but for train accuracy.
    Higher DROPOUT_EXPONENT = more aggressive curve.
    Lower  DROPOUT_EXPONENT = gentler curve.
    DROPOUT_MAX scales the overall level up/down.
    """
    return DROPOUT_MAX * (1.0 - train_acc) ** DROPOUT_EXPONENT

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def train_one_epoch() -> float:
    model.train()
    correct, total = 0, 0

    perm       = torch.randperm(N_TRAIN, device=DEVICE)
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
print("Training with adaptive LR and mirror-LR train-driven dropout...\n")
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
print(f"Dropout max        : {DROPOUT_MAX}")
print(f"Dropout exponent   : {DROPOUT_EXPONENT}")
print(f"BatchNorm          : {USE_BATCHNORM}")
print(f"LR exponent        : {LR_EXPONENT}\n")