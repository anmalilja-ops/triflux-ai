#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small demo that trains a tiny MLP on scikit‑learn's `digits` dataset.
It uses PyTorch (CUDA if available) and an *adaptive* learning‑rate schedule
that depends on the current test accuracy, with optional adaptive dropout.

Dropout now scales with train accuracy only — the better the model
memorises the training data, the harder dropout fights back.

Author:  <your name>
Date:    2026‑03‑28
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
EPOCHS      = 20000            # how many epochs to run
BATCH_SIZE  = 128              # batch size for training
NUM_CLASSES = 10               # digits 0‑9

# Model hyper‑parameters
LAYERS          = [64,64,64]      # width of each hidden layer (empty → linear)
DROPOUT         = 0.08           # base dropout (lower since train_acc is large)
USE_BATCHNORM   = True           # whether to add BatchNorm1d after each layer
DTYPE           = torch.float32

# Adaptive dropout settings
DROPOUT_MAX      = 0.45
DROPOUT_EXPONENT = 1.5           # how aggressively dropout scales with train acc

# Learning‑rate schedule
LR_START    = 0.0013
LR_EXPONENT = 1.132

# --------------------------------------------------------------------------- #
# Device setup
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}\n")

# --------------------------------------------------------------------------- #
# Load data (scikit‑learn digits)
# --------------------------------------------------------------------------- #
digits = load_digits()
X_raw  = digits.data.astype(np.float32)          # shape (1797, 64)
y_raw  = digits.target.astype(np.int64)          # shape (1797,)

# Standardise features
scaler = StandardScaler()
X_std  = scaler.fit_transform(X_raw)

# Convert to tensors and move to device
X_tensor = torch.tensor(X_std, dtype=DTYPE, device=DEVICE)
y_tensor = torch.tensor(y_raw, dtype=torch.long, device=DEVICE)

INPUT_SIZE = X_tensor.shape[1]
N_TOTAL    = X_tensor.shape[0]

print(f"Dataset: sklearn digits (1797 samples, 64 features)")
print(f"Train / Test split: {int(0.9 * N_TOTAL)} / {N_TOTAL - int(0.9 * N_TOTAL)}\n")

# --------------------------------------------------------------------------- #
# Train‑test split
# --------------------------------------------------------------------------- #
perm = torch.randperm(N_TOTAL, device=DEVICE)
train_idx = perm[: int(0.9 * N_TOTAL)]
test_idx  = perm[int(0.9 * N_TOTAL):]

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
        self.base_dropout = dropout
        self.use_bn = use_bn
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

        # output layer
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
# Adaptive learning‑rate helper
# --------------------------------------------------------------------------- #
def adjust_lr(current_acc: float) -> float:
    """Update optimizer LR based on current test accuracy."""
    new_lr = LR_START * (1.0 - current_acc) ** LR_EXPONENT
    for g in optimizer.param_groups:
        g["lr"] = new_lr
    return new_lr

# --------------------------------------------------------------------------- #
# Adaptive dropout — train accuracy only, test is irrelevant
# --------------------------------------------------------------------------- #
def compute_dropout(train_acc):
    """Dropout rises as train accuracy rises — independent of test."""
    dropout_current = DROPOUT * (1.0 + train_acc ** DROPOUT_EXPONENT)
    return min(dropout_current, DROPOUT_MAX)

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def train_one_epoch() -> float:
    """Train for one epoch and return training accuracy."""
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
    """Return test accuracy."""
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

    train_acc = train_one_epoch()
    test_acc  = evaluate()
    lr        = adjust_lr(test_acc)

    # dropout scales with train accuracy only
    dropout_current = compute_dropout(train_acc)
    model.set_dropout(dropout_current)

    if test_acc > best_test_acc:
        best_test_acc = test_acc

    print(f"{epoch:4d} | train: {train_acc*100:.2f}% | "
          f"test : {test_acc*100:.2f}% | LR: {lr:.6f} | "
          f"Dropout: {dropout_current:.3f} | time: {time.time()-t_start:.2f}s")

print("\n--- Summary ---")
print(f"Best test accuracy: {best_test_acc*100:.2f}%")
print(f"Model layers:      {LAYERS}")
print(f"Base dropout:      {DROPOUT}")
print(f"Dropout exponent:  {DROPOUT_EXPONENT}")
print(f"BatchNorm:         {USE_BATCHNORM}")
print(f"LR exponent:       {LR_EXPONENT}\n")