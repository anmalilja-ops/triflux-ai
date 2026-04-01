#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP trained on full MNIST (70 000 samples, 28x28 = 784 features).
Adaptive LR based on test accuracy, adaptive dropout based on train
accuracy only — with spring/damping smoothing so dropout can't change
too fast between epochs.
  LR      = LR_START * (1 - test_acc)  ** LR_EXPONENT
  Dropout = DROPOUT_SCALE * (1 - train_acc) ** DROPOUT_EXPONENT  (capped at DROPOUT_MAX)
  Smoothing: current = current + (target - current) / DROPOUT_SPRING
  Update frequency: every UPDATE_EVERY epochs for extra stability
Gen 4 — mirror-LR dropout + spring damping + update frequency.
Author:  <your name>
Date:    2026‑03‑30
"""
# --------------------------------------------------------------------------- #
# Imports (safe to import at module level)
# --------------------------------------------------------------------------- #
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler

# ⚠️ Do NOT put any code here that runs immediately on import!
# All execution happens below in `if __name__ == "__main__":`

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
EPOCHS      = 40000
BATCH_SIZE  = 6400
NUM_CLASSES = 10
LAYERS        = [2048,1024,1024,1024,]
USE_BATCHNORM = True
DTYPE         = torch.float32

# Adaptive dropout
DROPOUT_SCALE    = 0.18
DROPOUT_MAX      = 0.9
DROPOUT_EXPONENT = -0.232

# Smoothing parameters
DROPOUT_SPRING   = 8
UPDATE_EVERY     = 20

# Learning rate schedule
LR_START    = 0.0013
LR_EXPONENT = 1.231

# --------------------------------------------------------------------------- #
# Device setup (safe to do at module level)
# --------------------------------------------------------------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# Model definition (also safe — class definitions don't run code)
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

# --------------------------------------------------------------------------- #
# Adaptive helpers (safe — pure functions)
# --------------------------------------------------------------------------- #
def adjust_lr(current_acc: float) -> float:
    """LR shrinks as test accuracy rises."""
    new_lr = LR_START * (1.0 - current_acc) ** LR_EXPONENT
    for g in optimizer.param_groups:
        g["lr"] = new_lr
    return new_lr

def compute_dropout(train_acc: float) -> float:
    """Target dropout — rises as train acc rises, capped at DROPOUT_MAX."""
    return min(DROPOUT_SCALE * (1.0 - train_acc) ** DROPOUT_EXPONENT, DROPOUT_MAX)

# --------------------------------------------------------------------------- #
# Main execution block
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # -----------------------------
    # Load MNIST — fast in-memory style
    # -----------------------------
    print(f"\nUsing device: {DEVICE}\n")
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test  = X_test.reshape(X_test.shape[0],  -1).astype(np.float32)

    scaler = StandardScaler()
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

    print(f"Dropout : {DROPOUT_SCALE} * (1 - train_acc) ^ {DROPOUT_EXPONENT}  capped at {DROPOUT_MAX}")
    print(f"Spring  : {DROPOUT_SPRING} epochs of inertia")
    print(f"Update  : every {UPDATE_EVERY} epochs")
    print(f"LR      : {LR_START} * (1 - test_acc) ^ {LR_EXPONENT}\n")

    # Preview the curve at key accuracies
    print("Dropout curve preview (before spring smoothing):")
    for acc in [0.85, 0.90, 0.95, 0.98, 0.99, 0.999]:
        d = min(DROPOUT_SCALE * (1.0 - acc) ** DROPOUT_EXPONENT, DROPOUT_MAX)
        print(f"  train {acc*100:.1f}%  →  dropout {d:.4f}")
    print()

    # -----------------------------
    # Initialize model, optimizer
    # -----------------------------
    model = TinyMLP(INPUT_SIZE, LAYERS, NUM_CLASSES,
                    DROPOUT_SCALE, USE_BATCHNORM).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)

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

    # -----------------------------
    # Main training loop
    # -----------------------------
    print("Training with adaptive LR and spring-damped train-driven dropout...\n")

    best_test_acc   = 0.0
    current_dropout = DROPOUT_SCALE   # start at base, smoothly tracks target

    for epoch in range(1, EPOCHS + 1):
        t_start = time.time()

        # Train
        train_acc = train_one_epoch()
        test_acc  = evaluate()

        # Adjust LR based on *test* accuracy
        lr = adjust_lr(test_acc)

        # Only update dropout every UPDATE_EVERY epochs (stability)
        if epoch % UPDATE_EVERY == 0:
            target_dropout = compute_dropout(train_acc)
            current_dropout += (target_dropout - current_dropout) / DROPOUT_SPRING
            model.set_dropout(current_dropout)

        # Track best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pt')

        print(f"{epoch:4d} | train: {train_acc*100:.3f}% | "
              f"test : {test_acc*100:.3f}% | LR: {lr:.8f} | "
              f"Dropout: {current_dropout:.3f} | time: {time.time()-t_start:.3f}s")

    print("\n--- Summary ---")
    print(f"Best test accuracy : {best_test_acc*100:.2f}%")
    print(f"Model layers       : {LAYERS}")
    print(f"Dropout scale      : {DROPOUT_SCALE}")
    print(f"Dropout max        : {DROPOUT_MAX}")
    print(f"Dropout exponent   : {DROPOUT_EXPONENT}")
    print(f"Dropout spring     : {DROPOUT_SPRING}")
    print(f"Update every       : {UPDATE_EVERY} epochs")
    print(f"BatchNorm          : {USE_BATCHNORM}")
    print(f"LR exponent        : {LR_EXPONENT}\n")
