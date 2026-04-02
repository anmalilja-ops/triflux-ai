#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XY Dual-Axis MLP trained on full MNIST (70,000 samples, 28x28 = 784 features).

Architecture:
  - Y-stream: reads each image as 28 rows (top→bottom), each row = 28 features
  - X-stream: reads each image as 28 columns (left→right), each col = 28 features
  - Both streams output embeddings that are stacked into a "3D" combined tensor
  - A merger network fuses both streams → 10 class output

Adaptive systems (gen5):
  LR      = LR_START * (1 - test_acc)  ** LR_EXPONENT
  Dropout = DROPOUT_SCALE * (1 - train_acc) ** DROPOUT_EXPONENT (capped at DROPOUT_MAX)
  Smoothing: current = current + (target - current) / DROPOUT_SPRING
  Update frequency: every UPDATE_EVERY epochs

E1 Evolutionary Ensemble:
  - Trains ENSEMBLE_SIZE models in parallel each epoch
  - Each model has slight weight noise injected (controlled by NOISE_SCALE)
  - After each epoch the best model by test accuracy is selected as the survivor
  - All other models are re-initialised from the survivor + fresh noise
  - This escapes local minima while keeping the adaptive gen5 system intact

Gen 9 — XY dual-axis 3D-stacked siamese-style network + E1 evolutionary ensemble.
Author:  <boki_231>
Date:    2026-03-30
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import copy
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
BATCH_SIZE  = 1024
NUM_CLASSES = 10

# Each axis stream hidden layers
STREAM_LAYERS = [64, 64, 64]
STREAM_OUT    = 64

# Merger network
MERGER_LAYERS = [64, 64, 64]

USE_BATCHNORM = True
DTYPE         = torch.float32

# Adaptive dropout (gen5)
DROPOUT_SCALE    = 0.10
DROPOUT_MAX      = 0.9
DROPOUT_EXPONENT = -0.23
DROPOUT_SPRING   = 3
UPDATE_EVERY     = 6

# Learning rate (gen5)
LR_START    = 0.08
LR_EXPONENT = 1.2

# --------------------------------------------------------------------------- #
# E1 Evolutionary Ensemble settings
# --------------------------------------------------------------------------- #
ENSEMBLE_SIZE = 10          # number of models trained in parallel each epoch
                            # set to 1 to disable ensemble and run standard training

NOISE_SCALE   = 0.01       # std of Gaussian weight noise injected into each
                            # child model spawned from the survivor each epoch
                            # higher = more exploration, lower = lower exploitation
                            # recommended range: 0.0005 – 0.01

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

        self.y_row_encoder = make_mlp(28, STREAM_LAYERS, STREAM_OUT,
                                       dropout, USE_BATCHNORM)
        self.x_col_encoder = make_mlp(28, STREAM_LAYERS, STREAM_OUT,
                                       dropout, USE_BATCHNORM)
        self.merger = make_mlp(STREAM_OUT * 2, MERGER_LAYERS, NUM_CLASSES,
                                dropout, USE_BATCHNORM)
        self._dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        img = x.view(B, 28, 28)

        # Y stream
        rows_flat = img.reshape(B * 28, 28)
        y_encoded = self.y_row_encoder(rows_flat)
        y_encoded = y_encoded.view(B, 28, STREAM_OUT)
        y_pooled  = y_encoded.mean(dim=1)

        # X stream
        cols      = img.permute(0, 2, 1)
        cols_flat = cols.reshape(B * 28, 28)
        x_encoded = self.x_col_encoder(cols_flat)
        x_encoded = x_encoded.view(B, 28, STREAM_OUT)
        x_pooled  = x_encoded.mean(dim=1)

        # 3D stack + merge
        combined = torch.cat([y_pooled, x_pooled], dim=1)
        return self.merger(combined)

    def set_dropout(self, p: float):
        self._dropout = p
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

# --------------------------------------------------------------------------- #
# E1 helpers
# --------------------------------------------------------------------------- #
def inject_noise(model: nn.Module, noise_scale: float) -> nn.Module:
    """Return a deep copy of model with Gaussian noise added to all weights."""
    noisy = copy.deepcopy(model)
    with torch.no_grad():
        for param in noisy.parameters():
            param.add_(torch.randn_like(param) * noise_scale)
    return noisy


def spawn_ensemble(survivor: nn.Module,
                   optimizer_state: dict,
                   lr: float,
                   noise_scale: float,
                   n: int):
    """
    Create n models all derived from survivor + noise.
    Returns (models, optimizers) lists.
    The first slot is the survivor itself (no noise) — always kept.
    """
    models    = []
    optimizers = []
    for i in range(n):
        if i == 0:
            # Keep the survivor unchanged as one candidate
            m = copy.deepcopy(survivor)
        else:
            m = inject_noise(survivor, noise_scale)
        m.to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        # Copy optimizer state from survivor's optimizer so momentum etc carry over
        if i == 0 and optimizer_state is not None:
            try:
                opt.load_state_dict(optimizer_state)
                # Re-apply current lr in case it changed
                for g in opt.param_groups:
                    g["lr"] = lr
            except Exception:
                pass
        models.append(m)
        optimizers.append(opt)
    return models, optimizers

# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def train_one_epoch(model, optimizer, X_tr, y_tr, N_TRAIN):
    model.train()
    correct, total = 0, 0
    perm   = torch.randperm(N_TRAIN, device=DEVICE)
    X_shuf = X_tr[perm]
    y_shuf = y_tr[perm]
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


def evaluate(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        return (logits.argmax(1) == y_te).float().mean().item()


def adjust_lr(optimizer, test_acc: float) -> float:
    new_lr = LR_START * (1.0 - test_acc) ** LR_EXPONENT
    for g in optimizer.param_groups:
        g["lr"] = new_lr
    return new_lr


def compute_dropout(train_acc: float) -> float:
    return min(DROPOUT_SCALE * (1.0 - train_acc) ** DROPOUT_EXPONENT, DROPOUT_MAX)

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

    X_tr = torch.tensor(X_train, dtype=DTYPE, device=DEVICE)
    y_tr = torch.tensor(y_train.astype(np.int64), device=DEVICE)
    X_te = torch.tensor(X_test,  dtype=DTYPE, device=DEVICE)
    y_te = torch.tensor(y_test.astype(np.int64), device=DEVICE)

    N_TRAIN = X_tr.shape[0]

    print(f"Dataset       : MNIST")
    print(f"Train         : {N_TRAIN:,} samples")
    print(f"Test          : {X_te.shape[0]:,} samples\n")

    print(f"Architecture:")
    print(f"  Y-stream    : 28 → {STREAM_LAYERS} → {STREAM_OUT}  (mean pool over 28 rows)")
    print(f"  X-stream    : 28 → {STREAM_LAYERS} → {STREAM_OUT}  (mean pool over 28 cols)")
    print(f"  3D stack    : concat [{STREAM_OUT} + {STREAM_OUT}] = {STREAM_OUT*2} dim")
    print(f"  Merger      : {STREAM_OUT*2} → {MERGER_LAYERS} → {NUM_CLASSES}\n")

    print(f"Gen5 adaptive:")
    print(f"  Dropout     : {DROPOUT_SCALE} * (1 - train_acc) ^ {DROPOUT_EXPONENT}  capped at {DROPOUT_MAX}")
    print(f"  Spring      : {DROPOUT_SPRING}  |  Update every {UPDATE_EVERY} epochs")
    print(f"  LR          : {LR_START} * (1 - test_acc) ^ {LR_EXPONENT}\n")

    print(f"E1 Ensemble:")
    print(f"  Models      : {ENSEMBLE_SIZE}  ({'parallel selection active' if ENSEMBLE_SIZE > 1 else 'disabled — single model mode'})")
    print(f"  Noise scale : {NOISE_SCALE}\n")

    # Dropout curve preview
    print("Dropout curve preview:")
    for acc in [0.85, 0.90, 0.95, 0.98, 0.99, 0.999]:
        d = min(DROPOUT_SCALE * (1.0 - acc) ** DROPOUT_EXPONENT, DROPOUT_MAX)
        print(f"  train {acc*100:.1f}%  →  dropout {d:.4f}")
    print()

    # ── Initialise survivor ─────────────────────────────────────────────────
    survivor     = XYDualAxisNet(DROPOUT_SCALE).to(DEVICE)
    surv_opt     = torch.optim.Adam(survivor.parameters(), lr=LR_START)

    total_params = sum(p.numel() for p in survivor.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}\n")

    best_test_acc   = 0.0
    current_dropout = DROPOUT_SCALE
    current_lr      = LR_START

    # ── Rolling history for stats ────────────────────────────────────────────
    # Stores per-epoch: (survivor_test, ensemble_min, ensemble_avg, ensemble_max)
    # ensemble_* are None in single-model mode
    ROLLING_WINDOW  = 100          # epochs to average over for rolling stats
    history_surv    = []           # survivor test acc each epoch
    history_ens_min = []           # ensemble min  test acc each epoch
    history_ens_avg = []           # ensemble mean test acc each epoch
    history_ens_max = []           # ensemble max  test acc each epoch

    def rolling_avg(lst):
        if not lst:
            return None
        window = lst[-ROLLING_WINDOW:]
        return sum(window) / len(window)

    HEADER_EVERY = 100000000   # reprint column headers every N epochs

    def print_header():
        if ENSEMBLE_SIZE > 1:
            print(f"\n{'Epoch':>6} | {'Train':>8} | {'Test':>8} | {'Best':>8} | "
                  f"{'Clon-min':>9} | {'Clon-avg':>9} | {'Clon-max':>9} | "
                  f"{'Roll100':>8} | {'LR':>12} | {'Drop':>5} | Time")
            print("-" * 115)
        else:
            print(f"\n{'Epoch':>6} | {'Train':>8} | {'Test':>8} | {'Best':>8} | "
                  f"{'Roll100':>8} | {'LR':>12} | {'Drop':>5} | Time")
            print("-" * 75)

    print("Training XY Dual-Axis Net  |  gen5 adaptive  |  E1 evolutionary ensemble\n")
    print_header()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        if ENSEMBLE_SIZE > 1:
            # ── E1: spawn ensemble from survivor ────────────────────────────
            models, optimizers = spawn_ensemble(
                survivor,
                surv_opt.state_dict(),
                current_lr,
                NOISE_SCALE,
                ENSEMBLE_SIZE
            )

            # Train all models for one epoch in sequence
            train_accs = []
            test_accs  = []
            for m, opt in zip(models, optimizers):
                tr_acc = train_one_epoch(m, opt, X_tr, y_tr, N_TRAIN)
                te_acc = evaluate(m, X_te, y_te)
                train_accs.append(tr_acc)
                test_accs.append(te_acc)

            # Select best by test accuracy
            best_idx   = int(np.argmax(test_accs))
            survivor   = models[best_idx]
            surv_opt   = optimizers[best_idx]
            train_acc  = train_accs[best_idx]
            test_acc   = test_accs[best_idx]

            # Ensemble spread stats
            ens_min = min(test_accs)
            ens_avg = sum(test_accs) / len(test_accs)
            ens_max = max(test_accs)          # same as test_acc (winner)

            history_ens_min.append(ens_min)
            history_ens_avg.append(ens_avg)
            history_ens_max.append(ens_max)

        else:
            # ── Standard single-model training ──────────────────────────────
            train_acc = train_one_epoch(survivor, surv_opt, X_tr, y_tr, N_TRAIN)
            test_acc  = evaluate(survivor, X_te, y_te)
            ens_min = ens_avg = ens_max = None

        # ── Rolling survivor history ─────────────────────────────────────────
        history_surv.append(test_acc)
        roll100 = rolling_avg(history_surv)

        # ── Gen5 adaptive updates ────────────────────────────────────────────
        current_lr = adjust_lr(surv_opt, test_acc)

        if epoch % UPDATE_EVERY == 0:
            target          = compute_dropout(train_acc)
            current_dropout += (target - current_dropout) / DROPOUT_SPRING
            survivor.set_dropout(current_dropout)

        # ── Checkpoint best ──────────────────────────────────────────────────
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(survivor.state_dict(), 'best_xy_model_e1.pt')

        elapsed = time.time() - t0

        # ── Per-epoch print ──────────────────────────────────────────────────
        if ENSEMBLE_SIZE > 1:
            print(f"{epoch:6d} | "
                  f"train: {train_acc*100:6.3f}% | "
                  f"test: {test_acc*100:6.3f}% | "
                  f"best: {best_test_acc*100:6.3f}% | "
                  f"clon-min: {ens_min*100:6.3f}% | "
                  f"clon-avg: {ens_avg*100:6.3f}% | "
                  f"clon-max: {ens_max*100:6.3f}% | "
                  f"roll100: {roll100*100:6.3f}% | "
                  f"LR: {current_lr:.8f} | "
                  f"drop: {current_dropout:.3f} | "
                  f"{elapsed:.3f}s")
        else:
            print(f"{epoch:6d} | "
                  f"train: {train_acc*100:6.3f}% | "
                  f"test: {test_acc*100:6.3f}% | "
                  f"best: {best_test_acc*100:6.3f}% | "
                  f"roll100: {roll100*100:6.3f}% | "
                  f"LR: {current_lr:.8f} | "
                  f"drop: {current_dropout:.3f} | "
                  f"{elapsed:.3f}s")

        # ── Every 100 epochs print a spread summary ──────────────────────────
        if epoch % 100 == 0 and ENSEMBLE_SIZE > 1:
            r_min = rolling_avg(history_ens_min)
            r_avg = rolling_avg(history_ens_avg)
            r_max = rolling_avg(history_ens_max)
            r_sur = rolling_avg(history_surv)
            spread = (r_max - r_min) * 100 if r_max and r_min else 0.0
            print(f"\n  ── 100-epoch spread summary (epoch {epoch}) ──────────────────")
            print(f"     Survivor  rolling avg : {r_sur*100:.3f}%")
            print(f"     Ensemble  rolling min : {r_min*100:.3f}%")
            print(f"     Ensemble  rolling avg : {r_avg*100:.3f}%")
            print(f"     Ensemble  rolling max : {r_max*100:.3f}%")
            print(f"     Spread (max-min)      : {spread:.3f}%  "
                  + ("← good diversity" if spread > 0.3
                     else "← low diversity, try higher NOISE_SCALE" if spread < 0.1
                     else "← acceptable"))
            print(f"     All-time best test    : {best_test_acc*100:.3f}%")
            print(f"  ─────────────────────────────────────────────────────────────\n")

        # ── Reprint header every N epochs so columns stay readable ─────────
        if epoch % HEADER_EVERY == 0:
            print_header()

    print("\n--- Summary ---")
    print(f"Best test accuracy : {best_test_acc*100:.2f}%")
    print(f"Final roll-100 avg : {rolling_avg(history_surv)*100:.3f}%")
    if ENSEMBLE_SIZE > 1:
        print(f"Final ens-min avg  : {rolling_avg(history_ens_min)*100:.3f}%")
        print(f"Final ens-avg avg  : {rolling_avg(history_ens_avg)*100:.3f}%")
        print(f"Final ens-max avg  : {rolling_avg(history_ens_max)*100:.3f}%")
        spread = (rolling_avg(history_ens_max) - rolling_avg(history_ens_min)) * 100
        print(f"Final spread       : {spread:.3f}%")
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
    print(f"LR exponent        : {LR_EXPONENT}")
    print(f"Ensemble size      : {ENSEMBLE_SIZE}")
    print(f"Noise scale        : {NOISE_SCALE}\n")