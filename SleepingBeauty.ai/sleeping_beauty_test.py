#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SleepingBeauty.ai - Real Dual-Axis MLP Training with Adaptive REM Sleep.
Runs real epochs. When test accuracy stalls, triggers a "Sleep Phase" 
to dream on noise and consolidate weights before waking back up.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Core Configuration
# --------------------------------------------------------------------------- #
EPOCHS               = 2000
BATCH_SIZE           = 1024
NUM_CLASSES          = 10

POS_EMBED_DIM        = 4     
ROW_RANDOM_RATE      = 0.01  

STREAM_IN     = 28 + POS_EMBED_DIM
STREAM_LAYERS = [256, 128]       
STREAM_OUT    = 128          
MERGER_LAYERS = [1024, 512, 256, 128]       

USE_BATCHNORM = True
DTYPE         = torch.float32
LR_START      = 0.002
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Sleep Parameters
SLEEP_PATIENCE   = 100  # Epochs without improvement before going to sleep
SLEEP_DURATION   = 50   # How many epochs the dream phase lasts


# --------------------------------------------------------------------------- #
# Helper Functions & Model
# --------------------------------------------------------------------------- #
def make_mlp(in_dim: int, hidden: list, out_dim: int, dropout: float, use_bn: bool) -> nn.Sequential:
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

def apply_controlled_random_rows(img: torch.Tensor, rate: float) -> torch.Tensor:
    if rate <= 0.0: return img
    B = img.shape[0]
    mask = torch.rand(B, 28, device=img.device) < rate
    if mask.any():
        rand_idx = torch.randint(0, 28, (B, 28), device=img.device)
        batch_idx = torch.arange(B, device=img.device).unsqueeze(1).expand(B, 28)
        perturbed = img[batch_idx, rand_idx]
        img = torch.where(mask.unsqueeze(2), perturbed, img)
    return img

class XYDualAxisNetWithPos(nn.Module):
    def __init__(self, dropout: float, pos_dim: int = POS_EMBED_DIM):
        super().__init__()
        self.row_pos_embed = nn.Embedding(28, pos_dim)
        self.col_pos_embed = nn.Embedding(28, pos_dim)
        self.y_row_encoder = make_mlp(STREAM_IN, STREAM_LAYERS, STREAM_OUT, dropout, USE_BATCHNORM)
        self.x_col_encoder = make_mlp(STREAM_IN, STREAM_LAYERS, STREAM_OUT, dropout, USE_BATCHNORM)
        
        merger_in_dim = (28 * STREAM_OUT) * 2 
        self.merger = make_mlp(merger_in_dim, MERGER_LAYERS, NUM_CLASSES, dropout, USE_BATCHNORM)
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
        y_encoded     = self.y_row_encoder(rows_with_pos.reshape(B * 28, STREAM_IN))
        y_structural_map = y_encoded.view(B, -1)

        cols = img.permute(0, 2, 1)                       
        cols_with_pos = torch.cat([cols, col_pos], dim=2)
        x_encoded     = self.x_col_encoder(cols_with_pos.reshape(B * 28, STREAM_IN))
        x_structural_map = x_encoded.view(B, -1)          

        return self.merger(torch.cat([y_structural_map, x_structural_map], dim=1))


# --------------------------------------------------------------------------- #
# Real Training Loop with Adaptive Sleep
# --------------------------------------------------------------------------- #
def run_experiment(use_sleeping_beauty=False):
    # 1. Load Real Data
    print("Loading Fashion MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Flatten and scale
    x_train = x_train.astype(np.float32).reshape(-1, 784)
    x_test = x_test.astype(np.float32).reshape(-1, 784)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Convert to PyTorch tensors and move to GPU
    x_train = torch.tensor(x_train, dtype=DTYPE, device=DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    x_test = torch.tensor(x_test, dtype=DTYPE, device=DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.long, device=DEVICE)
    
    model = XYDualAxisNetWithPos(dropout=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_START, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Starting Real Experiment: SleepingBeauty.ai = {use_sleeping_beauty} ---")
    
    best_test_acc = 0.0
    epochs_no_improve = 0
    is_sleeping = False
    sleep_counter = 0
    n_train = x_train.size(0)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        # 💤 TRIGGER SLEEPING BEAUTY ADAPTIVELY 💤
        if use_sleeping_beauty and not is_sleeping and epochs_no_improve >= SLEEP_PATIENCE:
            is_sleeping = True
            sleep_counter = SLEEP_DURATION
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00005  # Drop LR for dream phase
            print(f"  [💤 ENTERING REM SLEEP] Epoch {epoch} | Brain is dreaming to consolidate weights...")
        
        # If currently sleeping, run dream batch instead of real data
        if is_sleeping:
            for _ in range(5): # Do 5 mini-dream batches per epoch
                X_noise = torch.randn(BATCH_SIZE, 784, device=DEVICE) * 0.05
                optimizer.zero_grad()
                mock_out = model(X_noise, perturb_rate=0.03)
                # Push towards class 0 (or uniform) to smooth weights
                loss = F.cross_entropy(mock_out, torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE))
                loss.backward()
                optimizer.step()
            
            sleep_counter -= 1
            if sleep_counter <= 0:
                is_sleeping = False
                epochs_no_improve = 0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LR_START * 0.3  # Wake up with fresh, slightly lower LR
                print(f"  [☀️ WAKING UP] Epoch {epoch} | Resuming normal training with refreshed weights.")
            
            if epoch % 50 == 0:
                print(f"  [💤 Dreaming] Epoch {epoch} | Consolidating...")
            continue

        # Normal Waking Training Phase
        perm = torch.randperm(n_train, device=DEVICE)
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            inputs, targets = x_train[idx], y_train[idx]
            
            optimizer.zero_grad(set_to_none=True)
            out = model(inputs, perturb_rate=ROW_RANDOM_RATE)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.item() * inputs.size(0)
            tr_correct += out.argmax(1).eq(targets).sum().item()
            tr_total += targets.size(0)
            
        tr_acc = 100. * tr_correct / tr_total
        tr_loss /= tr_total

        # Real Evaluation
        model.eval()
        te_loss, te_correct, te_total = 0.0, 0, 0
        with torch.no_grad():
            for i in range(0, x_test.size(0), BATCH_SIZE):
                inputs, targets = x_test[i:i+BATCH_SIZE], y_test[i:i+BATCH_SIZE]
                out = model(inputs)
                loss = criterion(out, targets)
                te_loss += loss.item() * inputs.size(0)
                te_correct += out.argmax(1).eq(targets).sum().item()
                te_total += targets.size(0)
                
        te_acc = 100. * te_correct / te_total
        te_loss /= te_total
        
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epoch % 50 == 0 or epoch == EPOCHS:
            gap = tr_acc - te_acc
            print(f"  Epoch {epoch:4d} | train: {tr_acc:.3f}% | test: {te_acc:.3f}% (best: {best_test_acc:.3f}%) | gap: {gap:.3f}% | Stalled: {epochs_no_improve}")
            
    return best_test_acc

# Execute Side-by-Side
acc_off = run_experiment(use_sleeping_beauty=False)
acc_on  = run_experiment(use_sleeping_beauty=True)

print("\n" + "="*50 + "\nFINAL EXPERIMENT ACCURACY COMPARISON\n" + "="*50)
print(f"Run 1: SleepingBeauty Mode OFF -> Final Peak Test Acc: {acc_off:.3f}%")
print(f"Run 2: SleepingBeauty Mode ON  -> Final Peak Test Acc: {acc_on:.3f}%")
print(f"Net Architectural Improvement  : +{acc_on - acc_off:.3f}%")