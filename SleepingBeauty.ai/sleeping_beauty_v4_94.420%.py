#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XY Dual-Axis MLP with Positional IDs, Row Sampling, and 
Sleeping Beauty v4 (Real-Memory Diffusion Dreams).
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
BATCH_SIZE           = 2048
NUM_CLASSES          = 10
PASSES_PER_EPOCH     = 2      # Set to 1 or 2

POS_EMBED_DIM        = 4     
ROW_RANDOM_RATE      = 0.01  

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
SLEEP_PATIENCE = 15  
SLEEP_DURATION = 5
DREAM_NOISE_LEVEL = 0.75  # How much to scramble the real memories (0.0 = perfect recall, 1.0 = pure noise)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# --------------------------------------------------------------------------- #
# Neural Baker (Conditional VAE)
# --------------------------------------------------------------------------- #
class NeuralBaker(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784 + NUM_CLASSES, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + NUM_CLASSES, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 784)
        )

    def encode(self, x, y):
        h = self.encoder(torch.cat([x, y], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        return torch.sigmoid(self.decoder(torch.cat([z, y], dim=1)))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# --------------------------------------------------------------------------- #
# Main Model Helpers
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

def fast_gpu_augment(x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    img = x.view(B, 28, 28)
    flip_h = torch.rand(B, 1, 1, device=x.device) > 0.5
    img = torch.where(flip_h, torch.flip(img, dims=[2]), img)
    dx = torch.randint(-1, 2, (1,)).item()
    dy = torch.randint(-1, 2, (1,)).item()
    img = torch.roll(img, shifts=(dy, dx), dims=(1, 2))
    return img.view(B, 784)

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
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_test  = X_test.reshape(X_test.shape[0],  -1).astype(np.float32) / 255.0

    X_tr = torch.tensor(X_train, dtype=DTYPE, device=DEVICE)
    y_tr = torch.tensor(y_train.astype(np.int64), device=DEVICE)
    X_te = torch.tensor(X_test,  dtype=DTYPE, device=DEVICE)
    y_te = torch.tensor(y_test.astype(np.int64), device=DEVICE)

    N_TRAIN = X_tr.shape[0]

    # ── Pre-train the Neural Baker ──────────────────────────────────────────
    print("Pre-training Neural Baker to generate dream images...")
    baker = NeuralBaker(latent_dim=32).to(DEVICE)
    baker_opt = torch.optim.AdamW(baker.parameters(), lr=0.001)
    
    for epoch in range(1, 21):
        baker.train()
        perm = torch.randperm(N_TRAIN, device=DEVICE)
        for i in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = X_tr[idx], y_tr[idx]
            yb_oh = F.one_hot(yb, NUM_CLASSES).float()
            
            recon, mu, logvar = baker(xb, yb_oh)
            bce = F.binary_cross_entropy(recon, xb, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = bce + kld
            
            baker_opt.zero_grad()
            loss.backward()
            baker_opt.step()
    print("Neural Baker is ready to bake dreams!\n")

    # ── Main Model + optimizer ──────────────────────────────────────────────
    model     = XYDualAxisNetWithPos(DROPOUT_SCALE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_START, weight_decay=1e-4)

    # ── Adaptive helpers ────────────────────────────────────────────────────
    def adjust_lr(test_acc: float) -> float:
        new_lr = LR_START * (1.0 - test_acc) ** LR_EXPONENT
        for g in optimizer.param_groups:
            g["lr"] = new_lr
        return new_lr

    def compute_dropout(train_acc: float) -> float:
        return min(DROPOUT_SCALE * (1.0 - train_acc) ** DROPOUT_EXPONENT, DROPOUT_MAX)

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
    print("TRAINING WITH POSITIONAL ID ENCODING + ROW SAMPLING + DIFFUSION SLEEP")
    print("=" * 80)

    best_test_acc     = 0.0
    current_dropout   = DROPOUT_SCALE
    epochs_no_improve = 0
    
    is_sleeping = False
    sleep_counter = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # 💤 TRIGGER SLEEPING BEAUTY v4 (Diffusion Dreams) 💤
        if not is_sleeping and epochs_no_improve >= SLEEP_PATIENCE:
            is_sleeping = True
            sleep_counter = SLEEP_DURATION
            for g in optimizer.param_groups:
                g["lr"] = 0.00005  
            print(f"\n  [💤 ENTERING REM SLEEP] Epoch {epoch} | Baker is scrambling real memories to dream...")

        if is_sleeping:
            model.train()
            baker.eval() 
            
            for _ in range(5):
                # 1. Grab a batch of REAL images and labels (memories)
                perm = torch.randperm(N_TRAIN, device=DEVICE)
                idx = perm[:BATCH_SIZE]
                x_real = X_tr[idx]
                y_dream = y_tr[idx]
                y_dream_oh = F.one_hot(y_dream, NUM_CLASSES).float()
                
                # 2. Pass real images through Baker's encoder
                with torch.no_grad():
                    mu, logvar = baker.encode(x_real, y_dream_oh)
                    z = baker.reparameterize(mu, logvar)
                    
                    # 3. Add noise to the latent space (Scramble the memory!)
                    z_dream = z + torch.randn_like(z) * DREAM_NOISE_LEVEL
                    
                    # 4. Decode back into an image (The Dream)
                    X_dream = baker.decode(z_dream, y_dream_oh)
                
                # 5. Feed dream to AI and force it to predict the real label
                optimizer.zero_grad(set_to_none=True)
                logits = model(X_dream, perturb_rate=0.03)
                loss = F.cross_entropy(logits, y_dream)
                loss.backward()
                optimizer.step()
            
            sleep_counter -= 1
            if sleep_counter <= 0:
                is_sleeping = False
                epochs_no_improve = 0
                print(f"  [☀️ WAKING UP] Epoch {epoch} | Resuming normal training with refreshed weights.\n")
            
            elapsed = time.time() - t0
            print(f"{epoch:5d} | [💤 Memory Dreaming] consolidation phase... | time: {elapsed:.2f}s")
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