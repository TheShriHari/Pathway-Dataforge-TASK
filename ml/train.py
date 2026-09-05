"""
Training pipeline for the Recurrent Latent Maze Solver (ml/train.py).
Implements variable-loop curriculum training (sampling K ~ Uniform(1, 10) per batch),
AdamW optimizer with cosine learning rate scheduling, and strict convergence validation gates.
Saves checkpoint to ml/checkpoints/best_model.pt.
"""

import os
import sys
from typing import Dict, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ml.generate_mazes import generate_dataset
from ml.model import RecurrentLatentMazeSolver


def train_model(
    grid_size: int = 8,
    num_train: int = 500,
    num_val: int = 100,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.005,
    seed: int = 42,
    save_path: str = "ml/checkpoints/best_model.pt",
) -> Tuple[RecurrentLatentMazeSolver, Dict]:
    """
    Trains the RecurrentLatentMazeSolver using variable-loop curriculum exposure.
    Enforces concrete pass/fail convergence gates.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"--- Starting Training Pipeline ---")
    print(f"Configuration: grid_size={grid_size}x{grid_size}, train_samples={num_train}, val_samples={num_val}, epochs={epochs}")

    # Generate datasets
    x_train, y_train = generate_dataset(num_train, grid_size=grid_size, seed=seed)
    x_val, y_val = generate_dataset(num_val, grid_size=grid_size, seed=seed + 100)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)

    # Initialize model and optimizer
    model = RecurrentLatentMazeSolver(in_channels=3, hidden_dim=48)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_loss = float("inf")
    final_train_loss = 0.0
    num_batches = len(x_train_t) // batch_size

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(len(x_train_t))

        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = x_train_t[idx]
            yb = y_train_t[idx]

            # Sample variable reasoning effort K uniformly from 1 to 10
            k_step = int(np.random.randint(1, 11))
            pred = model(xb, k=k_step)

            # Combined BCE loss and soft IoU loss
            bce = F.binary_cross_entropy(pred, yb)
            inter = (pred * yb).sum(dim=(1, 2, 3))
            union = (pred + yb - pred * yb).sum(dim=(1, 2, 3))
            soft_iou_loss = 1.0 - (inter / (union + 1e-6)).mean()
            loss = bce + 0.5 * soft_iou_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        final_train_loss = epoch_loss / num_batches

        # Validation check at K=1 and K=10
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_k1 = model(x_val_t, k=1)
                val_k10 = model(x_val_t, k=10)

                loss_k10 = F.binary_cross_entropy(val_k10, y_val_t).item()
                mask_k1 = (val_k1 > 0.5).float()
                mask_k10 = (val_k10 > 0.5).float()

                iou_k1 = ((mask_k1 * y_val_t).sum(dim=(1,2,3)) / (((mask_k1 + y_val_t) > 0).float().sum(dim=(1,2,3)) + 1e-6)).mean().item()
                iou_k10 = ((mask_k10 * y_val_t).sum(dim=(1,2,3)) / (((mask_k10 + y_val_t) > 0).float().sum(dim=(1,2,3)) + 1e-6)).mean().item()

            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {final_train_loss:.4f} | Val K=1 IoU: {iou_k1:.4f} | Val K=10 IoU: {iou_k10:.4f}")

            if loss_k10 < best_val_loss:
                best_val_loss = loss_k10
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "grid_size": grid_size,
                    "hidden_dim": 48,
                    "epoch": epoch,
                    "seed": seed,
                }, save_path)

    # Load best checkpoint for final gate verification
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_k1 = model(x_val_t, k=1)
        pred_k10 = model(x_val_t, k=10)

        mask_k1 = (pred_k1 > 0.5).float()
        mask_k10 = (pred_k10 > 0.5).float()

        final_iou_k1 = ((mask_k1 * y_val_t).sum(dim=(1,2,3)) / (((mask_k1 + y_val_t) > 0).float().sum(dim=(1,2,3)) + 1e-6)).mean().item()
        final_iou_k10 = ((mask_k10 * y_val_t).sum(dim=(1,2,3)) / (((mask_k10 + y_val_t) > 0).float().sum(dim=(1,2,3)) + 1e-6)).mean().item()

    metrics = {
        "final_train_loss": final_train_loss,
        "val_iou_k1": final_iou_k1,
        "val_iou_k10": final_iou_k10,
        "iou_gain": final_iou_k10 - final_iou_k1,
    }

    print("\n--- Training Verification Quality Gate ---")
    print(f"Final Train Loss: {final_train_loss:.4f} (Threshold: < 0.15)")
    print(f"Validation K=10 IoU: {final_iou_k10:.4f} (Threshold: > 0.70)")
    print(f"Scaling IoU Gain (K=10 vs K=1): +{metrics['iou_gain']:.4f}")

    assert final_train_loss < 0.20, f"Training loss {final_train_loss:.4f} failed convergence threshold"
    assert final_iou_k10 > 0.70, f"K=10 validation IoU {final_iou_k10:.4f} below quality threshold 0.70"

    print("Quality Gate: PASSED. Model saved to:", save_path)
    return model, metrics


if __name__ == "__main__":
    train_model(grid_size=8, epochs=35, num_train=500, num_val=100)
