"""
train.py
========
Training loop for the car classifier.

Usage:
    python src/train.py --data_dir dataset/Images --epochs 20 --batch_size 16
"""

import argparse
import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── local imports ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_dataloaders
from model   import build_model, count_parameters


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = total_n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        preds        = out.argmax(dim=1)
        total_loss  += loss.item() * imgs.size(0)
        total_correct += (preds == labels).sum().item()
        total_n      += imgs.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        preds        = out.argmax(dim=1)
        total_loss  += loss.item() * imgs.size(0)
        total_correct += (preds == labels).sum().item()
        total_n      += imgs.size(0)
    return total_loss / total_n, total_correct / total_n


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(history: dict, out_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train", marker="o")
    ax1.plot(epochs, history["val_loss"],   label="Val",   marker="s")
    ax1.set_title("Loss per Epoch"); ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train", marker="o")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val",   marker="s")
    ax2.set_title("Accuracy per Epoch"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[train] Saved training curves → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device : {device}")

    # Data
    train_loader, val_loader, class_to_idx, num_classes = get_dataloaders(
        data_dir   = args.data_dir,
        img_size   = args.img_size,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
    )

    # Save class map
    with open(os.path.join(args.out_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    # Model
    model = build_model(num_classes, freeze_backbone=False).to(device)
    print(f"[train] Trainable params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = os.path.join(args.out_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss={tr_loss:.4f} acc={tr_acc*100:.1f}% | "
              f"Val loss={va_loss:.4f} acc={va_acc*100:.1f}% | "
              f"{time.time()-t0:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "val_acc": va_acc}, best_path)
            print(f"  ✓ New best model saved (val_acc={va_acc*100:.1f}%)")

    print(f"\n[train] Best val accuracy: {best_val_acc*100:.1f}%")
    plot_history(history, args.out_dir)

    # Save final history
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Classification Training")
    parser.add_argument("--data_dir",    default="dataset/Images",
                        help="Root folder containing Train/ and Test/ sub-folders")
    parser.add_argument("--out_dir",     default="outputs",
                        help="Where to save checkpoints and plots")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--img_size",    type=int,   default=224)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=0)
    args = parser.parse_args()
    main(args)
