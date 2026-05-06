import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from data.data_loader import get_loaders
from models.resnet_teacher import get_teacher_model
from utils.seed import set_seed

set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS        = 120   # early stopping will decide when to stop
LR            = 0.1
PATIENCE      = 20    # stop if val_acc doesn't improve for 20 epochs
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def plot_curves(history: dict, out_path: str):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(epochs, history["train_acc"], label="Train",      color="#4C72B0")
    axes[0].plot(epochs, history["val_acc"],   label="Validation", color="#DD8452")
    axes[0].plot(epochs, history["test_acc"],  label="Test",       color="#55A868", linestyle="--")
    axes[0].set_title("Teacher — Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(epochs, history["train_loss"], label="Train",      color="#4C72B0")
    axes[1].plot(epochs, history["val_loss"],   label="Validation", color="#DD8452")
    axes[1].set_title("Teacher — Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curves saved → {out_path}")


def train():
    train_loader, val_loader, test_loader = get_loaders()

    model     = get_teacher_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-4
    )

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/figures",     exist_ok=True)
    os.makedirs("outputs/logs",        exist_ok=True)

    history = {"train_acc": [], "val_acc": [], "test_acc": [],
               "train_loss": [], "val_loss": []}

    best_val_acc    = 0.0
    epochs_no_improve = 0

    # CSV log
    log_path = "outputs/logs/teacher_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_acc", "val_acc", "test_acc",
                         "train_loss", "val_loss"])

    for epoch in range(EPOCHS):
        model.train()
        total = correct = running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct      += (outputs.argmax(1) == labels).sum().item()
            total        += labels.size(0)
            running_loss += loss.item() * labels.size(0)

        train_acc  = correct / total
        train_loss = running_loss / total

        # Validation loss
        model.eval()
        val_loss_total = val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss_total += criterion(outputs, labels).item() * labels.size(0)
                val_correct    += (outputs.argmax(1) == labels).sum().item()
                val_total      += labels.size(0)

        val_acc  = val_correct / val_total
        val_loss = val_loss_total / val_total
        test_acc = evaluate(model, test_loader)

        scheduler.step()

        # Log
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch+1, f"{train_acc:.4f}", f"{val_acc:.4f}",
                 f"{test_acc:.4f}", f"{train_loss:.4f}", f"{val_loss:.4f}"]
            )

        print(
            f"Epoch {epoch+1:4d} | "
            f"Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f} | "
            f"Loss train {train_loss:.4f} val {val_loss:.4f}"
        )

        # ── Early stopping ────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "outputs/checkpoints/resnet34_teacher.pth")
            print(f"  ✓ Checkpoint saved (best val={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n⏹  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    plot_curves(history, "outputs/figures/teacher_curves.png")
    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()