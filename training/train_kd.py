import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.data_loader import get_loaders
from models.resnet_student import get_student_model
from models.resnet_teacher import get_teacher_model
from utils.seed import set_seed

set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Hyperparameters ───────────────────────────────────────────────────────────
TEMPERATURE = 4.0
ALPHA       = 0.7
EPOCHS      = 120
LR          = 0.1
PATIENCE    = 20
# ─────────────────────────────────────────────────────────────────────────────


def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits    / T, dim=1)
    distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    ce_loss      = F.cross_entropy(student_logits, labels)
    return alpha * distill_loss + (1.0 - alpha) * ce_loss


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

    axes[0].plot(epochs, history["train_acc"], label="Train",      color="#4C72B0")
    axes[0].plot(epochs, history["val_acc"],   label="Validation", color="#DD8452")
    axes[0].plot(epochs, history["test_acc"],  label="Test",       color="#55A868", linestyle="--")
    axes[0].set_title("KD Student — Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_loss"], label="Train (KD loss)", color="#4C72B0")
    axes[1].plot(epochs, history["val_loss"],   label="Validation (CE)", color="#DD8452")
    axes[1].set_title("KD Student — Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curves saved → {out_path}")


def train():
    train_loader, val_loader, test_loader = get_loaders()

    # Frozen teacher
    teacher = get_teacher_model().to(device)
    teacher.load_state_dict(
        torch.load("outputs/checkpoints/resnet34_teacher.pth", map_location=device)
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student   = get_student_model().to(device)
    optimizer = optim.SGD(student.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
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

    best_val_acc      = 0.0
    epochs_no_improve = 0
    ce_criterion      = nn.CrossEntropyLoss()

    log_path = "outputs/logs/kd_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_acc", "val_acc", "test_acc",
                                 "train_loss", "val_loss"])

    for epoch in range(EPOCHS):
        student.train()
        total = correct = running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = kd_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct      += (student_logits.argmax(1) == labels).sum().item()
            total        += labels.size(0)
            running_loss += loss.item() * labels.size(0)

        train_acc  = correct / total
        train_loss = running_loss / total

        student.eval()
        val_loss_total = val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                val_loss_total += ce_criterion(outputs, labels).item() * labels.size(0)
                val_correct    += (outputs.argmax(1) == labels).sum().item()
                val_total      += labels.size(0)

        val_acc  = val_correct / val_total
        val_loss = val_loss_total / val_total
        test_acc = evaluate(student, test_loader)
        scheduler.step()

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch+1, f"{train_acc:.4f}", f"{val_acc:.4f}",
                 f"{test_acc:.4f}", f"{train_loss:.4f}", f"{val_loss:.4f}"])

        print(
            f"Epoch {epoch+1:4d} | "
            f"Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f} | "
            f"Loss train {train_loss:.4f} val {val_loss:.4f} | T={TEMPERATURE} α={ALPHA}"
        )

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(student.state_dict(), "outputs/checkpoints/resnet18_kd.pth")
            print(f"  ✓ Checkpoint saved (best val={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n⏹  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    plot_curves(history, "outputs/figures/kd_curves.png")
    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()