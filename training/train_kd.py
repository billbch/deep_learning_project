import os

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

# ── Hyperparameters ──────────────────────────────────────────────────────────
TEMPERATURE = 4.0   # T: softens the teacher's probability distribution
ALPHA       = 0.7   # α: weight of KD loss  (1-α = weight of CE loss)
EPOCHS      = 100
LR          = 0.1
# ─────────────────────────────────────────────────────────────────────────────


def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    Combined Knowledge-Distillation loss.

    L = alpha * KL(soft teacher || soft student)   <- distillation loss
      + (1 - alpha) * CrossEntropy(student, labels) <- task loss

    The KL term is scaled by T^2 to keep gradients stable when T > 1.
    """
    # Soft targets  ──────────────────────────────────────────────────────────
    soft_student  = F.log_softmax(student_logits / T, dim=1)
    soft_teacher  = F.softmax(teacher_logits    / T, dim=1)

    distill_loss  = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

    # Hard targets (standard cross-entropy)  ─────────────────────────────────
    ce_loss = F.cross_entropy(student_logits, labels)

    return alpha * distill_loss + (1.0 - alpha) * ce_loss


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def train():
    train_loader, test_loader = get_loaders()

    # Load pre-trained teacher (frozen)
    teacher = get_teacher_model().to(device)
    teacher.load_state_dict(
        torch.load("outputs/checkpoints/resnet34_teacher.pth", map_location=device)
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student to train with KD
    student   = get_student_model().to(device)
    optimizer = optim.SGD(student.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    os.makedirs("outputs/checkpoints", exist_ok=True)
    best_acc = 0.0

    for epoch in range(EPOCHS):
        student.train()
        total = correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss = kd_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds    = student_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_acc = correct / total
        test_acc  = evaluate(student, test_loader)
        scheduler.step()

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc:  {test_acc:.4f} | "
            f"T={TEMPERATURE}, α={ALPHA}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), "outputs/checkpoints/resnet18_kd.pth")
            print(f"  ✓ Saved KD checkpoint  (best={best_acc:.4f})")

    print(f"\nTraining complete. Best KD student accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    train()