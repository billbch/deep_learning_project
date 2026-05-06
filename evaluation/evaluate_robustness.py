"""
evaluation/evaluate_robustness.py
──────────────────────────────────
Evaluates all three models (Teacher, Baseline Student, KD Student) on:
  - Clean CIFAR-100 test set
  - Three corruptions: gaussian_noise, blur, brightness
  - Three severity levels: 1 (mild), 3 (medium), 5 (severe)

Prints a summary table and saves results to outputs/results.json
"""

import json
import os

import torch

from data.data_loader import get_loaders
from data.corruptions import get_corrupted_loader, ALL_CORRUPTIONS
from models.resnet_student import get_student_model
from models.resnet_teacher import get_teacher_model

device = "cuda" if torch.cuda.is_available() else "cpu"

SEVERITIES = [1, 3, 5]
CHECKPOINTS = {
    "Teacher (ResNet-34)":   "outputs/checkpoints/resnet34_teacher.pth",
    "Baseline (ResNet-18)":  "outputs/checkpoints/resnet18_baseline.pth",
    "KD Student (ResNet-18)":"outputs/checkpoints/resnet18_kd.pth",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(name: str, path: str):
    if "Teacher" in name:
        model = get_teacher_model()
    else:
        model = get_student_model()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def accuracy(model, loader) -> float:
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return round(correct / total * 100, 2)   # return as %


def robustness_gap(clean_acc: float, corrupted_accs: list) -> float:
    """Mean drop in accuracy across all corrupted conditions."""
    mean_corr = sum(corrupted_accs) / len(corrupted_accs)
    return round(clean_acc - mean_corr, 2)


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate_all():
    _,_, clean_loader = get_loaders()

    results = {}

    print("\n" + "=" * 70)
    print(f"{'Model':<28} {'Condition':<30} {'Accuracy':>10}")
    print("=" * 70)

    for model_name, ckpt_path in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {model_name}: checkpoint not found at {ckpt_path}")
            continue

        model = load_model(model_name, ckpt_path)
        results[model_name] = {}

        # ── Clean ──
        clean_acc = accuracy(model, clean_loader)
        results[model_name]["clean"] = clean_acc
        print(f"{model_name:<28} {'clean':<30} {clean_acc:>9.2f}%")

        # ── Corrupted ──
        all_corrupted_accs = []

        for corruption in ALL_CORRUPTIONS:
            results[model_name][corruption] = {}

            for sev in SEVERITIES:
                loader      = get_corrupted_loader(corruption, severity=sev)
                corr_acc    = accuracy(model, loader)
                label       = f"{corruption} (sev={sev})"

                results[model_name][corruption][sev] = corr_acc
                all_corrupted_accs.append(corr_acc)

                print(f"{'':28} {label:<30} {corr_acc:>9.2f}%")

        # ── Robustness gap ──
        gap = robustness_gap(clean_acc, all_corrupted_accs)
        results[model_name]["robustness_gap"] = gap
        print(f"{'':28} {'→ Robustness Gap':<30} {gap:>9.2f}%")
        print("-" * 70)

    print("=" * 70)

    # ── Save JSON ──
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}\n")

    return results


if __name__ == "__main__":
    evaluate_all()