import json
import os

import torch

from data.corruptions import ALL_CORRUPTIONS, get_corrupted_loader
from data.data_loader import get_loaders
from models.resnet_student import get_student_model
from models.resnet_teacher import get_teacher_model


device = "cuda" if torch.cuda.is_available() else "cpu"

SEVERITIES = [1, 3, 5]
CHECKPOINTS = {
    "Teacher (ResNet-34)": "outputs/checkpoints/resnet34_teacher.pth",
    "Baseline (ResNet-18)": "outputs/checkpoints/resnet18_baseline.pth",
    "KD Student (ResNet-18)": "outputs/checkpoints/resnet18_kd.pth",
}


def build_model(model_name: str):
    if "Teacher" in model_name:
        return get_teacher_model()
    return get_student_model()


def load_model(model_name: str, checkpoint_path: str):
    model = build_model(model_name).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.eval()


@torch.no_grad()
def accuracy(model, loader) -> float:
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return round((correct / total) * 100, 2)


def robustness_gap(clean_acc: float, corrupted_accs: list[float]) -> float:
    if not corrupted_accs:
        return 0.0
    mean_corrupted_acc = sum(corrupted_accs) / len(corrupted_accs)
    return round(clean_acc - mean_corrupted_acc, 2)


def evaluate_all(batch_size: int = 128):
    _, clean_loader = get_loaders(batch_size=batch_size)
    results = {}

    print("\n" + "=" * 78)
    print(f"{'Model':<28} {'Condition':<34} {'Accuracy':>10}")
    print("=" * 78)

    for model_name, checkpoint_path in CHECKPOINTS.items():
        if not os.path.exists(checkpoint_path):
            print(f"[SKIP] {model_name}: checkpoint not found at {checkpoint_path}")
            continue

        model = load_model(model_name, checkpoint_path)
        results[model_name] = {}

        clean_acc = accuracy(model, clean_loader)
        results[model_name]["clean"] = clean_acc
        print(f"{model_name:<28} {'clean':<34} {clean_acc:>9.2f}%")

        corrupted_accs = []
        for corruption in ALL_CORRUPTIONS:
            results[model_name][corruption] = {}

            for severity in SEVERITIES:
                loader = get_corrupted_loader(
                    corruption=corruption,
                    severity=severity,
                    batch_size=batch_size,
                )
                corr_acc = accuracy(model, loader)
                corrupted_accs.append(corr_acc)
                results[model_name][corruption][str(severity)] = corr_acc

                label = f"{corruption} (severity={severity})"
                print(f"{'':28} {label:<34} {corr_acc:>9.2f}%")

        gap = robustness_gap(clean_acc, corrupted_accs)
        results[model_name]["robustness_gap"] = gap
        print(f"{'':28} {'robustness gap':<34} {gap:>9.2f}%")
        print("-" * 78)

    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}\n")
    return results


if __name__ == "__main__":
    evaluate_all()
