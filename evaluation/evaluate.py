import torch

from data.data_loader import get_loaders
from models.resnet_student import get_student_model
from models.resnet_teacher import get_teacher_model

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# load data
_, test_loader = get_loaders()


# === TEST TEACHER ===
teacher = get_teacher_model().to(device)
teacher.load_state_dict(torch.load("outputs/checkpoints/resnet34_teacher.pth"))

teacher_acc = evaluate(teacher, test_loader)
print(f"Teacher Test Accuracy: {teacher_acc:.4f}")


# === TEST BASELINE ===
student = get_student_model().to(device)
student.load_state_dict(torch.load("outputs/checkpoints/resnet18_baseline.pth"))

student_acc = evaluate(student, test_loader)
print(f"Baseline Student Test Accuracy: {student_acc:.4f}")