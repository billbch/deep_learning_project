import os 

import torch
import torch.nn as nn
import torch.optim as optim

from data.data_loader import get_loaders
from models.resnet_student import get_student_model

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader = get_loaders()

model = get_student_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[60, 80],
    gamma=0.1
)

os.makedirs("outputs/checkpoints", exist_ok=True)
best_acc = 0.0

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()

    total = 0
    correct = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total

EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()

    total = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    test_acc = evaluate(model, test_loader, device)
    scheduler.step()

    print(
        f"Epoch {epoch+1}/{EPOCHS}: "
        f"Baseline Student Train Accuracy = {train_acc:.4f}, "
        f"Baseline Student Test Accuracy = {test_acc:.4f}"
    )

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "outputs/checkpoints/resnet18_baseline.pth")
        print("Saved baseline checkpoint")    
