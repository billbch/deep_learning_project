import os 

import torch
import torch.nn as nn
import torch.optim as optim

from data.data_loader import get_loaders
from models.resnet_teacher import get_teacher_model

# Device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Andmed
train_loader, test_loader = get_loaders()

# Mudel
model = get_teacher_model().to(device)

# Loss + optimizer
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

# Training loop
epochs = 1  # small number for testing!

for epoch in range(epochs):
    model.train()

    total = 0
    correct = 0

    for i, (images, labels) in enumerate(train_loader):
        if i > 20:
            break
        print(f"Batch {i}")
    #for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # reset gradients
        optimizer.zero_grad()

        # forward
        outputs = model(images)

        # loss
        loss = criterion(outputs, labels)

        # backward
        loss.backward()

        # update
        optimizer.step()

        # accuracy
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total


    # SAVING THE BEST CHECKPOINT
    test_acc = evaluate(model, test_loader, device)
    scheduler.step()

    print(
        f"Epoch {epoch+1}: "
        f"Teacher Train Accuracy = {train_acc:.4f}, "
        f"Teacher Test Accuracy = {test_acc:.4f}"
    )

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "outputs/checkpoints/resnet34_teacher.pth")
        print("Saved teacher checkpoint")    
