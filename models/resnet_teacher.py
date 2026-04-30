import torch.nn as nn
from torchvision.models import resnet34

def get_teacher_model(num_classes=100):
    model = resnet34(weights=None)
    
    # Changing the last output layer so that it would have 100 classes (because we use CIFAR-100)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model