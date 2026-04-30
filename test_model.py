import torch

from models.resnet_teacher import get_teacher_model
from models.resnet_student import get_student_model

# Dummy input (4 pictures)
x = torch.randn(4, 3, 32, 32)

teacher = get_teacher_model()
student = get_student_model()

teacher_out = teacher(x)
student_out = student(x)

print("Teacher output shape:", teacher_out.shape)
print("Student output shape:", student_out.shape)