"""
Domain-shift corruptions for CIFAR-100 test images.

Supported corruptions:
- gaussian_noise: adds Gaussian noise in tensor space.
- blur: applies Gaussian blur on the PIL image.
- brightness: changes image brightness.
- contrast: changes image contrast.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF


class GaussianNoise:
    """Add Gaussian noise after ToTensor, when pixels are in [0, 1]."""

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)

    def __repr__(self):
        return f"GaussianNoise(std={self.std})"


class GaussianBlur:
    """Apply Gaussian blur before ToTensor."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.5):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        return TF.gaussian_blur(img, kernel_size=self.kernel_size, sigma=self.sigma)

    def __repr__(self):
        return f"GaussianBlur(kernel_size={self.kernel_size}, sigma={self.sigma})"


class BrightnessShift:
    """Adjust brightness before ToTensor."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def __call__(self, img):
        return TF.adjust_brightness(img, self.factor)

    def __repr__(self):
        return f"BrightnessShift(factor={self.factor})"


class ContrastShift:
    """Adjust contrast before ToTensor."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def __call__(self, img):
        return TF.adjust_contrast(img, self.factor)

    def __repr__(self):
        return f"ContrastShift(factor={self.factor})"


NOISE_STD = {1: 0.04, 2: 0.07, 3: 0.10, 4: 0.14, 5: 0.20}
BLUR_SIGMA = {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0}
BRIGHTNESS_FACTOR = {1: 1.2, 2: 1.4, 3: 1.6, 4: 1.8, 5: 2.0}
CONTRAST_FACTOR = {1: 1.2, 2: 1.4, 3: 1.6, 4: 1.8, 5: 2.0}

ALL_CORRUPTIONS = ["gaussian_noise", "blur", "brightness", "contrast"]
NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def get_corruption_transform(corruption: str, severity: int = 3):
    if severity not in range(1, 6):
        raise ValueError(f"severity must be between 1 and 5, got {severity}")

    if corruption == "gaussian_noise":
        return transforms.Compose([
            transforms.ToTensor(),
            GaussianNoise(std=NOISE_STD[severity]),
            NORMALIZE,
        ])

    if corruption == "blur":
        kernel_size = max(3, int(BLUR_SIGMA[severity] * 3) | 1)
        return transforms.Compose([
            GaussianBlur(kernel_size=kernel_size, sigma=BLUR_SIGMA[severity]),
            transforms.ToTensor(),
            NORMALIZE,
        ])

    if corruption == "brightness":
        return transforms.Compose([
            BrightnessShift(factor=BRIGHTNESS_FACTOR[severity]),
            transforms.ToTensor(),
            NORMALIZE,
        ])

    if corruption == "contrast":
        return transforms.Compose([
            ContrastShift(factor=CONTRAST_FACTOR[severity]),
            transforms.ToTensor(),
            NORMALIZE,
        ])

    choices = ", ".join(ALL_CORRUPTIONS)
    raise ValueError(f"Unknown corruption '{corruption}'. Choose from: {choices}.")


def get_corrupted_loader(
    corruption: str = "gaussian_noise",
    severity: int = 3,
    batch_size: int = 128,
) -> DataLoader:
    """Return a CIFAR-100 test DataLoader with one corruption applied."""
    dataset = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=get_corruption_transform(corruption, severity),
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_all_corrupted_loaders(severity: int = 3, batch_size: int = 128) -> dict:
    """Return {corruption_name: DataLoader} for all supported corruptions."""
    return {
        corruption: get_corrupted_loader(corruption, severity, batch_size)
        for corruption in ALL_CORRUPTIONS
    }
