from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_loaders(batch_size=128, val_split=0.1):

    # Augmented transform — only for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),        # random crop with padding
        transforms.RandomHorizontalFlip(),            # flip 50% of the time
        transforms.ColorJitter(brightness=0.2,        # slight color variation
                               contrast=0.2,
                               saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # Clean transform — for val and test (no augmentation, ever)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    train_full = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_test_full = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=test_transform
    )
    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )

    # Same split, same seed
    val_size   = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size

    import torch
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(train_full,    [train_size, val_size], generator=generator)
    _, val_dataset   = random_split(val_test_full, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader