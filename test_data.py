from data.data_loader import get_loaders

train_loader, test_loader = get_loaders()

for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    break