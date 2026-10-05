# Download, transform, and split Fashion-MNIST data reproducibly.
from __future__ import annotations
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

CLASSES = ["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

def make_datasets(root: str = "data", fake: bool = False, fake_size: int = 1000, augment: bool = True):
    train_ops = [transforms.Grayscale(num_output_channels=3), transforms.Resize((224, 224))]
    if augment:
        train_ops += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(224, padding=8)]
    train_tf = transforms.Compose(train_ops + [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if fake:
        train = datasets.FakeData(size=fake_size, image_size=(3, 224, 224), num_classes=10, transform=train_tf, random_offset=0)
        val = datasets.FakeData(size=max(100, fake_size // 5), image_size=(3, 224, 224), num_classes=10, transform=eval_tf, random_offset=10000)
        test = datasets.FakeData(size=max(100, fake_size // 5), image_size=(3, 224, 224), num_classes=10, transform=eval_tf, random_offset=20000)
        return train, val, test
    raw = datasets.FashionMNIST(root=root, train=True, download=True)
    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(len(raw), generator=generator).tolist()
    train_end = int(0.8 * len(raw))
    val_end = int(0.9 * len(raw))
    train_set = datasets.FashionMNIST(root=root, train=True, download=False, transform=train_tf)
    eval_set = datasets.FashionMNIST(root=root, train=True, download=False, transform=eval_tf)
    test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=eval_tf)
    return (
        Subset(train_set, permutation[:train_end]),
        Subset(eval_set, permutation[train_end:val_end]),
        test_set,
    )


