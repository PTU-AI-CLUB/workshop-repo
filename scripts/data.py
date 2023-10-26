from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from typing import Tuple

BATCH_SIZE = 64

def load_data() -> Tuple[DataLoader]:

    train_dataset = MNIST(
        root="../data",
        train=True,
        transform=ToTensor(),
        target_transform=None,
        download=True
    )

    test_dataset = MNIST(
        root="../data",
        train=False,
        transform=ToTensor(),
        download=True
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE
    )

    return train_loader, test_loader