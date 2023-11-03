from torch.utils.data import DataLoader
from typing import Tuple
import torch
import pandas as pd

TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
BATCH_SIZE = 64

class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self,
                 path: str,
                 train: bool=True) -> None:
        
        df = pd.read_csv(path)
        self.train = train
        if train:
            self.X = df.iloc[:, 1:].values
            self.y = df.iloc[:, 0].values
        else:
            self.X = df.values

    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int):
        if self.train:
            return torch.tensor(self.X[idx, :], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)
        else:
            return torch.tensor(self.X[idx, :], dtype=torch.float32)


def load_data() -> Tuple[DataLoader]:

    train_dataset = MNISTDataset(
        path=TRAIN_PATH,
        train=True
    )

    test_dataset = MNISTDataset(
        path=TEST_PATH,
        train=False
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