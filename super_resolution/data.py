import enum
import pickle as pkl

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import DataConfig, TrainingConfig


class ImageDataset(Dataset):
    def __init__(self, data: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.data = data

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class DataClass(enum.Enum):
    DOG = enum.auto()
    CAT = enum.auto()


class SingleClassModule(LightningDataModule):
    def __init__(self, class_name: DataClass):
        super().__init__()
        self.class_name = class_name

    def prepare_data(self):
        print("Loading data...")
        if self.class_name is DataClass.DOG:
            with open(DataConfig.processed_dogs, "rb") as f:
                targets = torch.tensor(pkl.load(f))
        elif self.class_name is DataClass.CAT:
            with open(DataConfig.processed_cats, "rb") as f:
                targets = torch.tensor(pkl.load(f))
        else:
            raise ValueError(f"Unknown class name: {self.class_name}")

        print("Creating dataset...")
        val_transform = T.Resize(
            (DataConfig.compressed_height, DataConfig.compressed_width)
        )
        values = val_transform(targets)

        data = list(zip(values, targets))
        self.train_data = ImageDataset(data[: int(len(data) * DataConfig.split_ration)])
        self.val_data = ImageDataset(data[int(len(data) * DataConfig.split_ration) :])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=TrainingConfig.batch_size,
            num_workers=TrainingConfig.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=TrainingConfig.batch_size,
            num_workers=TrainingConfig.num_workers,
        )
