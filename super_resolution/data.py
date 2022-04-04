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
    MIX = enum.auto()


class ImageModule(LightningDataModule):
    def __init__(self, train_class: DataClass, val_class: DataClass) -> None:
        super().__init__()
        self.train_class = train_class
        self.val_class = val_class

    def prepare_data(self):
        print("Loading data...")
        with open(DataConfig.processed_dogs, "rb") as f:
            dog_targets = torch.tensor(pkl.load(f))
        with open(DataConfig.processed_cats, "rb") as f:
            cat_targets = torch.tensor(pkl.load(f))

        print("Preparing dataset...")
        val_transform = T.Resize(
            (DataConfig.compressed_height, DataConfig.compressed_width)
        )
        dog_transformed = val_transform(dog_targets)
        cat_transformed = val_transform(cat_targets)

        dog_data = list(zip(dog_transformed, dog_targets))
        cat_data = list(zip(cat_transformed, cat_targets))

        dog_train = dog_data[: int(len(dog_data) * DataConfig.split_ration)]
        dog_val = dog_data[int(len(dog_data) * DataConfig.split_ration) :]
        cat_train = cat_data[: int(len(cat_data) * DataConfig.split_ration)]
        cat_val = cat_data[int(len(cat_data) * DataConfig.split_ration) :]

        if self.train_class is DataClass.DOG:
            self.train_data = ImageDataset(dog_train)
        elif self.train_class is DataClass.CAT:
            self.train_data = ImageDataset(cat_train)
        elif self.train_class is DataClass.MIX:
            self.train_data = ImageDataset(dog_train + cat_train)

        if self.val_class is DataClass.DOG:
            self.val_data = ImageDataset(dog_val)
        elif self.val_class is DataClass.CAT:
            self.val_data = ImageDataset(cat_val)
        elif self.val_class is DataClass.MIX:
            self.val_data = ImageDataset(dog_val + cat_val)

        print("Done.")

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
