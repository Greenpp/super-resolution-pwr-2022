import enum
import pickle as pkl
import random
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, TrainingConfig


class DataClass(enum.Enum):
    DOG = enum.auto()
    CAT = enum.auto()
    MIX = enum.auto()


class ImageDataset(Dataset):
    def __init__(self, data: list[list[Path]]) -> None:
        self.data = data
        self.transform = T.ToTensor()

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        paths = self.data[index]
        imgs = [Image.open(path) for path in paths]
        tensors = [self.transform(img) for img in imgs]

        return tensors

    def __len__(self) -> int:
        return len(self.data)


class ImageModule(LightningDataModule):
    def __init__(self, train_class: DataClass, batch_size: int) -> None:
        super().__init__()
        self.train_class = train_class
        self.batch_size = batch_size

    def _load_paths(
        self, names_file: str, base_dir: str, scaled_dir: str
    ) -> list[list[Path]]:
        base_path = Path(base_dir)
        scaled_path = Path(scaled_dir)

        with open(names_file, "rb") as f:
            names = pkl.load(f)

        paths = [[scaled_path / name, base_path / name] for name in names]

        return paths

    def prepare_data(self):
        train_paths: list[list[Path]] = []
        if self.train_class is DataClass.DOG:
            paths = self._load_paths(
                DataConfig.train_dogs,
                DataConfig.processed_dogs,
                DataConfig.scaled_dogs,
            )
            train_paths.extend(paths)
        elif self.train_class is DataClass.CAT:
            paths = self._load_paths(
                DataConfig.train_cats,
                DataConfig.processed_cats,
                DataConfig.scaled_cats,
            )
            train_paths.extend(paths)
        elif self.train_class is DataClass.MIX:
            cat_paths = self._load_paths(
                DataConfig.train_cats,
                DataConfig.processed_cats,
                DataConfig.scaled_cats,
            )
            dog_paths = self._load_paths(
                DataConfig.train_dogs,
                DataConfig.processed_dogs,
                DataConfig.scaled_dogs,
            )
            random.shuffle(cat_paths)
            random.shuffle(dog_paths)

            mixed_paths = (
                cat_paths[: len(cat_paths) // 2] + dog_paths[: len(dog_paths) // 2]
            )
            train_paths.extend(mixed_paths)

        dog_val_paths = self._load_paths(
            DataConfig.val_dogs, DataConfig.processed_dogs, DataConfig.scaled_dogs
        )
        cat_val_paths = self._load_paths(
            DataConfig.val_cats, DataConfig.processed_cats, DataConfig.scaled_cats
        )
        val_paths: list[list[Path]] = []
        for d, c in zip(dog_val_paths, cat_val_paths):
            val_paths.append([*d, *c])

        self.train_data = ImageDataset(train_paths)
        self.val_data = ImageDataset(val_paths)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=TrainingConfig.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=TrainingConfig.num_workers,
        )
