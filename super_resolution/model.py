import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from ISR.models import RDN

import wandb

from .config import DataConfig

logging.getLogger("matplotlib").setLevel(level=logging.ERROR)


class SREnhancingModel(pl.LightningModule):
    """Super-resolution model enhancing previously scaled images."""

    def __init__(self) -> None:
        super().__init__()


class SRScalingModel(pl.LightningModule):
    """Super-resolution model scaling images to a given resolution."""

    def __init__(
        self, learning_rate: float, example_cat: str, example_dog: str
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.lr = learning_rate

        example_original_cat_file = Path(DataConfig.processed_cats) / example_cat
        example_scaled_cat_file = Path(DataConfig.scaled_cats) / example_cat
        self.example_original_cat = plt.imread(str(example_original_cat_file))
        self.example_scaled_cat = plt.imread(str(example_scaled_cat_file))

        example_original_dog_file = Path(DataConfig.processed_dogs) / example_dog
        example_scaled_dog_file = Path(DataConfig.scaled_dogs) / example_dog
        self.example_original_dog = plt.imread(str(example_original_dog_file))
        self.example_scaled_dog = plt.imread(str(example_scaled_dog_file))

        self.model = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
        )

        metrics = torchmetrics.MetricCollection(
            [
                # torchmetrics.StructuralSimilarityIndexMeasure(),
                torchmetrics.PeakSignalNoiseRatio(),
            ]
        )
        self.training_metrics = metrics.clone("train_")
        self.val_dog_metrics = metrics.clone("val_dog_")
        self.val_cat_metrics = metrics.clone("val_cat_")

        self.rdn = RDN(weights='psnr-small')
        self.rdn_cat_prediction = self.rdn.predict(self.example_scaled_cat)
        self.rdn_dog_prediction = self.rdn.predict(self.example_scaled_dog)

        self._log_restoration()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _unpack_batch(self, batch: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        return [t.float() for t in batch]

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        X, y = self._unpack_batch(batch)
        y_hat = self(X)

        loss = F.mse_loss(y_hat, y)
        metrics = self.training_metrics(y_hat, y)

        self.log_dict(metrics, on_step=True, prog_bar=False, on_epoch=False)
        self.log("train_loss", loss, on_step=True, prog_bar=True, on_epoch=False)

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        X_dog, y_dgo, X_cat, y_cat = self._unpack_batch(batch)
        y_hat_dog = self(X_dog)
        y_hat_cat = self(X_cat)

        loss_dog = F.mse_loss(y_hat_dog, y_dgo)
        loss_cat = F.mse_loss(y_hat_cat, y_cat)

        metrics_dog = self.val_dog_metrics(y_hat_dog, y_dgo)
        metrics_cat = self.val_cat_metrics(y_hat_cat, y_cat)

        self.log_dict(metrics_dog, on_step=True, prog_bar=True, on_epoch=True)
        self.log_dict(metrics_cat, on_step=True, prog_bar=True, on_epoch=True)

        self.log("val_dog_loss", loss_dog, on_step=True, prog_bar=True, on_epoch=True)
        self.log("val_cat_loss", loss_cat, on_step=True, prog_bar=True, on_epoch=True)

    def on_validation_end(self) -> None:
        self._log_restoration()

    def _log_restoration(self) -> None:
        scaled_img_tensor = (
            torch.stack(
                [
                    torch.tensor(self.example_scaled_cat).permute(2, 0, 1),
                    torch.tensor(self.example_scaled_dog).permute(2, 0, 1),
                ]
            )
            .float()
            .to(self.device)
            / 255.0
        )
        with torch.no_grad():
            restored_img = self(scaled_img_tensor).detach().cpu().numpy()

        restored_cat = restored_img[0].transpose(1, 2, 0)
        restored_dog = restored_img[1].transpose(1, 2, 0)

        plt.figure(figsize=(12, 12))
        plt.subplot(2, 4, 1)
        plt.imshow(self.example_original_cat)
        plt.title("Original cat")
        plt.subplot(2, 4, 2)
        plt.imshow(self.example_scaled_cat)
        plt.title("Scaled cat")
        plt.subplot(2, 4, 3)
        plt.imshow(restored_cat)
        plt.title("Restored cat")
        plt.subplot(2, 4, 4)
        plt.imshow(self.rdn_cat_prediction)
        plt.title("RDN Restored cat")
        plt.subplot(2, 4, 5)
        plt.imshow(self.example_original_dog)
        plt.title("Original dog")
        plt.subplot(2, 4, 6)
        plt.imshow(self.example_scaled_dog)
        plt.title("Scaled dog")
        plt.subplot(2, 4, 7)
        plt.imshow(restored_dog)
        plt.title("Restored dog")
        plt.subplot(2, 4, 8)
        plt.imshow(self.rdn_dog_prediction)
        plt.title("RDN Restored dog")

        wandb.log({"Restoration": plt})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
