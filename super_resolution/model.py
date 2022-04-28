from typing import Iterable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class SREnhancingModel(pl.LightningModule):
    """Super-resolution model enhancing previously scaled images."""

    def __init__(self) -> None:
        super().__init__()


class SRScalingModel(pl.LightningModule):
    """Super-resolution model scaling images to a given resolution."""

    def __init__(self, learning_rate: float) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.lr = learning_rate
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
