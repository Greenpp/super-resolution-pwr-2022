import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainingConfig


class SREnhancingModel(pl.LightningModule):
    """Super-resolution model enhancing previously scaled images."""

    def __init__(self) -> None:
        super().__init__()


class SRScalingModel(pl.LightningModule):
    """Super-resolution model scaling images to a given resolution."""

    def __init__(self) -> None:
        super().__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _unpack_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = batch
        return X.float(), y.float()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict:
        X, y = self._unpack_batch(batch)
        y_hat = self(X)

        loss = F.mse_loss(y_hat, y)
        return {"loss": loss}

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        X, y = self._unpack_batch(batch)
        y_hat = self(X)

        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, on_step=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=TrainingConfig.lr)
