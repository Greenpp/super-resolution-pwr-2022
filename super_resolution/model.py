import pytorch_lightning as pl


class SREnhancingModel(pl.LightningModule):
    """Super-resolution model enhancing previously scaled images."""

    def __init__(self) -> None:
        super().__init__()


class SRScalingModel(pl.LightningModule):
    """Super-resolution model scaling images to a given resolution."""

    def __init__(self) -> None:
        super().__init__()


# Img
