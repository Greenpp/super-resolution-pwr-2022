import click
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from super_resolution.config import RANDOM_SEED, TrainingConfig
from super_resolution.data import DataClass, ImageModule
from super_resolution.model import SRScalingModel


@click.command()
@click.option("-t", "--train-class", type=click.Choice(["dog", "cat", "mix"]))
@click.option("-b", "--batch-size", type=int, default=TrainingConfig.batch_size)
@click.option("-lr", "--learning-rate", type=float, default=TrainingConfig.lr)
@click.option("--test", is_flag=True)
def main(train_class: str, batch_size: int, learning_rate: float, test: bool) -> None:
    pl.seed_everything(RANDOM_SEED)

    if train_class == "dog":
        tc = DataClass.DOG
    elif train_class == "cat":
        tc = DataClass.CAT
    elif train_class == "mix":
        tc = DataClass.MIX
    else:
        raise ValueError("Invalid train class")

    logger = WandbLogger(
        project="super-resolution-pwr-2022",
        name="new_model",
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=TrainingConfig.max_epochs,
        logger=logger,
        fast_dev_run=test,
    )
    data_module = ImageModule(tc, batch_size)
    model = SRScalingModel(learning_rate)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
