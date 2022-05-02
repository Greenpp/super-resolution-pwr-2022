from __future__ import annotations

import pickle as pkl

import click
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from super_resolution.config import RANDOM_SEED, TrainingConfig
from super_resolution.data import DataClass, ImageModule
from super_resolution.model import SRScalingModel


@click.command()
@click.option(
    "-t",
    "--train-class",
    type=click.Choice(["dog", "cat", "mix"]),
    help="Train on dogs, cats or both?",
    default="mix",
)
@click.option(
    "-n", "--name", type=str, help="Name of the experiment on wandb.", default=None
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=TrainingConfig.batch_size,
    help="Batch size.",
)
@click.option(
    "-lr",
    "--learning-rate",
    type=float,
    default=TrainingConfig.lr,
    help="Learning rate.",
)
@click.option("-r", "--random-seed", type=int, default=RANDOM_SEED, help="Random seed.")
@click.option("--test", is_flag=True, help="Test run with 1 batch only.")
def main(
    train_class: str,
    name: str | None,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
    test: bool,
) -> None:
    pl.seed_everything(random_seed)

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
        name=name,
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=TrainingConfig.max_epochs,
        logger=logger,
        fast_dev_run=test,
    )
    data_module = ImageModule(tc, batch_size)

    with open("data/cat_val.pkl", "rb") as f:
        names = pkl.load(f)
    model = SRScalingModel(learning_rate, names[0], names[0])

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
