import pytorch_lightning as pl

from super_resolution.config import RANDOM_SEED, TrainingConfig
from super_resolution.data import DataClass, ImageModule
from super_resolution.model import SRScalingModel

pl.seed_everything(RANDOM_SEED)

trainer = pl.Trainer(gpus=1, max_epochs=TrainingConfig.max_epochs)
data_module = ImageModule(DataClass.DOG, DataClass.DOG)
model = SRScalingModel()

trainer.fit(model, data_module)
