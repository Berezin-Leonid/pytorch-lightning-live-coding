from lightning.pytorch import Trainer
from module import LitResnet
from datamodule import MNISTDataModule, CIFAR10DataModule

dm = MNISTDataModule(batch_size=64)
model = LitResnet(input_channels=1, num_classes=10, lr=0.01)

trainer = Trainer(
    max_epochs=1, 
    accelerator="auto",
    devices=1
)

trainer.fit(model, datamodule=dm)

trainer.test(model, datamodule=dm)
