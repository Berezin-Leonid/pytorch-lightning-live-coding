import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lightning.pytorch import LightningModule

def create_model(input_channels=1):
    # Используем ResNet18, как в твоем исходном коде, но адаптируем под MNIST (1 канал)
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(input_channels=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
