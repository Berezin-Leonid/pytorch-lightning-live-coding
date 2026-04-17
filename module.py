import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchmetrics
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, Specificity
from lightning.pytorch import LightningModule

def create_model(input_channels=1):
    # Используем ResNet18, как в твоем исходном коде, но адаптируем под MNIST (1 канал)
    model = torchvision.models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

class LitResnet(LightningModule):
    def __init__(self, num_classes=10, lr=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(input_channels=1)

        # Демонстрируем легкость композиции метрик через арифметику
        rec = Recall(task="multiclass", num_classes=num_classes, average='macro')
        spec = Specificity(task="multiclass", num_classes=num_classes, average='macro')
        gmean = (rec * spec) ** 0.5
        
        metrics = MetricCollection({
            'acc': Accuracy(task="multiclass", num_classes=num_classes),
            'prec': Precision(task="multiclass", num_classes=num_classes, average='macro'),
            'rec': rec,
            'f1': F1Score(task="multiclass", num_classes=num_classes, average='macro'),
            'gmean': gmean
        })
        
        # Создаем отдельные коллекции для каждой стадии
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

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
        
        # Обновляем состояние метрик
        self.val_metrics(logits, y)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Обновляем состояние метрик
        self.test_metrics(logits, y)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
