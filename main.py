import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from module import LitResnet

# Настройки трансформаций для ResNet (желательно 32x32 или 224x224, но оставим как есть для минимализма)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # ResNet лучше работает, когда картинка чуть больше 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Загрузка данных
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

# Инициализация модели и трейнера
model = LitResnet(lr=0.01)
trainer = Trainer(
    max_epochs=1,
    devices=[1],
    )

# Запуск обучения
trainer.fit(model, train_loader, val_loader)
trainer.test(model, val_loader)