from jsonargparse import CLI

def train_model(
    lr: float = 0.01, 
    batch_size: int = 32, 
    model_name: str = "resnet18"
):
    """
    Пример тренировочной функции.
    
    Args:
        lr: Скорость обучения (float)
        batch_size: Размер батча (int)
        model_name: Название архитектуры
    """
    print(f"Запуск обучения: модель={model_name}, lr={lr}, batch_size={batch_size}")

if __name__ == "__main__":
    # jsonargparse автоматически распарсит аргументы, типы и docstring
    CLI(train_model)
