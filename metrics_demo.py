import torch
import torchmetrics


metric = torchmetrics.Accuracy(task="multiclass", num_classes=4)

for i in range(3):
    preds = torch.randint(0, 4, (10,))
    target = torch.randint(0, 4, (10,))
    
    batch_acc = metric(preds, target)
    print(f"Batch {i} Accuracy: {batch_acc:.2f}")

total_acc = metric.compute()
print(f"\nTotal Accuracy (all batches): {total_acc:.2f}")

metric.reset()
