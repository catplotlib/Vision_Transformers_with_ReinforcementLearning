import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
from models import VisionTransformer
from utils.data_utils import load_data
from utils.training_utils import train_model
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = (1024, 1024)
patch_size = (16, 16)
num_classes = 1
batch_size = 8
num_epochs = 200
learning_rate = 1e-4
patience = 100  

train_loader, val_loader = load_data("data", batch_size=batch_size, num_workers=4)

train_dataset = train_loader.dataset
val_dataset = val_loader.dataset

model = VisionTransformer(input_size, patch_size).to(device)

loss_fn = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

deepspeed_config = {
    "train_batch_size": batch_size,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": learning_rate
        }
    },
    "fp16": {
        "enabled": True
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    training_data=train_dataset,  # Pass the training dataset
    config_params=deepspeed_config
)

train_loader = model_engine.deepspeed_io(train_dataset)

train_losses = []
best_loss = float('inf')
epochs_without_improvement = 0

os.makedirs("results/plots", exist_ok=True)

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model_engine(data)
        loss = loss_fn(output, target)
        model_engine.backward(loss)
        model_engine.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {batch_loss:.4f}')

    average_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(average_epoch_loss)
    print(f'End of Epoch {epoch+1}, Average Loss: {average_epoch_loss:.4f}')

    # Check for improvement
    if average_epoch_loss < best_loss:
        best_loss = average_epoch_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Stop training if loss plateaus
    if epochs_without_improvement >= patience:
        print(f"No improvement in loss for {patience} epochs. Stopping training.")
        break

torch.save(model_engine.state_dict(), "trained_model.pth")
print("Training completed and model saved.")

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/training_loss.png")
plt.close()