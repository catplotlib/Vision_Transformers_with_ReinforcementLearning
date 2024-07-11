import torch
import torch.optim as optim
import deepspeed
from torchvision import transforms

from models import DeepSpeedReward
from models import VisionTransformer
from utils import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define training parameters
input_size = (1024, 1024)  # Input image dimensions
patch_size = (16, 16)      # Patch size for Vision Transformer
hidden_dim = 256           # Hidden dimensions for DQN
state_dim = input_size[0] * input_size[1] * 3  # State dimension, assuming RGB images
action_dim = 9             # Number of possible actions
batch_size = 8
num_epochs = 200
learning_rate = 1e-4
gamma = 0.99               # Discount factor for DQN
epsilon = 0.1              # Exploration rate for epsilon-greedy policy
target_update = 10         # Frequency of target network update

# Load the dataset
train_loader, _ = load_data('data', batch_size=batch_size, num_workers=4)

# Initialize the model
model = VisionTransformer(input_size, patch_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

# Initialize DeepSpeedReward
rl_model = DeepSpeedReward(model, optimizer, loss_fn, state_dim, action_dim, hidden_dim, learning_rate, gamma, epsilon, target_update, device, batch_size)

# Setup DeepSpeed
# Setup DeepSpeed
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

# Initialize DeepSpeed
rl_model.initialize_deepspeed(deepspeed_config)  # Pass the config directly

# Training loop
for epoch in range(num_epochs):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        # Update the model with the current batch
        rl_model.update_model(data, target, batch_size)
        
        # Calculate metrics for logging
        with torch.no_grad():
            output = rl_model.model(data)
            mse_loss = rl_model.loss_fn(output, target).item()
            ssim_value = ssim(output, target, data_range=1, size_average=True).item()
            lpips_value = rl_model.lpips_fn(output, target).mean().item()
        
        print(f"Epoch {epoch}, MSE Loss: {mse_loss:.4f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}")
    
    if epoch % target_update == 0:
        rl_model.rl_agent.target_network.load_state_dict(rl_model.rl_agent.q_network.state_dict())

print("Training completed.")
