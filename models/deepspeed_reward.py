import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import numpy as np
import random
from pytorch_msssim import ssim
import lpips

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # state_dim needs to match the input feature count
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon, target_update, device,batch_size):
        self.device = device
        self.batch_size = batch_size
        self.action_dim = action_dim  # Store action_dim as an instance attribute
        self.q_network = DQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = DQN(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.update_counter = 0
        self.replay_buffer = ReplayBuffer(capacity=10000)
    
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()  # Use the instance attribute
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state).argmax().item()
    
    def update(self, state, action, reward, next_state, done, batch_size):
        self.replay_buffer.push(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:  # Use the instance variable
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action).unsqueeze(1).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)
        
        q_values = self.q_network(state).gather(1, action)
        next_q_values = self.target_network(next_state).max(1)[0].unsqueeze(1)
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

class DeepSpeedReward(nn.Module):
    def __init__(self, model, optimizer, loss_fn, state_dim, action_dim, hidden_dim, lr, gamma, epsilon, target_update, device, batch_size, scale_factor=0.1):
        super(DeepSpeedReward, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.rl_agent = DQNAgent(state_dim, action_dim, hidden_dim, lr, gamma, epsilon, target_update, device, batch_size)
        self.scale_factor = scale_factor
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    def forward(self, generated_images, target_images):
        output = self.model(generated_images)
        mse_loss = self.loss_fn(output, target_images)
        
        # Calculate SSIM
        ssim_value = ssim(output, target_images, data_range=1, size_average=True)
        
        # Calculate LPIPS
        lpips_value = self.lpips_fn(output, target_images).mean()
        
        # Combine metrics for reward
        reward = self.scale_factor * (ssim_value - lpips_value - mse_loss)
        
        return reward

    def update_model(self, generated_images, target_images, batch_size):
        self.optimizer.zero_grad()
        reward = self.forward(generated_images, target_images)

        # Rest of the method remains the same...
        
        # When applying the action, ensure the result is differentiable
        generated_images_updated = self.apply_action(generated_images, action)
        if not generated_images_updated.requires_grad:
            generated_images_updated = generated_images_updated.detach().requires_grad_(True)
        
        reward_updated = self.forward(generated_images_updated, target_images)
        reward_updated.backward()
        self.optimizer.step()
    
    def apply_action(self, images, action):
        # Decode action here:
        if action == 0:
            # No action
            return images
        elif action == 1:
            # Increase brightness
            return images + 0.1
        elif action == 2:
            # Decrease brightness
            return images - 0.1
        elif action == 3:
            # Increase contrast
            return images * 1.1
        elif action == 4:
            # Decrease contrast
            return images * 0.9
        elif action == 5:
            # Increase sharpness
            sharpen_kernel = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return F.conv2d(images.unsqueeze(0), sharpen_kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze(0)
        elif action == 6:
            # Decrease sharpness (blur)
            blur_kernel = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
            return F.conv2d(images.unsqueeze(0), blur_kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze(0)
        elif action == 7:
            # Adjust color balance
            # Example: Increase red channel
            images[:, 0, :, :] *= 1.1
            return images
        elif action == 8:
            # Patch-based enhancement
            # Example: Apply different enhancements on top left and bottom right quarters
            quarter_size = images.size(2) // 2
            top_left = images[:, :, :quarter_size, :quarter_size] * 1.2
            bottom_right = images[:, :, quarter_size:, quarter_size:] + 0.1
            images[:, :, :quarter_size, :quarter_size] = top_left
            images[:, :, quarter_size:, quarter_size:] = bottom_right
            return images
        
        if not modified_images.requires_grad:
            raise RuntimeError("Action application results in non-differentiable tensor.")

        return modified_images
        # return images.clamp(0, 1)
    
    def initialize_deepspeed(self, deepspeed_config):
        # Initialize DeepSpeed directly with the provided config dictionary
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            optimizer=self.optimizer,
            config_params=deepspeed_config  # Pass the configuration dictionary directly
        )

