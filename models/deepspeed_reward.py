import torch
import torch.nn as nn
import deepspeed

class DeepSpeedReward(nn.Module):
    def __init__(self, model, optimizer, loss_fn, scale_factor=0.1):
        super(DeepSpeedReward, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scale_factor = scale_factor
    
    def forward(self, generated_images, target_images):
        loss = self.loss_fn(generated_images, target_images)
        reward = -self.scale_factor * loss
        return reward
    
    def update_model(self, generated_images, target_images):
        self.optimizer.zero_grad()
        reward = self.forward(generated_images, target_images)
        reward.backward()
        self.optimizer.step()
    
    def initialize_deepspeed(self, args):
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=self.model,
            model_parameters=self.model.parameters(),
            optimizer=self.optimizer
        )