import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class VisionTransformer(nn.Module):
    def __init__(self, input_size, patch_size):
        super(VisionTransformer, self).__init__()
        
        self.input_size = input_size
        self.patch_size = patch_size
        num_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])
        
        # Define the ViT configuration
        config = ViTConfig(
            image_size=input_size[0],
            patch_size=patch_size[0],
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
        )
        
        self.vit = ViTModel(config)
        
        # Define the upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_size, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        outputs = self.vit(x)
        last_hidden_state = outputs.last_hidden_state
        batch_size, seq_length, hidden_size = last_hidden_state.size()
        num_patches = seq_length - 1  
        patches_per_dim = int(num_patches ** 0.5)
        features = last_hidden_state[:, 1:, :].permute(0, 2, 1).view(batch_size, hidden_size, patches_per_dim, patches_per_dim)
        output = self.upsample(features)
        return output