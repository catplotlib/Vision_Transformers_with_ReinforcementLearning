import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from models import VisionTransformer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define hyperparameters
input_size = (1024, 1024)
patch_size = (16, 16)

# Initialize the model with the correct architecture
model = VisionTransformer(input_size, patch_size).to(device)

# Load the trained model, handling the state_dict if trained using DDP
state_dict = torch.load("trained_model.pth", map_location=device)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# Path to the low-resolution image
low_res_image_path = 'data/train/low_res/lr_0.png'

# Load and transform the image
transform = transforms.Compose([
    transforms.Resize(input_size, Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
low_res_image = Image.open(low_res_image_path).convert("RGB")
low_res_image = transform(low_res_image).unsqueeze(0).to(device)  # Add batch dimension

# Generate the high-resolution image
with torch.no_grad():
    generated_image = model(low_res_image)

# Unnormalize the image
generated_image = generated_image.squeeze(0)
generated_image = (generated_image * 0.5) + 0.5  # Adjust depending on the normalization used during training

# Clamp the pixel values to the valid range [0, 1]
generated_image = torch.clamp(generated_image, min=0, max=1)

# Convert the tensor to PIL Image
generated_image = transforms.ToPILImage()(generated_image.cpu())

# Create a directory to save the generated images
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Save the image with the highest quality
generated_image.save(os.path.join(output_dir, "generated_high_res_image.png"), optimize=True, quality=100)

print("High-resolution image has been generated and saved with the highest quality using the GPU.")