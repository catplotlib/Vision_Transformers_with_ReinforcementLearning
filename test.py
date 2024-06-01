import os
import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from PIL import Image
from models import VisionTransformer
import matplotlib.pyplot as plt

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_size = (1024, 1024)
    patch_size = (16, 16)
    model = VisionTransformer(input_size, patch_size).to(device)
    state_dict = torch.load("trained_model.pth", map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    low_res_dir = 'data/val/low_res'
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(input_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    for filename in os.listdir(low_res_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            low_res_path = os.path.join(low_res_dir, filename)
            low_res_image = Image.open(low_res_path).convert("RGB")
            low_res_image = transform(low_res_image).unsqueeze(0).to(device)
            with torch.no_grad():
                generated_image = model(low_res_image)
            generated_image = generated_image.squeeze(0)
            generated_image = (generated_image * 0.5) + 0.5
            generated_image = torch.clamp(generated_image, min=0, max=1)
            save_path = os.path.join(output_dir, f"generated_{filename}")
            save_image(generated_image, save_path)

            print(f"Generated image saved: {save_path}")
    fig, axs = plt.subplots(2, len(os.listdir(low_res_dir)), figsize=(20, 10))
    axs = axs.ravel()
    for i, filename in enumerate(os.listdir(low_res_dir)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            low_res_path = os.path.join(low_res_dir, filename)
            low_res_image = Image.open(low_res_path)
            axs[i].imshow(low_res_image)
            axs[i].set_title(f"Low-Res: {filename}")
            axs[i].axis('off')
            generated_path = os.path.join(output_dir, f"generated_{filename}")
            generated_image = Image.open(generated_path)
            axs[i + len(os.listdir(low_res_dir))].imshow(generated_image)
            axs[i + len(os.listdir(low_res_dir))].set_title(f"Generated High-Res: {filename}")
            axs[i + len(os.listdir(low_res_dir))].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test()