import os
import torch
from torchvision.utils import save_image
from models import VisionTransformer
from utils.data_utils import load_data
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_metrics(hr_img, sr_img):
    hr_img = hr_img.astype(np.float32)
    sr_img = sr_img.astype(np.float32)

    hr_img = hr_img.reshape(-1)  # Flatten the array
    sr_img = sr_img.reshape(-1)  # Flatten the array

    mse = mean_squared_error(hr_img, sr_img)
    nmae = np.mean(np.abs(hr_img - sr_img)) / np.mean(hr_img)
    psnr_value = psnr(hr_img.reshape(1024, 1024, 3), sr_img.reshape(1024, 1024, 3), data_range=255)
    ssim_value = ssim(hr_img.reshape(1024, 1024, 3), sr_img.reshape(1024, 1024, 3), data_range=255, multichannel=True, win_size=3)

    return psnr_value, ssim_value, mse, nmae

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

    _, val_loader = load_data("data", batch_size=1, num_workers=4)

    output_dir = "generated_images/gen"
    os.makedirs(output_dir, exist_ok=True)

    psnr_values = []
    ssim_values = []
    mse_values = []
    nmae_values = []

    for low_res_image, high_res_image in val_loader:
        low_res_image = low_res_image.to(device)
        high_res_image = high_res_image.to(device)

        with torch.no_grad():
            generated_image = model(low_res_image)

        generated_image = generated_image.squeeze(0).cpu().numpy()
        generated_image = np.transpose(generated_image, (1, 2, 0))
        generated_image = ((generated_image * 0.5) + 0.5) * 255.0
        generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)

        high_res_image = high_res_image.squeeze(0).cpu().numpy()
        high_res_image = np.transpose(high_res_image, (1, 2, 0))
        high_res_image = ((high_res_image * 0.5) + 0.5) * 255.0
        high_res_image = np.clip(high_res_image, 0, 255).astype(np.uint8)

        psnr_value, ssim_value, mse, nmae = evaluate_metrics(high_res_image, generated_image)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        mse_values.append(mse)
        nmae_values.append(nmae)

        save_path = os.path.join(output_dir, f"generated_{len(psnr_values)}.png")
        save_image(torch.from_numpy(generated_image).permute(2, 0, 1).float() / 255.0, save_path)

        print(f"Generated image saved: {save_path}")

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_mse = np.mean(mse_values)
    avg_nmae = np.mean(nmae_values)

    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE: {avg_mse:.2f}")
    print(f"Average NMAE: {avg_nmae:.4f}")

if __name__ == '__main__':
    test()