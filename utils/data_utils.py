import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class SuperResolutionDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        self.low_res_dir = os.path.join(data_dir, split, "low_res")
        self.high_res_dir = os.path.join(data_dir, split, "high_res")
        
        self.low_res_files = sorted(os.listdir(self.low_res_dir))
        self.high_res_files = sorted(os.listdir(self.high_res_dir))
    
    def __len__(self):
        return len(self.low_res_files)
    
    def __getitem__(self, index):
        low_res_path = os.path.join(self.low_res_dir, self.low_res_files[index])
        high_res_path = os.path.join(self.high_res_dir, self.high_res_files[index])
        
        low_res_image = Image.open(low_res_path).convert("RGB")
        high_res_image = Image.open(high_res_path).convert("RGB")
        
        low_res_image = low_res_image.resize((1024, 1024), Image.BICUBIC)
        high_res_image = high_res_image.resize((1024, 1024), Image.BICUBIC)
        
        if self.transform:
            low_res_image = self.transform(low_res_image)
            high_res_image = self.transform(high_res_image)
        
        return low_res_image, high_res_image

def load_data(data_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = SuperResolutionDataset(data_dir, split="train", transform=transform)
    val_dataset = SuperResolutionDataset(data_dir, split="val", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader