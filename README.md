# Vision Transformer Super-Resolution Project

## Overview
This project utilizes a Vision Transformer (ViT) backbone integrated with DeepSpeed to create a super-resolution model that upscales low-resolution images to high-resolution outputs. Incorporating a reinforcement learning-based approach with the DeepSpeedReward module, the system adapts during training to optimize image quality through reward maximization.

## Key Features
Vision Transformer Architecture: Employs self-attention mechanisms to process image patches for detailed feature extraction.
Reinforcement Learning Optimization: Uses the DeepSpeedReward module to calculate rewards based on the negative loss, actively guiding the model towards better performance.

DeepSpeed Integration: Enhances training efficiency and scalability, making it feasible to train on larger datasets with reduced computational resources.
Dataset Handling: Includes scripts for loading and preprocessing both training and validation datasets.

Visual Evaluation: Scripts provided for qualitative analysis of the model's performance by comparing low-resolution inputs with generated high-resolution images.

## Prerequisites
Ensure you have the following installed before proceeding:

- Python 3.8 or newer
- PyTorch 1.8 or higher
- DeepSpeed
- Transformers
- torchvision
- PIL
- matplotlib

## Installation
Start by cloning the project repository:

```bash
git clone https://github.com/catplotlib/Vision_Transformers_with_ReinforcementLearning
cd Vision_Transformers_with_ReinforcementLearning
```

Install the required dependencies:

```bash
pip install torch torchvision transformers deepspeed pillow matplotlib
```

## Configuration and Training
Modify the DeepSpeed configuration within train.py as needed:

```python
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
```

```bash
python train.py
```

## Testing and Evaluation
To evaluate the model, run the testing script:

```bash
python test.py
```

This script will generate high-resolution images and save them in the generated_images/ directory. It also displays a comparison plot for visual assessment.