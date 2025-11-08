# femnist_loader.py
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch

# Step 4: Load FEMNIST dataset
dataset = load_dataset("femnist")

# Step 5: Check available splits
print(dataset)

# Step 6: Define transformation (resize, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Step 7: Apply transform to the images
def transform_batch(example):
    example["image"] = transform(example["image"])
    return example

dataset = dataset.map(transform_batch)

# Example: create a DataLoader for the training set
train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True)

# Quick test to confirm it works
for batch in train_loader:
    images, labels = batch["image"], batch["label"]
    print(images.shape, labels.shape)
    break
