import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import PersonDataset
from model import ResNetForImageProcessing
from tqdm import tqdm

import torch.nn.functional as F


def masked_mse_loss(pred, target, mask):
    diff = pred - target
    diff = diff * mask
    mse_loss = (diff**2).sum() / mask.sum()
    return mse_loss


is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

# Define your dataset
dataset = PersonDataset("datasets/daytime.csv")

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model
model = ResNetForImageProcessing(output_size=19)

if is_cuda:
    model = model.cuda()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 100
placeholder_value = float("inf")


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks, image_params in tqdm(data_loader):
        images, masks, image_params = (
            images.to(device),
            masks.to(device),
            image_params.to(device),
        )

        inputs = torch.cat(
            (images, masks), dim=1
        )  # Concatenate RGB and mask to form 4-channel input

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss only for non-placeholder values
        mask = image_params != placeholder_value
        valid_outputs = outputs[mask]
        valid_targets = image_params[mask]

        if valid_outputs.numel() > 0:  # Ensure there are valid elements
            loss = criterion(valid_outputs, valid_targets).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print(loss.item())

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}")
