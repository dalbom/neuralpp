import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
from datetime import datetime
from torch.utils.data import DataLoader
from data_loader import PersonDataset
from model import ResNetForImageProcessing
from image_processing import denormalize_parameters, apply_image_operations
from tqdm import tqdm


def write_image_log(images, masks, params, results_dir, epoch):
    # Get the first element of the first batch
    sample_image = images[0].permute(1, 2, 0).cpu().numpy() * 255
    sample_mask = masks[0].permute(1, 2, 0).cpu().numpy()
    sample_image_param = params[0].detach().cpu().numpy()
    sample_image_param = denormalize_parameters(sample_image_param)

    # Apply inverse image operations to recover the original image
    recovered_image = (
        apply_image_operations(sample_image / 255, sample_image_param) * 255
    )
    recovered_image = recovered_image.astype(np.uint8)

    # Convert mask to 3 channels
    sample_mask3 = np.dstack([sample_mask] * 3)

    # Recover only the masked region
    recovered_image = np.multiply(sample_image, 1 - sample_mask3) + np.multiply(
        recovered_image, sample_mask3
    )

    # Concatenate images horizontally
    concatenated_image = np.hstack((sample_image, sample_mask3 * 255, recovered_image))

    # Save the concatenated image
    filename = os.path.join(results_dir, f"epoch_{epoch:03d}.png")
    cv2.imwrite(
        filename, concatenated_image[:, :, ::-1]
    )  # Convert RGB to BGR for OpenCV


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
num_epochs = 1000

# Create a results folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join("results", timestamp)
os.makedirs(results_dir, exist_ok=True)

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
        mask = image_params >= 0
        mask_invalid = image_params < 0

        valid_outputs = outputs[mask]
        valid_targets = image_params[mask]

        invalid_outputs = outputs[mask_invalid]

        loss_valid = criterion(valid_outputs, valid_targets).mean()
        loss_invalid = torch.mean((invalid_outputs + 1) ** 2)

        loss = loss_valid + loss_invalid
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Write image log
    write_image_log(images, masks, outputs, results_dir, epoch)

    # Save model weights every 20 epochs
    if (epoch + 1) % 20 == 0:
        weight_path = os.path.join(results_dir, f"weights_epoch_{epoch + 1:04d}.pth")
        torch.save(model.state_dict(), weight_path)

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}")
