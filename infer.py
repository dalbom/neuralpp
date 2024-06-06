import torch
import numpy as np
import cv2
import os
from datetime import datetime
from torch.utils.data import DataLoader
from data_loader import PersonInferenceDataset
from model import ResNetForImageProcessing
from image_processing import denormalize_parameters, apply_image_operations
from tqdm import tqdm


# Function to perform inference
def perform_inference(model, data_loader, device, results_dir):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for idx, (images, masks) in enumerate(tqdm(data_loader)):
            images, masks = images.to(device), masks.to(device)

            inputs = torch.cat(
                (images, masks), dim=1
            )  # Concatenate RGB and mask to form 4-channel input

            # Forward pass
            outputs = model(inputs)

            for i in range(outputs.size(0)):
                # Process each image in the batch
                sample_image = images[i].permute(1, 2, 0).cpu().numpy() * 255
                sample_mask = masks[i].permute(1, 2, 0).cpu().numpy()
                sample_image_param = outputs[i].detach().cpu().numpy()
                sample_image_param = denormalize_parameters(sample_image_param)

                # Apply inverse image operations to recover the original image
                recovered_image = (
                    apply_image_operations(sample_image / 255, sample_image_param) * 255
                )
                recovered_image = recovered_image.astype(np.uint8)

                # Convert mask to 3 channels
                sample_mask3 = np.dstack([sample_mask] * 3)

                # Recover only the masked region
                recovered_image = np.multiply(
                    sample_image, 1 - sample_mask3
                ) + np.multiply(recovered_image, sample_mask3)

                # Concatenate images horizontally
                concatenated_image = np.hstack(
                    (sample_image, sample_mask3 * 255, recovered_image)
                )

                # Save the concatenated image
                filename = os.path.join(results_dir, f"inference_{idx:04d}_{i:02d}.png")
                cv2.imwrite(
                    filename, concatenated_image[:, :, ::-1]
                )  # Convert RGB to BGR for OpenCV


# Parameters
EPOCH = 200
JPEG_QF = 20
BLUR = 20


# Check if CUDA is available and set the device
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

# Define your dataset
dataset = PersonInferenceDataset("datasets/inference.csv", jpeg_qf=JPEG_QF, blur=BLUR)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Initialize the model
model = ResNetForImageProcessing(output_size=20)

if is_cuda:
    model = model.cuda()

# Load pre-trained model weights
pretrained_weights = f"results/20240605-084327/weights_epoch_{EPOCH:04d}.pth"
model.load_state_dict(torch.load(pretrained_weights))

# Create a results folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join(
    "results", timestamp + "-inference_%d_jpeg%d_blur%d" % (EPOCH, JPEG_QF, BLUR)
)
os.makedirs(results_dir, exist_ok=True)

# Perform inference
perform_inference(model, data_loader, device, results_dir)

print("Inference complete. Results saved in:", results_dir)
