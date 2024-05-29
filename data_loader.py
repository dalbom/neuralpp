import json
import random
import torch
from torch.utils.data import Dataset
from skimage.io import imread
import numpy as np
import cv2
from image_processing import (
    generate_random_parameters,
    apply_image_operations,
    normalize_parameters,
)


class PersonDataset(Dataset):
    def __init__(self, data_path, transform=None, max_person_size=300):
        self.data_path = data_path
        self.transform = transform
        self.max_person_size = max_person_size
        self.image_paths = []
        self.annotation_paths = []
        self._load_data(data_path)

    def _load_data(self, data_path):
        with open(data_path, "r") as reader:
            reader.readline()  # Skip header

            for line in reader:
                image_path, annotation_path = line.strip().split(",")

                annotations = json.load(open(annotation_path))
                valid_persons = self.get_valid_person_instances(annotations)

                if len(valid_persons) > 0:
                    self.image_paths.append(image_path)
                    self.annotation_paths.append(annotation_path)

        print("Loaded", len(self.image_paths), "data samples.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        image = imread(image_path)
        annotations = json.load(open(annotation_path))

        cropped_image, binary_mask = self.process_image(image, annotations)

        if self.transform:
            augmented = self.transform(image=cropped_image, mask=binary_mask)
            cropped_image = augmented["image"]
            binary_mask = augmented["mask"]

        # cv2.imwrite("data_%04d_rgb.png" % idx, cropped_image)
        # cv2.imwrite("data_%04d_mask.png" % idx, binary_mask * 255)

        image_param = generate_random_parameters()
        corrupted_image = (
            apply_image_operations(cropped_image / 255, image_param) * 255
        ).astype(np.uint8)

        # cv2.imwrite("data_%04d_corrupted.png" % idx, corrupted_image)

        binary_mask3 = np.dstack([binary_mask] * 3)
        cropped_image = np.multiply(cropped_image, binary_mask3) + np.multiply(
            corrupted_image, 1 - binary_mask3
        )

        # cv2.imwrite("data_%04d_combined.png" % idx, cropped_image)

        cropped_image = torch.from_numpy(cropped_image).permute(2, 0, 1).float() / 255.0
        binary_mask = torch.from_numpy(binary_mask).unsqueeze(0).float()

        # Normalize image_param
        normalized_image_param = normalize_parameters(image_param)
        image_param_tensor = torch.tensor(normalized_image_param, dtype=torch.float32)

        return cropped_image, binary_mask, image_param_tensor

    def process_image(self, image, annotations):
        person_instances = self.get_valid_person_instances(annotations)
        instance = random.choice(person_instances)
        x_min, y_min, x_max, y_max = self.get_instance_bounds(instance["polygon"])

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Calculate the starting and ending points for the crop
        x_start = center_x - 256
        y_start = center_y - 256
        x_end = center_x + 256
        y_end = center_y + 256

        # Ensure the coordinates stay within the image boundaries
        x_end += (x_start < 0) * abs(x_start)
        x_start = max(0, x_start)
        y_end += (y_start < 0) * abs(y_start)
        y_start = max(0, y_start)

        x_start -= (x_end > image.shape[1]) * (x_end - image.shape[1])
        x_end = min(image.shape[1], x_end)
        y_start -= (y_end > image.shape[0]) * (y_end - image.shape[0])
        y_end = min(image.shape[0], y_end)

        cropped_image = image[y_start:y_end, x_start:x_end]

        # Create the binary mask for the person instance
        binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        polygon = np.array([instance["polygon"]], dtype=np.int32)
        cv2.fillPoly(binary_mask, polygon, 1)
        binary_mask = binary_mask[y_start:y_end, x_start:x_end]

        return cropped_image, binary_mask

    def get_valid_person_instances(self, annotations):
        objects = []

        for obj in annotations["objects"]:
            if obj["label"] != "person":
                continue

            x_min, y_min, x_max, y_max = self.get_instance_bounds(obj["polygon"])
            if (
                x_max - x_min <= self.max_person_size
                and y_max - y_min <= self.max_person_size
            ):
                objects.append(obj)

        return objects

    def get_instance_bounds(self, polygon):
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def create_binary_mask(self, image_shape, polygon):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        polygon = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 1)
        return mask
