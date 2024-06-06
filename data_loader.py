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
    apply_jpeg_artifacts,
    apply_blur,
)


class PersonInferenceDataset(Dataset):
    def __init__(self, data_path, jpeg_qf, blur=0):
        self.data_path = data_path
        self.image_paths = []
        self.mask_paths = []
        self.blur = blur * 0.1
        self.jpeg_qf = jpeg_qf
        self._load_data(data_path)

    def _load_data(self, data_path):
        with open(data_path, "r") as reader:
            reader.readline()  # Skip header

            for line in reader:
                image_path, mask_path = line.strip().split(",")

                self.image_paths.append(image_path)
                self.mask_paths.append(mask_path)

        print("Loaded", len(self.image_paths), "data samples.")

    def __len__(self):
        return len(self.image_paths)

    def _corrupt_image(self, image):
        # Apply artifacts
        corrupted_image = apply_blur(image / 255, self.blur)
        corrupted_image = apply_jpeg_artifacts(corrupted_image, self.jpeg_qf)

        return corrupted_image * 255

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        corrupted_image = self._corrupt_image(image)

        binary_mask3 = np.dstack([mask / 255] * 3)
        image = np.multiply(corrupted_image, binary_mask3) + np.multiply(
            image, 1 - binary_mask3
        )

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return image, mask


class PersonDataset(Dataset):
    def __init__(self, data_path, transform=None, max_person_size=500):
        self.data_path = data_path
        self.transform = transform
        self.max_person_size = max_person_size
        self.image_paths = []
        self.annotation_paths = []
        self.polygon_key = None
        self.object_key = None
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

        image_param = generate_random_parameters()
        corrupted_image = (
            apply_image_operations(cropped_image / 255, image_param) * 255
        ).astype(np.uint8)

        binary_mask3 = np.dstack([binary_mask] * 3)
        cropped_image = np.multiply(cropped_image, binary_mask3) + np.multiply(
            corrupted_image, 1 - binary_mask3
        )

        cropped_image = torch.from_numpy(cropped_image).permute(2, 0, 1).float() / 255.0
        binary_mask = torch.from_numpy(binary_mask).unsqueeze(0).float()

        # Normalize image_param
        normalized_image_param = normalize_parameters(image_param)
        image_param_tensor = torch.tensor(normalized_image_param, dtype=torch.float32)

        return cropped_image, binary_mask, image_param_tensor

    def process_image(self, image, annotations, margin=20):
        person_instances = self.get_valid_person_instances(annotations)
        instance = random.choice(person_instances)
        x_min, y_min, x_max, y_max = self.get_instance_bounds(
            instance[self.polygon_key]
        )

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Apply jittering
        margin_x = min(margin, (x_max - x_min) // 4)
        jitter_x = random.randint(-margin_x, margin_x)
        margin_y = min(margin, (y_max - y_min) // 4)
        jitter_y = random.randint(-margin_y, margin_y)
        center_x += jitter_x
        center_y += jitter_y

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
        polygon = np.array([instance[self.polygon_key]], dtype=np.int32)
        cv2.fillPoly(binary_mask, polygon, 1)
        binary_mask = binary_mask[y_start:y_end, x_start:x_end]

        return cropped_image, binary_mask

    def get_valid_person_instances(self, annotations):
        objects = []

        if self.object_key is None:
            if "objects" not in annotations:
                self.object_key = "annotations"
                self.polygon_key = "points"
            else:
                self.object_key = "objects"
                self.polygon_key = "polygon"

        for obj in annotations[self.object_key]:
            if obj["label"] != "person":
                continue

            x_min, y_min, x_max, y_max = self.get_instance_bounds(obj[self.polygon_key])
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
