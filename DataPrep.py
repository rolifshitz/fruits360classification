"""This module defines a class to load and prep data."""
from typing import Optional
import os

from FruitsDataset import FruitsDataset

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
import cv2


class DataPrep:
    @staticmethod
    def load_data_into_loaders(data_path: str, batch_size: int, shuffle: bool, seed: int,
                               split_data: bool = True, val_size: Optional[float] = 0.2):
        """Load the dataset given the path and return the pytorch data loader(s).

        Args:
            data_path: Path to all class folders where each class folder contains the images for that class.
            batch_size: Batch size for the data loaders.
            shuffle: Whether to shuffle data at each epoch.
            seed: Random seed for RNG.
            split_data: Whether to split data into trn and val sets and return two data loaders instead of one.
            val_size: Size of val set relative to whole when split_data is True.
        """
        folder_names = sorted(os.listdir(data_path))

        # First, create a list of all the class names
        class_names = []
        for folder_name in folder_names:
            class_name = DataPrep.get_class_name(folder_name)
            if class_name not in class_names:
                class_names.append(DataPrep.get_class_name(folder_name))

        # Then, loop through each class folder, loading all of the images in the folder into images.
        # While doing this, append the index of the class name in class_names to the labels list (the label).
        images = []
        labels = []
        for folder_name in folder_names:
            class_name = DataPrep.get_class_name(folder_name)
            label = class_names.index(class_name)

            # Load images for this folder
            folder_path = os.path.join(data_path, folder_name)
            image_names = os.listdir(folder_path)
            for image_name in image_names:
                image_path = os.path.join(folder_path, image_name)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv loads images as BGR

                # Normalize
                image = image / 255

                images.append(image)
                labels.append(label)

        # Convert to numpy arrays and change images shape from (N, 100, 100, 3) to (N, 3, 100, 100)
        x_numpy = np.array(images).reshape((len(images), 3, 100, 100))
        y_numpy = np.array(labels)

        if split_data:
            # Randomly split dataset into trn and val sets (using seed)
            x_trn, x_val, y_trn, y_val = train_test_split(x_numpy, y_numpy, test_size=val_size, random_state=seed)

            # Create dataset and data loader pytorch objects
            trn_dataset = FruitsDataset(x_trn, y_trn)
            trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=shuffle)

            val_dataset = FruitsDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)  # Batch size = 1
            return trn_loader, val_loader
        else:
            # Create dataset and data loader pytorch object
            dataset = FruitsDataset(x_numpy, y_numpy)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            return loader

    @staticmethod
    def get_class_name(folder_name: str):
        """Get the class name from a folder name (some have an extra number at the end but are the same class)."""
        name = folder_name.strip()
        if name[-1].isdigit():
            name = name[:-1].strip()
        return name
