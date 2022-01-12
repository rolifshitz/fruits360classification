"""This module defines the pytorch dataset wrapper."""
import numpy as np
import torch


class FruitsDataset(torch.utils.data.Dataset):
    def __init__(self, x_numpy: np.ndarray, y_numpy: np.ndarray):
        """PyTorch wrapper for data."""
        # Assert shapes
        assert x_numpy.shape == (x_numpy.shape[0], 3, 100, 100)
        assert x_numpy.shape[0] == y_numpy.shape[0]

        # Store x and y as torch tensors
        self.x = torch.from_numpy(x_numpy).float()
        self.y = torch.from_numpy(y_numpy).long()

    def __getitem__(self, index: int):
        input, target = self.x[index], self.y[index]
        return input, target

    def __len__(self):
        return len(self.x)

