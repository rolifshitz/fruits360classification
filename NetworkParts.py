"""This module defines the network layers we will use in our models."""
from torch import nn


class SingleConv(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int):
        """Initialize sConv2d -> BatchNorm2d -> ReLU

        Args:
            channels_in (int): Number of input channels.
            channels_out (int): Number of convolutional filters.
            kernel_size (int): Size of convolution filters.
        """
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU()
        )

    def forward(self, input):
        return self.single_conv(input)


class LinearWithBN(nn.Module):
    def __init__(self, features_in, features_out):
        """Initialize fully connected layer followed by Batch Norm and ReLU activation."""
        super().__init__()
        self.linear_bn_relu = nn.Sequential(
            nn.Linear(features_in, features_out),
            nn.BatchNorm1d(features_out),
            nn.ReLU()
        )

    def forward(self, input):
        return self.linear_bn_relu(input)
