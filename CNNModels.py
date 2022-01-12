"""This module defines the Tiny, Small, Medium, and Big CNN models."""
import torch
from torch import nn
from NetworkParts import SingleConv, LinearWithBN


class CNN_Big(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SingleConv(3, 64, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = SingleConv(64, 128, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = SingleConv(128, 128, 3)
        self.linear1 = LinearWithBN(25 * 25 * 128, 4096)
        self.linear2 = LinearWithBN(4096, 1028)
        self.linear3 = LinearWithBN(1028, 113)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        """Input has shape (N, C, H, W)."""
        output = self.conv1(input)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.pool2(output)
        output = self.conv3(output)
        output = self.linear1(output.view(input.shape[0], 25 * 25 * 128))
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.softmax(output)
        return output


class CNN_Medium(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SingleConv(3, 64, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = SingleConv(64, 128, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.linear1 = LinearWithBN(25 * 25 * 128, 4096)
        self.linear2 = LinearWithBN(4096, 1028)
        self.linear3 = LinearWithBN(1028, 113)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        output = self.conv1(input)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.pool2(output)
        output = self.linear1(output.view(input.shape[0], 25 * 25 * 128))
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.softmax(output)
        return output


class CNN_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SingleConv(3, 64, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.linear1 = LinearWithBN(50 * 50 * 64, 2048)
        self.linear2 = LinearWithBN(2048, 1028)
        self.linear3 = LinearWithBN(1028, 113)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        output = self.conv1(input)
        output = self.pool1(output)
        output = self.linear1(output.view(input.shape[0], 50 * 50 * 64))
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.softmax(output)
        return output


class CNN_Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SingleConv(3, 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.linear1 = LinearWithBN(50 * 50 * 32, 2048)
        self.linear2 = LinearWithBN(2048, 1028)
        self.linear3 = LinearWithBN(1028, 113)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        output = self.conv1(input)
        output = self.pool1(output)
        output = self.linear1(output.view(input.shape[0], 50 * 50 * 32))
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.softmax(output)
        return output
