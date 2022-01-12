"""This module defines the Tiny, Small, Medium, and Big Dense 1Hidden models."""
import torch
from torch import nn
from NetworkParts import LinearWithBN


class Dense1Hidden_Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = LinearWithBN(3 * 100 * 100, 512)
        self.linear2 = LinearWithBN(512, 113)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        output = self.linear1(input.view(input.shape[0], 3 * 100 * 100))
        output = self.linear2(output)
        output = self.softmax(output)
        return output


class Dense1Hidden_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = LinearWithBN(3 * 100 * 100, 1028)
        self.linear2 = LinearWithBN(1028, 113)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        output = self.linear1(input.view(input.shape[0], 3 * 100 * 100))
        output = self.linear2(output)
        output = self.softmax(output)
        return output


class Dense1Hidden_Medium(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = LinearWithBN(3 * 100 * 100, 2048)
        self.linear2 = LinearWithBN(2048, 113)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        output = self.linear1(input.view(input.shape[0], 3 * 100 * 100))
        output = self.linear2(output)
        output = self.softmax(output)
        return output


class Dense1Hidden_Big(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = LinearWithBN(3 * 100 * 100, 4096)
        self.linear2 = LinearWithBN(4096, 113)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        output = self.linear1(input.view(input.shape[0], 3 * 100 * 100))
        output = self.linear2(output)
        output = self.softmax(output)
        return output

