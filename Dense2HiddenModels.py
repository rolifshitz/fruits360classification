"""This module defines the Tiny, Small, Medium, and Big Dense 2Hidden models."""
import torch
from torch import nn
from NetworkParts import LinearWithBN


class Dense2Hidden_Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = LinearWithBN(3 * 100 * 100, 512)
        self.linear2 = LinearWithBN(512, 512)
        self.linear3 = LinearWithBN(512, 113)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        # assert input.shape[1:] == (3, 100, 100)
        output = self.linear1(input.view(input.shape[0], 3 * 100 * 100))
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.softmax(output)
        return output


class Dense2Hidden_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = LinearWithBN(3 * 100 * 100, 1028)
        self.linear2 = LinearWithBN(1028, 1028)
        self.linear3 = LinearWithBN(1028, 113)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        # assert input.shape[1:] == (3, 100, 100)
        output = self.linear1(input.view(input.shape[0], 3 * 100 * 100))
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.softmax(output)
        return output


class Dense2Hidden_Medium(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = LinearWithBN(3 * 100 * 100, 2048)
        self.linear2 = LinearWithBN(2048, 2048)
        self.linear3 = LinearWithBN(2048, 113)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        # assert input.shape[1:] == (3, 100, 100)
        output = self.linear1(input.view(input.shape[0], 3 * 100 * 100))
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.softmax(output)
        return output


class Dense2Hidden_Big(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = LinearWithBN(3 * 100 * 100, 4096)
        self.linear2 = LinearWithBN(4096, 4096)
        self.linear3 = LinearWithBN(4096, 113)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        # assert input.shape[1:] == (3, 100, 100)
        output = self.linear1(input.view(input.shape[0], 3 * 100 * 100))
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.softmax(output)
        return output
