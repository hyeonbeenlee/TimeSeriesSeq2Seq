import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import platform
from copy import deepcopy
import random
import time
import os, sys
from typing import List, Tuple
from torch import Tensor
import warnings
import math

sys.path.append(os.getcwd())
from architectures.skeleton import Skeleton


class ResidualConvolution(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0,
        activation: str = "gelu",
        pool: str = "max",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activation = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
        }[activation]
        self.pool = {
            "max": nn.MaxPool1d(kernel_size, stride),
            "avg": nn.AvgPool1d(kernel_size, stride),
        }[pool]
        self.dropout = nn.Dropout(dropout)

        # convolution layers
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # downsample layers
        self.downsample = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)

        residual = self.downsample(x)
        residual = self.dropout(residual)

        out = out + residual
        out = self.bn2(out)
        out = self.activation(out)

        out = self.pool(out)
        return out


class StackedResidualConvolution1D(Skeleton):
    def __init__(
        self,
        in_channels,
        n_stacks=1,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0,
        cnn_activation="gelu",
    ):
        super().__init__()
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.update_attributes(locals())
        self.update_init_args(locals())

        # CNNs
        base = 1.3
        self.stacks = nn.ModuleList()
        for i in range(n_stacks):
            if i == 0:
                self.stacks.append(
                    ResidualConvolution(
                        round(in_channels * base**i),
                        round(in_channels * base ** (i + 1)),
                        kernel_size=kernel_size,
                        stride=stride,
                        dropout=dropout,
                        activation=cnn_activation,
                        pool="max",
                    )
                )
            else:
                self.stacks.append(
                    ResidualConvolution(
                        round(in_channels * base**i),
                        round(in_channels * base ** (i + 1)),
                        kernel_size=kernel_size,
                        stride=stride,
                        dropout=dropout,
                        activation=cnn_activation,
                        pool="avg",
                    )
                )
        self.output_size = round(in_channels * base**n_stacks)

    def forward(self, x: torch.Tensor):
        output = x  # (N,C,L)
        for cnn in self.stacks:
            output = cnn.forward(output)  # (N,Cout,Lout)
        output = output.permute(0, 2, 1)  # (N,Lout,Cout)
        return output


if __name__ == "__main__":
    B = 32
    Cin = 27
    L = 20
    x = torch.randn(B, Cin, L)
    net = StackedResidualConvolution1D(in_channels=Cin, n_stacks=4)
    out = net(x)
    pass
