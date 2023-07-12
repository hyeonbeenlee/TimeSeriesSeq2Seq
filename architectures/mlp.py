import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import platform
from copy import deepcopy

import os, sys

sys.path.append(os.getcwd())
from architectures.skeleton import Skeleton
from architectures.init import initialize


class MLP(Skeleton):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers: int = 3,
    ) -> None:
        super(MLP, self).__init__()
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.update_attributes(locals())
        self.update_init_args(locals())
        self.set_device()

        self.num_hidden_layers = num_layers - 1
        assert num_layers >= 3
        self.mlp = []
        for i in range(num_layers):
            if i == 0:  # input layer
                self.mlp.append(nn.Linear(input_size, hidden_size, device=self.device))
                self.mlp.append(nn.GELU())
            elif i == num_layers - 1:  # output layer
                self.mlp.append(nn.Linear(hidden_size, output_size, device=self.device))
            else:  # hidden layers
                self.mlp.append(nn.Linear(hidden_size, hidden_size, device=self.device))
                self.mlp.append(nn.GELU())
        self.mlp = nn.Sequential(*self.mlp)
        initialize(self)

    def initialize(self):
        for n, p in self.named_parameters():
            if "weight" in n:
                nn.init.kaiming_normal_(p)
            elif "bias" in n:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        out = self.mlp.forward(x)
        return out


if __name__ == "__main__":
    x = torch.rand(64, 10)
    model = MLP(10, 6, 512, num_layers=3)
    y = model(x)
