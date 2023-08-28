import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def initialize(model):
    for n, p in model.named_parameters():
        if "bias" in n:
            nn.init.zeros_(p)
        else:
            nn.init.normal_(p, mean=0, std=0.01)
