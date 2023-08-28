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
from architectures.rnn import *
from architectures.cnn import *
from architectures.mlp import *


class BahdanauAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.D = 2 if bidirectional else 1
        # Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate.
        self.Ws = nn.Linear(self.D * hidden_size, self.D * hidden_size)
        self.Va = nn.Linear(self.D * hidden_size, 1)
        self.Wa = nn.Linear(self.D * hidden_size, self.D * hidden_size)
        self.Ua = nn.Linear(self.D * hidden_size, self.D * hidden_size)

    def forward(self, q, k, v):
        batch_size, length, hidden_size = k.size()
        # q,k,v=dec,enc_all,enc_all
        # project decoder initial states
        q = q.unsqueeze(1) if len(q.shape) == 2 else q

        attn_scores = F.tanh(self.Wa(q) + self.Ua(k))  # (B,L,D*H)
        attn_scores = self.Va(attn_scores).squeeze(-1)  # BL
        attn_scores = F.softmax(attn_scores, dim=1).unsqueeze(2)  # BL1
        attn_v = torch.mul(attn_scores, v).sum(dim=1, keepdim=True)  # B1H
        return attn_v, attn_scores  # (B,D*H), (B,L)


class DotProductAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.D = 2 if bidirectional else 1
        self.Wq = nn.Linear(self.D * hidden_size, self.D * hidden_size)
        self.Wk = nn.Linear(self.D * hidden_size, self.D * hidden_size)
        self.Wv = nn.Linear(self.D * hidden_size, self.D * hidden_size)
        self.Wo = nn.Linear(self.D * hidden_size, self.D * hidden_size)

    def forward(self, q, k, v):
        batch_size, length, hidden_size = k.size()
        q = q.unsqueeze(1) if len(q.shape) == 2 else q
        q = self.Wq(q)  # B1H
        k = self.Wk(k)  # BLH
        v = self.Wv(v)
        attn_scores = torch.bmm(k, q.transpose(1, 2))  # BL1
        attn_scores /= math.sqrt(self.D * hidden_size)
        attn_scores = F.softmax(attn_scores, dim=1)  # BL1
        attn_v = self.Wo(torch.mul(attn_scores, v)).sum(dim=1, keepdim=True)
        return attn_v, attn_scores  # (B,D*H), (B,L)


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.D = 2 if bidirectional else 1
        self.Wq = nn.Linear(self.D * hidden_size, self.D * hidden_size)
        self.Wk = nn.Linear(self.D * hidden_size, self.D * hidden_size)
        self.Wv = nn.Linear(self.D * hidden_size, self.D * hidden_size)
        self.Wo = nn.Linear(self.D * hidden_size, self.D * hidden_size)

    def forward(self, q, k, v, mask=None):
        batch_size, length, hidden_size = k.size()
        q = q.unsqueeze(1) if len(q.shape) == 2 else q
        q = self.Wq(q)  # BLH
        k = self.Wk(k)  # BLH
        v = self.Wv(v)  # BLH
        attn_scores = torch.bmm(k, q.transpose(1, 2))  # BLL
        attn_scores /= math.sqrt(self.D * hidden_size)
        if mask is not None:
            mask = mask.expand(batch_size, length, length)
            attn_score = attn_score.masked_fill_(mask == 0, -1e12)
        attn_scores = F.softmax(attn_scores, dim=1)
        attn_v = self.Wo(torch.einsum("blL,blH->bLH", attn_scores, v)).sum(
            dim=1, keepdim=True
        )
        return attn_v, attn_scores  # (B,D*H), (B,L)
