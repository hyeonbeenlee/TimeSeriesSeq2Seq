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

    def forward(
        self, encoder_states: List[List[Tuple[Tensor]]], decoder_state: Tensor, t: int
    ):
        # project decoder initial states
        if t == 0:
            query = self.Ws(decoder_state)  # (B,D*H)
        else:
            query = decoder_state  # (B,D*H)
        # pre-compute encoder state projections
        proj_keys = []
        keys = []
        for d, direction in enumerate(encoder_states):
            keys += [
                torch.stack([state[-1][0] for state in direction], dim=1)
            ]  # (B,L,H)
        keys = torch.cat(keys, dim=-1)  # (B,L,D*H) (bidirectional) encoder states
        proj_keys = self.Ua(keys)  # (B,L,D*H) projected encoder states
        values = keys  # (B,L,D*H) encoder states

        query = self.Wa(query)  # (B,D*H)
        query = query.unsqueeze(1).repeat(1, keys.shape[1], 1)
        attn_scores = F.tanh(query + proj_keys)  # (B,L,D*H)
        attn_scores = self.Va(attn_scores).squeeze(-1)  # BL
        attn_scores = F.softmax(attn_scores, dim=1)  # BL
        attn_scores = attn_scores.unsqueeze(2)
        attn_values = torch.mul(attn_scores, values).sum(dim=1)  # BH
        return attn_values, attn_scores  # (B,D*H), (B,L)


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.D = 2 if bidirectional else 1

    def forward(
        self, encoder_states: List[List[Tuple[Tensor]]], decoder_state: Tensor, t: int
    ):
        query = decoder_state  # (B,D*H)
        # pre-compute encoder state projections
        keys = []
        for d, direction in enumerate(encoder_states):
            keys += [
                torch.stack([state[-1][0] for state in direction], dim=1)
            ]  # (B,L,H)
        keys = torch.cat(keys, dim=-1)  # (B,L,D*H) (bidirectional) encoder states
        values = keys  # (B,L,D*H) encoder states

        query = query.unsqueeze(1).repeat(1, keys.shape[1], 1)  # (B,L,D*H)
        attn_scores = torch.mul(query, keys).sum(dim=-1)  # (B,L) dotproduct
        attn_scores /= math.sqrt(self.hidden_size)  # scaled
        attn_scores = F.softmax(attn_scores, dim=1)  # (B,L)
        attn_scores = attn_scores.unsqueeze(2)
        attn_values = torch.mul(attn_scores, values).sum(dim=1)  # BH
        return attn_values, attn_scores  # (B,D*H), (B,L)


class SelfAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.D = 2 if bidirectional else 1
        self.Q=nn.Linear(input_size, self.D*hidden_size)
        self.K=nn.Linear(input_size, self.D*hidden_size)
        self.V=nn.Linear(input_size, self.D*hidden_size)

    def forward(
        self, input, encoder_states: List[List[Tuple[Tensor]]], decoder_state: Tensor, t: int
    ):
        B,L,C=input.shape
        q=self.Q(input)
        k=self.K(input)
        v=self.V(input)
        query = self.Q(input)  # (B,D*H)
        # pre-compute encoder state projections
        keys = []
        for d, direction in enumerate(encoder_states):
            keys += [
                torch.stack([state[-1][0] for state in direction], dim=1)
            ]  # (B,L,H)
        keys = torch.cat(keys, dim=-1)  # (B,L,D*H) (bidirectional) encoder states
        values = keys  # (B,L,D*H) encoder states

        query = query.unsqueeze(1).repeat(1, keys.shape[1], 1)  # (B,L,D*H)
        attn_scores = torch.mul(query, keys).sum(dim=-1)  # (B,L) dotproduct
        attn_scores /= math.sqrt(self.hidden_size)  # scaled
        attn_scores = F.softmax(attn_scores, dim=1)  # (B,L)
        attn_scores = attn_scores.unsqueeze(2)
        attn_values = torch.mul(attn_scores, values).sum(dim=1)  # BH
        return attn_values, attn_scores  # (B,D*H), (B,L)