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
from architectures.rnn_seq2seq import *


class LSTMSeq2Point(Skeleton):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0,
        layernorm: bool = False,
    ) -> None:
        super(LSTMSeq2Point, self).__init__()
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.update_attributes(locals())
        self.update_init_args(locals())
        self.set_device()
        self.D = 2 if bidirectional else 1
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            layernorm=layernorm,
        )
        self.decoder = []
        for i in range(num_layers - 1):
            self.decoder += [nn.Linear(self.D * hidden_size, self.D * hidden_size)]
        self.decoder += [nn.Linear(self.D * hidden_size, output_size)]
        self.decoder = nn.Sequential(*self.decoder)
        self.init_h = Parameter(torch.randn(1, self.hidden_size), requires_grad=True)
        self.init_c = Parameter(torch.randn(1, self.hidden_size), requires_grad=True)
        initialize(self)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        B, Lin, Cin = input.shape
        initial_states = [
            (
                self.init_h.to(input.device).repeat(B, 1),
                self.init_c.to(input.device).repeat(B, 1),
            )
            for _ in range(self.D * self.num_layers)
        ]
        assert len(initial_states) == self.D * self.num_layers
        # RNN encoder
        # h_out: [forward, backward] hidden states of the last layer
        # encoder_states: [forward, backward] hidden and cell states of all steps and layers
        # each contains Lin * num_layers * (hidden,cell)
        # input.shape == (B,L,Cin)

        h_out, encoder_states = self.encoder.forward(input, initial_states)

        # RNN decoder
        # out_seq.shape == (B,lout,Cout) if y is None else (B,y.shape[1],Cout)
        if self.bidirectional:
            encoder_state = torch.cat(
                [encoder_states[0][-1][-1][0], encoder_states[1][-1][-1][0]], dim=-1
            )  # direction, step, num_layers, (hx,cx)
        else:
            encoder_state = encoder_states[0][-1][-1][0]
        out = self.decoder.forward(encoder_state)
        return out


class CNNLSTMSeq2Point(Skeleton):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0,
        layernorm: bool = False,
    ) -> None:
        super(CNNLSTMSeq2Point, self).__init__()
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.update_attributes(locals())
        self.update_init_args(locals())
        self.set_device()
        self.D = 2 if bidirectional else 1
        self.encoder_cnn = StackedResidualConvolution(
            input_size, n_stacks=num_layers, dropout=dropout
        )
        self.encoder_rnn = LSTMEncoder(
            input_size=self.encoder_cnn.output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            layernorm=layernorm,
        )
        self.decoder = []
        for i in range(num_layers - 1):
            self.decoder += [nn.Linear(self.D * hidden_size, self.D * hidden_size)]
        self.decoder += [nn.Linear(self.D * hidden_size, output_size)]
        self.decoder = nn.Sequential(*self.decoder)
        self.init_h = Parameter(torch.randn(1, self.hidden_size), requires_grad=True)
        self.init_c = Parameter(torch.randn(1, self.hidden_size), requires_grad=True)
        initialize(self)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        B, Lin, Cin = input.shape
        out = self.encoder_cnn(input.permute(0, 2, 1))  # (B,Cin,Lin) -> (B,Lout,Cout)

        initial_states = [
            (
                self.init_h.to(out.device).repeat(B, 1),
                self.init_c.to(out.device).repeat(B, 1),
            )
            for _ in range(self.D * self.num_layers)
        ]
        assert len(initial_states) == self.D * self.num_layers
        # RNN encoder
        # _: [forward, backward] hidden states of the last layer
        # encoder_states: [forward, backward] hidden and cell states of all steps and layers
        # each contains Lin * num_layers * (hidden,cell)
        # input.shape == (B,L,Cin)

        _, encoder_states = self.encoder_rnn.forward(out, initial_states)

        # RNN decoder
        # out_seq.shape == (B,lout,Cout) if y is None else (B,y.shape[1],Cout)
        if self.bidirectional:
            encoder_state = torch.cat(
                [encoder_states[0][-1][-1][0], encoder_states[1][-1][-1][0]], dim=-1
            )  # direction, step, num_layers, (hx,cx)
        else:
            encoder_state = encoder_states[0][-1][-1][0]
        out = self.decoder.forward(encoder_state)
        return out


if __name__ == "__main__":
    B, L, Cin, Cout = 16, 20, 27, 6
    x = torch.randn(B, L, Cin)
    net = CNNLSTMSeq2Point(Cin, Cout, 256, 3, bidirectional=True)
    for i in range(10):
        t1 = time.perf_counter()
        net(x)
        t2 = time.perf_counter()
        print(t2 - t1)
