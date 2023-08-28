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
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from architectures.skeleton import Skeleton
from architectures.rnn import *
from architectures.cnn import *
from architectures.mlp import *
from architectures.attention import *
from architectures.init import initialize

# Reference:
# https://github.com/pytorch/pytorch/blob/main/benchmarks/fastrnns/custom_lstms.py


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 4 = input, forget, cell, output
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    def forward(
        self, x: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h0, c0 = state  # BH,BH
        gates = (
            torch.mm(x, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(h0, self.weight_hh.t())
            + self.bias_hh
        )  # B,4H
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)  # 4 chunks in dim 1

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c1 = (forgetgate * c0) + (ingate * cellgate)  # BH
        h1 = outgate * torch.tanh(c1)  # BH

        return h1, (h1, c1)


class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        # The layernorms provide learnable biases
        self.layernorm_i = nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)

    def forward(
        self, x: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h0, c0 = state
        igates = self.layernorm_i(torch.mm(x, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(h0, self.weight_hh.t()))
        gates = igates + hgates  # B,4H
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c1 = self.layernorm_c((forgetgate * c0) + (ingate * cellgate))  # BH
        h1 = outgate * torch.tanh(c1)  # BH

        return h1, (h1, c1)


class StackedLSTMCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers: int = 1,
        dropout: float = 0,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.__dict__.update(locals())
        cell = LayerNormLSTMCell if layernorm else LSTMCell
        self.stacks = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.stacks.append(cell(input_size, hidden_size))
            else:
                self.stacks.append(cell(hidden_size, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state: Tuple[Tensor]) -> Tuple[Tensor]:
        h0_all, c0_all = state
        if len(h0_all.size()) == 2:
            h0_all = h0_all.unsqueeze(0)
            c0_all = c0_all.unsqueeze(0)
        num_layers, batch_size, hidden_size = h0_all.size()
        h1_all, c1_all = [], []
        out = x
        for i, rnn in enumerate(self.stacks):
            out, (h1, c1) = rnn(out, (h0_all[i], c0_all[i]))
            h1, c1 = self.dropout(h1), self.dropout(c1)
            h1_all += [h1]
            c1_all += [c1]
        h1_all = torch.stack(h1_all)
        c1_all = torch.stack(c1_all)
        return h1_all, c1_all


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers: int = 1,
        dropout: float = 0,
        layernorm: bool = False,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.D = 2 if bidirectional else 1
        self.__dict__.update(locals())
        self.cell_1 = StackedLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layernorm=layernorm,
        )
        if bidirectional:
            self.cell_2 = StackedLSTMCell(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                layernorm=layernorm,
            )

    def forward_unidirectional(self, x, state):
        batch_size, length, input_size = x.size()
        h0, c0 = state
        if len(h0.size()) == 2:
            h0 = h0.unsqueeze(0)
            c0 = c0.unsqueeze(0)
        num_layers, batch_size, hidden_size = h0.size()
        x = x.unbind(1)
        ht, ct = h0, c0
        h_all, c_all = [], []
        for t in range(length):
            ht, ct = self.cell_1(x[t], (ht, ct))
            h_all += [ht]
            c_all += [ct]
        h_all = torch.stack(h_all)
        c_all = torch.stack(c_all)
        return h_all, c_all

    def forward_bidirectional(self, x, state):
        batch_size, length, input_size = x.size()
        h0, c0 = state
        if len(h0.size()) == 2:
            h0 = h0.unsqueeze(0)
            c0 = c0.unsqueeze(0)
        D_num_layers, batch_size, hidden_size = h0.size()
        num_layers = D_num_layers // self.D
        x = x.unbind(1)
        x_ = x[::-1]
        ht, ct = h0[:num_layers], c0[:num_layers]
        ht_, ct_ = h0[num_layers:], c0[num_layers:]
        h_all, c_all = [], []
        for t in range(length):
            ht, ct = self.cell_1(x[t], (ht, ct))
            ht_, ct_ = self.cell_2(x_[t], (ht_, ct_))
            h_all += [torch.cat([ht, ht_])]
            c_all += [torch.cat([ct, ct_])]
        h_all = torch.stack(h_all)
        c_all = torch.stack(c_all)
        return h_all, c_all

    def forward(self, x, state):
        if self.bidirectional:
            h_all, c_all = self.forward_bidirectional(x, state)
        else:
            h_all, c_all = self.forward_unidirectional(x, state)
        return h_all, c_all


class LSTMEncoder(Skeleton):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0,
        layernorm: bool = False,
    ) -> None:
        super(LSTMEncoder, self).__init__()
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.update_attributes(locals())
        self.update_init_args(locals())
        self.set_device()
        self.D = 2 if bidirectional else 1
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layernorm=layernorm,
            bidirectional=bidirectional,
        )
        self.h0 = nn.Parameter(
            torch.empty(self.D * num_layers, 1, hidden_size).normal_(mean=0, std=1e-2)
        )
        self.c0 = nn.Parameter(
            torch.empty(self.D * num_layers, 1, hidden_size).normal_(mean=0, std=1e-2)
        )
        self.register_parameter("h0", self.h0)
        self.register_parameter("c0", self.c0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, input_size = x.size()
        state = (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1))
        h_all, c_all = self.lstm.forward(x, state)
        return h_all, c_all


class LSTMDecoder(Skeleton):
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
        super().__init__()
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.update_attributes(locals())
        self.update_init_args(locals())
        self.set_device()
        self.D = 2 if bidirectional else 1
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=self.D * hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layernorm=layernorm,
            bidirectional=False,
        )
        self.mlp = []
        for _ in range(num_layers - 1):
            self.mlp.append(
                nn.Linear(
                    self.D * hidden_size,
                    self.D * hidden_size,
                )
            )
            self.mlp.append(nn.GELU())
        self.mlp.append(
            nn.Linear(
                self.D * hidden_size,
                output_size,
            )
        )
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, y, enc) -> torch.Tensor:
        batch_size, length, output_size = y.size()
        h_enc, c_enc = enc
        D_num_layers, batch_size, hidden_size = h_enc.size()
        h_all, c_all = self.lstm.forward(y, enc)
        out = self.mlp(h_all[:, -1, ...]).transpose(0, 1)
        return out, (h_all.squeeze(0), c_all.squeeze(0))


class LSTMSeq2Seq(Skeleton):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        layernorm: bool = False,
        cnn: bool = False,
        attention: str = "bahdanau",
    ) -> None:
        super(LSTMSeq2Seq, self).__init__()
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.update_attributes(locals())
        self.update_init_args(locals())
        self.set_device()
        self.D = 2 if bidirectional else 1
        if cnn:
            self.encoder_cnn = StackedResidualConvolution1D(
                input_size, n_stacks=num_layers, dropout=dropout
            )
            input_size = self.encoder_cnn.output_size
        self.encoder_lstm = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            layernorm=layernorm,
        )
        self.decoder_lstm = LSTMDecoder(
            input_size=output_size + self.D * hidden_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            layernorm=layernorm,
        )
        self.attn_encdec = {
            "bahdanau": BahdanauAttention,
            "dotproduct": DotProductAttention,
        }[attention](hidden_size=hidden_size, bidirectional=bidirectional)
        self.Ws_h = nn.Linear(self.D * hidden_size, self.D * hidden_size)
        self.Ws_c = nn.Linear(self.D * hidden_size, self.D * hidden_size)
        initialize(self)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        teacher_forcing: float = -1,
        trg_len: int = 1,
    ) -> torch.Tensor:
        batch_size, length_x, input_size = x.size()
        trg_len = y.shape[1] if y is not None else trg_len

        # encoder
        if self.cnn:
            x = self.encoder_cnn(x.transpose(1, 2))
        h_enc, c_enc = self.encoder_lstm(x)

        # decoder initialization
        h_dec = h_enc[0]
        c_dec = c_enc[0]
        if self.bidirectional:
            h_dec = torch.cat(
                [h_dec[: self.num_layers], h_dec[self.num_layers :]], dim=-1
            )
            c_dec = torch.cat(
                [c_dec[: self.num_layers], c_dec[self.num_layers :]], dim=-1
            )
            k = torch.cat(
                [h_enc[:, self.num_layers - 1, ...], h_enc[:, -1, ...]], dim=-1
            ).transpose(0, 1)
            v = k
        else:
            k = h_enc[:, -1, ...].transpose(0, 1)
            v = k
        h_dec = F.tanh(self.Ws_h(h_dec))
        c_dec = F.tanh(self.Ws_c(c_dec))

        # autoregressive decoding
        y = [yt.unsqueeze(1) for yt in y.unbind(1)] if y is not None else y
        sos = torch.zeros((batch_size, 1, self.output_size), device=x.device)
        outs = [sos]
        for t in range(trg_len):
            q = h_dec[-1].unsqueeze(1)
            attn_v, attn_scores = self.attn_encdec.forward(q=q, k=k, v=v)
            p = random.uniform(0, 1)
            if p < teacher_forcing:
                out, (h_dec, c_dec) = self.decoder_lstm(
                    torch.cat([y[t], attn_v], dim=-1), (h_dec, c_dec)
                )
            else:
                out, (h_dec, c_dec) = self.decoder_lstm(
                    torch.cat([outs[t], attn_v], dim=-1), (h_dec, c_dec)
                )
            outs += [out]
        outs = torch.cat(outs[1:], dim=1)
        return outs


if __name__ == "__main__":
    x = torch.rand(32, 100, 27)
    y = torch.rand(32, 20, 6)
    model = LSTMSeq2Seq(
        27,
        6,
        256,
        num_layers=3,
        bidirectional=True,
    )
    model.forward(x, y=None, trg_len=57)