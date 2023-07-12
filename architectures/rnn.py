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
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state  # BH,BH
        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )  # B,4H
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)  # 4 chunks in dim 1

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)  # BH
        hy = outgate * torch.tanh(cy)  # BH

        return hy, (hy, cy)


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
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates  # B,4H
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))  # BH
        hy = outgate * torch.tanh(cy)  # BH

        return hy, (hy, cy)


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
        self.stacks = nn.ModuleList()
        cell = LayerNormLSTMCell if layernorm else LSTMCell
        for i in range(num_layers):
            if i == 0:
                self.stacks.append(cell(input_size, hidden_size))
            else:
                self.stacks.append(cell(hidden_size, hidden_size))

        # if num_layers == 1:
        #     warnings.warn(
        #         "dropout lstm adds dropout layers after all but last "
        #         "recurrent layer, it expects num_layers greater than "
        #         "1, but got num_layers = 1"
        #     )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, states: Tuple[Tensor]) -> Tuple[Tensor]:
        output_states = []
        output = x  # B,Cin
        for i, rnn in enumerate(self.stacks):
            state = states[i]  # (hx,cx) each shape of BH
            output, out_state = rnn(output, state)
            # if i < self.num_layers - 1:
            output = self.dropout(output)
            out_state=(output, out_state[1])
            output_states += [out_state]
        return output, output_states  # last layer hx, all layers (hx,cx)
nn.LSTM

class ForwardLSTMLayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers: int = 1,
        dropout: float = 0,
        layernorm: bool = False,
    ):
        # Recurrence-looping module
        super().__init__()
        self.__dict__.update(locals())
        self.cell = StackedLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layernorm=layernorm,
        )

    def forward(
        self, input: Tensor, initial_states: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(1)  # B,L,Hin -> [BHin1, BHin2, ..., BHinL]
        outputs = []
        state = initial_states
        states = []
        for i in range(len(inputs)):
            h_out, state = self.cell.forward(inputs[i], state)
            outputs += [h_out]
            states += [state]  # states of all layers
        return (
            torch.stack(outputs, dim=1),
            states,
        )  # last stack hx*L, (steps,num_layers,(hx,cx))


def reverse(lst: List[Tensor]) -> List[Tensor]:
    return lst[::-1]


class BackwardLSTMLayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers: int = 1,
        dropout: float = 0,
        layernorm: bool = False,
    ):
        # Recurrence-looping module
        super().__init__()
        self.__dict__.update(locals())
        self.cell = StackedLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layernorm=layernorm,
        )

    def forward(
        self, input: Tensor, initial_states: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = reverse(input.unbind(1))  # B,L,Hin -> [BHin1, BHin2, ..., BHinL]
        outputs = []
        state = initial_states
        states = []
        for i in range(len(inputs)):
            h_out, state = self.cell.forward(inputs[i], state)
            outputs += [h_out]
            states += [state]  # states of all layers
        return (
            torch.stack(outputs, dim=1),
            states,
        )  # last stack hx*L, (steps,num_layers,(hx,cx))


class BidirLSTMLayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers: int = 1,
        dropout: float = 0,
        layernorm: bool = False,
    ):
        # Recurrence-looping module
        super().__init__()
        self.__dict__.update(locals())
        self.directions = nn.ModuleList(
            [
                ForwardLSTMLayer(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    layernorm=layernorm,
                ),
                BackwardLSTMLayer(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    layernorm=layernorm,
                ),
            ]
        )

    def forward(
        self, input: Tensor, initial_states: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        outputs = []
        output_states = []
        initial_state = initial_states
        for i, direction in enumerate(self.directions):
            initial_state = initial_states[
                i * self.num_layers : (i + 1) * self.num_layers
            ]
            h_out, out_state = direction(
                input, initial_state
            )  # last layer, (steps,num_layers,(hx,cx))
            outputs += [h_out]
            output_states += [out_state]
        return outputs, output_states  # last layer, (steps,num_layers,(hx,cx))
