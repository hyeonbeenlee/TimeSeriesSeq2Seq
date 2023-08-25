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
from architectures.rnn_attention import *
from architectures.init import initialize


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
        self.rnn = BidirLSTMLayer if bidirectional else ForwardLSTMLayer
        self.rnn = self.rnn(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layernorm=layernorm,
        )

    def forward(self, x: torch.Tensor, initial_states: list) -> torch.Tensor:
        B, Lin, Cin = x.shape
        assert Cin == self.input_size
        assert len(initial_states) == self.D * self.num_layers
        assert initial_states[0][0].shape[1] == self.hidden_size

        h_out, encoder_states = self.rnn(x, initial_states)
        encoder_states = [encoder_states] if not self.bidirectional else encoder_states
        return h_out, encoder_states


class LSTMDecoder(Skeleton):
    def __init__(
        self,
        output_size,
        hidden_size,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0,
        layernorm: bool = False,
        attention: bool = "none",
    ) -> None:
        super().__init__()
        assert attention == "none"
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.update_attributes(locals())
        self.update_init_args(locals())
        self.set_device()
        self.D = 2 if bidirectional else 1
        self.rnn = StackedLSTMCell(
            input_size=output_size,
            hidden_size=self.D * hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layernorm=layernorm,
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

    def forward(
        self,
        encoder_states,
        y: torch.Tensor = None,
        teacher_forcing: float = -1,
        trg_len=1,
    ) -> torch.Tensor:
        if self.bidirectional:
            decoder_state = []
            for i in range(self.num_layers):
                h0 = torch.cat(
                    [encoder_states[0][-1][i][0], encoder_states[1][-1][i][0]], dim=1
                )  # direction,step,num_layers,hc
                c0 = torch.cat(
                    [encoder_states[0][-1][i][1], encoder_states[1][-1][i][1]], dim=1
                )
                decoder_state += [(h0, c0)]
        else:
            decoder_state = encoder_states[0][-1]

        trg_len = y.shape[1] if y is not None else trg_len
        B = decoder_state[0][0].shape[0]
        sos = torch.zeros(B, self.output_size, device=decoder_state[0][0].device)
        outputs = [sos]
        # Recurrence loop
        for t in range(trg_len):
            # teacher forcing
            if random.random() < teacher_forcing and y is not None:
                B, Lout, Cout = y.shape
                assert Cout == self.output_size
                assert len(decoder_state) == self.num_layers
                assert decoder_state[0][0].shape[1] == self.D * self.hidden_size
                inputs = y.unbind(1)

                output, decoder_state = self.rnn(
                    inputs[t],
                    decoder_state,
                )
                outputs += [self.mlp(output)]
            # autoregressive
            else:
                output, decoder_state = self.rnn(
                    outputs[t],
                    decoder_state,
                )
                outputs += [self.mlp(output)]
        outputs = torch.stack(outputs, dim=1)[:, 1:, :]  # remove sos, B,Lout,Cout
        return outputs


class LSTMAttentionDecoder(Skeleton):
    def __init__(
        self,
        output_size,
        hidden_size,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0,
        layernorm: bool = False,
        attention: str = "bahdanau",
    ) -> None:
        super().__init__()
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.update_attributes(locals())
        self.update_init_args(locals())
        self.set_device()
        self.D = 2 if bidirectional else 1
        self.rnn = StackedLSTMCell(
            input_size=output_size + self.D * hidden_size,
            hidden_size=self.D * hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layernorm=layernorm,
        )
        self.attention = {
            "bahdanau": BahdanauAttention,
            "dotproduct": DotProductAttention,
        }[attention]
        self.attention = self.attention(
            hidden_size=hidden_size, bidirectional=bidirectional
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

    def forward(
        self,
        encoder_states,
        y: torch.Tensor = None,
        teacher_forcing: float = -1,
        trg_len=1,
    ) -> torch.Tensor:
        if self.bidirectional:
            decoder_state = []
            for i in range(self.num_layers):
                h0 = torch.cat(
                    [encoder_states[0][-1][i][0], encoder_states[1][-1][i][0]], dim=1
                )  # direction,step,num_layers,hc
                c0 = torch.cat(
                    [encoder_states[0][-1][i][1], encoder_states[1][-1][i][1]], dim=1
                )
                decoder_state += [(h0, c0)]
        else:
            decoder_state = encoder_states[0][-1]

        trg_len = y.shape[1] if y is not None else trg_len
        B = decoder_state[0][0].shape[0]
        sos = torch.zeros(B, self.output_size, device=decoder_state[0][0].device)
        outputs = [sos]
        # Recurrence loop
        for t in range(trg_len):
            # teacher forcing
            if random.random() < teacher_forcing and y is not None:
                B, Lout, Cout = y.shape
                assert Cout == self.output_size
                assert len(decoder_state) == self.num_layers
                assert decoder_state[0][0].shape[-1] == self.D * self.hidden_size
                inputs = y.unbind(1)
                attn_values, attn_scores = self.attention(
                    encoder_states, decoder_state[-1][0], t
                )
                decoder_input = torch.cat([inputs[t], attn_values], dim=-1)
                _, decoder_state = self.rnn(
                    decoder_input,
                    decoder_state,
                )
                outputs += [self.mlp(decoder_state[-1][0])]
            # autoregressive
            else:
                attn_values, attn_scores = self.attention(
                    encoder_states, decoder_state[-1][0], t
                )
                decoder_input = torch.cat([outputs[t], attn_values], dim=-1)
                _, decoder_state = self.rnn(
                    decoder_input,
                    decoder_state,
                )
                outputs += [self.mlp(decoder_state[-1][0])]
        outputs = torch.stack(outputs, dim=1)[:, 1:, :]  # remove sos, B,Lout,Cout
        return outputs


class LSTMSeq2Seq(Skeleton):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0,
        layernorm: bool = False,
        attention: str = "none",
    ) -> None:
        super(LSTMSeq2Seq, self).__init__()
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.update_attributes(locals())
        self.update_init_args(locals())
        self.set_device()
        self.D = 2 if bidirectional else 1
        assert attention in ["bahdanau", "dotproduct", "none"]
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            layernorm=layernorm,
        )
        self.decoder = LSTMDecoder if attention == "none" else LSTMAttentionDecoder
        self.decoder = self.decoder(
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            layernorm=layernorm,
            attention=attention,
        )
        self.init_h = Parameter(torch.randn(1, self.hidden_size), requires_grad=True)
        self.init_c = Parameter(torch.randn(1, self.hidden_size), requires_grad=True)
        initialize(self)

    def forward(
        self,
        input: torch.Tensor,
        y: torch.Tensor = None,
        teacher_forcing: float = -1,
        trg_len: int = 1,
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
        out_seq = self.decoder.forward(
            encoder_states, y, teacher_forcing=teacher_forcing, trg_len=trg_len
        )
        return out_seq


class CNNLSTMSeq2Seq(Skeleton):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0,
        layernorm: bool = False,
        attention: str = "none",
    ) -> None:
        super(CNNLSTMSeq2Seq, self).__init__()
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.update_attributes(locals())
        self.update_init_args(locals())
        self.set_device()
        self.D = 2 if bidirectional else 1
        assert attention in ["bahdanau", "dotproduct", "none"]
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
        self.decoder = LSTMDecoder if attention == "none" else LSTMAttentionDecoder
        self.decoder = self.decoder(
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            layernorm=layernorm,
            attention=attention,
        )
        self.init_h = Parameter(torch.randn(1, self.hidden_size), requires_grad=True)
        self.init_c = Parameter(torch.randn(1, self.hidden_size), requires_grad=True)
        initialize(self)

    def forward(
        self,
        input: torch.Tensor,
        y: torch.Tensor = None,
        teacher_forcing: float = -1,
        trg_len: int = 1,
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
        out = self.decoder.forward(
            encoder_states,
            y,
            teacher_forcing=teacher_forcing,
            trg_len=trg_len,
        )
        return out


def test_seq2seq():
    B = 1
    L = 1
    Cin = 1
    Cout = 1
    H = 256
    num_layers = 3
    bidirectional = True

    x = torch.randn(B, L, Cin)
    y = torch.randn(B, L, Cout)

    num_layers_ = [1, 3]
    bidirectional_ = [False, True]
    dropout_ = [0, 0.7]
    layernorm_ = [False, True]
    attention_ = [
        "bahdanau",
        "dotproduct",
    ]

    count = 1
    for num_layers in num_layers_:
        for bidirectional in bidirectional_:
            for dropout in dropout_:
                for layernorm in layernorm_:
                    for attention in attention_:
                        D = 2 if bidirectional else 1
                        initial_states = [
                            (torch.zeros(B, H), torch.zeros(B, H))
                            for i in range(D * num_layers)
                        ]
                        model = LSTMSeq2Seq(
                            Cin,
                            Cout,
                            H,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            layernorm=layernorm,
                            attention=attention,
                        )
                        t1 = time.perf_counter()
                        # TF
                        model.forward(x, y, teacher_forcing=0.5)
                        t2 = time.perf_counter()
                        print(f"{t2-t1}")
                        # Fully-Autoregressive
                        t1 = time.perf_counter()
                        model.forward(x, trg_len=20)
                        t2 = time.perf_counter()
                        print(f"{t2-t1}")

                        writer = SummaryWriter(
                            f"architectures/visualization/model{count:02d}"
                        )
                        writer.add_graph(model, x)
                        count += 1


def test_cnnlstm():
    B = 32
    Cin = 27
    Cout = 6
    L = 20

    x = torch.randn(B, L, Cin)
    net = CNNLSTMSeq2Seq(
        input_size=Cin,
        output_size=Cout,
        hidden_size=256,
        num_layers=4,
        bidirectional=False,
        attention="none",
    )
    out = net(x)


if __name__ == "__main__":
    test_seq2seq()
    test_cnnlstm()
