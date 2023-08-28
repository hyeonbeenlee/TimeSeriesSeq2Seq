import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

import os, sys

sys.path.append(os.getcwd())
from architectures.skeleton import Skeleton
from architectures.rnn import StackedLSTMCell
from architectures.init import initialize

"""
References:
Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting, 37(4), 1748-1764.
"""


class GRN(Skeleton):
    def __init__(self, d_model, dropout) -> None:
        super(GRN, self).__init__()
        self.__dict__.update(locals())
        self.update_init_args(locals())
        self.update_attributes(locals())
        self.set_device()
        self.norm = nn.LayerNorm(d_model)
        self.glu = GLU(d_model)
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        initialize(self)

    def forward(self, x, context=None):
        if context is None:
            context = torch.zeros_like(x, device=x.device)
        elif len(x.size()) == 4 and len(context.size()) == 3:
            context = context.unsqueeze(2)
        out = F.elu(self.w2(x) + self.w3(context))
        out = self.dropout(self.w1(out))
        out = self.norm(x + self.glu(out))
        return out


class GLU(Skeleton):
    """
    Component gating layers based on Gated Linear Units by Dauphin et al.(2017)
    if GLU outputs -> 0, nonlinear(ELU in GRN) processing is supressed.
    """

    def __init__(self, d_model) -> None:
        super(GLU, self).__init__()
        self.__dict__.update(locals())
        self.update_init_args(locals())
        self.update_attributes(locals())
        self.set_device()
        self.w4 = nn.Linear(d_model, d_model)
        self.w5 = nn.Linear(d_model, d_model)
        initialize(self)

    def forward(self, x):
        gates = F.sigmoid(self.w4(x))
        feature = self.w5(x)
        return torch.mul(gates, feature)


class VariableSelection(Skeleton):
    def __init__(self, n_features, d_model, dropout) -> None:
        super(VariableSelection, self).__init__()
        self.__dict__.update(locals())
        self.update_init_args(locals())
        self.update_attributes(locals())
        self.set_device()
        self.grn_w = GRN(d_model, dropout)
        self.grn_f = nn.ModuleList([GRN(d_model, dropout) for _ in range(n_features)])
        self.transform1 = nn.Linear(1, d_model)
        self.transform2 = nn.Linear(d_model, 1)

    def forward(self, x, context=None):
        x = self.transform1(x.unsqueeze(3))
        weights = self.grn_w(x, context)
        weights = self.transform2(weights)
        weights = F.softmax(weights, dim=2)
        f = torch.stack(
            [self.grn_f[j](x[..., j, :]) for j in range(self.n_features)], dim=2
        )
        out = torch.mul(weights, f).sum(dim=2)
        return out, weights


class InterpretableMultiHeadAttention(Skeleton):
    def __init__(self, d_model, n_heads) -> None:
        super(InterpretableMultiHeadAttention, self).__init__()
        self.__dict__.update(locals())
        self.update_init_args(locals())
        self.update_attributes(locals())
        self.set_device()
        self.d_attn = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, self.d_attn)
        self.w_o = nn.Linear(self.d_attn, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def split(self, q, k, v):
        batch_size_q, length_q, d_model_q = q.size()
        batch_size_k, length_k, d_model_k = k.size()
        batch_size_v, length_v, d_model_v = v.size()
        # Parallel projections d_model -> d_k
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        # Divide into batch, head, length, d_k
        q = q.reshape(batch_size_q, length_q, self.n_heads, self.d_attn).transpose(1, 2)
        k = k.reshape(batch_size_k, length_k, self.n_heads, self.d_attn).transpose(1, 2)
        v = v.reshape(batch_size_v, length_v, 1, self.d_attn).transpose(1, 2)
        return q, k, v

    def forward(self, q, k, v, mask=None):
        batch_size, length, d_model = q.size()
        q, k, v = self.split(q, k, v)
        batch_size, n_heads, length, d_k = q.size()
        attn_score = torch.einsum("bhLd,bhld->bhLl", q, k)  # matmul
        attn_score /= math.sqrt(d_k)  # scale
        attn_score = attn_score.mean(dim=1, keepdim=True)  # mean

        # Mask
        if mask is not None:
            # mask = torch.tril(torch.ones(length, length, device=q.device))
            mask = mask.expand(batch_size, 1, length, length)
            attn_score = attn_score.masked_fill_(mask == 0, -1e12)

        # Attend values
        attn_score = self.softmax(attn_score)
        attn_v = torch.matmul(attn_score, v)
        attn_v = self.w_o(attn_v).squeeze(1)

        return attn_v, attn_score


class StaticCovariateEncoder(Skeleton):
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        # z: observed (unknown in future)
        # x: known (in future)
        super(StaticCovariateEncoder, self).__init__()
        self.__dict__.update(locals())
        self.update_init_args(locals())
        self.update_attributes(locals())
        self.set_device()
        self.vs_s = VariableSelection(input_size, d_model, dropout)
        self.grn = GRN(d_model, dropout)

    def forward(self, x):
        contxt_static, _ = self.vs_s(x)
        contxt_static = self.grn(contxt_static).mean(dim=1, keepdim=True)
        return contxt_static


class TFT_LSTMEncoder(Skeleton):
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        # z: observed (unknown in future)
        # x: known (in future)
        super(TFT_LSTMEncoder, self).__init__()
        self.__dict__.update(locals())
        self.update_init_args(locals())
        self.update_attributes(locals())
        self.set_device()
        self.vs_x = VariableSelection(input_size, d_model, dropout)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.glu = GLU(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, c_s, c_hc):
        batch_size, length, input_size = x.size()
        contxt_x, _ = self.vs_x(x, c_s)
        h0, c0 = c_hc.transpose(0, 1).repeat(self.num_layers, 1, 1).chunk(2, dim=-1)
        contxt_temporal, enc_last = self.lstm.forward(
            contxt_x, (h0.contiguous(), c0.contiguous())
        )
        enc_all = self.norm(contxt_x + self.glu(contxt_temporal))
        return enc_all, enc_last


class TFT_LSTMDecoder(Skeleton):
    def __init__(
        self,
        output_size: int,
        d_model: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        # z: observed (unknown in future)
        # x: known (in future)
        super(TFT_LSTMDecoder, self).__init__()
        self.__dict__.update(locals())
        self.update_init_args(locals())
        self.update_attributes(locals())
        self.set_device()
        self.vs_y = VariableSelection(output_size, d_model, dropout)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.glu = GLU(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, y, enc_last, c_s):
        batch_size, length, input_size = y.size()
        contxt_y, _ = self.vs_y(y, c_s)
        contxt_temporal, dec_last = self.lstm.forward(
            contxt_y,
            enc_last,
        )
        dec_all = self.norm(contxt_y + self.glu(contxt_temporal))
        return dec_all


class TemporalFusionTransformer(Skeleton):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 256,
        num_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """
        Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting, 37(4), 1748-1764.

        Modifications made by Hyeonbeen(me):
        1. Static metadata inputs S => Temporal mean of past time-varying inputs
        2. Known future time-varying inputs x in LSTM decoder => Outputs of TFT (autoregressive)

        Args:
            input_size (int): dimension of time-varying input data
            output_size (int): dimension of time-varying output data
            d_model (int, optional): base dimension of model. Defaults to 256.
            num_layers (int, optional): number of layers in LSTM encoder-decoder. Defaults to 1.
            n_heads (int, optional): number of heads in multiheaded attention layer. Defaults to 8.
            dropout (float, optional): dropout rate. Defaults to 0.1.
        """
        super(TemporalFusionTransformer, self).__init__()
        self.__dict__.update(locals())
        self.update_init_args(locals())
        self.update_attributes(locals())
        self.set_device()
        self.static_enc1 = StaticCovariateEncoder(
            input_size, d_model, num_layers, dropout
        )
        self.static_enc2 = StaticCovariateEncoder(
            input_size, 2 * d_model, num_layers, dropout
        )
        self.static_enc3 = StaticCovariateEncoder(
            input_size, d_model, num_layers, dropout
        )
        self.static_enrichment = GRN(d_model, dropout)
        self.lstm_encoder = TFT_LSTMEncoder(input_size, d_model, num_layers, dropout)
        self.lstm_decoder = TFT_LSTMDecoder(output_size, d_model, num_layers, dropout)
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.glu1 = GLU(d_model)
        self.glu2 = GLU(d_model)
        self.grn = GRN(d_model, dropout)
        self.dense = nn.Linear(d_model, output_size)
        initialize(self)

    def encode(self, x):
        batch_size, length_x, input_size = x.size()

        # sequence-to-sequence
        c_s = self.static_enc1(x)
        c_hc = self.static_enc2(x)
        c_e = self.static_enc3(x)
        enc_all, enc_last = self.lstm_encoder.forward(x, c_s, c_hc)
        return (enc_all, enc_last), (c_s, c_hc, c_e)

    def decode(self, y, enc_states, cell_states):
        enc_all, enc_last = enc_states
        c_s, c_hc, c_e = cell_states
        batch_size, length_x, d_model = enc_all.size()
        batch_size, length_y, d_model = y.size()

        # sequence-to-sequence
        dec_all = self.lstm_decoder.forward(y, enc_last, c_s)

        # static enrichment layer
        phi = torch.cat([enc_all, dec_all], dim=1)
        qkv = self.static_enrichment.forward(phi, c_e)

        # temporal self-attention
        mask = torch.ones(length_x + length_y, length_x + length_y, device=y.device)
        mask[length_x:, length_x:] = torch.tril(
            torch.ones(length_y, length_y, device=y.device)
        )
        attn_v, attn_scores = self.attention.forward(q=qkv, k=qkv, v=qkv, mask=mask)
        attn_v = self.norm1(qkv + self.glu1(attn_v))

        # positionwise feed-forward
        attn_v = self.grn.forward(attn_v)
        attn_v = self.norm2(phi + self.glu2(attn_v))

        # dense, future outputs only
        out = self.dense(attn_v[:, length_x:, :])
        return out

    def forward(self, x, y=None, trg_len: int = 1, teacher_forcing: float = -1):
        batch_size, length_x, input_size = x.size()
        trg_len = y.shape[1] if y is not None else trg_len

        sos = torch.zeros(batch_size, 1, self.output_size, device=x.device)
        dec_input = [sos]
        enc_states, cell_states = self.encode(x)
        for t in range(trg_len):
            out = self.decode(torch.cat(dec_input, dim=1), enc_states, cell_states)
            p = random.uniform(0, 1)
            if p < teacher_forcing and y is not None:
                dec_input += [y[:, t, :].unsqueeze(1)]
            else:
                dec_input += [out[:, -1, :].unsqueeze(1)]
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    length_x = 100
    input_size = 27
    length_y = 20
    output_size = 6
    x = torch.rand(batch_size, length_x, input_size, device=device)
    y = torch.rand(batch_size, length_y, output_size, device=device)

    model = TemporalFusionTransformer(
        input_size=input_size, output_size=output_size
    ).to(device)
    out = model.forward(x, y)  # with decoder inputs
    out.mean().backward()
    out = model.forward(x, trg_len=50)  # autoregressive
    out.mean().backward()
