import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

import os, sys

sys.path.append(os.getcwd())
from architectures.skeleton import Skeleton
from architectures.init import initialize

"""
References:
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
https://github.com/hyunwoongko/transformer
https://github.com/UdbhavPrasad072300/Transformer-Implementations
"""


class LayerNorm(nn.Module):
    def __init__(self, d_model: int = 512, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.__dict__.update(locals())
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super(MultiHeadAttention, self).__init__()
        self.__dict__.update(locals())
        self.d_k = d_model // n_heads
        assert round(self.d_k * self.n_heads) == d_model
        self.w_q = nn.Linear(d_model, d_model)  # parallel projection and divide
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
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
        q = q.reshape(batch_size_q, length_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch_size_k, length_k, self.n_heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch_size_v, length_v, self.n_heads, self.d_k).transpose(1, 2)
        return q, k, v

    def forward(self, q, k, v, mask=None):
        batch_size, length, d_model = q.size()
        q, k, v = self.split(q, k, v)
        batch_size, n_heads, length, d_k = q.size()
        attn_score = torch.einsum("bhLd,bhld->bhLl", q, k)  # matmul
        attn_score /= math.sqrt(d_k)  # scale

        # Mask
        if mask is not None:
            mask = mask.expand(batch_size, n_heads, length, length)
            attn_score = attn_score.masked_fill_(mask == 0, -1e12)

        # Attend values
        attn_score = self.softmax(attn_score)
        attn_v = torch.matmul(attn_score, v)

        # Concat heads and project
        attn_v = attn_v.reshape(batch_size, length, d_model)
        attn_v = self.w_o(attn_v)

        return attn_v, attn_score


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super(PositionwiseFeedForward, self).__init__()
        self.__dict__.update(locals())
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # d_model -> d_ff -> d_model
        x = self.ff(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 1024):
        super(PositionalEncoding, self).__init__()
        self.__dict__.update(locals())
        pos = torch.arange(0, max_len).float().unsqueeze(dim=1)
        self.pe = torch.zeros(max_len, d_model)
        evens = torch.arange(0, d_model, 2).float()
        self.pe[:, 0::2] = torch.sin(pos / (10000 ** (evens / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (evens / d_model)))
        assert torch.unique(self.pe, dim=0).shape[0] == max_len

    def forward(self, ebd_x):
        batch_size, length, d_model = ebd_x.size()
        pe = self.pe[:length, :].unsqueeze(0).repeat(batch_size, 1, 1).to(ebd_x.device)
        return pe


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.__dict__.update(locals())
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        initialize(self)

    def forward(self, ebd_x):
        batch_size, length, d_model = ebd_x.size()
        out1, score1 = self.attn(q=ebd_x, k=ebd_x, v=ebd_x, mask=None)
        out1 = self.norm1(ebd_x + self.dropout(out1))
        out2 = self.ff(out1)
        out2 = self.norm2(out1 + self.dropout(out2))
        return out2


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ) -> None:
        super(TransformerDecoder, self).__init__()
        self.__dict__.update(locals())
        self.attn_masked = MultiHeadAttention(d_model, n_heads)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        initialize(self)

    def forward(self, enc, ebd_y):
        batch_size, length_x, d_model = enc.size()
        batch_size, length_y, d_model = ebd_y.size()
        mask = torch.tril(torch.ones(length_y, length_y, device=enc.device))
        out1, score1 = self.attn_masked(q=ebd_y, k=ebd_y, v=ebd_y, mask=mask)
        out1 = self.norm1(ebd_y + self.dropout(out1))
        out2, score2 = self.attn(q=out1, k=enc, v=enc, mask=None)
        out2 = self.norm2(out1 + self.dropout(out2))
        out3 = self.ff(out2)
        out3 = self.norm3(out2 + self.dropout(out3))
        return out3


class Transformer(Skeleton):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ) -> None:
        super(Transformer, self).__init__()
        self.__dict__.update(locals())
        self.update_init_args(locals())
        self.update_attributes(locals())
        self.set_device()

        self.embed_x = nn.Linear(input_size, d_model)
        self.embed_y = nn.Linear(output_size, d_model)
        self.linear = nn.Linear(d_model, output_size)
        self.pe_x = PositionalEncoding(d_model)
        self.pe_y = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            self.encoders.append(TransformerEncoder(d_model, n_heads, dropout, d_ff))
            self.decoders.append(TransformerDecoder(d_model, n_heads, dropout, d_ff))
        initialize(self)

    def embedding_x(self, x):
        return self.dropout(self.embed_x(x) * math.sqrt(self.d_model) + self.pe_x(x))

    def embedding_y(self, y):
        return self.dropout(self.embed_y(y) * math.sqrt(self.d_model) + self.pe_y(y))

    def encode(self, x):
        batch_size, length_x, input_size = x.size()
        ebd_x = self.embedding_x(x)
        for encoder in self.encoders:
            ebd_x = encoder(ebd_x)
        return ebd_x

    def decode(self, ebd_x, y):
        batch_size, length_x, d_model = ebd_x.size()
        batch_size, length_y, output_size = y.size()
        ebd_y = self.embedding_y(y)
        for decoder in self.decoders:
            ebd_y = decoder.forward(ebd_x, ebd_y)
        out = self.linear(ebd_y)
        return out

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        teacher_forcing: float = -1,
        trg_len: int = 1,
    ):
        batch_size, length_x, input_size = x.size()
        ebd_x = self.encode(x)
        sos = torch.zeros(batch_size, 1, self.output_size, device=x.device)
        dec_input = [sos]
        for t in range(trg_len):
            out = self.decode(ebd_x, torch.cat(dec_input, dim=1))
            p = random.uniform(0, 1)
            if p < teacher_forcing and y is not None:
                dec_input += [y[:, t, :].unsqueeze(1)]
            else:
                dec_input += [out[:, -1, :].unsqueeze(1)]
        return out  # (batch_size, trg_len, output_size)


class nnTransformer(Skeleton):
    def __init__(
        self,
        input_size,
        output_size,
        num_layers: int = 6,
        n_heads: int = 8,
        d_model: int = 512,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ) -> None:
        super(nnTransformer, self).__init__()
        self.__dict__.update(locals())
        self.update_init_args(locals())
        self.update_attributes(locals())
        self.set_device()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.embed_x = nn.Linear(input_size, d_model)
        self.embed_y = nn.Linear(output_size, d_model)
        self.linear = nn.Linear(d_model, output_size)
        for i in range(num_layers):
            self.encoders.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation=nn.GELU(),
                    batch_first=True,
                )
            )
            self.decoders.append(
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation=nn.GELU(),
                    batch_first=True,
                )
            )
        initialize(self)

    def encode(self, x):
        x = self.embed_x(x)
        for encoder in self.encoders:
            x = encoder(x)
        return x

    def decode(self, enc, y):
        y = self.embed_y(y)
        for decoder in self.decoders:
            y = decoder(y, enc)
        y = self.linear(y)
        return y

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        teacher_forcing: float = -1,
        trg_len: int = 1,
    ):
        batch_size, length_x, input_size = x.size()
        ebd_x = self.encode(x)
        sos = torch.zeros(batch_size, 1, self.output_size, device=x.device)
        dec_input = [sos]
        for t in range(trg_len):
            out = self.decode(ebd_x, torch.cat(dec_input, dim=1))
            p = random.uniform(0, 1)
            if p < teacher_forcing and y is not None:
                dec_input += [y[:, t, :].unsqueeze(1)]
            else:
                dec_input += [out[:, -1, :].unsqueeze(1)]
        return out  # (batch_size, trg_len, output_size)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size=16
    length_x=100
    input_size=27
    length_y=20
    output_size=6
    x = torch.rand(batch_size, length_x, input_size, device=device)
    y = torch.rand(batch_size, length_y, output_size, device=device)

    model = nnTransformer(input_size=input_size,output_size=output_size).to(device)
    out = model.forward(x, y) # with decoder inputs
    out.mean().backward()
    out=model.forward(x, trg_len=50) # autoregressive
    out.mean().backward()
    
    model=Transformer(input_size=input_size,output_size=output_size).to(device)
    out=model.forward(x,y) # with decoder inputs
    out.mean().backward()
    out=model.forward(x, trg_len=50) # autoregressive
    out.mean().backward()