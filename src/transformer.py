import copy
import math

import numpy as np

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class Embed(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embed, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return x + self.dropout(self.pe[:, :x.size(1)])


class Decoder(nn.Module):
    def __init__(self, layer, norm, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = norm

    def forward(self, src_feature, src_mask, tgt_embed, tgt_mask):
        for layer in self.layers:
            tgt_embed = layer(src_feature, src_mask, tgt_embed, tgt_mask)
        return self.norm(tgt_embed)


def clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, norm, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = norm

    def forward(self, src_embed, src_mask):
        for layer in self.layers:
            src_embed = layer(src_embed, src_mask)
        return self.norm(src_embed)


class EncoderLayer(nn.Module):
    def __init__(self, attn, ff, su):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.ff = ff
        self.su = clone(su, 2)

    def forward(self, src, src_mask):
        src = self.su[0](src, lambda x: self.attn(src, src, src, src_mask))
        return self.su[1](src, self.ff)


class DecoderLayer(nn.Module):
    def __init__(self, attn, attn2, ff, su):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.attn2 = attn2
        self.ff = ff
        self.su = clone(su, 3)

    def forward(self, src, src_mask, tgt, tgt_mask):
        tgt = self.su[0](tgt, lambda x: self.attn(x, x, x, tgt_mask))
        tgt = self.su[1](tgt, lambda x: self.attn2(x, src, src, src_mask))
        return self.su[2](tgt, self.ff)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiheadAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert d_model % h == 0
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h

        self.w = clone(nn.Linear(d_model, d_model), 3)
        self.out = nn.Linear(d_model, d_model)

        self.attn = None

    def forward(self, q, k, v, mask):
        b, n, d_model = list(q.shape)
        q, k, v = [rearrange(f(x), 'b n (h d) -> b h n d', h=self.h, d=self.d_k) for f, x in zip(self.w, [q, k, v])]
        score = q @ k.transpose(-1, -2)
        if mask is not None:
            score = score.masked_fill(mask, -1e9)
        self.attn = attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)
        x = rearrange(attn @ v, 'b h n d -> b n (h d)', h=self.h, d=self.d_k)
        x = self.out(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2*(x-mean)/(std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))


class Transformer(nn.Module):
    def __init__(self, src_emb: nn.Sequential, tgt_emb: nn.Sequential, encoder: Encoder, decoder: Decoder,
                 generator: nn.Linear):
        super(Transformer, self).__init__()
        self.generator = generator
        self.decoder = decoder
        self.encoder = encoder
        self.tgt_emb = tgt_emb
        self.src_emb = src_emb

    def forward(self, src, src_mask, tgt, tgt_mask):
        src_feature = self.encode(src, src_mask)
        tgt_feature = self.decode(src_feature, src_mask, tgt, tgt_mask)
        out = self.generator(tgt_feature)
        return out

    def decode(self, src_feature, src_mask, tgt, tgt_mask):
        tgt_embed = self.tgt_emb(tgt)
        tgt_feature = self.decoder(src_feature, src_mask, tgt_embed, tgt_mask)
        return tgt_feature

    def encode(self, src, src_mask):
        src_embed = self.src_emb(src)
        src_feature = self.encoder(src_embed, src_mask)
        return src_feature

def transformer(src_vocab_size, tgt_vocab_size, d_model=512, h=2, d_ff=20, N=6):
    x = torch.arange(100).view(10, 10)
    mask = torch.zeros(10, 1, 10, 10)
    tgt = torch.arange(1, 101).view(10, 10)
    tgt_mask = (torch.zeros(10, 1, 10, 10) == 1) | (torch.from_numpy(np.triu(np.ones((10, 10)), 1)) == 1)

    c = copy.deepcopy

    src_emb = nn.Sequential(Embed(d_model, src_vocab_size), PositionalEncoding(d_model))
    tgt_emb = nn.Sequential(Embed(d_model, tgt_vocab_size), PositionalEncoding(d_model))

    attn = MultiheadAttention(d_model, h, dropout=0.1)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout=0.1)
    connection = SublayerConnection(d_model, dropout=0.1)
    norm = LayerNorm(d_model)

    encoder_layer = EncoderLayer(c(attn), c(ff), c(connection))
    encoder = Encoder(encoder_layer, c(norm), N)

    decoder_layer = DecoderLayer(c(attn), c(attn), c(ff), c(connection))
    decoder = Decoder(decoder_layer, c(norm), N)

    generator = nn.Linear(d_model, tgt_vocab_size)
    model = Transformer(src_emb, tgt_emb, encoder, decoder, generator)
    out = model(x, mask, tgt, tgt_mask)
    assert list(out.shape) == [10, 10, 101]

    return model
