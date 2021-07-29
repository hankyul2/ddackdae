import copy
import math

import numpy as np

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, 1).unsqueeze(1)
        mul_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, ::2] = torch.sin(position * mul_term)
        pe[:, 1::2] = torch.cos(position * mul_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)].clone().detach())


def clone(layer, n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model//h
        self.w = clone(nn.Linear(d_model, d_model), 3)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        q, k, v = [rearrange(f(x), 'b s (h k) -> b h s k', h=self.h) for f, x in zip(self.w, [q, k, v])]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        scores = torch.masked_fill(scores, mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        w_v = attn @ v
        c_v = rearrange(w_v, 'b h s k -> b s (h k)')
        return self.out(c_v)


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x)))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, attn, ff, su):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.ff = ff
        self.su = clone(su, 2)

    def forward(self, src, src_mask):
        src = self.su[0](src, lambda x: self.attn(x, x, x, src_mask))
        return self.su[0](src, self.ff)


class DecoderLayer(nn.Module):
    def __init__(self, attn, attn2, ff, su):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.attn2 = attn2
        self.ff = ff
        self.su = clone(su, 3)

    def forward(self, src, tgt, src_mask, tgt_mask):
        tgt = self.su[0](tgt, lambda x: self.attn(x, x, x, tgt_mask))
        tgt = self.su[1](tgt, lambda x: self.attn2(x, src, src, src_mask))
        return self.su[1](tgt, self.ff)


class Encoder(nn.Module):
    def __init__(self, layer, norm, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = norm

    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)


class Decoder(nn.Module):
    def __init__(self, layer, norm, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = norm

    def forward(self, src, tgt, src_mask, tgt_mask):
        for layer in self.layers:
            tgt = layer(src, tgt, src_mask, tgt_mask)
        return self.norm(tgt)


class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_rep = self.encode(src, src_mask)
        tgt_rep = self.decode(src_rep, tgt, src_mask, tgt_mask)
        return self.generator(tgt_rep)

    def decode(self, src_rep, tgt, src_mask, tgt_mask):
        return self.decoder(src_rep, self.tgt_embed(tgt), src_mask, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def greedy_decode(self, src, src_mask, device, max_len=20, BOS=1, EOS=None):
        memory = self.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(BOS).type_as(src.data)
        for i in range(max_len - 1):
            out = self.decode(memory, ys, src_mask, get_seq_mask(ys).to(device))
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == EOS:
                break
        return ys


def get_model(src_vocab_size, tgt_vocab_size, d_model=512, N=6, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy

    src_embed = nn.Sequential(Embedding(src_vocab_size, d_model), PositionalEncoding(d_model, dropout=dropout))
    tgt_embed = nn.Sequential(Embedding(tgt_vocab_size, d_model), PositionalEncoding(d_model, dropout=dropout))

    attn = MultiheadAttention(d_model, h, dropout=dropout)
    ff = PositionwiseFeedforward(d_model, d_ff, dropout=dropout)
    norm = LayerNorm(d_model)
    su = SublayerConnection(d_model, dropout=dropout)

    encoder = Encoder(EncoderLayer(c(attn), c(ff), c(su)), c(norm), N)
    decoder = Decoder(DecoderLayer(c(attn), c(attn), c(ff), c(su)), c(norm), N)
    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(src_embed, tgt_embed, encoder, decoder, generator)

    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param.data)

    return model


def get_sample_input(pad_idx=0, batch_size=10):
    # src = torch.arange(2, 102, 1).view(10, 10)
    # tgt_output = torch.arange(3, 103, 1).view(10, 10)
    # tgt_input = torch.cat([torch.ones(10).unsqueeze(1), tgt_output[:, :-1].clone().detach()], dim=1).long()
    src = torch.cat([torch.ones(batch_size).unsqueeze(1), torch.randint(2, 12, size=(batch_size, 9))], dim=1).long()
    tgt_output = src.clone().detach()
    tgt_input = torch.cat([torch.ones(batch_size).unsqueeze(1), tgt_output[:, :-1].clone().detach()], dim=1).long()
    src_mask, tgt_mask = get_mask(src, tgt_input, pad_idx)

    return src, tgt_input, tgt_output, src_mask, tgt_mask


def get_mask(src, tgt, pad_idx):
    src_mask = get_pad_mask(src, pad_idx)
    tgt_mask = get_pad_mask(tgt, pad_idx) | get_seq_mask(tgt)
    return src_mask, tgt_mask


def get_seq_mask(tgt):
    return torch.from_numpy(np.triu(np.ones(tgt.size(1)), k=1)) == 1


def get_pad_mask(src, pad_idx):
    return (src == pad_idx).unsqueeze(1).unsqueeze(1)