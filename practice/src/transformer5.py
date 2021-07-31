import copy
import math

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)].clone().detach().requires_grad_(False))


def clone(layer, n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h

        self.w = clone(nn.Linear(d_model, d_model), 3)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        q, k, v = [rearrange(f(x), 'b s (h k) -> b h s k', h=self.h) for f, x in zip(self.w, [q, k, v])]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        w_v = attn @ v
        c_v = rearrange(w_v, 'b h s k -> b s (h k)')
        return self.out(c_v)


class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Feedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean, std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
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
        return self.su[1](src, self.ff)


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
        return self.su[2](tgt, self.ff)


class Encoder(nn.Module):
    def __init__(self, layer, norm, N):
        super(Encoder, self).__init__()
        self.norm = norm
        self.layers = clone(layer, N)

    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)


class Decoder(nn.Module):
    def __init__(self, layer, norm, N):
        super(Decoder, self).__init__()
        self.norm = norm
        self.layers = clone(layer, N)

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

    def greedy_decode(self, src, src_mask, BOS=1, max_len=5000, EOS=-1):
        src_rep = self.encode(src, src_mask)
        ys = torch.full((1, 1), BOS, device=src.device, dtype=src.dtype)
        for iter in range(max_len-1):
            fc_out = self.generator(self.decode(src_rep, ys, src_mask, get_mask(src, ys)[1].to(src.device)))
            _, next_word = fc_out[:, -1].max(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word.clone().detach()], dim=1)
            if next_word.clone().detach().item() == EOS:
                break
        return ys


def get_model(src_vocab_size, tgt_vocab_size, N=6, d_model=512, h=8, d_ff=2048, dropout=0.1):
    c = copy.deepcopy

    src_embed = nn.Sequential(Embedding(d_model, src_vocab_size), PositionalEncoding(d_model, dropout))
    tgt_embed = nn.Sequential(Embedding(d_model, tgt_vocab_size), PositionalEncoding(d_model, dropout))

    attn = MultiheadAttention(d_model, h, dropout)
    ff = Feedforward(d_model, d_ff, dropout)
    norm = LayerNorm(d_model)
    su = SublayerConnection(d_model, dropout)

    encoder = Encoder(EncoderLayer(c(attn), c(ff), c(su)), c(norm), N)
    decoder = Decoder(DecoderLayer(c(attn), c(attn), c(ff), c(su)), c(norm), N)
    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(src_embed, tgt_embed, encoder, decoder, generator)

    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return model


def get_sample_input(src_vocab_size, tgt_vocab_size, batch_size, seq=10, pad_idx=0):
    src = torch.randint(1, src_vocab_size, size=(batch_size, seq)).long()
    src[:, 0] = 1
    tgt = src.clone().detach()
    tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
    src_mask, tgt_mask = get_mask(src, tgt_input, pad_idx)
    return src, tgt_input, tgt_output, src_mask, tgt_mask


def get_mask(src, tgt, pad_idx=0):
    src_mask = get_pad_mask(src, pad_idx)
    tgt_mask = get_pad_mask(tgt, pad_idx) | get_seq_mask(tgt)
    return src_mask, tgt_mask


def get_seq_mask(tgt):
    return torch.triu(torch.ones((tgt.size(1), tgt.size(1)), device=tgt.device), diagonal=1) == 1


def get_pad_mask(src, pad_idx):
    return (src == pad_idx).unsqueeze(1).unsqueeze(1)
