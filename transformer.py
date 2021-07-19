import math, os, random, copy, re, time

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchtext.datasets import IWSLT2016
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, FastText
from torchtext.data.functional import to_map_style_dataset
from einops import rearrange, repeat, reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm.notebook import tqdm

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_size=5000, dropout=0.1):
        super().__init__()
        
        pe = torch.zeros(max_size, d_model)
        position = torch.arange(0, max_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0)/d_model))
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        return x + self.dropout(self.pe[:, :x.size(1)].clone().detach().requires_grad_(False))
      
 def clone(layer, n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])

class EncoderLayer(nn.Module):
    def __init__(self, at, ff, su):
        super().__init__()
        self.at = at
        self.ff = ff
        self.su = clone(su, 2)
    
    def forward(self, x, mask):
        x = self.su[0](x, lambda x: self.at(x, x, x, mask))
        return self.su[1](x, self.ff)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        self.L = clone(nn.Linear(d_model, d_model), 3)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None
        self.qkv = None
        self.v = None
        
    def forward(self, q, k, v, mask=None):
        b, n, d_m = list(q.shape)
        
        self.qkv = q, k, v = [rearrange(f(x), 'b n (h d) -> b h n d', h=self.h, d=self.d_k) for f, x in zip(self.L, (q, k, v))]
        
        score = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        if mask is not None:
            score = score.masked_fill(mask, -1e9)
        self.attn = attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)
        x = attn @ v
        
        self.v = x = rearrange(x, 'b h n d -> b n (h d)')
        return self.out(x)

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (self.eps + std) + self.b_2

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
def make_pad_mask(src, pad=1):
    mask = (src == pad).unsqueeze(0)
    return mask

def make_seq_mask(size):
    mask = torch.from_numpy(np.triu(np.ones((1, size, size)), 1)) == 1
    return mask
  
class DecoderLayer(nn.Module):
    def __init__(self, at, att, ff, su):
        super().__init__()
        self.at = at
        self.att = att
        self.ff = ff
        self.su = clone(su, 3)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.su[0](x, lambda x: self.at(x, x, x, tgt_mask))
        x = self.su[1](x, lambda x: self.att(x, memory, memory, src_mask))
        return self.su[2](x, self.ff)
      
class Encoder(nn.Module):
    def __init__(self, layer, norm, N):
        super().__init__()
        self.layers = clone(layer, N)
        self.norm = norm
    
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer, norm, N):
        super().__init__()
        self.layers = clone(layer, N)
        self.norm = norm
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, fc):
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.fc = fc
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), tgt, src_mask, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, tgt, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
      
def make_model(src_vocab_size, tgt_vocab_size, d_model=10, N=6, h=2, d_ff=20, dropout=0.1):
    c = copy.deepcopy
    
    src_embedding = Embedding(d_model, src_vocab_size)
    tgt_embedding = Embedding(d_model, tgt_vocab_size)
    
#     src_embedding.embed.weight.data.copy_(src_pretrained_embedding)
#     tgt_embedding.embed.weight.data.copy_(tgt_pretrained_embedding)
    
    src_embed = nn.Sequential(src_embedding, PositionalEncoding(d_model))
    tgt_embed = nn.Sequential(tgt_embedding, PositionalEncoding(d_model))
    
    attn = MultiheadAttention(d_model, h)
    ff = PositionwiseFeedForward(d_model, d_ff)
    su = SublayerConnection(d_model)
    norm = LayerNorm(d_model)
    
    encoder = Encoder(EncoderLayer(c(attn), c(ff), c(su)), c(norm), N)
    decoder = Decoder(DecoderLayer(c(attn), c(attn), c(ff), c(su)), c(norm), N)
    fc = nn.Linear(d_model, tgt_vocab_size)
    
    model = Transformer(src_embed, tgt_embed, encoder, decoder, fc)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model
