import math

import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F
import copy


def is_pair(size):
    return size if type(size) == type((1,)) else (size, size)

class Embedding(nn.Module):
    def __init__(self, img_size, patch_size, d_model, dropout=0.1):
        super(Embedding, self).__init__()
        i_h, i_w = is_pair(img_size)
        p_h, p_w = is_pair(patch_size)
        p_num = (i_h // p_h) * (i_w // p_w)
        p_size = p_h * p_w * 3
        assert i_h % p_h == 0 and i_w % p_w == 0
    
        self.f_p = lambda x: rearrange(x, 'b c (p h) (q w) -> b (p q) (h w c)', h=p_h, w=p_w)
        self.f_l = nn.Linear(p_size, d_model)
        self.cls_token = nn.Parameter(torch.rand(d_model))
        self.pos_embed = nn.Parameter(torch.rand(size=(p_num+1, d_model)))
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.f_p(x)
        b, s, _ = x.shape
        x = torch.cat([repeat(self.cls_token, 'd -> b 1 d', b=b), self.f_l(x)], dim=1) + self.pos_embed[:s+1, :]
        return self.dropout(x)


def clone(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h

        self.w = clone(nn.Linear(d_model, d_model), 3)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        q, k, v = [rearrange(f(x), 'b s (h k) -> b h s k', h=self.h) for f, x in zip(self.w, [q, k, v])]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        w_v = attn @ v
        c_v = rearrange(w_v, 'b h s k -> b s (h k)')
        return self.out(c_v)


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-9):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, attn, ff, su):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.ff = ff
        self.su = clone(su, 2)

    def forward(self, x):
        x = self.su[0](x, lambda x: self.attn(x, x, x))
        return self.su[1](x, self.ff)


class Encoder(nn.Module):
    def __init__(self, layer, norm, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = norm

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, embed, encoder, mlp_head):
        super(ViT, self).__init__()
        self.embed = embed
        self.encoder = encoder
        self.mlp_head = mlp_head

    def forward(self, x):
        x_rep = self.encode(x)
        return self.mlp_head(x_rep[:, 0])

    def encode(self, x):
        return self.encoder(self.embed(x))


def get_model(img_size=224, patch_size=16, num_class=100, N=6, d_model=512, h=8, d_ff=2048, dropout=0.1):
    c = copy.deepcopy
    embed = Embedding(img_size, patch_size, d_model, dropout)
    attn = MultiheadAttention(d_model, h, dropout)
    ff = PositionwiseFeedforward(d_model, d_ff, dropout)
    norm = LayerNorm(d_model)
    su = SublayerConnection(d_model, dropout)
    encoder = Encoder(EncoderLayer(c(attn), c(ff), c(su)), c(norm), N)
    mlp_head = nn.Linear(d_model, num_class)

    model = ViT(embed, encoder, mlp_head)

    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return model