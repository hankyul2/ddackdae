import copy

import numpy as np

import pytest
import torch
from torch import nn

from src.transformer import Transformer, Encoder, Decoder, PositionalEncoding, Embed, MultiheadAttention, \
    PositionwiseFeedForward, SublayerConnection, EncoderLayer, LayerNorm, DecoderLayer


def test_transformer_constructor():
    d_model = 10
    h = 2
    d_ff = 20
    N = 6
    src_vocab_size = 10
    tgt_vocab_size = 10
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
    assert model is not None


def test_transformer_forward():
    x = torch.arange(100).view(10, 10)
    mask = torch.zeros(10, 1, 10, 10)
    tgt = torch.arange(1, 101).view(10, 10)
    tgt_mask = (torch.zeros(10, 1, 10, 10) == 1) | (torch.from_numpy(np.triu(np.ones((10, 10)), 1)) == 1)

    d_model = 10
    h = 2
    d_ff = 20
    N = 6
    src_vocab_size = 101
    tgt_vocab_size = 101
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


def test_embed_constructor():
    embed = Embed(10, 10)
    assert embed.d_model == 10


def test_embed_forward():
    embed = Embed(10, 10)
    x = torch.arange(10).unsqueeze(0)
    x_embed = embed(x)
    assert list(x_embed.shape) == [1, 10, 10]


def test_positional_encoding_constructor():
    encoding = PositionalEncoding(d_model=10, max_len=5000, dropout=0.1)
    assert list(encoding.pe.shape) == [1, 5000, 10]


def test_positional_encoding_forward():
    x = torch.rand(10, 10, 10)
    encoding = PositionalEncoding(d_model=10, max_len=5000, dropout=0.1)
    x = encoding(x)
    assert list(x.shape) == [10, 10, 10]


def test_multihead_attention_forward():
    x = torch.rand(10, 10, 10)
    mask = torch.zeros(10, 1, 10, 10)
    attn = MultiheadAttention(d_model=10, h=2, dropout=0.1)
    x = attn(x, x, x, mask)
    assert list(x.shape) == [10, 10, 10]


def test_positionwisefeedforward_forward():
    x = torch.rand(10, 10, 10)
    ff = PositionwiseFeedForward(d_model=10, d_ff=20, dropout=0.1)
    x = ff(x)
    assert list(x.shape) == [10, 10, 10]


def test_sublayerconnection_forward():
    x = torch.rand(10, 10, 10)
    connection = SublayerConnection(d_model=10, dropout=0.1)
    ff = PositionwiseFeedForward(d_model=10, d_ff=20, dropout=0.1)
    x = connection(x, ff)
    assert list(x.shape) == [10, 10, 10]


def test_encoder_layer_forward():
    x = torch.rand(10, 10, 10)
    mask = torch.zeros(10, 1, 10, 10)
    attn = MultiheadAttention(d_model=10, h=2, dropout=0.1)
    ff = PositionwiseFeedForward(d_model=10, d_ff=20, dropout=0.1)
    connection = SublayerConnection(d_model=10, dropout=0.1)
    encoder_layer = EncoderLayer(attn, ff, connection)
    x = encoder_layer(x, mask)
    assert list(x.shape) == [10, 10, 10]


def test_encoder_forward():
    c = copy.deepcopy
    x = torch.rand(10, 10, 10)
    mask = torch.zeros(10, 1, 10, 10)
    attn = MultiheadAttention(d_model=10, h=2, dropout=0.1)
    ff = PositionwiseFeedForward(d_model=10, d_ff=20, dropout=0.1)
    connection = SublayerConnection(d_model=10, dropout=0.1)
    encoder_layer = EncoderLayer(c(attn), c(ff), c(connection))
    norm = LayerNorm(d_model=10)
    encoder = Encoder(encoder_layer, norm, 6)
    x = encoder(x, mask)
    assert list(x.shape) == [10, 10, 10]


def test_decoder_forward():
    c = copy.deepcopy
    x = torch.rand(10, 10, 10)
    mask = torch.zeros(10, 1, 10, 10)
    tgt = torch.rand(10, 10, 10)
    tgt_mask = (torch.zeros(10, 1, 10, 10) == 1) | (torch.from_numpy(np.triu(np.ones((10, 10)), 1)) == 1)

    attn = MultiheadAttention(d_model=10, h=2, dropout=0.1)
    ff = PositionwiseFeedForward(d_model=10, d_ff=20, dropout=0.1)
    connection = SublayerConnection(d_model=10, dropout=0.1)
    norm = LayerNorm(d_model=10)

    encoder_layer = EncoderLayer(c(attn), c(ff), c(connection))
    encoder = Encoder(encoder_layer, c(norm), 6)
    src_feature = encoder(x, mask)

    decoder_layer = DecoderLayer(c(attn), c(attn), c(ff), c(connection))
    decoder = Decoder(decoder_layer, c(norm), 6)
    x = decoder(src_feature, mask, tgt, tgt_mask)
    assert list(x.shape) == [10, 10, 10]