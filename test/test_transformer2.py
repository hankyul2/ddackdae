from practice.src.transformer2 import get_transformer, get_sample_data, Embedding, PositionalEncoding, \
    MultiheadAttention, FeedForward, LayerNorm, SublayerConnection


def test_transformer():
    src, tgt, src_mask, tgt_mask = get_sample_data()
    model = get_transformer(101, 102, 6, 512, 8, 2048, 0.1)
    out = model(src, tgt, src_mask, tgt_mask)
    assert list(out.shape) == [10, 10, 102]


def test_get_sample_data():
    src, tgt, src_mask, tgt_mask = get_sample_data()
    assert list(src_mask.shape) == [10, 1, 1, 10]
    assert list(tgt_mask.shape) == [10, 1, 10, 10]
    assert list(src.shape) == [10, 10]
    assert list(tgt.shape) == [10, 10]


def test_embedding():
    src, tgt, src_mask, tgt_mask = get_sample_data()
    embed = Embedding(50, 101)
    src = embed(src)
    assert list(src.shape) == [10, 10, 50]


def test_positional_encoding():
    src, tgt, src_mask, tgt_mask = get_sample_data()
    embed = Embedding(50, 101)
    encoding = PositionalEncoding(50)
    src = embed(src)
    src = encoding(src)
    assert list(src.shape) == [10, 10, 50]


def test_multihead_attention():
    src, tgt, src_mask, tgt_mask = get_sample_data()
    embed = Embedding(50, 101)
    encoding = PositionalEncoding(50)
    src = embed(src)
    src = encoding(src)
    attn = MultiheadAttention(50, 5)
    src = attn(src, src, src, src_mask)
    assert list(src.shape) == [10, 10, 50]


def test_feedforward():
    src, tgt, src_mask, tgt_mask = get_sample_data()
    embed = Embedding(50, 101)
    encoding = PositionalEncoding(50)
    src = embed(src)
    src = encoding(src)
    attn = MultiheadAttention(50, 5)
    src = attn(src, src, src, src_mask)
    ff = FeedForward(50, 100)
    src = ff(src)
    assert list(src.shape) == [10, 10, 50]


def test_norm_layer():
    src, tgt, src_mask, tgt_mask = get_sample_data()
    embed = Embedding(50, 101)
    encoding = PositionalEncoding(50)
    src = embed(src)
    src = encoding(src)
    attn = MultiheadAttention(50, 5)
    src = attn(src, src, src, src_mask)
    ff = FeedForward(50, 100)
    src = ff(src)
    norm = LayerNorm(50)
    src = norm(src)
    assert list(src.shape) == [10, 10, 50]


def test_sublayer_connection():
    src, tgt, src_mask, tgt_mask = get_sample_data()
    embed = Embedding(50, 101)
    encoding = PositionalEncoding(50)
    src = embed(src)
    src = encoding(src)
    ff = FeedForward(50, 100)
    norm = LayerNorm(50)
    sub = SublayerConnection(norm)
    src = sub(src, ff)
    assert list(src.shape) == [10, 10, 50]
