from src.transformer import get_model, get_sample_input, Embedding, PositionalEncoding, \
    MultiheadAttention, PositionwiseFeedforward, LayerNorm, SublayerConnection


def test_transformer():
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_input()
    model = get_model(102, 103, 512, 6, 2048, 8, 0.1)
    out = model(src, tgt_input, src_mask, tgt_mask)
    assert list(out.shape) == [10, 10, 103]


def test_get_sample_data():
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_input()
    assert list(src_mask.shape) == [10, 1, 1, 10]
    assert list(tgt_mask.shape) == [10, 1, 10, 10]
    assert list(src.shape) == [10, 10]
    assert list(tgt_input.shape) == [10, 10]


def test_embedding():
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_input()
    embed = Embedding(102, 50)
    src = embed(src)
    assert list(src.shape) == [10, 10, 50]


def test_positional_encoding():
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_input()
    embed = Embedding(102, 50)
    encoding = PositionalEncoding(50)
    src = embed(src)
    src = encoding(src)
    assert list(src.shape) == [10, 10, 50]


def test_multihead_attention():
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_input()
    embed = Embedding(102, 50)
    encoding = PositionalEncoding(50)
    src = embed(src)
    src = encoding(src)
    attn = MultiheadAttention(50, 5)
    src = attn(src, src, src, src_mask)
    assert list(src.shape) == [10, 10, 50]


def test_feedforward():
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_input()
    embed = Embedding(102, 50)
    encoding = PositionalEncoding(50)
    src = embed(src)
    src = encoding(src)
    attn = MultiheadAttention(50, 5)
    src = attn(src, src, src, src_mask)
    ff = PositionwiseFeedforward(50, 100)
    src = ff(src)
    assert list(src.shape) == [10, 10, 50]


def test_norm_layer():
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_input()
    embed = Embedding(102, 50)
    encoding = PositionalEncoding(50)
    src = embed(src)
    src = encoding(src)
    attn = MultiheadAttention(50, 5)
    src = attn(src, src, src, src_mask)
    ff = PositionwiseFeedforward(50, 100)
    src = ff(src)
    norm = LayerNorm(50)
    src = norm(src)
    assert list(src.shape) == [10, 10, 50]


def test_sublayer_connection():
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_input()
    embed = Embedding(102, 50)
    encoding = PositionalEncoding(50)
    src = embed(src)
    src = encoding(src)
    ff = PositionwiseFeedforward(50, 100)
    norm = LayerNorm(50)
    sub = SublayerConnection(50)
    src = sub(src, ff)
    assert list(src.shape) == [10, 10, 50]
