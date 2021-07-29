from practice.src.transformer4 import get_sample_input, get_model


def test_transformer_get_model():
    src, tgt, src_mask, tgt_mask = get_sample_input()
    model = get_model(101, 102)
    output = model(src, tgt, src_mask, tgt_mask)
    assert list(output.shape) == [10, 10, 102]