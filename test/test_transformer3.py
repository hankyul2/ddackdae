from practice.src.transformer3 import get_transformer, get_sample_input


def test_transformer():
    src, tgt, src_mask, tgt_mask = get_sample_input()
    model = get_transformer(101, 102, 512, 6, 8, 2048, 0.1)
    assert model is not None