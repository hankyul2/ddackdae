import torch

from practice.src.transformer import get_transformer, get_mask


def test_get_transformer():
    src = torch.arange(1, 101, 1).view(10, 10)
    tgt = torch.arange(2, 102, 1).view(10, 10)

    src_mask, tgt_mask = get_mask(src, tgt, pad_idx=0)

    model = get_transformer(101, 102, 512, 8, 6, 2048)

    out = model(src, tgt, src_mask, tgt_mask)

    assert model is not None
    assert list(out.shape) == [10, 10, 102]