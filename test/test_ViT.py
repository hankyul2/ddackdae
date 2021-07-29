import torch

from src.ViT import get_model


def test_vit_construction():
    model = get_model()
    assert model is not None

def test_vit_forward():
    sample_input = torch.rand(size=(10, 3, 224, 224))
    model = get_model()
    fc_out = model(sample_input)
    assert list(fc_out.shape) == [10, 100]
