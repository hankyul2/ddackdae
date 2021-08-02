from einops import rearrange
from torch.optim import Adam

from practice.src.transformer6 import get_model, get_sample_data
from practice.src.transformer_train2 import run, LabelSmoothing, MyOpt


def test_run():
    run(1)

def test_sample_data():
    data = get_sample_data()
    assert data is not None

def test_sample_data2():
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_data(high=5, batch_size=20, seq_len=20)
    assert list(src.shape) == [20, 20]
    assert list(tgt_input.shape) == [20, 19]
    assert list(tgt_output.shape) == [20, 19]
    assert list(src_mask.shape) == [20, 1, 1, 20]
    assert list(tgt_mask.shape) == [20, 1, 19, 19]

def test_get_model():
    model = get_model()
    assert model is not None

def test_model_forward():
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_data(high=10, batch_size=20, seq_len=20)
    model = get_model()
    fc_out = model(src, tgt_input, src_mask, tgt_mask)
    assert list(fc_out.shape) == [20, 19, 10]

def test_criterion():
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_data(high=10, batch_size=20, seq_len=20)
    model = get_model()
    criterion = LabelSmoothing(smoothness=0.1)

    fc_out = model(src, tgt_input, src_mask, tgt_mask)
    loss = criterion(rearrange(fc_out, 'b s c -> (b s) c'), rearrange(tgt_output, 'b s -> (b s)'))

    assert loss.detach().item()/400 < 20

def test_optimizer():
    model = get_model()
    optimizer = MyOpt(Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9))
    assert optimizer.rate(1) != 0
