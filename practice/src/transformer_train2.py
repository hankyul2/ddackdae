import os, sys

from einops import rearrange

sys.path.append('.')
sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from practice.src.transformer6 import get_model, get_sample_data


class LabelSmoothing(nn.Module):
    def __init__(self, smoothness=0.1, pad_idx=0):
        super(LabelSmoothing, self).__init__()
        self.smoothness = smoothness
        self.certain = 1.0 - smoothness
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.pad_idx = pad_idx

    def forward(self, x, y):
        b, c = x.shape
        truth_dist = torch.full((b, c), self.smoothness / (c - 2), device=x.device)
        truth_dist.scatter_(1, y.unsqueeze(1), self.certain)
        truth_dist[:, self.pad_idx] = 0
        truth_dist[torch.nonzero(y == self.pad_idx, as_tuple=True)] = 0
        return self.criterion(F.log_softmax(x, dim=-1), truth_dist)


class MyOpt(nn.Module):
    def __init__(self, optimizer, d_model=512, factor=2, warmup=4000):
        super(MyOpt, self).__init__()
        self.d_model = d_model
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self.step_ = 0

    def step(self):
        self.step_ += 1
        lr = self.rate()
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.optimizer.step()

    def rate(self, step=None):
        step = step if step is not None else self.step_
        return self.factor * (self.d_model ** (-0.5)) * min(step ** (-0.5), step * (self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


class ModelWrapper(object):
    def __init__(self, model, criterion, optimizer, device, pad_idx):
        super(ModelWrapper, self).__init__()
        self.pad_idx = pad_idx
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def fit(self, train_dl, valid_dl, nepoch):
        for epoch in range(nepoch):
            train_loss = self.train(train_dl())
            valid_loss = self.valid(valid_dl(nbatch=5))
            print('EPOCH {:03d} | Train Loss {:07.4f} | Test Loss {:07.4f}'.format(epoch + 1, train_loss, valid_loss))

    def train(self, train_dl):
        self.model.train()
        total_loss = 0
        total_token = 0
        for step, (x, y) in enumerate(train_dl):
            x, y = [item.to(self.device) for item in x], y.to(self.device)
            ntoken = (y != self.pad_idx).sum().item()
            fc_out = self.forward(x)
            loss = self.criterion(rearrange(fc_out, 'b s c -> (b s) c'), rearrange(y, 'b s -> (b s)')) / ntoken
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.detach().item() * ntoken
            total_token += ntoken

        return total_loss / total_token

    def forward(self, x):
        return self.model(*x)

    def valid(self, valid_dl):
        self.model.eval()
        total_loss = 0
        total_token = 0
        for step, (x, y) in enumerate(valid_dl):
            x, y = [item.to(self.device) for item in x], y.to(self.device)
            ntoken = (y != self.pad_idx).sum().item()
            fc_out = self.forward(x)
            loss = self.criterion(rearrange(fc_out, 'b s c -> (b s) c'), rearrange(y, 'b s -> (b s)')) / ntoken

            total_loss += loss.detach().item() * ntoken
            total_token += ntoken

        return total_loss / total_token

    def translate(self, ds):
        for src, tgt_input, tgt_output, src_mask, tgt_mask in zip(*ds):
            tsl = self.model.greedy_decode(src.unsqueeze(0).to(self.device), src_mask.unsqueeze(0).to(self.device), max_len=20)

            print('src text) ', src.detach().tolist())
            print('tgt text) ', [1] + tgt_output.detach().tolist())
            print('tsl text) ', tsl[0].detach().tolist())
            print()


def run(epoch):
    src_vocab_size = 11
    tgt_vocab_size = 11
    train_dl = get_sample_dl
    valid_dl = get_sample_dl

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_model(src_vocab_size, tgt_vocab_size, N=2).to(device)
    criterion = LabelSmoothing(smoothness=0.1)
    optimizer = get_my_opt(model)
    model = ModelWrapper(model, criterion, optimizer, device, 0)

    model.fit(train_dl, valid_dl, epoch)

    model.translate(get_sample_data(batch_size=10, seq_len=20))


def get_my_opt(model, factor=1, warmup=400):
    return MyOpt(Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9), model.src_embed[0].d_model,
                 factor=factor,
                 warmup=warmup)


def get_sample_dl(nbatch=20, seq_len=20, batch_size=30, high=10):
    for iter in range(nbatch):
        src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_data(high=high, batch_size=batch_size,
                                                                         seq_len=seq_len)
        yield (src, tgt_input, src_mask, tgt_mask), tgt_output


if __name__ == '__main__':
    run(10)