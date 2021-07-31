import os, sys

from einops import rearrange
from torch.optim import Adam

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append('.')
sys.path.append('..')

import torch
from torch import nn
import torch.nn.functional as F

from practice.src.transformer5 import get_sample_input, get_model


class LabelSmoothing(nn.Module):
    def __init__(self, smoothness=0.1, pad_idx=0):
        super(LabelSmoothing, self).__init__()
        self.pad_idx = pad_idx
        self.smoothness = smoothness
        self.confidence = 1.0 - smoothness
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, x, y):
        b, c = x.shape
        truth_dist = torch.full((b, c), self.smoothness / (c - 2)).to(x.device)
        truth_dist.scatter_(1, y.unsqueeze(1), self.confidence)
        truth_dist[:, self.pad_idx] = 0
        truth_dist[torch.nonzero(y == self.pad_idx, as_tuple=True)] = 0
        return self.criterion(F.log_softmax(x, dim=-1), truth_dist) ##


class MyOpt(object):
    def __init__(self, d_model, optimizer, factor=2, warmup=4000):
        super(MyOpt, self).__init__()
        self.d_model = d_model
        self.optimizer = optimizer
        self.factor = factor
        self.warmup = warmup
        self.step_ = 0

    def step(self):
        self.step_ += 1
        lr = self.rate()
        for param in self.optimizer.param_groups: ##
            param['lr'] = lr
        self.optimizer.step()

    def rate(self):
        return self.factor * (self.d_model ** -0.5) * min(self.step_ ** -0.5, self.step_ * (self.warmup ** -1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()


class ModelWrapper(nn.Module):
    def __init__(self, model, criterion, optimizer, device):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def forward(self, x):
        return self.model(*x)

    def run_epoch(self, dl, train=False):
        if train: ##################
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        total_token = 0
        for iter, (x, y) in enumerate(dl):
            x, y = self.to_device(x), y.to(self.device)
            token = (y != 0).sum().detach().item()
            fc_out = self.forward(x)
            loss = self.criterion(rearrange(fc_out, 'b s c -> (b s) c'), rearrange(y, 'b s -> (b s)')) / token
            if train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.detach().item() * token ##
            total_token += token

        return total_loss / total_token

    def fit(self, train_dl, valid_dl, nepoch):
        for epoch in range(nepoch):
            train_loss = self.run_epoch(train_dl(), train=True)
            valid_loss = self.run_epoch(valid_dl(batch_num=5))
            print('EPOCH {:03d} | Train Loss {:07.4f} | Valid Loss {:07.4f}'.format(epoch + 1, train_loss, valid_loss))

    def translate(self, ds):
        self.model.eval() ##
        for iter, (src, tgt_input, tgt_output, src_mask, tgt_mask) in enumerate(zip(*ds)):
            tsl = self.model.greedy_decode(src.unsqueeze(0).to(self.device), src_mask.unsqueeze(0).to(self.device), max_len=10) ##

            print('src text) ', src.detach().tolist())
            print('tgt text) ', [1] + tgt_output.detach().tolist())
            print('tsl text) ', tsl[0].detach().tolist())
            print()

    def to_device(self, x):
        return [item.to(self.device) for item in x]


def run():
    src_vocab_size = 11
    tgt_vocab_size = 11
    train_dl = get_sample_dl
    valid_dl = get_sample_dl

    device = torch.device('cuda')
    model = get_model(src_vocab_size, tgt_vocab_size, N=2).to(device)

    criterion = LabelSmoothing(smoothness=0.0)
    optimizer = get_opt(model, factor=1, warmup=400)
    model = ModelWrapper(model, criterion, optimizer, device)

    model.fit(train_dl, valid_dl, nepoch=10)

    model.translate(get_sample_input(src_vocab_size, tgt_vocab_size, 10))


def get_opt(model, factor=2, warmup=4000):
    return MyOpt(model.src_embed[0].d_model, Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), factor=factor,
                 warmup=warmup)


def get_sample_dl(batch_num=20, batch_size=30, src_vocab_size=11, tgt_vocab_size=11): ################
    for batch in range(batch_num):
        src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_input(src_vocab_size, tgt_vocab_size,
                                                                          batch_size)
        yield (src, tgt_input, src_mask, tgt_mask), tgt_output


if __name__ == '__main__':
    run()
