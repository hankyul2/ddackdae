import os

import torch
from einops import rearrange
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from torch import nn

from src.transformer import get_model, get_sample_input, get_mask


class ModelWrapper(nn.Module):
    def __init__(self, model, criterion, optimizer):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(*x)

    def train(self, dl, device):
        self.model.train()
        total_loss = 0
        for iter, (x, y) in enumerate(dl):
            self.optimizer.zero_grad()
            x, y = self.to_device(x, device), y.to(device)
            n_token = (y != 0).sum().item()
            fc_out = self.forward(x)
            loss = self.criterion(rearrange(fc_out, 'b s c -> (b s) c'), rearrange(y, 'b s -> (b s)')) / n_token
            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach().item() * n_token

        return total_loss

    @torch.no_grad()
    def valid(self, dl, device):
        self.model.eval()
        total_loss = 0
        for iter, (x, y) in enumerate(dl):
            x, y = self.to_device(x, device), y.to(device)
            fc_out = self.forward(x)
            loss = self.criterion(rearrange(fc_out, 'b s c -> (b s) c'), rearrange(y, 'b s -> (b s)'))
            total_loss += loss.detach().item()

        return total_loss

    def fit(self, train_dl, valid_dl, num_epoch, device):
        for epoch in range(num_epoch):
            train_loss = self.train(train_dl(), device)
            valid_loss = self.valid(valid_dl(batch_iter=5), device)
            print('EPOCH {:03} | train loss {:07.4f} | valid loss {:07.4f}'.format(epoch+1, train_loss, valid_loss))

    def translate(self, dl, device):
        self.model.eval()
        for iter, (src, tgt_input, tgt_output, src_mask, tgt_mask) in enumerate(zip(*dl)):
            src, src_mask = src.unsqueeze(0), src_mask.unsqueeze(0)
            y_hat = self.model.greedy_decode(src.to(device), src_mask.to(device), device, max_len=10)

            print("src text) ", src[0].detach().tolist())
            print("tgt text) ", [1] + tgt_output.detach().tolist())
            print("tsl text) ", y_hat[0].detach().tolist())
            print()

            if iter == 10:
                break

    def to_device(self, x, device):
        return [item.to(device) for item in x]


class MyOpt:
    def __init__(self, d_model, optimizer, factor=2, warmup=4000):
        self.d_model = d_model
        self.optimizer = optimizer
        self.factor = factor
        self.warmup = warmup
        self.step_ = 0

    def step(self):
        self.step_ += 1
        rate = self.get_rate()
        for param in self.optimizer.param_groups:
            param['lr'] = rate
        self.optimizer.step()

    def get_rate(self):
        return self.factor * (self.d_model ** -0.5) * min(self.step_ ** -0.5, self.step_ * (self.warmup ** -1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(model, factor=2, warmup=4000):
    return MyOpt(model.src_embed[0].d_model, Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), factor, warmup)


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, pad_idx=0):
        super(LabelSmoothing, self).__init__()
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1 - smoothing
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x, y):
        (n, c)= x.shape
        truth_dist = torch.full((n, c), self.smoothing / (c-2), device=x.device)
        truth_dist.scatter_(1, y.unsqueeze(1), self.confidence)
        truth_dist[:, self.pad_idx] = 0
        truth_dist[torch.nonzero(y == self.pad_idx, as_tuple=True)] = 0
        return self.criterion(F.log_softmax(x, dim=-1), truth_dist)


def get_dl(batch_iter=20, batch_size=30):
    for batch in range(batch_iter):
        src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_input(batch_size=30)
        yield (src, tgt_input, src_mask, tgt_mask), tgt_output


def run():
    # step 1. prepare dataset
    src_vocab_size = 12
    tgt_vocab_size = 12

    train_dl = get_dl
    valid_dl = get_dl
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_input(batch_size=30)

    # step 2. prepare model
    device = torch.device('cuda')
    model = get_model(src_vocab_size, tgt_vocab_size, d_model=512, N=2, d_ff=2048, h=8).to(device)

    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    # step 3. prepare training tools (criterion, optimizer, etc)
    criterion = LabelSmoothing(smoothing=0.0)
    optimizer = get_std_opt(model=model, factor=1, warmup=400)
    model = ModelWrapper(model=model, criterion=criterion, optimizer=optimizer)

    # step 4. train
    model.fit(train_dl, valid_dl, num_epoch=10, device=device)

    # step 5. validate
    model.translate([src, tgt_input, tgt_output, src_mask, tgt_mask], device)


if __name__ == '__main__':
    run()
