import os

import torch
from einops import rearrange
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

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
            fc_out = self.forward(x)
            loss = self.criterion(rearrange(fc_out, 'b s c -> (b s) c'), rearrange(y, 'b s -> (b s)'))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach().item()

        return total_loss

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
            train_loss = self.train(train_dl, device)
            valid_loss = self.valid(valid_dl, device)
            print('EPOCH {} | train loss {} | valid loss {}'.format(epoch, train_loss, valid_loss))

    def translate(self, dl, device):
        self.model.eval()
        for iter, (src, tgt_input, tgt_output, src_mask, tgt_mask) in enumerate(zip(*dl)):
            src, src_mask = src.unsqueeze(0), src_mask.unsqueeze(0)
            y_hat = self.model.greedy_decode(src.to(device), src_mask.to(device), device, max_len=20)

            print("src text) ", src[0].detach().tolist())
            print("target text) ", tgt_output.detach().tolist())
            print("target translated text)", y_hat[0].detach().tolist())
            print()

            if iter == 10:
                break

    def to_device(self, x, device):
        return [item.to(device) for item in x]


def run():
    src_vocab_size = 102
    tgt_vocab_size = 103

    device = torch.device('cuda')

    model = get_model(src_vocab_size, tgt_vocab_size, d_model=64, N=3, d_ff=128, h=4).to(device)

    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    criterion = CrossEntropyLoss()
    # optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)

    model = ModelWrapper(model=model, criterion=criterion, optimizer=optimizer)
    src, tgt_input, tgt_output, src_mask, tgt_mask = get_sample_input(batch_size=600)

    train_dl = [((src, tgt_input, tgt_mask, src_mask), tgt_output)]
    valid_dl = [((src, tgt_input, tgt_mask, src_mask), tgt_output)]

    model.fit(train_dl, valid_dl, 50, device)

    model.translate([src, tgt_input, tgt_output, src_mask, tgt_mask], device)


if __name__ == '__main__':
    run()
