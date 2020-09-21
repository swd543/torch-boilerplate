#!/usr/bin/env python3

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class SomeNet(nn.Module):
    """Some Information about SomeNet"""
    def __init__(self, input_shape=(32,32), output_shape=100):
        super(SomeNet, self).__init__()
        self.input_shape=input_shape
        self.output_shape=output_shape

        self.i=nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.c=nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.l=nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, self.output_shape),
            nn.LogSoftmax(dim=1)
        )
        self.apply(init_weights)

    def forward(self, x):
        x=self.i(x)
        x=self.c(x)
        x=x.view(x.size(0),-1)
        # print(x.shape)
        x=self.l(x)
        return x

if __name__ == "__main__":
    model=SomeNet()
    print(model)
    dummy=torch.rand((1,3,32,32))
    print(model(dummy))