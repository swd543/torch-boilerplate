#!/usr/bin/env python3

# %%
from torch.nn.modules.activation import LogSoftmax
from utils.utils import calculate_mean_std, AverageMeter
from utils import metrics
import os
from torch.utils.data import DataLoader
import tempfile
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomRotation
import wandb
import torch
from torchvision import datasets, transforms
import numpy as np

wandb.init(project='test', dir=tempfile.gettempdir())
print(
    f'Using cudnn version {torch.backends.cudnn.version()}, pytorch version {torch.__version__}')
trainset = datasets.CIFAR100(
    '~/data', True, transform=transforms.ToTensor(), download=True)

# %%
print(trainset[0][0])
# %%
wandb.log({'sample images': [wandb.Image(
    trainset[i][0], caption=trainset[i][1]) for i in range(32)]})

# %%
train_loader = DataLoader(trainset, 1024, num_workers=os.cpu_count())
[mean, std] = calculate_mean_std(train_loader)
print(f'Dataset attributes : {mean}, {std}')

# %%
trainset = datasets.CIFAR100('~/data', True, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
]), download=True)

train_loader = DataLoader(
    trainset, 1024, num_workers=os.cpu_count(), shuffle=True)

wandb.log({'augmented training images': [wandb.Image(i) for i in next(iter(train_loader))[0]]})

validset = datasets.CIFAR100('~/data', False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
]), download=True)

valid_loader = DataLoader(validset, 1024, num_workers=os.cpu_count())

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

class SomeNet(nn.Module):
    """Some Information about SomeNet"""
    def __init__(self, input_shape=trainset[0][0].shape, output_shape=len(trainset.classes)):
        super(SomeNet, self).__init__()
        self.input_shape=input_shape
        self.output_shape=output_shape

        self.i=nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU()
        )
        self.c=nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
        )
        self.l=nn.Sequential(
            nn.Linear(73728, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.output_shape),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x=self.i(x)
        x=self.c(x)
        x=x.view(x.shape[0],-1)
        x=self.l(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model=nn.DataParallel(SomeNet()).to(device) if torch.cuda.device_count() > 1 else SomeNet().to(device) 

optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
# lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-2, cycle_momentum=False)
criterion = nn.CrossEntropyLoss()

train_loss=AverageMeter('train_loss')
valid_loss=AverageMeter('valid_loss')
train_acc =AverageMeter('train_acc')
valid_acc =AverageMeter('valid_acc')

epochs=100
for e in range(0,epochs):
    wandb.log({'epoch':e, 'lr':[p['lr'] for p in optimizer.param_groups][-1]})

    model=model.train()
    with tqdm(train_loader, desc=f'[{e}/{epochs}] Train') as progress, torch.enable_grad():
        for i, l in progress:
            i, l = i.to(device), l.to(device)
            optimizer.zero_grad()
            predictions=model(i)
            losses=criterion(predictions, l)
            losses.backward()
            train_acc.update(metrics.accuracy(predictions, l))
            optimizer.step()
            train_loss.update(losses)
            wandb.log({'train_loss':losses, 'train_acc':train_acc.avg})
            progress.set_postfix_str(f'{train_loss} {train_acc}')
        train_loss.commit()
        train_acc.commit()

    model=model.eval()
    with tqdm(valid_loader, desc=f'[{e}/{epochs}] Valid') as progress, torch.no_grad():
        for i, l in progress:
            i, l = i.to(device), l.to(device)
            optimizer.zero_grad()
            predictions=model(i)
            losses=criterion(predictions, l)
            valid_acc.update(metrics.accuracy(predictions, l))
            valid_loss.update(losses)
            wandb.log({'valid_loss':losses})
            progress.set_postfix_str(f'{valid_loss}')
        valid_loss.commit()
        valid_acc.commit()
    
    lr_scheduler.step(valid_loss.avg)