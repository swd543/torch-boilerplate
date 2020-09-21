#!/usr/bin/env python3

# %%
import torch
print(torch.version.cuda)
import cupy as cp

data=cp.random.rand(100,100)
print(data)

torch.cuda.device_count()