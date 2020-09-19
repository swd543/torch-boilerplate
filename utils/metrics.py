# %%
import torch
from abc import abstractmethod

def _check_valid(outputs:torch.Tensor, actual:torch.Tensor):
    assert outputs.shape == actual.shape and len(outputs.shape) == 2

def accuracy(outputs:torch.Tensor, actual:torch.Tensor):
    _check_valid(outputs, actual)
    accuracy = (actual.argmax(-1) == outputs.argmax(-1))
    return 100 * accuracy.sum() // len(accuracy)

def loss(outputs:torch.Tensor, actual:torch.Tensor):
    _check_valid(outputs, actual)

a=torch.Tensor([[0,0.1,0.9,0],[1,0,0,0]])
b=torch.Tensor([[0,0,1,0],[0,0,0,0]])
# a=torch.Tensor([0,0.1,0.9,0])
# b=torch.Tensor([0,0.1,0.8,0])

print(a.shape)
print(accuracy(a,b))