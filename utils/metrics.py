# %%
import torch
from torch.tensor import Tensor

def accuracy(outputs:Tensor, actual:Tensor, dim=-1)->Tensor:
    correct = (actual == outputs.argmax(dim))
    return 100 * correct.sum() // len(correct)