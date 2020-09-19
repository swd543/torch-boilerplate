from torch.utils.data import DataLoader
import torch

def calculate_mean_std(loader: DataLoader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return mean, std

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.vals = []
        self.avgs = []
        self.sums = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val.item() if type(val) is torch.Tensor else val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def commit(self):
        self.avgs.append(self.avg)
        self.sums.append(self.sum)
        self.vals.append(self.val)
        self.reset()

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

