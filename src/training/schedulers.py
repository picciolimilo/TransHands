import math
import torch


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if epoch < self.warmup_epochs:
                lr = base_lr * (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs
