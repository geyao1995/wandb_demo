import sys

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from config import ParamConfig
from help_funcs_wandb import define_wandb_lr_metrics


class Trainer:
    def __init__(self, device: str, model: torch.nn.Module, config: ParamConfig, train_loader,
                 optimizer, lr_scheduler):

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.total_epoch = config.epoch_total
        self.train_loader = train_loader
        self.lr_scheduler = lr_scheduler

        # use wandb to record change of lr
        self.wandb_metric_batch, self.wandb_metric_lr = define_wandb_lr_metrics()

        self.bs_print = 100
        self.tqdm_bar = tqdm(total=len(self.train_loader) * self.total_epoch,
                             file=sys.stdout, position=0, ncols=100)

    def train_epoch(self, idx_epoch):
        """
        idx_epoch should start from 1
        """

        self.model.train()

        loss = 0.
        lr = 0.

        for batch_idx, (data, target) in enumerate(self.train_loader, 1):

            self.optimizer.zero_grad()
            lr = self.lr_scheduler.get_last_lr()[0]

            logits = self.model(data.to(self.device))
            loss = F.cross_entropy(logits, target.to(self.device))

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            if batch_idx % self.bs_print == 0:
                self.tqdm_bar.write(f'[{idx_epoch:<2}, {batch_idx + 1:<2}] '
                                    f'loss: {loss:<6.4f} '
                                    f'lr: {lr:.4f} ')

            self.tqdm_bar.update(1)
            self.tqdm_bar.set_description(f'epoch-{idx_epoch:<3} '
                                          f'batch-{batch_idx + 1:<3} '
                                          f'loss-{loss:<.2f} '
                                          f'lr-{lr:.3f}')

            idx_batch_total = (idx_epoch - 1) * len(self.train_loader) + batch_idx
            wandb.log({self.wandb_metric_lr: lr,
                       self.wandb_metric_batch: idx_batch_total})

        if idx_epoch >= self.total_epoch:
            self.tqdm_bar.close()

        return loss.item(), lr
