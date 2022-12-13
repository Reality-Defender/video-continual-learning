# Stealed from https://github.com/moskomule/ewc.pytorch/blob/master/utils.py and re-elaborated

from copy import deepcopy

import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count


class EWC(object):
    def __init__(self, model: nn.Module, dataloader, lmbda, importance_method: str = 'fisher'):

        self.model = model
        self.dataloader = DataLoader(dataloader.dataset, batch_size=1, shuffle=False, num_workers=10)
        # self.dataloader = dataloader
        self.criterion = nn.CrossEntropyLoss()

        self.importance_method = importance_method
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._importance_computation()
        self.lmbda = lmbda

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data

    def _importance_computation(self):
        precision_matrices = {}

        if self.importance_method in ['fisher', 'mas']:
            for n, p in deepcopy(self.params).items():
                p.data.zero_()
                precision_matrices[n] = p.data
        elif self.importance_method == 'fisher_complete':
            for n, p in deepcopy(self.params).items():
                p1 = p.data.view(1, -1)
                p1.data.zero_()
                h1 = torch.matmul(p1.T, p1)
                precision_matrices[n] = h1.data
        else:
            raise ValueError(f'Invalid importance method: {self.importance_method}')

        self.model.eval()
        for batch_input in tqdm(self.dataloader, total=len(self.dataloader), leave=False,
                                desc=f'Computing {self.importance_method} importance'):
            input = batch_input[0].cuda()
            label = batch_input[1].cuda()
            self.model.zero_grad()
            output = self.model(input)

            if self.importance_method in ['fisher', 'fisher_complete']:
                loss = self.criterion(output, label)
            elif self.importance_method == 'mas':
                loss = torch.nn.functional.softmax(output, dim=1).square().sum()

            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    if self.importance_method == 'fisher':
                        precision_matrices[n].data += p.grad.data ** 2  # / len(self.dataloader)
                    elif self.importance_method == 'fisher_complete':
                        p1 = p.grad.data.view(1, -1)
                        precision_matrices[n].data += torch.matmul(p1.T, p1)
                    elif self.importance_method == 'mas':
                        precision_matrices[n].data += p.grad.data.abs()

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                precision_matrices[n].data /= len(self.dataloader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module, lr: float, correct: bool = False):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                if correct:
                    _loss = (self._precision_matrices[n] * self.lmbda * (p - self._means[n]) ** 2 /
                             (self._precision_matrices[n] * self.lmbda * lr + 1)).sum()
                else:
                    if self.importance_method in ['fisher', 'mas']:
                        _loss = (self._precision_matrices[n] * (p - self._means[n]) ** 2).sum()
                    elif self.importance_method == 'fisher_complete':
                        error = ((p - self._means[n]) ** 2).view(1, -1)
                        _loss = torch.matmul(torch.matmul(error, self._precision_matrices[n]), error.T)

                loss += _loss

        return loss * self.lmbda


def ewc_train(model: nn.Module, optimizer: torch.optim, criterion, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, epoch: int, tb: SummaryWriter, global_step: int, ewc_correction: bool):
    model.train()
    epoch_loss = 0
    epoch_ce = 0
    epoch_penalty = 0
    for batch_data in tqdm(data_loader, total=len(data_loader), desc=f'Training epoch {epoch}', leave=False):
        input = batch_data[0].cuda()
        target = batch_data[1].cuda()
        optimizer.zero_grad()
        output = model(input)
        ce = criterion(output, target)
        penalty = ewc.penalty(model, optimizer.param_groups[0]['lr'], ewc_correction)
        loss = ce + penalty

        # tracking loss terms
        if global_step % 10 == 0:
            tb.add_scalar('train/loss', loss.item(), global_step)
            tb.add_scalar('train/cross_entropy', ce.item(), global_step)
            tb.add_scalar('train/penalty', penalty.item(), global_step)
            tb.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        epoch_loss += loss.item()
        epoch_ce += ce.item()
        epoch_penalty += penalty.item()

        loss.backward()
        optimizer.step()
        global_step += 1
    return epoch_loss / len(data_loader), epoch_ce / len(data_loader), epoch_penalty / len(data_loader), global_step


@torch.no_grad()
def test(model: nn.Module, data_loader: torch.utils.data.DataLoader, dataset_name: str, criterion, ewc, lr: float, phase: str,
         tb: SummaryWriter = None, global_step: int = 0):
    model.eval()
    correct = 0
    loss = 0
    for batch_data in tqdm(data_loader, desc=f'{phase}', total=len(data_loader), leave=False):
        input = batch_data[0].cuda()
        target = batch_data[1].cuda()
        output = model(input)
        ce = criterion(output, target)
        penalty = ewc.penalty(model, lr)
        loss += ce + penalty
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()

    total_loss = loss.item() / len(data_loader)
    acc = correct / len(data_loader.dataset)

    if phase == 'validation':
        tb.add_scalar(f'{phase}/{dataset_name}/loss', total_loss, global_step=global_step)
        tb.add_scalar(f'{phase}/{dataset_name}/accuracy', acc, global_step=global_step)

    return acc, total_loss
