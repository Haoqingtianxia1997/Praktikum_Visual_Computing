import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
from utils.dice_score import dice_loss
from utils.data_loading import BasicDataset, CarvanaDataset
import os
import random
import pickle


def compute_fisher_information(model, data_loader, criterion, device):
    fisher_information = {}

    # 初始化 Fisher Information 矩阵
    for name, param in model.named_parameters():
        fisher_information[name] = torch.zeros_like(param)

    model.eval()
    for batch in data_loader:
        data, target = batch['image'], batch['mask']
        data = data.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        target = target.to(device=device, dtype=torch.long)
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=False):
            output = model(data)
            if model.n_classes == 1:
                loss = criterion(output.squeeze(1), target.float())
                loss += dice_loss(F.sigmoid(output.squeeze(1)), target.float(), multiclass=False)
            else:
                loss = criterion(output, target)
                loss += dice_loss(
                    F.softmax(output, dim=1).float(),
                    F.one_hot(target, model.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
        model.zero_grad()
        loss.backward()
        # 累加梯度的平方
        for name, param in model.named_parameters():
            fisher_information[name] += param.grad ** 2
    # 求平均
    for name in fisher_information:
        fisher_information[name] /= len(data_loader)

    return fisher_information


def save_information(information, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(information, f)


def load_information(filepath):
    with open(filepath, 'rb') as f:
        information = pickle.load(f)
    return information


def ewc_loss(model, fisher_information, optimal_params, lambda_ewc, base_loss, device):
    ewc_term = 0
    for name, param in model.named_parameters():
        fisher = fisher_information.get(name, torch.zeros_like(param))
        opt_param = optimal_params[name]
        param = param.to(device)
        opt_param = opt_param.to(device)
        fisher = fisher.to(device)
        ewc_term += (fisher * (param - opt_param) ** 2).sum()
    return base_loss + lambda_ewc * ewc_term
