import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def quantile_regression_loss(coefficient, source, target):
    diff = target.unsqueeze(-1) - source.unsqueeze(1)
    loss = huber(diff) * (coefficient - (diff.detach() < 0).float()).abs()
    loss = loss.mean()
    return loss


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def unsqe_cat_gather(tensor_list, idx, dim = 1 ):
    tensor_list = [tensor.unsqueeze(dim) for tensor in tensor_list]
    tensors = torch.cat(tensor_list, dim = dim)

    target_shape = list(tensors.shape)
    target_shape[dim] = 1

    view_shape = list(idx.shape) + [1] * (len(target_shape) - len(idx.shape))
    idx = idx.view(view_shape)
    idx = idx.expand(tuple(target_shape))
    tensors = tensors.gather(dim, idx).squeeze(dim)
    return tensors


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
   
def linear_schedule(start, end, start_epochs, epochs):
    dx = (end-start)/epochs
    curr = start
    for t in range(start_epochs):
        yield curr
    for t in range(epochs):
        curr += dx
        yield curr
    while True: yield end

def _sigmoid(x: torch.Tensor, hard: bool=True, threshold:float=0.5):
    if hard:
        soft_sig = torch.sigmoid(x)
        ret = torch.where(soft_sig > threshold, 1.0, 0.0)
        ## straight through - let gradient flow
        ret = ret - x.detach() + soft_sig
    else: ret = torch.functional.sigmoid(x)
    return ret

def _argmax(x: torch.Tensor):
    index = x.max(-1, keepdim=True)[1]
    ret = torch.zeros_like(x).scatter_(-1, index, 1.0)
    ## straight through - let gradient flow
    ret = ret - x.detach() + x
    return ret

'''
@{param} x: torch.Tensor
@{param} threshold: float
@{returns} a tensor of same shape as x, 1.0 for > 0.8*max value and 0.0 for otherwise.
'''
def _threshold(x: torch.Tensor, threshold: float):
    max = x.max(-1, keepdim=True)[0]
    ret = torch.where(x > threshold*max.detach(), x, 0.0)
    ret = ret - x.detach() + x
    return ret


'''
@{param} x: torch.Tensor
@{returns} a probability tensor with small values going to zero.
'''
def _sparsemax(x: torch.Tensor):
    sorted, _ = torch.sort(-x)
    sorted = -sorted
    pfx = sorted.cumsum(dim = -1)
    ks = torch.arange(0, x.shape[-1], dtype=torch.int64).expand(*sorted.shape).to(x.device)
    z = 1 + (1+ks) * sorted
    ks = torch.where(pfx < z, 1+ks, 0).to(x.device) #1 based indexing
    index = ks.max(dim=-1, keepdim=True)[0]
    tau = pfx.gather(dim = -1, index=index-1) #0 based indexing
    tau = (tau-1)/index
    return torch.relu(x - tau)


    
    