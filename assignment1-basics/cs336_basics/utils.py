from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor
from einops import rearrange, einsum
import numpy as np
import swanlab


def data_loading(dataset, batch_size, context_length, device="cuda"):
    starts = np.random.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,)
    )

    inputs = np.stack([dataset[start:start+context_length] for start in starts])
    labels = np.stack([dataset[start+1:start+context_length+1] for start in starts])

    return (
        torch.from_numpy(inputs).long().to(device),
        torch.from_numpy(labels).long().to(device)
    )

def valid_data_loading(dataset, batch_size, context_length, device="cuda"):
    inputs = []
    labels = []
    cnt = 0

    for start in range(0, len(dataset)-context_length, context_length):
        inputs.append(dataset[start:start+context_length])
        labels.append(dataset[start+1:start+context_length+1])
        cnt+=1
        if cnt == batch_size:
            yield (
                torch.from_numpy(np.stack(inputs)).long().to(device),
                torch.from_numpy(np.stack(labels)).long().to(device)
            )
            cnt = 0
            inputs = []
            labels = []
    if cnt != 0:
        yield (
            torch.from_numpy(np.stack(inputs)).long().to(device),
            torch.from_numpy(np.stack(labels)).long().to(device)
        )


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    logits = logits - logits.max(dim=-1, keepdim=True).values
    o = -logits.gather(1, targets.unsqueeze(1)).unsqueeze(1)
    frac = torch.log(torch.sum(torch.exp(logits), dim=-1))
    return torch.mean(o+frac)

# def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
#     # print(logits.shape)
#     # print(targets.shape)
#     logits = logits - logits.max(dim=-1, keepdim=True).values
#     o = -logits.gather(-1, targets.unsqueeze(-1)).unsqueeze(-1)
#     frac = torch.log(torch.sum(torch.exp(logits), dim=-1))
#     return torch.mean(o+frac)


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    p_with_grad = [p for p in parameters if p.grad is not None]
    
    if len(p_with_grad) == 0:
        return
    l2 = torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in p_with_grad))
    factor = max_l2_norm/(l2+eps)

    swanlab.log({"grad_norm": l2})

    if max_l2_norm < l2:
        for p in p_with_grad:
            p.grad.mul_(factor)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, learning_rate_schedule,
        lr_max=1e-3, lr_min=0, 
        warmup_iters=None, cosine_cycle_iters=None,
        betas=(0.9, 0.999), 
        eps=1e-8, weight_decay=1e-2
    ):
        defaults = {
            "lr_max": lr_max,
            "lr_min": lr_min,
            "warmup_iters": warmup_iters,
            "cosine_cycle_iters": cosine_cycle_iters,
            "betas":betas, "eps":eps, "weight_decay":weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr_max = group["lr_max"]
            lr_min = group["lr_min"]
            warmup_iters = group["warmup_iters"]
            cosine_cycle_iters = group["cosine_cycle_iters"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            lr = None

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                t = state["t"]
                m = state["m"]
                v = state["v"]
                if lr == None:
                    lr = learning_rate_schedule(
                        t, lr_max, lr_min, warmup_iters, cosine_cycle_iters
                    )
                    if t % 10 == 0:
                        swanlab.log({"lr": lr}, step=t)
                
                grad = p.grad.data

                m.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                v.mul_(betas[1]).addcmul_(grad,grad,value=1-betas[1])
                lr_t = lr*((1-betas[1]**t)**0.5)/(1-betas[0]**t)
                tmp_v = v.sqrt().add_(eps)
                p.data.addcdiv_(m, tmp_v, value=-lr_t)
                p.data.mul_(1-lr*weight_decay)

                state["t"] += 1
                state["m"] = m
                state["v"] = v
        return loss


def learning_rate_schedule(it: int, 
    max_learning_rate: float, min_learning_rate: float,
    warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return (it/warmup_iters)*max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        cos = np.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*np.pi)
        return min_learning_rate + 0.5*(1+cos)*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    
    torch.save(checkpoint, out)


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)

    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    iteration = checkpoint["iteration"]

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    return iteration

