import argparse
import numpy as np
import torch
import json
from tqdm import tqdm
import swanlab
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import timeit
from tqdm.contrib import tzip
import random
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

from cs336_basics.nn_utils import *
from cs336_basics.model import *


def get_args():
    parser = argparse.ArgumentParser(description="train_info")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--valid_data_path", type=str)

    args = parser.parse_args()
    return args


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function,
    train_data, valid_data,
    max_steps: int,
    batch_size: int,
    max_seq_len: int,
    log_step: int,
    valid_step: int,
    valid_batch_size: int,
    save_step: int,
    save_path: str,
    start_t = 0,
    train_name = "train_test",
    clip_grad_norm = 1.0,
    ddp=False,
    rank=0,
    world_size=1,
    do_bench=False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = 0
    SAVE_DIR = os.path.join(save_path, train_name)
    data_iter = queue_data_loading(train_data, batch_size, max_seq_len, device, ddp, rank, world_size)

    step_time = 0
    comm_time = 0

    model.train()
    for x, labels in (tqdm(data_iter) if rank == 0 else data_iter):
        # print(f"rank {rank}: step {t}")
        # if do_bench:
        #     step_start_time = timeit.default_timer()
        optimizer.zero_grad()
        y = model(x)
        loss = loss_function(
            y.reshape(-1, y.shape[-1]),
            labels.reshape(-1)
        )
        loss.backward()
        torch.cuda.synchronize()

        # if do_bench:
        #     comm_start_time = timeit.default_timer()
        if ddp:
            model.finish_gradient_synchronization()

            # grad_list = [p.grad for p in model.parameters()]
            # all_grad = torch._utils._flatten_dense_tensors(grad_list)
            # dist.all_reduce(all_grad, async_op=False)
            # all_grad/=world_size
            # all_grad = torch._utils._unflatten_dense_tensors(all_grad, grad_list)
            # for grad, value in zip(grad_list, all_grad):
            #     grad.copy_(value)

            # for p in model.parameters():
            #     dist.all_reduce(p.grad, async_op=False)
            #     p.grad/=world_size
            # torch.cuda.synchronize()
        # if do_bench:
        #     comm_end_time = timeit.default_timer()

        gradient_clipping(model.parameters(), clip_grad_norm, rank=rank)
        optimizer.step(rank=rank)
        # torch.cuda.synchronize()

        # if do_bench:
        #     step_end_time = timeit.default_timer()
        #     step_time+=step_end_time-step_start_time
        #     comm_time+=comm_end_time-comm_start_time
        
        if ddp:
            dist.all_reduce(loss, async_op=False)
            loss/=world_size
        if t % log_step == 0 and rank==0:
            swanlab.log({"train_loss": loss} , step=t)
        
        if t % valid_step == 0 and rank==0:
            model.eval()
            with torch.no_grad():
                data_iter = valid_data_loading(valid_data, valid_batch_size, max_seq_len, device)
                loss = 0.0
                item_sum = 0
                valid_t = len(valid_data+max_seq_len*valid_batch_size-1)//(max_seq_len*valid_batch_size)
                for x, labels in data_iter:
                    y = model(x)
                    loss += loss_function(
                        y.reshape(-1, y.shape[-1]),
                        labels.reshape(-1)
                    ).item()*len(y)
                    item_sum += len(y)
                    torch.cuda.empty_cache()
                valid_loss = loss/item_sum
                # print(f"Step [{t}]: valid loss = {valid_loss}")
                swanlab.log({"valid_loss": valid_loss}, step=t)
        
        if t % save_step == 0 and rank==0:
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            SAVE_PATH = os.path.join(SAVE_DIR, f"checkpoint-{t}.pth")
            save_checkpoint(model, optimizer, t, SAVE_PATH)
        t+=1
    
    # if do_bench:
    #     print(f"rank {rank}:\nSTEP: {step_time:.2f}s\nCOMM: {comm_time:.2f}s\nPER: {comm_time/step_time*100:.2f}%")


def main():
    args = get_args()
    CONFIG_PATH = args.config_path
    TRAIN_DATA_PATH = args.train_data_path
    VALID_DATA_PATH = args.valid_data_path

    CONFIG_PATH = "./config_set/config_single.json"

    train_data = np.load(TRAIN_DATA_PATH, mmap_mode="r")
    valid_data = np.load(VALID_DATA_PATH, mmap_mode="r")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(CONFIG_PATH, "r") as f:
        configs = json.load(f)
    
    for config in configs:
        model_config = config["model_config"]
        optimizer_config = config["optimizer_config"]
        train_config = config["train_config"]
        
        model = BasicsTransformerLM(**model_config).to(device)
    
        optimizer = AdamW(
            model.parameters(), learning_rate_schedule, **optimizer_config)
        loss_function = cross_entropy
        
        train_name = config["train_name"]
        swanlab.init(
            mode="disabled",
            project = "CS336",
            name = train_name,
            config = train_config,
        )
        train(model, optimizer, loss_function, train_data, valid_data, **train_config, train_name=train_name)
        
        
        swanlab.finish()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def ddp_main(rank, world_size):
    setup(rank, world_size)
    args = get_args()
    CONFIG_PATH = args.config_path
    TRAIN_DATA_PATH = args.train_data_path
    VALID_DATA_PATH = args.valid_data_path

    # CONFIG_PATH = args.config_path
    CONFIG_PATH = "./config_set/config_ddp.json"

    torch.cuda.set_device(rank)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = np.load(TRAIN_DATA_PATH, mmap_mode="r")
    valid_data = np.load(VALID_DATA_PATH, mmap_mode="r")

    with open(CONFIG_PATH, "r") as f:
        configs = json.load(f)
    
    for config in configs:
        model_config = config["model_config"]
        optimizer_config = config["optimizer_config"]

        train_config = config["train_config"]
        
        # model = DDP(BasicsTransformerLM(**model_config).to(device))
        model = DDP_bucket(BasicsTransformerLM(**model_config).to(device), 0.00032)
        optimizer = AdamW(
            model.parameters(), learning_rate_schedule, **optimizer_config)

        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                dist.broadcast(p.data, src=0)
        
        loss_function = cross_entropy
        
        train_name = config["train_name"]
        if rank==0:
            swanlab.init(
                mode="disabled",
                project = "CS336",
                name = train_name,
                config = train_config,
            )
        train(model, optimizer, loss_function, train_data, valid_data, **train_config,
         train_name=train_name, ddp=True, rank=rank, world_size=world_size, do_bench=True)
        
        torch.cuda.synchronize()
        step_start = timeit.default_timer()
        train(model, optimizer, loss_function, train_data, valid_data, **train_config,
         train_name=train_name, ddp=True, rank=rank, world_size=world_size, do_bench=True)
        torch.cuda.synchronize()
        step_end = timeit.default_timer()
        print(f"rank {rank}: STEP {step_end-step_start:.2f}s")

        if rank==0:
            swanlab.finish()


def ddp():
    world_size = 4
    mp.spawn(ddp_main, args=(world_size, ), nprocs=world_size, join=True)


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.module = module
        self.grad_handles = []
        self._sync_weight()
        self._register_gradient_hooks()
    
    def _register_gradient_hooks(self):
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad, p=p: self.gradient_synchronization(grad, p))
    
    def _sync_weight(self):
        weight_list = []
        for p in self.module.parameters():
            weight_list.append(p.data)
        weights = torch._utils._flatten_dense_tensors(weight_list)
        dist.broadcast(weights, src=0)
        weights = torch._utils._unflatten_dense_tensors(weights, weight_list)
        for weight, value in zip(weight_list, weights):
            weight.copy_(value)
    
    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)
    
    def gradient_synchronization(self, grad, p):
        if grad is not None:
            grad = grad.contiguous()
            handle = dist.all_reduce(grad, async_op=True)
            self.grad_handles.append((handle, grad, p))
            return grad
    
    def finish_gradient_synchronization(self):
        for handle, grad, p in self.grad_handles:
            handle.wait()
        
        for handle, grad, p in self.grad_handles:
            grad/=self.world_size
            p.grad.copy_(grad)
        self.grad_handles.clear()


class DDP_bucket(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.grad_handles = []
        self._sync_weight()
    
    def _sync_weight(self):
        weight_list = []
        for p in self.module.parameters():
            weight_list.append(p.data)
        weights = torch._utils._flatten_dense_tensors(weight_list)
        dist.broadcast(weights, src=0)
        weights = torch._utils._unflatten_dense_tensors(weights, weight_list)
        for weight, value in zip(weight_list, weights):
            weight.copy_(value)
    
    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)
    
    def finish_gradient_synchronization(self):
        grad_list = []
        current_size = 0
        for p in reversed(list(self.module.parameters())):
            if p.grad is None:
                continue
            p_size = p.grad.numel() * p.grad.element_size() / (1024*1024)
            if current_size + p_size < self.bucket_size_mb:
                grad_list.append(p.grad)
                current_size += p_size
            else:
                if current_size:
                    all_grad = torch._utils._flatten_dense_tensors(grad_list)
                    handle = dist.all_reduce(all_grad, async_op=True)
                    self.grad_handles.append((handle, list(grad_list), all_grad))
                current_size = 0
                grad_list.clear()
                grad_list.append(p.grad)
                current_size += p_size

        if current_size:
            all_grad = torch._utils._flatten_dense_tensors(grad_list)
            handle = dist.all_reduce(all_grad, async_op=True)
            self.grad_handles.append((handle, list(grad_list), all_grad))
            current_size = 0
            grad_list.clear()
        
        for handle, grad_list, all_grad in self.grad_handles:
            handle.wait()
        
        for _, grad_list, all_grad in self.grad_handles:
            all_grad/=self.world_size
            all_grad = torch._utils._unflatten_dense_tensors(all_grad, grad_list)
            for grad, value in zip(grad_list, all_grad):
                grad.copy_(value)
        
        self.grad_handles.clear()


if __name__ == "__main__":
    # main()
    ddp()