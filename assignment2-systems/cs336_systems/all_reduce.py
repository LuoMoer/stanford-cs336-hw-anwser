import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import timeit


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(BACKEND, rank=rank, world_size=world_size)


def distributed_demo(rank, world_size):
    setup(rank, world_size)
    if DEVICE=="cuda":
        torch.cuda.set_device(rank)
    data = torch.randn((DATA_SIZE,), dtype=torch.float32, device=DEVICE)

    warmup, active = 5, 10
    
    for _ in range(warmup):
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
    
    start_t = timeit.default_timer()
    for _ in range(active):
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
    end_t = timeit.default_timer()
    time = [0 for _ in range(world_size)]
    dist.all_gather_object(time, (end_t-start_t)/active)
    if rank==0:
        time = [f"{t*1000:.2f}ms" if t<1.0 else f"{t:.2f}s" for t in time]
        print(time)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--data_size", type=int, default=2**10)
    args = parser.parse_args()
    return args


args = get_args()
DEVICE = args.device
BACKEND = "nccl" if DEVICE=="cuda" else "gloo"
DATA_SIZE = args.data_size
world_size = args.world_size


if __name__ == '__main__':
    args = get_args()
    mp.spawn(distributed_demo, args=(world_size, ), nprocs=world_size, join=True)
