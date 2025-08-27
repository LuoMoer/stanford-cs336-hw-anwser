import cs336_basics.model
import cs336_basics.nn_utils
import torch
import argparse
import json
import timeit
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, record_function, ProfilerActivity

def benchmarking(model_args, warm_up, active, backward=False, show_time_list=False, use_profile=False, batch_size=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cs336_basics.model.BasicsTransformerLM(**model_args).to(device)
    data = torch.randint(
        low=0, high=model_args["vocab_size"], 
        size=(batch_size, 256), device=device)

    def _test():
        torch.cuda.empty_cache()
        y = model(data)
        if backward:
            loss = cs336_basics.nn_utils.cross_entropy(y, data)
            loss.backward()
        torch.cuda.empty_cache()
    
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    if use_profile:
        with profile(
            activities=[ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1,warmup=warm_up,active=active),
        ) as prof:
            for _ in range(1+warm_up+active):
                with record_function("test"):
                    _test()
                prof.step()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        for _ in range(warm_up):
            torch.cuda.synchronize()
            _test()
            torch.cuda.synchronize()
        
        avg_time = 0.0
        time_list = []
        
        for _ in range(active):
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            _test()
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            time_list.append(end_time-start_time)
            avg_time += time_list[-1]
            
        
        avg_time/=active
        unit = " s"
        if avg_time < 1:
            time_list = [t*1000 for t in time_list]
            avg_time*=1000
            unit = "ms"

        if show_time_list:
            out_str = [f"{t:.3f}{unit}" for t in time_list]
            print(out_str)
        print(f"GPU time usage: {avg_time:.3f}{unit}")
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--warm_up", type=int)
    parser.add_argument("--active", type=int)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    MODEL_CONFIG_PATH = args.model_config
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_config = json.load(f)
    
    with torch.autocast("cuda", dtype=torch.float16):
        benchmarking(
            model_args=model_config,
            warm_up=args.warm_up,
            active=args.active,
            backward=True,
            show_time_list = True,
            use_profile = True
        )


if __name__ == "__main__":
    main()