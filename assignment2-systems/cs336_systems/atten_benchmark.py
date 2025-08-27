import torch
import timeit
import cs336_basics.model
import triton

d_model_set = [16, 32, 64, 128]
seq_len_set = [256, 1024, 4096, 8192, 16384]
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
torch._dynamo.config.recompile_limit = 100

attens = {
    d_model: cs336_basics.model.CausalMultiHeadSelfAttention(
        d_model, 1, cs336_basics.model.RotaryEmbedding(16384, d_model)).to(device)
    for d_model in d_model_set
}

compiled_attens = {
    d_model: torch.compile(attens[d_model]) 
    for d_model in d_model_set
}

for d_model in d_model_set:
    for seq_len in seq_len_set:
        try:
            atten = attens[d_model]
            x = torch.randn((batch_size, seq_len, d_model), device=device)
            
            # warm_up
            for _ in range(10):
                y = atten(x)
                torch.cuda.synchronize()

            start_time = timeit.default_timer()
            for _ in range(100):
                y = atten(x)
                torch.cuda.synchronize()
            
            end_time = timeit.default_timer()
            avg_time = (end_time-start_time)/100.0
            unit = " s"
            if avg_time < 1:
                avg_time*=1000
                unit = "ms"
            print(f"Mine: d_model: {d_model}, seq_len: {seq_len}, Total time: {avg_time:.3f}{unit}")
        except:
            print(f"Mine: d_model: {d_model}, seq_len: {seq_len}, Failed Run")
        
        try:
            jit_atten = compiled_attens[d_model]
            for _ in range(10):
                y = jit_atten(x)
                torch.cuda.synchronize()

            start_time = timeit.default_timer()
            for _ in range(100):
                y = jit_atten(x)
                torch.cuda.synchronize()
            
            end_time = timeit.default_timer()
            avg_time = (end_time-start_time)/100.0
            unit = " s"
            if avg_time < 1:
                avg_time*=1000
                unit = "ms"
            print(f"JIT: d_model: {d_model}, seq_len: {seq_len}, Total time: {avg_time:.3f}{unit}")
        except:
            print(f"JIT: d_model: {d_model}, seq_len: {seq_len}, Failed Run")
        