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


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        if not dtype:
            dtype = torch.float32
        device = 'cpu'
        if torch.cuda.is_available():
            device='cuda'
        sigma = (2 / (in_features + out_features)) ** 0.5
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0*sigma, b=3.0*sigma)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.weight.device)
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        if not dtype:
            dtype = torch.float32
        device = 'cpu'
        if torch.cuda.is_available():
            device='cuda'
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim),device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(self.weight.device)
        return self.weight[token_ids]
        
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not dtype:
            dtype = torch.float32
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones((d_model,),device=device,dtype=dtype))
    
    def forward(self, x:torch.Tensor):
        in_dtype = x.dtype
        x = x.to(self.weight.device).to(torch.float32)
        frac = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps)
        x = x/frac
        x = einsum(self.weight, x, "d_model, ... d_model -> ... d_model")
        x = x.to(in_dtype)
        return x


class FFN_SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if d_ff % 64 != 0:
            self.d_ff = ((d_model*8+3-1)/3 + 64 - 1) // 64 * 64
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not dtype:
            dtype = torch.float32
        
        self.w1 = Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2 = Linear(d_ff,d_model,device=device,dtype=dtype)
        self.w3 = Linear(d_model,d_ff,device=device,dtype=dtype)

    def silu(self, x: torch.Tensor)->torch.Tensor:
        return x*torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = x.to(self.w1.weight.device)
        assert x.size(-1) == self.d_model
        silu_x = self.silu(self.w1(x))
        glu_x = silu_x * self.w3(x)
        return self.w2(glu_x)


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        assert d_k%2==0
        d = torch.arange(0,d_k,2)/d_k
        freqs = theta ** -d
        i = torch.arange(max_seq_len)
        freqs = einsum(i, freqs, "i, k-> i k")
        cos, sin = torch.cos(freqs), torch.sin(freqs)

        self.register_buffer("cos", cos.to(device), persistent=False)
        self.register_buffer("sin", sin.to(device), persistent=False)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = x.to(self.cos.device)
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        seq_len = x.size(-2)
        rotated_even = x_even*self.cos[:seq_len] - x_odd*self.sin[:seq_len]
        rotated_odd = x_even*self.sin[:seq_len] + x_odd*self.cos[:seq_len]
        result = torch.empty_like(x)
        result[..., ::2]=rotated_even
        result[..., 1::2]=rotated_odd
        return result

def softmax(x: torch.Tensor, dim: int = -1)->torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(x)
    frac = torch.sum(exp, dim=dim, keepdim=True)
    return exp/frac

def softmax_temperature(x: torch.Tensor, temperature: float = 1, dim: int = -1)->torch.Tensor:
    x/=temperature
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(x)
    frac = torch.sum(exp, dim=dim, keepdim=True)
    return exp/frac

def scaled_dot_product_attention(
    Q: torch.Tensor,K: torch.Tensor,
    V: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
    d_k = K.size(-1)
    score = einsum(Q,K,"... n d_k, ... m d_k -> ... n m")/(d_k**0.5)
    if mask is not None:
        score =  torch.where(mask, score, float("-inf"))
    score = softmax(score)
    return einsum(V, score, "... m d_v, ... n m -> ... n d_v")


class multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model, num_heads, position_embedding=None):
        super().__init__()
        assert d_model%num_heads==0
        self.d_model = d_model
        self.num_heads = num_heads
        self.position_embedding = position_embedding
        self.d_h = d_model//num_heads
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask:torch.Tensor = None )->torch.Tensor:
        assert self.d_model == x.size(-1) 
        batch = x.size(0)
        seq_len = x.size(-2)
        x = x.to(self.q_proj.weight.device)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q, K, V = (rearrange(X, "... seq (num_heads d_h) -> ... num_heads seq d_h", d_h = self.d_h) for X in (Q,K,V)) 

        if self.position_embedding:
            Q = self.position_embedding(Q)
            K = self.position_embedding(K)
        causal_mask = torch.tril(torch.ones(batch,self.num_heads,seq_len,seq_len,device=x.device)).bool()
        if mask:
            mask = mask.unsqueeze(1).unsqueeze(2)
            causal_mask &= mask
        atten_output = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal_mask)
        atten_output = rearrange(atten_output, "... num_heads seq d_h -> ... seq (num_heads d_h)").contiguous()
        return self.output_proj(atten_output)

class transformer_block(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, position_embedding=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ln1 = RMSNorm(d_model=d_model, device=self.device)
        self.attn = multihead_self_attention(d_model=d_model, num_heads=num_heads, position_embedding=position_embedding)
        self.ln2 = RMSNorm(d_model=d_model, device=self.device)
        self.ffn = FFN_SwiGLU(d_model=d_model, d_ff=d_ff, device=self.device)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None)->torch.Tensor:
        x = x.to(self.device)
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class transformer_lm(torch.nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int, rope_theta):

        super().__init__()
        assert d_model % num_heads == 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_model//num_heads, max_seq_len=context_length, device=device)
        self.layers = torch.nn.ModuleList([transformer_block(
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, position_embedding=self.rope) for i in range(num_layers)])
        self.ln_final = RMSNorm(d_model=d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        x.to(self.device)
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
    
    @torch.no_grad()
    def generate(self, x: str, tokenizer, max_seq_len: int = 256, temperature: float = 1.0, top_p: float = 1.0):
        EOF_TOKEN = tokenizer.encode("<|endoftext|>")[0]
        x = tokenizer.encode(x) 
        t = len(x)
        x = torch.from_numpy(np.array([x])).long().to(self.device)
        while t < max_seq_len:
            logits = softmax_temperature(
                self.forward(x)[: , -1, :].reshape(-1), temperature=temperature)
            
            logits, id = torch.sort(logits, descending=True, dim=-1)
            sum, threshold = 0, 0
            for item in logits:
                sum+=item
                if sum > top_p:
                    threshold = item
                    break
            
            top_p_mask = logits < threshold
            logits = logits.masked_fill(top_p_mask, float(0.0))/sum
            new_token_id = id[torch.multinomial(logits, num_samples=1, replacement=False)[0]]
            if(new_token_id.item()==EOF_TOKEN):
                break
            x = torch.cat((x, new_token_id.unsqueeze(0).unsqueeze(0)), dim=-1)
            t+=1
            
        return tokenizer.decode(x.squeeze(0).tolist())




        