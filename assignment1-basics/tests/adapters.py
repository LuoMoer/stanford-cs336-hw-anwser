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

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear = Linear(d_in, d_out)
    state_dict = {"w": weights} 
    linear.load_state_dict(state_dict)
    return linear(in_features)
    

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

    

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = Embedding(vocab_size, d_model)
    state_dict = {"embedding": weights}
    embedding.load_state_dict(state_dict)
    return embedding(token_ids)

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

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    ffn = FFN_SwiGLU(d_model, d_ff)
    state_dict = {"w1.w":w1_weight, "w2.w":w2_weight, "w3.w":w3_weight}
    ffn.load_state_dict(state_dict)
    return ffn(in_features)


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


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return scaled_dot_product_attention(Q,K,V,mask)
    

def scaled_dot_product_attention(
    Q: torch.Tensor,K: torch.Tensor,
    V: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
    d_k = K.size(-1)
    score = einsum(Q,K,"... n d_k, ... m d_k -> ... n m")/(d_k**0.5)
    assert mask.shape == score.shape, f"Mask shape {mask.shape} doesn't match score shape {score.shape}"
    if mask is not None:
        score =  torch.where(mask, score, float("-inf"))
    score = softmax(score)
    return einsum(V, score, "... m d_v, ... n m -> ... n d_v")



def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha = multihead_self_attention(d_model, num_heads)
    state_dict = {
        "q_proj.w":q_proj_weight,
        "k_proj.w":k_proj_weight,
        "v_proj.w":v_proj_weight,
        "o_proj.w":o_proj_weight,
    }
    mha.load_state_dict(state_dict)
    
    return mha(in_features) 

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

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    rope = RotaryPositionalEmbedding(theta=theta,d_k=d_model//num_heads,max_seq_len=max_seq_len)
    mha = multihead_self_attention(d_model, num_heads, rope)
    state_dict = {
        "q_proj.w":q_proj_weight,
        "k_proj.w":k_proj_weight,
        "v_proj.w":v_proj_weight,
        "o_proj.w":o_proj_weight,
    }
    mha.load_state_dict(state_dict)
    return mha(in_features) 
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)

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
        x.to(self.cos.device)
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        seq_len = x.size(-2)
        rotated_even = x_even*self.cos[:seq_len] - x_odd*self.sin[:seq_len]
        rotated_odd = x_even*self.sin[:seq_len] + x_odd*self.cos[:seq_len]
        result = torch.empty_like(x)
        result[..., ::2]=rotated_even
        result[..., 1::2]=rotated_odd
        return result

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model//num_heads, max_seq_len=max_seq_len)
    block = transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, position_embedding=rope)
    block.load_state_dict(weights)
    return block(in_features)

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


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    lm = transformer_lm(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        rope_theta=rope_theta)
    lm.load_state_dict(weights)
    return lm(in_indices)

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

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = RMSNorm(d_model, eps)
    state_dict = {"g":weights}
    rmsnorm.load_state_dict(state_dict)
    return rmsnorm(in_features)


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



def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    return data_loading(dataset, batch_size, context_length, device)

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



def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax(in_features, dim)

def softmax(x: torch.Tensor, dim: int=-1)->torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    exp = torch.exp(x)
    frac = torch.sum(exp, dim=dim, keepdim=True)
    return exp/frac


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return cross_entropy(inputs, targets)

# def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
#     logits = logits - logits.max(dim=-1, keepdim=True).values
#     o = -logits.gather(1, targets.unsqueeze(1)).unsqueeze(1)
#     frac = torch.log(torch.sum(torch.exp(logits), dim=-1))
#     return torch.mean(o+frac)

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    # print(logits.shape)
    # print(targets.shape)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    o = -logits.gather(-1, targets.unsqueeze(-1)).unsqueeze(-1)
    frac = torch.log(torch.sum(torch.exp(logits), dim=-1))
    return torch.mean(o+frac)






def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    gradient_clipping(parameters, max_l2_norm)

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    p_with_grad = [p for p in parameters if p.grad is not None]
    
    if len(p_with_grad) == 0:
        return
    
    l2 = torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in p_with_grad))
    factor = max_l2_norm/(l2+eps)

    if max_l2_norm < l2:
        for p in p_with_grad:
            p.grad.mul_(factor)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = {"lr": lr, "betas":betas, "eps":eps, "weight_decay":weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

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



def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    return learning_rate_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)

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


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    save_checkpoint(model, optimizer, iteration, out)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    
    torch.save(checkpoint, out)



def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    return load_checkpoint(src, model, optimizer)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)

    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    iteration = checkpoint["iteration"]

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    return iteration


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_token_bytes = [special_token.encode("utf-8") for special_token in self.special_tokens]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.token_to_id = {}
        for id, token in vocab.items():
            self.token_to_id[token]=id
        
        # Add special_token to vocab if not already present
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.token_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id
    
    def encode(self, text: str) -> list[int]:
        # str -> token_dis
        import regex as re
        token_ids = []
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))

        if pattern:
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]
        for part in parts:
            if part in self.special_tokens:
                token_ids.append(self.token_to_id[part.encode("utf-8")])
            else:
                token_ids.extend(self._get_token_ids(part))
        
        return token_ids
    
    def decode(self, ids: list[int]) -> str:
        # token_ids -> str
        bytes_seq = b"".join([self.vocab[id] for id in ids])
        return bytes_seq.decode("utf-8", errors="replace")
            
    def encode_iterable(self, iterable: Iterable[str]) -> iter:
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def _get_token_ids(self, text: str) -> list[int]:
        # print('!'*50+"DEBUG get_token_ids"+'!'*50)
        # print([text])
        import regex as re
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        iter = re.finditer(PAT, text)
        token_ids = []
        for item in iter:
            token_seq = item[0].encode("utf-8")
            token = self._merge(token_seq)
            token_ids.extend([self.token_to_id[b] for b in token])
        return token_ids
    
    def _merge(self, token_seq: bytes):
        token = [bytes([b]) for b in token_seq]
        def get_pairs(word):
            pairs = set()
            prev_b = word[0]
            for b in word[1:]:
                pairs.add((prev_b, b))
                prev_b = b
            return pairs
        pairs = get_pairs(token)

        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key = lambda pair:self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            a,b=bigram
            new_token=[]
            i=0
            while i < len(token):
                try:
                    j=token.index(a,i)
                except ValueError:
                    new_token.extend(token[i:])
                    break
                else:
                    new_token.extend(token[i:j])
                    i=j
                if token[i]==a and i+1<len(token) and token[i+1]==b:
                    new_token.append(a+b)
                    i+=2
                else:
                    new_token.append(token[i])
                    i+=1
            token = new_token
            if len(token) == 1:
                break
            else:
                pairs = get_pairs(token)
        return token

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # initialize vocab
    vocab = {}
    current_vocab_size = 0
    for special_token in special_tokens:
        vocab[current_vocab_size] = special_token.encode("utf-8")
        current_vocab_size+=1
    for i in range(256):
        vocab[current_vocab_size] = bytes([i])
        current_vocab_size+=1
    
    
    # pretokenization
    import regex as re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "r", encoding="utf-8") as f:
        texts = f.read()
        chunks = re.split("|".join(map(re.escape, special_tokens)), texts)
    
    word_num = {}
    words = {}

    for texts in chunks:
        iter = re.finditer(PAT, texts)
        for i in iter:
            word_num[i[0].encode("utf-8")] = word_num.get(i[0].encode("utf-8"), 0) + 1
            words[i[0].encode("utf-8")] = []
            for b in i[0].encode("utf-8"):
                words[i[0].encode("utf-8")].append(bytes([b]))
    
    # merge
    merges = []
    from tqdm import tqdm

    assert vocab_size >= current_vocab_size
    
    pairs = {}
    for word, word_bytes in words.items():
        for i in range(len(word_bytes)-1):
            pairs[(word_bytes[i],word_bytes[i+1])] = pairs.get((word_bytes[i],word_bytes[i+1]), 0) + word_num[word] 

    for loop in tqdm(range(vocab_size-current_vocab_size)):
        if len(pairs) == 0:
            print(f"Vocab_size too big! The max Vocab_size is [{current_vocab_size}].")
            break
        
        max_count = max(pairs.values())
        candidates = [k for k, v in pairs.items() if v == max_count]
        best_pair = max(candidates)

        merges.append(best_pair)
        vocab[current_vocab_size] = merges[-1][0]+merges[-1][1]
        current_vocab_size+=1
        
        for word in words.keys():
            new_word_bytes = []
            i = 0
            a = merges[-1][0]
            b = merges[-1][1]
            new_word = a+b
            if new_word not in word:
                continue
            while i < len(words[word]):
                if i+1 < len(words[word]) and words[word][i]==a and words[word][i+1]==b:
                    new_word_bytes.append(new_word)
                    num = word_num[word]
                    if len(new_word_bytes) > 1:
                        # 前面还有至少一个元素 new_word_bytes[-2]
                        left_word = new_word_bytes[-2]
                        pairs[(left_word,a)]-=num
                        if pairs[(left_word,a)] == 0:
                            del pairs[(left_word,a)]
                        pairs[(left_word,new_word)]=pairs.get((left_word,new_word),0)+num
                    if i+2 < len(words[word]):
                        # 后面至少还有一个元素
                        right_word = words[word][i+2]
                        pairs[(b,right_word)]-=num
                        if pairs[(b,right_word)] == 0:
                            del pairs[(b,right_word)]
                        pairs[(new_word,right_word)]=pairs.get((new_word,right_word),0)+num
                    i+=2
                else:
                    new_word_bytes.append(words[word][i])
                    i+=1
            words[word] = new_word_bytes
        del pairs[(merges[-1][0],merges[-1][1])]

    return vocab, merges

# vocab, merges = run_train_bpe("/home/swluo/data/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",500,["<|endoftext|>"])

# tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

# tokenizer.encode("hello my darling.\n <|endoftext|> I am luomo.\n")