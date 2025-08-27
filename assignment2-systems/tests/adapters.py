from __future__ import annotations

from typing import Type

import torch

from einops import rearrange, einsum
import triton
import triton.language as tl
import torch.distributed as dist


def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyFlashAttnAutogradFunctionClass
    return PyFlashAttention

def cdiv(x, y):
    return (x+y-1)//y

class PyFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        B_Q = cdiv(Q.size(-2), Q_TILE_SIZE)
        B_K = cdiv(K.size(-2), K_TILE_SIZE)
        Q = Q.to(torch.float32)
        K = K.to(torch.float32)
        V = V.to(torch.float32)
        O = torch.empty_like(Q)
        L = torch.zeros((Q.size(0), Q.size(-2)), device=Q.device).to(torch.float32)

        for i in range(B_Q):
            # ..., Q_TILE_SIZE, d
            Q_tile = Q[...,i*Q_TILE_SIZE:(i+1)*Q_TILE_SIZE,:]
            # ..., Q_TILE_SIZE, d
            O_tile = torch.empty_like(Q_tile)
            # ..., Q_TILE_SIZE
            L_tile = torch.zeros((Q_tile.size(0), Q_TILE_SIZE), device=Q.device)
            # ..., Q_TILE_SIZE
            M_tile = torch.full((Q_tile.size(0), Q_TILE_SIZE), -float("inf"), device=Q.device)
            for j in range(B_K):
                # ..., K_TILE_SIZE, d
                K_tile = K[...,j*K_TILE_SIZE:(j+1)*K_TILE_SIZE,:]
                # ..., Q_TILE_SIZE, d
                V_tile = V[...,j*K_TILE_SIZE:(j+1)*K_TILE_SIZE,:]

                ..., Q_TILE_SIZE, K_TILE_SIZE
                S_tile = einsum(
                    Q_tile, K_tile, 
                    "... Q_TILE_SIZE d, ... K_TILE_SIZE d -> ... Q_TILE_SIZE K_TILE_SIZE"
                )/(Q_tile.size(-1)**0.5)

                if is_causal:
                    S_tile = torch.where(
                        torch.arange(i*Q_TILE_SIZE, (i+1)*Q_TILE_SIZE, device=Q.device)[None, :, None] >= torch.arange(j*K_TILE_SIZE, (j+1)*K_TILE_SIZE, device=Q.device)[None, None, :],
                        S_tile, -1e6
                    )
            
                # ..., Q_TILE_SIZE
                new_M_tile = torch.max(S_tile, axis=-1)[0]
                new_M_tile = torch.where(new_M_tile>M_tile, new_M_tile, M_tile)

                # ..., Q_TILE_SIZE, K_TILE_SIZE
                P_tile = torch.exp(S_tile-new_M_tile.unsqueeze(-1))

                # ..., Q_TILE_SIZE
                L_tile = torch.exp(M_tile-new_M_tile)*L_tile + torch.sum(P_tile, axis=-1)

                # ..., Q_TILE_SIZE, d
                # O_tile = torch.diag_embed(torch.exp(M_tile-new_M_tile))@O_tile + P_tile@V_tile 
                O_tile = torch.exp(M_tile-new_M_tile).unsqueeze(-1)*O_tile + P_tile@V_tile 

                M_tile = new_M_tile

            O_tile = (L_tile**-1).unsqueeze(-1)*O_tile
            L_tile = M_tile + torch.log(L_tile)
            # if i==0:
            #     print("L: ", L_tile[0,...])

            O[...,i*Q_TILE_SIZE:(i+1)*Q_TILE_SIZE,:] = O_tile
            L[...,i*Q_TILE_SIZE:(i+1)*Q_TILE_SIZE] = L_tile


        # print(O)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        ctx.scale = Q.size(-1)**0.5
        return O

    @staticmethod
    @torch.compile
    def backward(ctx, grad_O):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        S = Q@K.transpose(-2,-1)/ctx.scale
        N_QUERIES = Q.size(-2)
        N_KEYS = K.size(-2)
        D = Q.size(-1)
        if is_causal:
            S = torch.where(
                torch.arange(N_QUERIES, device=Q.device)[None, :, None] >= torch.arange(N_KEYS, device=Q.device)[None, None, :],
                S, -1e6
            )
        
        # S: ..., N_Q, N_K
        # L: ..., N_Q
        P = torch.exp(S-L[:, :, None])

        # P: ..., N_Q, N_K
        # grad_O: ..., N_Q, d
        grad_V = (P.transpose(-2,-1))@grad_O
        grad_P = grad_O@(V.transpose(-2,-1))

        grad_S = P*(grad_P-torch.sum(O*grad_O, axis=-1, keepdim=True))
        grad_Q = grad_S@K/ctx.scale
        grad_K = grad_S.transpose(-2,-1)@Q/ctx.scale

        return grad_Q, grad_K, grad_V, None


def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyTritonFlashAttentionAutogradFunctionClass
    return TritonFlashAttention
    return PyFlashAttention


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    queries_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr+batch_index*stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(queries_tile_index*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr+batch_index*stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr+batch_index*stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr+batch_index*stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(queries_tile_index*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr+batch_index*stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(queries_tile_index*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)

    Q_tile_index = tl.arange(0, Q_TILE_SIZE) + queries_tile_index*Q_TILE_SIZE
    O_tile = tl.zeros_like(Q_tile)
    L_tile = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    M_tile = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    new_M_tile = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
        V_tile = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)

        S_tile = tl.dot(Q_tile, K_tile.T)/scale

        if is_causal:
            K_tile_index = tl.arange(0, K_TILE_SIZE) + j*K_TILE_SIZE
            S_tile = tl.where(Q_tile_index[:, None]>=K_tile_index[None, :], S_tile, -1e6)

        new_M_tile = tl.max(S_tile, axis=-1)
        new_M_tile = tl.where(new_M_tile > M_tile, new_M_tile, M_tile)

        P_tile = tl.exp(S_tile-new_M_tile[:, None])

        L_tile = tl.exp(M_tile-new_M_tile)*L_tile + tl.sum(P_tile, axis=-1)

        O_tile = tl.exp(M_tile-new_M_tile)[:, None]*O_tile + tl.dot(P_tile, V_tile)
        M_tile = new_M_tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        
    
    
    O_tile = O_tile/L_tile[:, None]
    
    L_tile = M_tile + tl.log(L_tile)
    # if (batch_index==0 and queries_tile_index==0):
    #     tl.device_print("L_tile: ", L_tile)
    
    tl.store(O_block_ptr, O_tile, boundary_check=(0,1))
    tl.store(L_block_ptr, L_tile, boundary_check=(0,))


@triton.jit
def flash_bkw_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    grad_Q_ptr, grad_K_ptr, grad_V_ptr,
    grad_O_ptr, do_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_gqb, stride_gqq, stride_gqd,
    stride_gkb, stride_gkk, stride_gkd,
    stride_gvb, stride_gvk, stride_gvd,
    stride_gob, stride_goq, stride_god,
    stride_dob, stride_doq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    kyes_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr+batch_index*stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr+batch_index*stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(kyes_tile_index*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr+batch_index*stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(kyes_tile_index*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr+batch_index*stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    grad_Q_block_ptr = tl.make_block_ptr(
        grad_Q_ptr+batch_index*stride_gqb,
        shape=(N_QUERIES, D),
        strides=(stride_gqq, stride_gqd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    grad_K_block_ptr = tl.make_block_ptr(
        grad_K_ptr+batch_index*stride_gkb,
        shape=(N_KEYS, D),
        strides=(stride_gkk, stride_gkd),
        offsets=(kyes_tile_index*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    
    grad_V_block_ptr = tl.make_block_ptr(
        grad_V_ptr+batch_index*stride_gvb,
        shape=(N_KEYS, D),
        strides=(stride_gvk, stride_gvd),
        offsets=(kyes_tile_index*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    grad_O_block_ptr = tl.make_block_ptr(
        grad_O_ptr+batch_index*stride_gob,
        shape=(N_QUERIES, D),
        strides=(stride_goq, stride_god),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    D_block_ptr = tl.make_block_ptr(
        do_ptr+batch_index*stride_dob,
        shape=(N_QUERIES,),
        strides=(stride_doq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    K_tile = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
    V_tile = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
    grad_K_tile = tl.zeros_like(K_tile)
    grad_V_tile = tl.zeros_like(V_tile)

    K_tile_index = tl.arange(0, K_TILE_SIZE) + kyes_tile_index*K_TILE_SIZE

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_tile = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
        grad_O_tile = tl.load(grad_O_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
        
        L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        D_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        
        S_tile = tl.dot(Q_tile, K_tile.T)/scale

        if is_causal:
            Q_tile_index = tl.arange(0, Q_TILE_SIZE) + i*Q_TILE_SIZE
            S_tile = tl.where(Q_tile_index[:, None]>=K_tile_index[None, :], S_tile, -1e6)
 
        P_tile = tl.exp(S_tile-L_tile[:, None])

        grad_V_tile += tl.dot(P_tile.T, grad_O_tile)

        grad_P_tile = tl.dot(grad_O_tile, V_tile.T)
        grad_S_tile = P_tile*(grad_P_tile - D_tile[:, None])/scale
        
        grad_K_tile += tl.dot(grad_S_tile.T, Q_tile)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        grad_Q_block_ptr = grad_Q_block_ptr.advance((Q_TILE_SIZE, 0))
        grad_O_block_ptr = grad_O_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
    
    tl.store(grad_K_block_ptr, grad_K_tile, boundary_check=(0,1))
    tl.store(grad_V_block_ptr, grad_V_tile, boundary_check=(0,1))


@triton.jit
def flash_bkw_q_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    grad_Q_ptr, grad_K_ptr, grad_V_ptr,
    grad_O_ptr, do_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_gqb, stride_gqq, stride_gqd,
    stride_gkb, stride_gkk, stride_gkd,
    stride_gvb, stride_gvk, stride_gvd,
    stride_gob, stride_goq, stride_god,
    stride_dob, stride_doq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    queries_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr+batch_index*stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(queries_tile_index*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr+batch_index*stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr+batch_index*stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr+batch_index*stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(queries_tile_index*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    grad_Q_block_ptr = tl.make_block_ptr(
        grad_Q_ptr+batch_index*stride_gqb,
        shape=(N_QUERIES, D),
        strides=(stride_gqq, stride_gqd),
        offsets=(queries_tile_index*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    grad_O_block_ptr = tl.make_block_ptr(
        grad_O_ptr+batch_index*stride_gob,
        shape=(N_QUERIES, D),
        strides=(stride_goq, stride_god),
        offsets=(queries_tile_index*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    D_block_ptr = tl.make_block_ptr(
        do_ptr+batch_index*stride_dob,
        shape=(N_QUERIES,),
        strides=(stride_doq,),
        offsets=(queries_tile_index*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    
    Q_tile = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
    grad_O_tile = tl.load(grad_O_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
    grad_Q_tile = tl.load(grad_Q_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
    
    L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    D_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    Q_tile_index = tl.arange(0, Q_TILE_SIZE) + queries_tile_index*K_TILE_SIZE

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
        V_tile = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)

        S_tile = tl.dot(Q_tile, K_tile.T)/scale
        if is_causal:
            K_tile_index = tl.arange(0, K_TILE_SIZE) + j*K_TILE_SIZE
            S_tile = tl.where(Q_tile_index[:, None]>=K_tile_index[None, :], S_tile, -1e6)
 
        P_tile = tl.exp(S_tile-L_tile[:, None])

        grad_P_tile = tl.dot(grad_O_tile, V_tile.T)
        grad_S_tile = P_tile*(grad_P_tile - D_tile[:, None])/scale

        grad_Q_tile +=  tl.dot(grad_S_tile, K_tile)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    tl.store(grad_Q_block_ptr, grad_Q_tile, boundary_check=(0,1))


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        BATCH_SIZE = Q.size(0)
        N_QUERIES = Q.size(-2)
        N_KEYS = K.size(-2)
        D = Q.size(-1)
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        B_Q = cdiv(N_QUERIES, Q_TILE_SIZE)
        B_K = cdiv(N_KEYS, K_TILE_SIZE)

        O = torch.empty_like(Q)
        L = torch.zeros((BATCH_SIZE, N_QUERIES), device=Q.device)
        ctx.scale = D**0.5
        flash_fwd_kernel[(cdiv(N_QUERIES, Q_TILE_SIZE), BATCH_SIZE)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            ctx.scale, D, Q_TILE_SIZE, K_TILE_SIZE, is_causal    
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_O):
        L, Q, K, V, O = ctx.saved_tensors
        BATCH_SIZE = Q.size(0)
        N_QUERIES = Q.size(-2)
        N_KEYS = K.size(-2)
        D = Q.size(-1)
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)

        # ..., N_Q
        D_o = torch.sum(O*grad_O, axis=-1)
        is_causal = ctx.is_causal

        flash_bkw_kernel[((cdiv(N_KEYS, K_TILE_SIZE), BATCH_SIZE))](
            Q, K, V, O, L, grad_Q, grad_K, grad_V, grad_O, D_o, 
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
            grad_K.stride(0), grad_K.stride(1), grad_K.stride(2),
            grad_V.stride(0), grad_V.stride(1), grad_V.stride(2),
            grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
            D_o.stride(0), D_o.stride(1),
            N_QUERIES, N_KEYS,
            ctx.scale, D, Q_TILE_SIZE, K_TILE_SIZE, is_causal 
        )
        flash_bkw_q_kernel[((cdiv(N_KEYS, K_TILE_SIZE), BATCH_SIZE))](
            Q, K, V, O, L, grad_Q, grad_K, grad_V, grad_O, D_o, 
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
            grad_K.stride(0), grad_K.stride(1), grad_K.stride(2),
            grad_V.stride(0), grad_V.stride(1), grad_V.stride(2),
            grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
            D_o.stride(0), D_o.stride(1),
            N_QUERIES, N_KEYS,
            ctx.scale, D, Q_TILE_SIZE, K_TILE_SIZE, is_causal 
        )
        return grad_Q, grad_K, grad_V, None
        

    
def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDPIndividualParameters(module)
    return DDP(module)


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
        id_p = []
        for p in self.module.parameters():
            if p.requires_grad:
                id_p.append(id(p))
                p.register_hook(lambda grad, p=p: self.gradient_synchronization(grad, p))
        if self.rank==0:
            print(id_p)
    
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


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    ddp_model.finish_gradient_synchronization()


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


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    return DDP_bucket(module, bucket_size_mb)


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    return


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    return Sharded_AdamW(params, optimizer_cls, **kwargs)

class Sharded_AdamW():
    def __init__(self, params, optimizer_cls, **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        params = [p for p in params]
        self.all_params = params
        rank_params = [p for i,p in enumerate(params) if i%self.world_size==self.rank]

        self.params = [p for i,p in enumerate(params) if i%self.world_size==self.rank]

        self.optimizer = optimizer_cls(rank_params, **kwargs)
        self.gather_params_list = []

        for rk in range(self.world_size):
            rank_params = [p for i,p in enumerate(params) if i%self.world_size==rk]
            self.gather_params_list.append(rank_params)


    def step(self, closure=None, **kwargs): 
        loss = self.optimizer.step(closure, **kwargs)

        dist.all_gather_object(self.gather_params_list, self.params)

        for rk, params in enumerate(self.gather_params_list):
            cnt=0
            for i, p in enumerate(self.all_params):
                if i%self.world_size==rk:
                    p.data.copy_(params[cnt].data)
                    cnt+=1
        return loss
    
    def add_param_group(self, param_group): 
        return
    
    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)



import math
from collections.abc import Callable, Iterable
import torch
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Can either apply weight decay here, or at the very end
                # p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]
                alpha = group["lr"]
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                t = state.get("t", 1)
                prev_m_t = state.get("m", torch.zeros_like(grad))
                prev_v_t = state.get("v", torch.zeros_like(grad))

                m_t = beta_1 * prev_m_t + ((1 - beta_1) * grad)
                v_t = beta_2 * prev_v_t + ((1 - beta_2) * torch.square(grad))

                alpha_t = alpha * (math.sqrt(1 - (beta_2**t)) / (1 - (beta_1**t)))
                p.data -= alpha_t * m_t / (torch.sqrt(v_t) + eps)
                # Apply weight decay
                p.data -= alpha * group["weight_decay"] * p.data

                state["m"] = m_t
                state["v"] = v_t
                state["t"] = t + 1

        return loss