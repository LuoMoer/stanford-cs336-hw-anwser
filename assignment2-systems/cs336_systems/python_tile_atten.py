import torch
from torch import nn
import triton
import triton.language as tl
from einops import rearrange, einsum
import cs336_basics.model

def cdiv(x, y):
    return (x+y-1)//y


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
        K_TILE_SIZE = 32
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
        K_TILE_SIZE = 32

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


batch_size = 1
n_queries = 4096
n_keys = 4096
D = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

flash_attention = TritonFlashAttention.apply

attention = cs336_basics.model.CausalMultiHeadSelfAttention(
    D, 1, cs336_basics.model.RotaryEmbedding(16384, D)).to(device)

def _make_attn_inputs(device=None):
    # torch.random.manual_seed(0)
    q = torch.randn(batch_size, n_queries, D, device=device, requires_grad=True)
    k = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
    v = torch.randn(batch_size, n_keys, D, device=device, requires_grad=True)
    do = torch.randn(batch_size, n_queries, D, device=device)

    return q, k, v, do

def _test():
    q, k, v, do = _make_attn_inputs(device)
    flash_attention(q, k, v, True).backward(do)

def _test_normal():
    q, k, v, do = _make_attn_inputs(device)
    attention(q).backward(do)

print("seq_len: ", n_keys)
print("dims : ", D)
print(triton.testing.do_bench(_test))
print(triton.testing.do_bench(_test_normal))