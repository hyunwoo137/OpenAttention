#!/usr/bin/env python3
"""OpenAttention.flash: fused decode-step attention over exact tokens and cluster proxies,
token-level (not block-sparse).

A cluster's proxy enters the online softmax exactly like a real token -- key `kbar_b`,
value `vbar_b`, and `+log(n_b)` on the logit -- so the tail costs one extra tile loop
rather than a second attention pass.

The selector returns arbitrary individual token indices, and forcing it to pick blocks
would coarsen selection by the block factor.  So the indices are TOKEN indices and the
K/V tile is gathered from scattered rows straight into SRAM.

That gather is the whole point.  Doing it with `index_select` in PyTorch costs a DRAM
round trip -- write the gathered tile out, read it back -- and measures 766 GB/s
against 3428 GB/s for a contiguous copy of the same bytes.  Inside the kernel the
tile never leaves the SM.

Three phases per (sequence, query head):
  P1  selected tokens   S: sink, local and top-k; scattered gather, exact logits
  P2  expanded tokens   X: the tokens ErrorGate reads from expanded clusters; scattered gather
  P3  cluster proxies   one virtual token per cluster, from cached (kbar, vbar)
"""

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _decode_expand_kernel(
    Q, K, V, Out,
    SelIdx,                 # (BQ, N_SEL)   int32  token indices from the selector
    ExpIdx,                 # (BQ, N_EXP)   int32  token indices of the expanded bins
    KBar, VBar, LogN,       # (BH, NB, D), (BH, NB, D), (BQ, NB) fp32; LogN in NATURAL log
    stride_kb, stride_ks,
    stride_qb,
    stride_ob,
    stride_kbar_b, stride_kbar_n,
    stride_ell_b,
    G,                      # query heads per KV head
    N_SEL, N_EXP, NB,
    sm_scale,
    BLOCK_N: tl.constexpr,
    BLOCK_B: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)              # one program per (sequence, query head)
    kvh = pid // G                      # which KV head this query head reads
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + pid * stride_qb + offs_d).to(tl.float32)
    qk_scale = sm_scale * LOG2E

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    k_base = kvh * stride_kb

    # ---- P1: selected tokens (scattered gather into SRAM) --------------------
    for start in range(0, N_SEL, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        valid = offs < N_SEL
        # -1 marks a padding slot: rows can hold different numbers of real tokens
        # (a bin clipped by the end of the sequence), and padding keeps the launch
        # shape rectangular without double-counting anything.
        tok = tl.load(SelIdx + pid * N_SEL + offs, mask=valid, other=-1)
        valid = valid & (tok >= 0)
        tok = tl.where(valid, tok, 0)
        kp = K + k_base + tok[:, None] * stride_ks + offs_d[None, :]
        k = tl.load(kp, mask=valid[:, None], other=0.0).to(tl.float32)
        z = tl.sum(q[None, :] * k, axis=1) * qk_scale
        z = tl.where(valid, z, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(z, axis=0))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(z - m_new)
        vp = V + k_base + tok[:, None] * stride_ks + offs_d[None, :]
        v = tl.load(vp, mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    # ---- P2: expanded bins (contiguous runs, still token-level) --------------
    for start in range(0, N_EXP, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        valid = offs < N_EXP
        # -1 marks a padding slot: rows can hold different numbers of real tokens
        # (a bin clipped by the end of the sequence), and padding keeps the launch
        # shape rectangular without double-counting anything.
        tok = tl.load(ExpIdx + pid * N_EXP + offs, mask=valid, other=-1)
        valid = valid & (tok >= 0)
        tok = tl.where(valid, tok, 0)
        kp = K + k_base + tok[:, None] * stride_ks + offs_d[None, :]
        k = tl.load(kp, mask=valid[:, None], other=0.0).to(tl.float32)
        z = tl.sum(q[None, :] * k, axis=1) * qk_scale
        z = tl.where(valid, z, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(z, axis=0))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(z - m_new)
        vp = V + k_base + tok[:, None] * stride_ks + offs_d[None, :]
        v = tl.load(vp, mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    # ---- P3: surviving bins as virtual tokens --------------------------------
    # identical shape to P1/P2: kbar takes the place of k, vbar of v, and the bin's
    # occupancy enters as +log2(n_b) which the caller folded into EllBias.
    kb_base = kvh * stride_kbar_b
    for start in range(0, NB, BLOCK_B):
        offs = start + tl.arange(0, BLOCK_B)
        valid = offs < NB
        kp = KBar + kb_base + offs[:, None] * stride_kbar_n + offs_d[None, :]
        kbar = tl.load(kp, mask=valid[:, None], other=0.0).to(tl.float32)
        # LogN is natural log(n_b); the kernel works in log2, so convert here rather
        # than making every caller remember the base.  -inf marks an expanded bin.
        logn = tl.load(LogN + pid * stride_ell_b + offs, mask=valid,
                       other=-float("inf")).to(tl.float32)
        z = (tl.sum(q[None, :] * kbar, axis=1) * sm_scale + logn) * LOG2E
        z = tl.where(valid, z, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(z, axis=0))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(z - m_new)
        vp = VBar + kb_base + offs[:, None] * stride_kbar_n + offs_d[None, :]
        vbar = tl.load(vp, mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * vbar, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    safe = tl.where(l_i == 0.0, 1.0, l_i)
    tl.store(Out + pid * stride_ob + offs_d, (acc / safe).to(Out.dtype.element_ty))


def decode_expand_attention(q, k, v, sel_idx, exp_idx, kbar, vbar, log_n,
                            sm_scale=None, block_n=64, block_b=64, num_warps=4):
    """q:(BQ,D)  k,v:(BH,K,D)  sel_idx:(BQ,N_SEL)  exp_idx:(BQ,N_EXP)
       kbar,vbar:(BH,NB,D)  log_n:(BQ,NB)  ->  (BQ,D)

    Index lists may be padded with -1 where a row has fewer real tokens.

    `log_n` is the NATURAL log of each bin's token count, with -inf in the slots of
    bins that were expanded -- that way the kernel needs no membership test to skip
    the bins it already attended exactly.
    """
    BQ, D = q.shape
    BH, _, _ = k.shape
    NB = kbar.shape[1]
    G = BQ // BH
    if sm_scale is None:
        sm_scale = D ** -0.5
    out = torch.empty_like(q)
    _decode_expand_kernel[(BQ,)](
        q, k, v, out, sel_idx, exp_idx, kbar, vbar, log_n,
        k.stride(0), k.stride(1),
        q.stride(0), out.stride(0),
        kbar.stride(0), kbar.stride(1), log_n.stride(0),
        G, sel_idx.shape[1], exp_idx.shape[1], NB, sm_scale,
        BLOCK_N=block_n, BLOCK_B=block_b, HEAD_DIM=D, num_warps=num_warps,
    )
    return out
