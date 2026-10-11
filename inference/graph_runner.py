#!/usr/bin/env python3
"""V65: the decode step of dense / HashAttention / ours as ONE CUDA graph per batch size.

ours_fused showed that after fusing ours' selection every decode step is CPU bound: the host spends
~37 ms issuing ~2,270 kernels (1,440 of the HF model, 832 of ours) while the GPU works 13-31 ms.  A CUDA
graph records the kernels of one decode step once and replays them with one host call.  For that, every
kernel of the step must see the same shapes and the same argument values at every step, so here

  * the current length n and the KV write position live in device memory (C.n_dev, C.kv_idx) and every
    kernel reads them there; all per-step buffers are sized by Lmax and positions >= n are masked
  * nothing in the step reads a GPU value on the host (ours and HAT have a fixed budget per step)
  * dense runs on a split-K FlashDecoding Triton kernel that reads n from device memory (SDPA's flash
    backend takes the length from the tensor shape); its kernel time is checked against SDPA flash

ours additionally gets three GPU-side changes over ours_fused (same selection):
  2  the scan writes each tile's maximum; the row top-k is then taken inside the k tiles with the
     largest maxima (exact: every tile holding a top-k key has a maximum >= the k-th key, so it is among
     the k largest maxima)
  5  within-cluster top-q: the same kernel, fewer warps (reductions stay inside a warp)
  6  residual statistics per (row, cluster) over that cluster's own segment of the expanded list (topq writes
     it grouped by cluster) plus its selected tokens: no atomics, no delta buffers, one kernel

    python graph_runner.py --selftest --device cuda:1
    python graph_runner.py --method ours  --rho 0.01 --batch 1 --mode graph --device cuda:1 --out ...
    python graph_runner.py --method dense --batch 8 --mode eager --device cuda:1 --out ...
"""
import argparse
import json
import os
import statistics
import sys
import time
import types

import torch
import torch.nn.functional as TF
import triton
import triton.language as tl
from triton.language.extra import libdevice

import os as _os
HUB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # repo root
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HUB)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import methods_eager as bl                                                        # noqa: E402
import ours_fused as F                                                            # noqa: E402
from methods_eager import C, CTX_LEN, D, G, HKV, HQ, NEG, SC, NL                  # noqa: E402
from ours_fused import _k_errorgate, _k_waterfill, _topq_loops, _topq_regs, _ukey  # noqa: E402,F401
sys.argv = _argv
LOG2E = 1.4426950408889634


# ============================================================ ours: kernels that read n from device memory
@triton.jit
def _k_index_update_d(K, V, Codes, U, Scale, W, Colmean, Cent, Asg, Nb, Sk, Sk2, Sv, Nptr,
                      Lmax, CN, G: tl.constexpr, D: tl.constexpr, RQ: tl.constexpr, RC: tl.constexpr,
                      BLOCK_C: tl.constexpr):
    """ours_fused _k_index_update with the new token's position read from device memory (n - 1)."""
    r = tl.program_id(0)
    pos = tl.load(Nptr) - 1
    kvr = r // G
    offs_d = tl.arange(0, D)
    kn = tl.load(K + kvr * Lmax * D + pos * D + offs_d)
    vn = tl.load(V + kvr * Lmax * D + pos * D + offs_d).to(tl.float32)
    knf = kn.to(tl.float32)
    if r % G == 0:
        offs_q = tl.arange(0, RQ)
        u = tl.load(U + kvr * D * RQ + offs_d[:, None] * RQ + offs_q[None, :])
        zc = tl.sum(knf[:, None] * u, axis=0)
        sc = tl.load(Scale + kvr * RQ + offs_q)
        code = tl.minimum(tl.maximum(libdevice.rint(zc / sc), -127.0), 127.0)
        tl.store(Codes + kvr * Lmax * RQ + pos * RQ + offs_q, code.to(Codes.dtype.element_ty))
    offs_c = tl.arange(0, RC)
    w = tl.load(W + r * D * RC + offs_d[:, None] * RC + offs_c[None, :]).to(tl.float32)
    zp = tl.sum(knf[:, None] * w, axis=0).to(tl.bfloat16).to(tl.float32)
    zp = zp - tl.load(Colmean + r * RC + offs_c)
    best_v = tl.full([], -float("inf"), tl.float32)
    best_i = tl.full([], 0, tl.int32)
    for c0 in range(0, CN, BLOCK_C):
        offs_b = c0 + tl.arange(0, BLOCK_C)
        cm = offs_b < CN
        cent = tl.load(Cent + r * CN * RC + offs_b[:, None] * RC + offs_c[None, :], mask=cm[:, None], other=0.0)
        dist = tl.sum(zp[None, :] * cent, axis=1) - 0.5 * tl.sum(cent * cent, axis=1)
        dist = tl.where(cm, dist, -float("inf"))
        m = tl.max(dist, axis=0)
        i = tl.min(tl.where(dist == m, offs_b, 2147483647), axis=0)
        upd = m > best_v
        best_i = tl.where(upd, i, best_i)
        best_v = tl.where(upd, m, best_v)
    tl.store(Asg + r * Lmax + pos, best_i.to(tl.int32))
    nbp = Nb + r * CN + best_i
    tl.store(nbp, tl.load(nbp) + 1.0)
    base = r * CN * D + best_i * D + offs_d
    tl.store(Sk + base, tl.load(Sk + base) + knf)
    tl.store(Sk2 + base, tl.load(Sk2 + base) + knf * knf)
    tl.store(Sv + base, tl.load(Sv + base) + vn)


@triton.jit
def _k_qgs_scan_tm(Zq, Codes, S, TM, Nptr, ns, nl, Lmax, SW, NT,
                   G: tl.constexpr, RQ: tl.constexpr, BLOCK_N: tl.constexpr):
    """QGS scores of one KV head's G query heads over one tile of int8 codes (bf16-rounded like the bf16
    matmul of methods_eager), sink/local and positions >= n set to -inf, plus the tile maximum per row."""
    kvr = tl.program_id(0)
    t = tl.program_id(1)
    n = tl.load(Nptr)
    offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
    inb = offs_n < Lmax
    offs_r = tl.arange(0, RQ)
    codes = tl.load(Codes + kvr * Lmax * RQ + offs_n[:, None] * RQ + offs_r[None, :],
                    mask=inb[:, None], other=0).to(tl.float32)
    dead = (offs_n < ns) | (offs_n >= n - nl)
    for g in tl.static_range(G):
        r = kvr * G + g
        zq = tl.load(Zq + r * RQ + offs_r).to(tl.float32)
        sc = tl.sum(codes * zq[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
        sc = tl.where(dead, -float("inf"), sc)
        tl.store(S + r * SW + offs_n, sc)
        tl.store(TM + r * NT + t, tl.max(sc, axis=0))


@triton.jit
def _k_select(S, TM, Sel, Nptr, ns, nl, NT, SW,
              K_TOP: tl.constexpr, KT2: tl.constexpr, BN: tl.constexpr, BLOCK_T: tl.constexpr,
              BLOCK_SL: tl.constexpr):
    """per row: the KT2 tiles with the largest maxima (ties -> lower tile), their keys in position order,
    the row top-K_TOP among them (ties -> lower position, as a top-k over the whole row), then
    sel = [sink 0..ns-1, local n-nl..n-1, top] and the selected keys' scores set to -inf."""
    r = tl.program_id(0)
    n = tl.load(Nptr)
    ot = tl.arange(0, BLOCK_T)
    tmv = tl.load(TM + r * NT + ot, mask=ot < NT, other=-float("inf"))
    # tile ids of the KT2 largest maxima, as a membership mask over tiles
    chosen = ot < 0
    for k in tl.static_range(KT2):
        m = tl.max(tl.where(chosen, -float("inf"), tmv), axis=0)
        i = tl.min(tl.where((tmv == m) & (~chosen) & (ot < NT), ot, 2147483647), axis=0)
        chosen = chosen | (ot == i)
    # chosen tile ids in position order (slot k = k-th chosen tile), then their keys as (KT2, BN)
    rank = tl.cumsum(chosen.to(tl.int32), axis=0) - 1
    slots = tl.arange(0, KT2)
    tile = tl.sum(tl.where(chosen[None, :] & (rank[None, :] == slots[:, None]), ot[None, :], 0), axis=1)
    pos = tile[:, None] * BN + tl.arange(0, BN)[None, :]
    v = tl.load(S + r * SW + pos)
    base = Sel + r * (ns + nl + K_TOP)
    for k in tl.static_range(K_TOP):
        m = tl.max(tl.max(v, axis=1), axis=0)
        j = tl.min(tl.min(tl.where(v == m, pos, 2147483647), axis=1), axis=0)
        tl.store(base + ns + nl + k, j)
        v = tl.where(pos == j, -float("inf"), v)
    tl.debug_barrier()
    for k in tl.static_range(K_TOP):                    # selected scores -> -inf, after all reads of S
        tl.store(S + r * SW + tl.load(base + ns + nl + k), -float("inf"))
    osl = tl.arange(0, BLOCK_SL)
    sl = tl.where(osl < ns, osl, n - nl + (osl - ns))
    slm = osl < ns + nl
    tl.store(base + osl, sl, mask=slm)
    tl.store(S + r * SW + sl, -float("inf"), mask=slm)


@triton.jit
def _topq_regs2(srow, Mem, Asg, Exp, r, c, qv, base, start, len0, n, n0, Lmax, E,
                BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr, SMALL_Q: tl.constexpr):
    """ours_fused _topq_regs with the small-quota path on one packed int64 key per token:
    (score key, then 0x7FFFFFFF - token) -> a max gives the best score and, among equal scores, the lowest
    token id (= members before generated tokens, then position: the radix path's order).  Two reductions
    per extracted token instead of four."""
    if qv <= SMALL_Q:
        oj = tl.arange(0, BLOCK_M)
        vm = oj < len0
        tok = tl.load(Mem + r * Lmax + start + oj, mask=vm, other=0)
        u = _ukey(tl.load(srow + tok, mask=vm, other=0.0))
        rt = n0 + tl.arange(0, BLOCK_R)
        rv = rt < n
        a = tl.load(Asg + r * Lmax + rt, mask=rv, other=-1)
        rv = rv & (a == c)
        ur = _ukey(tl.load(srow + rt, mask=rv, other=0.0))
        lo = 2147483647
        km = tl.where(vm, ((u ^ 0x80000000).to(tl.int32, bitcast=True).to(tl.int64) << 32) | (lo - tok).to(tl.int64),
                      -9223372036854775807)
        kr = tl.where(rv, ((ur ^ 0x80000000).to(tl.int32, bitcast=True).to(tl.int64) << 32) | (lo - rt).to(tl.int64),
                      -9223372036854775807)
        for k in range(qv):
            best = tl.maximum(tl.max(km, axis=0), tl.max(kr, axis=0))
            tl.store(Exp + r * E + base + k, (lo - (best & 0x7FFFFFFF)).to(tl.int32))
            km = tl.where(km == best, -9223372036854775807, km)
            kr = tl.where(kr == best, -9223372036854775807, kr)
    else:
        _topq_regs(srow, Mem, Asg, Exp, r, c, qv, base, start, len0, n, n0, Lmax, E, BLOCK_M, BLOCK_R, 0)


@triton.jit
def _k_topq_d(S, Mem, Cstart, Clen, Asg, Quota, Exp, Nptr, N0ptr, SW, Lmax, CN, E,
              BLOCK_T: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr,
              SMALL_Q: tl.constexpr):
    """ours_fused _k_topq with n / n0 from device memory and the score row stride SW."""
    r = tl.program_id(0)
    c = tl.program_id(1)
    qv = tl.load(Quota + r * CN + c).to(tl.int32)
    if qv > 0:
        n = tl.load(Nptr)
        n0 = tl.load(N0ptr)
        offs_q = tl.arange(0, BLOCK_Q)
        qa = tl.load(Quota + r * CN + offs_q, mask=offs_q < c, other=0.0)
        base = tl.sum(qa, axis=0).to(tl.int32)
        start = tl.load(Cstart + r * CN + c)
        len0 = tl.load(Clen + r * CN + c)
        srow = S + r * SW
        if (len0 <= BLOCK_M) & (n - n0 <= BLOCK_R):
            _topq_regs2(srow, Mem, Asg, Exp, r, c, qv, base, start, len0, n, n0, Lmax, E, BLOCK_M, BLOCK_R, SMALL_Q)
        else:
            _topq_loops(srow, Mem, Asg, Exp, r, c, qv, base, start, len0, n, n0, Lmax, E, BLOCK_T)


@triton.jit
def _k_residual_seg(K, V, Asg, Sel, Exp, Sk, Sv, Nt, Quota, KbarR, VbarR, LogN,
                    Lmax, CN, NS, E, G: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr,
                    BLOCK_Q: tl.constexpr, BLOCK_S: tl.constexpr):
    """per (row, cluster): pooled sums minus the cluster's selected tokens and its own segment of the
    expanded list [base, base + q_b) -> kbar', vbar' (bf16), log n'.  No atomics."""
    r = tl.program_id(0)
    c = tl.program_id(1)
    kvr = r // G
    offs_d = tl.arange(0, D)
    sp = r * CN * D + c * D + offs_d
    sk = tl.load(Sk + sp)
    sv = tl.load(Sv + sp)
    qf = tl.load(Quota + r * CN + c)
    nr = tl.load(Nt + r * CN + c) - qf
    # selected tokens of this cluster
    offs_s = tl.arange(0, BLOCK_S)
    sm = offs_s < NS
    tok = tl.load(Sel + r * NS + offs_s, mask=sm, other=0)
    cid = tl.load(Asg + r * Lmax + tok, mask=sm, other=-1)
    hit = sm & (cid == c)
    kp = kvr * Lmax * D + tok[:, None] * D + offs_d[None, :]
    sk -= tl.sum(tl.load(K + kp, mask=hit[:, None], other=0.0).to(tl.float32), axis=0)
    sv -= tl.sum(tl.load(V + kp, mask=hit[:, None], other=0.0).to(tl.float32), axis=0)
    qv = qf.to(tl.int32)
    if qv > 0:
        offs_q = tl.arange(0, BLOCK_Q)
        base = tl.sum(tl.load(Quota + r * CN + offs_q, mask=offs_q < c, other=0.0), axis=0).to(tl.int32)
        for j0 in range(0, qv, BLOCK_T):
            oj = j0 + tl.arange(0, BLOCK_T)
            m = oj < qv
            t2 = tl.load(Exp + r * E + base + oj, mask=m, other=-1)
            m = m & (t2 >= 0)
            t2 = tl.where(m, t2, 0)
            kp2 = kvr * Lmax * D + t2[:, None] * D + offs_d[None, :]
            sk -= tl.sum(tl.load(K + kp2, mask=m[:, None], other=0.0).to(tl.float32), axis=0)
            sv -= tl.sum(tl.load(V + kp2, mask=m[:, None], other=0.0).to(tl.float32), axis=0)
    inv = 1.0 / tl.maximum(nr, 1.0)
    tl.store(KbarR + sp, (sk * inv).to(tl.bfloat16))
    tl.store(VbarR + sp, (sv * inv).to(tl.bfloat16))
    tl.store(LogN + r * CN + c, tl.where(nr > 0.5, tl.log(tl.maximum(nr, 1.0)), -float("inf")))


# ============================================================ dense: split-K FlashDecoding, n from device
@triton.jit
def _k_dense_split(Q, K, V, Nptr, Pm, Pl, Pacc, Lmax, SPLIT, CHUNK, sm_scale,
                   G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    """one (KV head, key chunk): the G query heads of that KV head against keys [start, min(start+CHUNK, n)),
    online softmax, partial (m, l, acc) per query head (rows padded to GP >= 16 for tl.dot)."""
    kvr = tl.program_id(0)
    sp = tl.program_id(1)
    n = tl.load(Nptr)
    start = sp * CHUNK
    end = tl.minimum(start + CHUNK, n)
    offs_g = tl.arange(0, GP)
    gm = offs_g < G
    offs_d = tl.arange(0, D)
    q = tl.load(Q + (kvr * G + offs_g)[:, None] * D + offs_d[None, :], mask=gm[:, None], other=0.0)
    qk = sm_scale * 1.4426950408889634
    m_i = tl.full([GP], -float("inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    kb = kvr * Lmax * D
    for t0 in range(start, end, BLOCK_N):
        offs = t0 + tl.arange(0, BLOCK_N)
        vm = offs < end
        k = tl.load(K + kb + offs[:, None] * D + offs_d[None, :], mask=vm[:, None], other=0.0)
        z = tl.dot(q, tl.trans(k)) * qk                                          # (GP, BLOCK_N)
        z = tl.where(vm[None, :], z, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(z, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(z - m_safe[:, None])
        v = tl.load(V + kb + offs[:, None] * D + offs_d[None, :], mask=vm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
    row = (kvr * G + offs_g) * SPLIT + sp
    tl.store(Pm + row, m_i, mask=gm)
    tl.store(Pl + row, l_i, mask=gm)
    tl.store(Pacc + row[:, None] * D + offs_d[None, :], acc, mask=gm[:, None])


@triton.jit
def _k_dense_combine(Pm, Pl, Pacc, Out, SPLIT, D: tl.constexpr, BLOCK_S: tl.constexpr):
    r = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_S)
    sm = offs_s < SPLIT
    offs_d = tl.arange(0, D)
    m = tl.load(Pm + r * SPLIT + offs_s, mask=sm, other=-float("inf"))
    l = tl.load(Pl + r * SPLIT + offs_s, mask=sm, other=0.0)
    mx = tl.max(m, axis=0)
    w = tl.where(sm & (m > -float("inf")), tl.math.exp2(m - mx), 0.0)
    acc = tl.load(Pacc + (r * SPLIT + offs_s)[:, None] * D + offs_d[None, :], mask=sm[:, None], other=0.0)
    num = tl.sum(acc * w[:, None], axis=0)
    den = tl.sum(l * w, axis=0)
    tl.store(Out + r * D + offs_d, (num / den).to(Out.dtype.element_ty))


# ============================================================ graph-safe methods (read C.n_dev / C.n0_dev)
class DenseG(bl.Method):
    """dense decode attention as split-K FlashDecoding; n from device memory"""
    name = "dense"

    def __init__(self, a, B, Lmax, dev):
        super().__init__(a, B, Lmax, dev)
        self.SPLIT = max(4, 512 // (B * HKV))
        self.CHUNK = triton.cdiv(Lmax, self.SPLIT)
        R = B * HQ
        self.pm = torch.empty(R * self.SPLIT, device=dev)
        self.pl = torch.empty(R * self.SPLIT, device=dev)
        self.pacc = torch.empty(R * self.SPLIT, D, device=dev)

    def decode(self, l, q, n):
        R = self.R
        qr = q.reshape(R, D).contiguous()
        bl.mark()
        _k_dense_split[(self.B * HKV, self.SPLIT)](qr, C.K[l], C.V[l], C.n_dev, self.pm, self.pl, self.pacc,
                                                   self.Lmax, self.SPLIT, self.CHUNK, SC,
                                                   G=G, GP=16, D=D, BLOCK_N=64, num_warps=4)
        out = torch.empty(R, D, device=self.dev, dtype=q.dtype)
        _k_dense_combine[(R,)](self.pm, self.pl, self.pacc, out, self.SPLIT, D=D,
                               BLOCK_S=triton.next_power_of_2(self.SPLIT), num_warps=4)
        return out.view(self.B, HQ, 1, D)


class DenseSDPA(bl.Method):
    """eager reference: SDPA restricted to its FlashAttention backend (no per-length graph build)"""
    name = "densesdpa"

    def decode(self, l, q, n):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        bl.mark()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return TF.scaled_dot_product_attention(q, C.K[l][:, :, :n], C.V[l][:, :, :n], enable_gqa=True, scale=SC)


class DenseSDPADefault(bl.Method):
    """eager reference: SDPA with its default backend dispatch (cuDNN here: builds a graph per new length)"""
    name = "densesdpadef"

    def decode(self, l, q, n):
        bl.mark()
        return TF.scaled_dot_product_attention(q, C.K[l][:, :, :n], C.V[l][:, :, :n], enable_gqa=True, scale=SC)


class HATG(bl.HAT):
    """HashAttention (methods_eager HAT) with Lmax-sized buffers and device-side n.  The top-k size
    h = int(heavy_frac * n) is taken at the n the step is traced at (constant to within one token over a
    128-token window)."""
    name = "hat"

    def __init__(self, a, B, Lmax, dev):
        super().__init__(a, B, Lmax, dev)
        self.arL = torch.arange(Lmax, device=dev).view(1, -1)

    def decode(self, l, q, n):
        R, dev = self.R, self.dev
        kn = C.K[l].index_select(2, C.kv_idx).repeat_interleave(G, 1)
        self.sig[l].index_copy_(2, C.kv_idx, self.mlp(kn, self.kW[l], self.kB[l]))
        qs = self.mlp(q, self.qW[l], self.qB[l])
        s = torch.matmul(qs, self.sig[l].transpose(-1, -2)).view(R, self.Lmax)
        ns = nl = int(0.001 * n)
        h = int(self.heavy_frac * n)
        nd = C.n_dev.long()
        s = s.masked_fill((self.arL < ns) | (self.arL >= nd - nl), NEG)
        top = s.topk(h, dim=-1).indices
        loc = (nd - nl + torch.arange(nl, device=dev)).view(1, -1).expand(R, -1)
        idx = torch.cat([torch.arange(ns, device=dev).view(1, -1).expand(R, -1), loc, top], 1)
        bl.mark()
        kf, vf = self.flat(l)
        ks, vs = bl.gather_rows(kf, idx, self.off), bl.gather_rows(vf, idx, self.off)
        q2 = q.reshape(R, 1, D)
        z = torch.bmm(q2, ks.transpose(1, 2)).float() * SC
        p = torch.softmax(z, -1).to(vs.dtype)
        return torch.bmm(p, vs).view(self.B, HQ, 1, D)


class OursG(F.OursFast):
    """ours_fused OursFast, graph-safe, with the v65 scan / top-q / residual changes"""
    name = "ours"
    BN = 256
    SMALL_Q2 = 16         # quotas up to this take the packed-key extraction (sweep: 8-32 equal, radix-only 1.5-4x slower)
    TOPQ_WARPS = 0        # 0 = sweep rule: 4 warps for B*32 < 128 rows, else 2 (tile 512) / 1 (tile 256)

    def __init__(self, a, B, Lmax, dev):
        super().__init__(a, B, Lmax, dev)
        self.NT = triton.cdiv(Lmax, self.BN)
        self.SW = self.NT * self.BN                              # score row stride (tiles padded)
        self.arBN = torch.arange(self.BN, device=dev).view(1, 1, -1)

    def decode(self, l, q, n):
        B, R, Cn, E, dev, Lmax = self.B, self.R, self.C, self.E, self.dev, self.Lmax
        K, V = C.K[l], C.V[l]
        qr = q.reshape(R, D).contiguous()
        # 1. index update of the new token
        _k_index_update_d[(R,)](K, V, self.codes8[l], self.U[l], self.scale[l], self.W[l], self.colmean[l],
                                self.cent[l], self.asg[l], self.nb[l], self.Sk[l], self.Sk2[l], self.Sv[l],
                                C.n_dev, Lmax, Cn, G=G, D=D, RQ=self.rq, RC=self.rc, BLOCK_C=64, num_warps=4)
        # 2. QGS scan (tile maxima) and the exact row top-k inside the k best tiles
        zq = (torch.matmul(q.view(B, HKV, G, D).float(), self.U[l]) * self.scale[l]).to(torch.bfloat16)
        ns = nl = max(1, int(0.0001 * n))
        heavy = self.a.rho - 0.0002 - E / CTX_LEN - 1.5 * Cn / CTX_LEN
        k_top = max(1, int(heavy * n))
        s = torch.empty(R, self.SW, device=dev)
        tm = torch.empty(R, self.NT, device=dev)
        _k_qgs_scan_tm[(B * HKV, self.NT)](zq, self.codes8[l], s, tm, C.n_dev, ns, nl, Lmax, self.SW, self.NT,
                                           G=G, RQ=self.rq, BLOCK_N=self.BN, num_warps=1)
        # stage 2 in one kernel: 2k best tiles (a tie between tile maxima cannot drop a tied key), their keys
        # in position order, the row top-k with ties to the lower position, sel = [sink, local, top], and
        # the selected scores set to -inf (they go last in their cluster for the within-cluster top-q)
        NS = ns + nl + k_top
        sel = torch.empty(R, NS, device=dev, dtype=torch.int32)
        _k_select[(R,)](s, tm, sel, C.n_dev, ns, nl, self.NT, self.SW, K_TOP=k_top,
                        KT2=triton.next_power_of_2(min(2 * k_top, self.NT)), BN=self.BN,
                        BLOCK_T=triton.next_power_of_2(self.NT), BLOCK_SL=triton.next_power_of_2(ns + nl),
                        num_warps=4)
        # 3. ErrorGate, 4. water-filling (ours_fused kernels)
        nblk = triton.cdiv(Cn, self.BC)
        score = torch.empty(R, Cn, device=dev)
        n_t = torch.empty(R, Cn, device=dev)
        _k_errorgate[(R, nblk)](qr, K, self.asg[l], sel, self.Sk[l], self.Sk2[l], self.nb[l], score, n_t,
                                Lmax, Cn, NS, SC, G=G, D=D, BLOCK_C=self.BC, BLOCK_T=16, num_warps=4)
        quota = torch.empty(R, Cn, device=dev)
        _k_waterfill[(R,)](score, n_t, quota, Cn, float(E), BLOCK=self.BQ, num_warps=4)
        self._quota = quota
        # 5. within-cluster top-q_b (the selected scores are already -inf)
        exp_idx = torch.full((R, E), -1, device=dev, dtype=torch.int32)
        _k_topq_d[(R, Cn)](s, self.mem[l], self.cstart[l], self.clen[l], self.asg[l], quota, exp_idx,
                           C.n_dev, C.n0_dev, self.SW, Lmax, Cn, E, BLOCK_T=128, BLOCK_Q=self.BQ,
                           BLOCK_M=self.BM[l], BLOCK_R=triton.next_power_of_2(max(Lmax - self.n0, 16)),
                           SMALL_Q=self.SMALL_Q2,
                           num_warps=self.TOPQ_WARPS or (4 if R < 128 else (2 if self.BM[l] >= 512 else 1)))
        # 6. residual statistics (ours_fused: one atomic scatter per token read, then normalise)
        kbar_r = torch.empty(R, Cn, D, device=dev, dtype=torch.bfloat16)
        vbar_r = torch.empty(R, Cn, D, device=dev, dtype=torch.bfloat16)
        logn = torch.empty(R, Cn, device=dev)
        dk = torch.zeros(R, Cn, D, device=dev)
        dv = torch.zeros(R, Cn, D, device=dev)
        F._k_resid_scatter[(R, triton.cdiv(NS + E, self.BT))](K, V, self.asg[l], sel, exp_idx, dk, dv, Lmax, Cn,
                                                             NS, E, G=G, D=D, BLOCK_T=self.BT, num_warps=4)
        F._k_resid_norm[(R, nblk)](self.Sk[l], self.Sv[l], dk, dv, n_t, quota, kbar_r, vbar_r, logn, Cn,
                                   D=D, BLOCK_C=self.BC, num_warps=4)
        # 7. ours.flash
        bl.mark()
        out = bl.ours_flash(qr, K.view(-1, Lmax, D), V.view(-1, Lmax, D), sel, exp_idx, kbar_r, vbar_r, logn)
        return out.view(B, HQ, 1, D)


METHODS = {"dense": DenseG, "densesdpa": DenseSDPA, "densesdpadef": DenseSDPADefault, "hat": HATG, "ours": OursG}


# ============================================================ attention hook with device-side positions
def attn_hook(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
    l = module.layer_idx
    T = query.shape[2]
    if C.phase == "prefill":
        b = C.brow
        C.K[l][b:b + 1, :, :T] = key
        C.V[l][b:b + 1, :, :T] = value
        out = TF.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True, scale=SC)
        C.method.build(l, b, query, T)
        return out.transpose(1, 2).contiguous(), None
    C.K[l].index_copy_(2, C.kv_idx, key)
    C.V[l].index_copy_(2, C.kv_idx, value)
    out = C.method.decode(l, query, C.n)
    return out.transpose(1, 2).contiguous(), None


# ============================================================ self test: static paths vs the eager ones
def selftest(dev):
    print("== dense split-K kernel vs SDPA (fp32 math), and its kernel time vs SDPA flash")
    for B in (1, 8):
        Lmax, n = 30200, 30100
        C.K = [torch.randn(B, HKV, Lmax, D, device=dev, dtype=torch.bfloat16)]
        C.V = [torch.randn(B, HKV, Lmax, D, device=dev, dtype=torch.bfloat16)]
        C.n_dev = torch.tensor([n], device=dev, dtype=torch.int32)
        m = DenseG(types.SimpleNamespace(rho=0.01), B, Lmax, dev)
        q = torch.randn(B, HQ, 1, D, device=dev, dtype=torch.bfloat16)
        o = m.decode(0, q, n).float()
        kr = C.K[0][:, :, :n].float().repeat_interleave(G, 1)
        vr = C.V[0][:, :, :n].float().repeat_interleave(G, 1)
        ref = torch.softmax((q.float() @ kr.transpose(-1, -2)) * SC, -1) @ vr
        rel = ((o - ref).norm(dim=-1) / ref.norm(dim=-1)).max().item()
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def t(fn, it=50):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(it):
                fn()
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) / it * 1e3

        def sd():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                TF.scaled_dot_product_attention(q, C.K[0][:, :, :n], C.V[0][:, :, :n], enable_gqa=True, scale=SC)
        print(f"  B={B}: max rel err {rel:.2e} | split-K {t(lambda: m.decode(0, q, n)):.0f} us/layer "
              f"(SPLIT={m.SPLIT}) vs SDPA flash {t(sd):.0f} us/layer")
    print("== ours: OursG (v65) vs ours_fused OursFast on shared state (synthetic data of the real shapes)")
    F._record_exp()                                  # spies on OursFast.decode
    dec = OursG.decode

    def spied(self, l, q, n):                        # the same spy on OursG.decode
        old = bl.ours_flash

        def spy(qr, k, v, sel, exp_idx, kb, vb, ln, nb=None):
            F._last["exp"], F._last["sel"] = exp_idx.clone(), sel.clone()
            return old(qr, k, v, sel, exp_idx, kb, vb, ln, nb)
        bl.ours_flash = spy
        try:
            return dec(self, l, q, n)
        finally:
            bl.ours_flash = old
    OursG.decode = spied
    for rho in (0.01, 0.03):
        for B in (1, 8):
            a, Lmax, qp = F._synthetic(B, rho, dev)
            n0 = qp.shape[2]
            ref, new = F.OursFast(a, B, Lmax, dev), OursG(a, B, Lmax, dev)
            for b in range(B):
                ref.build(0, b, qp, n0)
            for nm in ("codes", "U", "scale", "W", "colmean", "cent", "asg", "nb", "Sk", "Sk2", "Sv"):
                getattr(new, nm)[0].copy_(getattr(ref, nm)[0])
            for b in range(B):
                new._build_csr(0, b, n0)
            C.n0_dev = torch.tensor([n0], device=dev, dtype=torch.int32)
            n, bad_sel, bad_exp, rows, relmax = n0, 0, 0, 0, 0.0
            for it in range(8):
                n += 1
                C.n_dev = torch.tensor([n], device=dev, dtype=torch.int32)
                q = torch.randn(B, HQ, 1, D, device=dev, dtype=torch.bfloat16)
                o_r = ref.decode(0, q, n).float()
                e_r, s_r = F._last["exp"].long().sort(-1).values, F._last["sel"].long().sort(-1).values
                o_n = new.decode(0, q, n).float()
                e_n, s_n = F._last["exp"].long().sort(-1).values, F._last["sel"].long().sort(-1).values
                bad_sel += int((s_r != s_n).any(-1).sum())
                bad_exp += int((e_r != e_n).any(-1).sum())
                rows += e_r.shape[0]
                relmax = max(relmax, ((o_n - o_r).norm(dim=-1) / o_r.norm(dim=-1)).max().item())
            print(f"  rho={rho} B={B}: rows with a different top-k {bad_sel}/{rows}, different expanded set "
                  f"{bad_exp}/{rows}, output rel err max {relmax:.2e}")
            del ref, new
            torch.cuda.empty_cache()


# ============================================================ driver
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=list(METHODS), default="ours")
    p.add_argument("--rho", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--mode", choices=["eager", "graph"], default="graph")
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", default="")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--fresh-lengths", action="store_true",
                   help="every timed rep decodes at lengths no earlier step used (no per-length cache can hit)")
    a = p.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    if a.selftest:
        selftest(dev)
        return
    sys.path.insert(0, os.path.join(bl.BASE, "common"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    state_before = bl.gpu_state()
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(bl.MODEL)
    model = AutoModelForCausalLM.from_pretrained(bl.MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    ids, qlen, refs = bl.load_prompts(tok, a.batch)
    B, L0 = ids.shape
    ids = ids.to(dev)
    W = 3                                                   # eager warm-up steps before the capture
    Lmax = L0 + a.gen + W + 8 + (24 + a.reps * a.gen if a.fresh_lengths else 0)
    C.K = [torch.zeros(B, HKV, Lmax, D, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
    C.V = [torch.zeros(B, HKV, Lmax, D, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
    C.qlen = qlen
    C.n_dev = torch.zeros(1, device=dev, dtype=torch.int32)
    C.n0_dev = torch.full((1,), L0, device=dev, dtype=torch.int32)
    C.kv_idx = torch.zeros(1, device=dev, dtype=torch.long)
    C.timing = False
    C.method = METHODS[a.method](types.SimpleNamespace(rho=a.rho, mp_K=9, mp_L=25), B, Lmax, dev)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = attn_hook
    pos0 = torch.arange(L0, device=dev).view(1, -1)
    tok_buf = torch.zeros(B, 1, device=dev, dtype=torch.long)
    pos_buf = torch.zeros(B, 1, device=dev, dtype=torch.long)
    # the per-step constants of ours / HAT (sink/local/top-k sizes) must not change inside the window
    for m_n in (L0 + 1, Lmax):
        assert max(1, int(0.0001 * m_n)) == max(1, int(0.0001 * (L0 + 1)))

    @torch.inference_mode()
    def prefill():
        C.phase = "prefill"
        nx = []
        for b in range(B):
            C.brow = b
            out = model(input_ids=ids[b:b + 1], position_ids=pos0, use_cache=False, logits_to_keep=1)
            nx.append(out.logits[:, -1].argmax(-1))
        return torch.stack(nx, 0)

    def set_step(s):                                     # device-side state of decode step s (n = L0 + s + 1)
        C.n = L0 + s + 1
        C.n_dev.fill_(C.n)
        C.kv_idx.fill_(C.n - 1)
        pos_buf.fill_(L0 + s)

    @torch.inference_mode()
    def eager_step(nxt, s):
        C.phase = "decode"
        set_step(s)
        out = model(input_ids=nxt, position_ids=pos_buf, use_cache=False)
        return out.logits[:, -1].argmax(-1, keepdim=True)

    graph = [None, None]

    @torch.inference_mode()
    def capture(s):
        C.phase = "decode"
        set_step(s)
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            out = model(input_ids=tok_buf, position_ids=pos_buf, use_cache=False)
            nxt_buf = out.logits[:, -1].argmax(-1, keepdim=True)
        graph[0], graph[1] = g, nxt_buf

    @torch.inference_mode()
    def graph_step(nxt, s):
        set_step(s)
        tok_buf.copy_(nxt)
        graph[0].replay()
        return graph[1]

    def run(nxt, steps, first=0, keep=False):
        toks = [nxt] if keep else None
        step = graph_step if a.mode == "graph" else eager_step
        for s in range(first, first + steps):
            nxt = step(nxt, s)
            if keep:
                toks.append(nxt.clone())
        return nxt, toks

    sync = lambda: torch.cuda.synchronize(dev)          # noqa: E731
    res = {"method": a.method, "rho": a.rho, "batch": B, "mode": a.mode, "fresh_lengths": a.fresh_lengths, "prompt_tokens": L0, "gen": a.gen,
           "gpu": torch.cuda.get_device_name(dev), "gpu_state_before": state_before}
    # warm-up: prefill, W eager steps (compiles Triton, cuBLAS handles), capture, answer check (24 tokens)
    nxt = prefill()
    first_tok = nxt
    toks_e = [nxt]
    for s in range(W):
        nxt = eager_step(nxt, s)
        toks_e.append(nxt.clone())
    t0 = time.perf_counter()
    if a.mode == "graph":
        capture(W)
    sync()
    res["capture_s"] = time.perf_counter() - t0
    _, toks_g = run(nxt, 24 - W, first=W, keep=True)
    sync()
    gen0 = torch.cat(toks_e + toks_g[1:], 1)[0].tolist()
    res["answer_row0"] = tok.decode(gen0, skip_special_tokens=True)
    res["reference_row0"] = refs[0]
    print(f"[{a.method} {a.mode} rho={a.rho} B={B} L0={L0}] answer: {res['answer_row0']!r}", flush=True)
    del first_tok
    # timed: prefill (+ build) then gen steps; in graph mode every step is a replay of the captured graph
    reps = []
    for r in range(a.reps):
        sync()
        t0 = time.perf_counter()
        nxt = prefill()
        sync()
        t1 = time.perf_counter()
        run(nxt, a.gen, first=(24 + r * a.gen) if a.fresh_lengths else 0)
        sync()
        t2 = time.perf_counter()
        reps.append({"prefill_ms": (t1 - t0) * 1e3, "decode_ms": (t2 - t1) * 1e3, "e2e_ms": (t2 - t0) * 1e3,
                     "tpot_ms": (t2 - t1) * 1e3 / a.gen})
        print(f"  rep{r}: prefill {reps[-1]['prefill_ms']:8.1f} ms  decode {reps[-1]['decode_ms']:8.1f} ms "
              f"({reps[-1]['tpot_ms']:.2f} ms/step)  E2E {reps[-1]['e2e_ms']:8.1f} ms", flush=True)
    res["reps"] = reps
    for k in ("prefill_ms", "decode_ms", "e2e_ms", "tpot_ms"):
        res[k] = statistics.median(x[k] for x in reps)
        res[k + "_rep0"] = reps[0][k]
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    res["wall_s"] = time.time() - t_start
    res["gpu_state_after"] = bl.gpu_state()
    print(f"[{a.method} {a.mode} rho={a.rho} B={B}] TPOT {res['tpot_ms']:.2f} ms (rep0 {res['tpot_ms_rep0']:.2f})  "
          f"E2E {res['e2e_ms']:.1f} ms  prefill {res['prefill_ms']:.1f} ms  capture {res['capture_s']:.1f} s  "
          f"peak {res['peak_mem_gb']:.1f} GB", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
