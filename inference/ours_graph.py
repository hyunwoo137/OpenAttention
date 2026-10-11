#!/usr/bin/env python3
"""V66: GPU-side changes on top of graph_runner OursG (same selection), for the graph-replayed decode step.

Under a CUDA graph the step costs its GPU time, and the real-model profile of v65 (results_v65_graph/prof_*)
put the remaining time in
  5  within-cluster top-q   every (row, cluster) program also scanned all generated tokens (a 256-lane tile)
                            to find the ones of its cluster
  6  residual statistics    one atomic add per value of every token read (3%: ~45 M atomics per layer at B=8)
  7  ours.flash             one program per query head (32 at B=1) walking up to ~900 entries serially
so here
  1  the index update also appends the new token to its cluster's list of generated tokens
  5  top-q reads that list (a 32-lane tile) instead of scanning every generated token
  6  per (row, block of 16 clusters): the block's own contiguous segment of the expanded list (top-q writes it
     grouped by cluster) and the selected tokens, one-hot fp32 dots, no atomics
  7  split-K over the [selected, expanded, proxies] list + the dense combine kernel

    python ours_graph.py --selftest --device cuda:1
    python ours_graph.py --method ours2 --rho 0.01 --batch 1 --mode graph --device cuda:1 --out ...
"""
import sys
import types

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

import os as _os
HUB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # repo root
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HUB)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import methods_eager as bl                                                  # noqa: E402
import ours_fused as F                                                      # noqa: E402
import graph_runner as G5                                                    # noqa: E402
from methods_eager import C, CTX_LEN, D, G, HKV, HQ, NEG, SC                # noqa: E402
from ours_fused import _k_errorgate, _k_waterfill, _topq_loops, _ukey       # noqa: E402,F401
from graph_runner import _k_dense_combine, _k_qgs_scan_tm, _k_select         # noqa: E402,F401
sys.argv = _argv
RMAX = 256          # generated tokens a cluster's list can hold (>= decode steps of a request)


@triton.jit
def _k_index_update_d2(K, V, Codes, U, Scale, W, Colmean, Cent, Asg, Nb, Sk, Sk2, Sv, RecList, RecCnt, Nptr,
                       Lmax, CN, RM, G: tl.constexpr, D: tl.constexpr, RQ: tl.constexpr, RC: tl.constexpr,
                       BLOCK_C: tl.constexpr, QMAX: tl.constexpr):
    """graph_runner _k_index_update_d + append the new token to its cluster's generated-token list
    (QMAX: code range, 127 = int8 as deployed; bench_v67 also runs 7 = int4)"""
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
        code = tl.minimum(tl.maximum(libdevice.rint(zc / sc), -QMAX), QMAX)
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
    cp = RecCnt + r * CN + best_i
    k = tl.load(cp)
    tl.store(RecList + (r * CN + best_i) * RM + k, pos, mask=k < RM)
    tl.store(cp, k + 1)


@triton.jit
def _topq_core(Exp, r, E, base, qv, tok, vm, u, rt, rv, ur, SMALL_Q: tl.constexpr):
    """top-qv of two register tiles (cluster members; the cluster's generated tokens).  Order: score, then
    lower token id (members before generated tokens, then position) -- the radix path's order."""
    if qv <= SMALL_Q:
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
        prefix = tl.full([], 0, tl.uint32)
        pmask = tl.full([], 0, tl.uint32)
        k_rem = qv
        bins = tl.arange(0, 256)
        for i in tl.static_range(4):
            shift = 24 - 8 * i
            mt = vm & ((u & pmask) == prefix)
            mr = rv & ((ur & pmask) == prefix)
            hist = tl.histogram(((u >> shift) & 0xFF).to(tl.int32), 256, mask=mt) + \
                tl.histogram(((ur >> shift) & 0xFF).to(tl.int32), 256, mask=mr)
            ge = tl.cumsum(hist, axis=0, reverse=True)
            gt = ge - hist
            hit = (ge >= k_rem) & (gt < k_rem)
            d = tl.sum(tl.where(hit, bins, 0), axis=0)
            k_rem = k_rem - tl.sum(tl.where(hit, gt, 0), axis=0)
            prefix = prefix | (d.to(tl.uint32) << shift)
            pmask = pmask | (tl.full([], 0xFF, tl.uint32) << shift)
        eq = vm & (u == prefix)
        er = tl.cumsum(eq.to(tl.int32), axis=0) - 1
        take = (vm & (u > prefix)) | (eq & (er < k_rem))
        tl.store(Exp + r * E + base + tl.cumsum(take.to(tl.int32), axis=0) - 1, tok, mask=take)
        taken = tl.sum(take.to(tl.int32), axis=0)
        ties = tl.sum(eq.to(tl.int32), axis=0)
        eqr = rv & (ur == prefix)
        err = tl.cumsum(eqr.to(tl.int32), axis=0) - 1 + ties
        taker = (rv & (ur > prefix)) | (eqr & (err < k_rem))
        tl.store(Exp + r * E + base + taken + tl.cumsum(taker.to(tl.int32), axis=0) - 1, rt, mask=taker)


@triton.jit
def _k_topq_d2(S, Mem, Cstart, Clen, Asg, Quota, Exp, RecList, RecCnt, Nptr, N0ptr, SW, Lmax, CN, E, RM,
               BLOCK_T: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr,
               SMALL_Q: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1)
    qv = tl.load(Quota + r * CN + c).to(tl.int32)
    if qv > 0:
        offs_q = tl.arange(0, BLOCK_Q)
        qa = tl.load(Quota + r * CN + offs_q, mask=offs_q < c, other=0.0)
        base = tl.sum(qa, axis=0).to(tl.int32)
        start = tl.load(Cstart + r * CN + c)
        len0 = tl.load(Clen + r * CN + c)
        cnt = tl.load(RecCnt + r * CN + c)
        srow = S + r * SW
        if (len0 <= BLOCK_M) & (cnt <= BLOCK_R):
            oj = tl.arange(0, BLOCK_M)
            vm = oj < len0
            tok = tl.load(Mem + r * Lmax + start + oj, mask=vm, other=0)
            u = _ukey(tl.load(srow + tok, mask=vm, other=0.0))
            lr = tl.arange(0, BLOCK_R)
            rv = lr < cnt
            rt = tl.load(RecList + (r * CN + c) * RM + lr, mask=rv, other=0)
            ur = _ukey(tl.load(srow + rt, mask=rv, other=0.0))
            _topq_core(Exp, r, E, base, qv, tok, vm, u, rt, rv, ur, SMALL_Q)
        else:
            n = tl.load(Nptr)
            n0 = tl.load(N0ptr)
            _topq_loops(srow, Mem, Asg, Exp, r, c, qv, base, start, len0, n, n0, Lmax, E, BLOCK_T)


@triton.jit
def _k_residual_blk(K, V, Asg, Sel, Exp, Sk, Sv, Nt, Quota, KbarR, VbarR, LogN, Lmax, CN, NS, E,
                    G: tl.constexpr, D: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
                    BLOCK_Q: tl.constexpr, BLOCK_S: tl.constexpr):
    """per (row, block of BLOCK_C clusters): pooled sums minus the selected tokens of the block and the
    block's own segment of the expanded list, one-hot fp32 dots; no atomics."""
    r = tl.program_id(0)
    cb = tl.program_id(1)
    offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = offs_c < CN
    offs_d = tl.arange(0, D)
    kvr = r // G
    sp = r * CN * D + offs_c[:, None] * D + offs_d[None, :]
    sk = tl.load(Sk + sp, mask=cm[:, None], other=0.0)
    sv = tl.load(Sv + sp, mask=cm[:, None], other=0.0)
    ot = tl.arange(0, BLOCK_S)
    tm = ot < NS
    tok = tl.load(Sel + r * NS + ot, mask=tm, other=0)
    cid = tl.load(Asg + r * Lmax + tok, mask=tm, other=-1)
    kp = kvr * Lmax * D + tok[:, None] * D + offs_d[None, :]
    oh = ((cid[None, :] == offs_c[:, None]) & tm[None, :]).to(tl.float32)
    sk -= tl.dot(oh, tl.load(K + kp, mask=tm[:, None], other=0.0).to(tl.float32), input_precision="ieee")
    sv -= tl.dot(oh, tl.load(V + kp, mask=tm[:, None], other=0.0).to(tl.float32), input_precision="ieee")
    oq = tl.arange(0, BLOCK_Q)
    qa = tl.load(Quota + r * CN + oq, mask=oq < CN, other=0.0)
    b0 = tl.sum(tl.where(oq < cb * BLOCK_C, qa, 0.0), axis=0).to(tl.int32)
    b1 = tl.sum(tl.where(oq < cb * BLOCK_C + BLOCK_C, qa, 0.0), axis=0).to(tl.int32)
    for j0 in range(b0, b1, BLOCK_T):
        oj = j0 + tl.arange(0, BLOCK_T)
        m = oj < b1
        t2 = tl.load(Exp + r * E + oj, mask=m, other=-1)
        m = m & (t2 >= 0)
        t2 = tl.where(m, t2, 0)
        c2 = tl.load(Asg + r * Lmax + t2, mask=m, other=-1)
        kp2 = kvr * Lmax * D + t2[:, None] * D + offs_d[None, :]
        oh2 = ((c2[None, :] == offs_c[:, None]) & m[None, :]).to(tl.float32)
        sk -= tl.dot(oh2, tl.load(K + kp2, mask=m[:, None], other=0.0).to(tl.float32), input_precision="ieee")
        sv -= tl.dot(oh2, tl.load(V + kp2, mask=m[:, None], other=0.0).to(tl.float32), input_precision="ieee")
    nr = tl.load(Nt + r * CN + offs_c, mask=cm, other=0.0) - tl.load(Quota + r * CN + offs_c, mask=cm, other=0.0)
    inv = 1.0 / tl.maximum(nr, 1.0)
    tl.store(KbarR + sp, (sk * inv[:, None]).to(tl.bfloat16), mask=cm[:, None])
    tl.store(VbarR + sp, (sv * inv[:, None]).to(tl.bfloat16), mask=cm[:, None])
    tl.store(LogN + r * CN + offs_c, tl.where(nr > 0.5, tl.log(tl.maximum(nr, 1.0)), -float("inf")), mask=cm)


@triton.jit
def _k_ours_flash_split(Q, K, V, Sel, Exp, KBar, VBar, LogN, Pm, Pl, Pacc,
                        Lmax, NS, NE, NB, SPLIT, CHUNK, sm_scale,
                        G: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    """ours.flash (P1 selected, P2 expanded, P3 proxies with +log n) over one chunk of the concatenated
    list; partial (m, l, acc) for graph_runner _k_dense_combine."""
    r = tl.program_id(0)
    sp = tl.program_id(1)
    kvr = r // G
    offs_d = tl.arange(0, D)
    q = tl.load(Q + r * D + offs_d).to(tl.float32)
    tot = NS + NE + NB
    start = sp * CHUNK
    end = tl.minimum(start + CHUNK, tot)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for t0 in range(start, end, BLOCK_N):
        j = t0 + tl.arange(0, BLOCK_N)
        valid = j < end
        is_s = j < NS
        is_e = (j >= NS) & (j < NS + NE)
        is_p = j >= NS + NE
        ts = tl.load(Sel + r * NS + j, mask=valid & is_s, other=-1)
        te = tl.load(Exp + r * NE + (j - NS), mask=valid & is_e, other=-1)
        tok = tl.where(is_s, ts, te)
        tv = valid & (~is_p) & (tok >= 0)
        tok = tl.where(tv, tok, 0)
        cb = tl.where(is_p, j - NS - NE, 0)
        pv = valid & is_p
        kt = tl.load(K + kvr * Lmax * D + tok[:, None] * D + offs_d[None, :], mask=tv[:, None], other=0.0).to(tl.float32)
        kb = tl.load(KBar + (r * NB + cb)[:, None] * D + offs_d[None, :], mask=pv[:, None], other=0.0).to(tl.float32)
        ln = tl.load(LogN + r * NB + cb, mask=pv, other=-float("inf"))
        zt = tl.sum(q[None, :] * kt, axis=1) * sm_scale * 1.4426950408889634
        zp = (tl.sum(q[None, :] * kb, axis=1) * sm_scale + ln) * 1.4426950408889634
        z = tl.where(tv, zt, tl.where(pv, zp, -float("inf")))
        m_new = tl.maximum(m_i, tl.max(z, axis=0))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(z - m_safe)
        vt = tl.load(V + kvr * Lmax * D + tok[:, None] * D + offs_d[None, :], mask=tv[:, None], other=0.0).to(tl.float32)
        vb = tl.load(VBar + (r * NB + cb)[:, None] * D + offs_d[None, :], mask=pv[:, None], other=0.0).to(tl.float32)
        v = tl.where(tv[:, None], vt, vb)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    tl.store(Pm + r * SPLIT + sp, m_i)
    tl.store(Pl + r * SPLIT + sp, l_i)
    tl.store(Pacc + (r * SPLIT + sp) * D + offs_d, acc)


class OursG2(G5.OursG):
    name = "ours2"
    BR = 32              # generated tokens of one cluster held in registers by top-q
    QMAX = 127.0         # QGS code range (int8)

    def __init__(self, a, B, Lmax, dev):
        super().__init__(a, B, Lmax, dev)
        Cn, R = self.C, self.R
        self.reclist = [torch.zeros(B, HQ, Cn, RMAX, device=dev, dtype=torch.int32) for _ in range(bl.NL)]
        self.reccnt = [torch.zeros(B, HQ, Cn, device=dev, dtype=torch.int32) for _ in range(bl.NL)]
        tot = self.C + self.E + 64                                  # >= NS + E + C
        self.FSPLIT = 1 if R >= 128 else max(2, min(16, 1024 // R))   # split only helped at B=1 (real-model profile)
        self.FCHUNK = triton.cdiv(triton.cdiv(tot, self.FSPLIT), 32) * 32
        self.pm = torch.empty(R * self.FSPLIT, device=dev)
        self.pl = torch.empty(R * self.FSPLIT, device=dev)
        self.pacc = torch.empty(R * self.FSPLIT, D, device=dev)

    def _build_csr(self, l, b, T):
        super()._build_csr(l, b, T)
        # the generated-token lists hold RMAX entries per cluster; a request generates at most
        # min(Lmax - T, max_new_tokens) tokens (bench_v67 sizes Lmax by its longest prompt, so only the latter
        # binds there: 120 < RMAX); a list that would overflow is guarded in the kernel and falls back to the
        # scanning path of top-q, so this is a performance bound, not a correctness one
        self.reccnt[l][b].zero_()

    def decode(self, l, q, n):
        B, R, Cn, E, dev, Lmax = self.B, self.R, self.C, self.E, self.dev, self.Lmax
        K, V = C.K[l], C.V[l]
        qr = q.reshape(R, D).contiguous()
        _k_index_update_d2[(R,)](K, V, self.codes8[l], self.U[l], self.scale[l], self.W[l], self.colmean[l],
                                 self.cent[l], self.asg[l], self.nb[l], self.Sk[l], self.Sk2[l], self.Sv[l],
                                 self.reclist[l], self.reccnt[l], C.n_dev, Lmax, Cn, RMAX,
                                 G=G, D=D, RQ=self.rq, RC=self.rc, BLOCK_C=64, QMAX=self.QMAX, num_warps=4)
        zq = (torch.matmul(q.view(B, HKV, G, D).float(), self.U[l]) * self.scale[l]).to(torch.bfloat16)
        ns = nl = max(1, int(0.0001 * n))
        heavy = self.a.rho - 0.0002 - E / CTX_LEN - 1.5 * Cn / CTX_LEN
        k_top = max(1, int(heavy * n))
        s = torch.empty(R, self.SW, device=dev)
        tm = torch.empty(R, self.NT, device=dev)
        _k_qgs_scan_tm[(B * HKV, self.NT)](zq, self.codes8[l], s, tm, C.n_dev, ns, nl, Lmax, self.SW, self.NT,
                                           G=G, RQ=self.rq, BLOCK_N=self.BN, num_warps=1)
        NS = ns + nl + k_top
        sel = torch.empty(R, NS, device=dev, dtype=torch.int32)
        _k_select[(R,)](s, tm, sel, C.n_dev, ns, nl, self.NT, self.SW, K_TOP=k_top,
                        KT2=triton.next_power_of_2(min(2 * k_top, self.NT)), BN=self.BN,
                        BLOCK_T=triton.next_power_of_2(self.NT), BLOCK_SL=triton.next_power_of_2(ns + nl), num_warps=4)
        nblk = triton.cdiv(Cn, self.BC)
        score = torch.empty(R, Cn, device=dev)
        n_t = torch.empty(R, Cn, device=dev)
        _k_errorgate[(R, nblk)](qr, K, self.asg[l], sel, self.Sk[l], self.Sk2[l], self.nb[l], score, n_t,
                                Lmax, Cn, NS, SC, G=G, D=D, BLOCK_C=self.BC, BLOCK_T=16, num_warps=4)
        quota = torch.empty(R, Cn, device=dev)
        _k_waterfill[(R,)](score, n_t, quota, Cn, float(E), BLOCK=self.BQ, num_warps=4)
        self._quota = quota
        exp_idx = torch.full((R, E), -1, device=dev, dtype=torch.int32)
        _k_topq_d2[(R, Cn)](s, self.mem[l], self.cstart[l], self.clen[l], self.asg[l], quota, exp_idx,
                            self.reclist[l], self.reccnt[l], C.n_dev, C.n0_dev, self.SW, Lmax, Cn, E, RMAX,
                            BLOCK_T=128, BLOCK_Q=self.BQ, BLOCK_M=self.BM[l], BLOCK_R=self.BR, SMALL_Q=self.SMALL_Q2,
                            num_warps=self.TOPQ_WARPS or (4 if R < 128 else (2 if self.BM[l] >= 512 else 1)))
        kbar_r = torch.empty(R, Cn, D, device=dev, dtype=torch.bfloat16)
        vbar_r = torch.empty(R, Cn, D, device=dev, dtype=torch.bfloat16)
        logn = torch.empty(R, Cn, device=dev)
        # 6. residual: the ours_fused atomic scatter + normalise measured faster than _k_residual_blk on the real
        #    model (1% B=1 0.31 vs 0.62, 3% B=8 3.98 vs 5.49 ms/step), so it stays
        dk = torch.zeros(R, Cn, D, device=dev)
        dv = torch.zeros(R, Cn, D, device=dev)
        F._k_resid_scatter[(R, triton.cdiv(NS + E, self.BT))](K, V, self.asg[l], sel, exp_idx, dk, dv, Lmax, Cn,
                                                             NS, E, G=G, D=D, BLOCK_T=self.BT, num_warps=4)
        F._k_resid_norm[(R, nblk)](self.Sk[l], self.Sv[l], dk, dv, n_t, quota, kbar_r, vbar_r, logn, Cn,
                                   D=D, BLOCK_C=self.BC, num_warps=4)
        bl.mark()
        if self.FSPLIT == 1:                     # enough rows to fill the GPU: the one-pass kernel
            out = bl.ours_flash(qr, K.view(-1, Lmax, D), V.view(-1, Lmax, D), sel, exp_idx, kbar_r, vbar_r, logn)
        else:                                    # few rows (B=1): split the list across programs
            _k_ours_flash_split[(R, self.FSPLIT)](qr, K, V, sel, exp_idx, kbar_r, vbar_r, logn, self.pm, self.pl,
                                                  self.pacc, Lmax, NS, E, Cn, self.FSPLIT, self.FCHUNK, SC,
                                                  G=G, D=D, BLOCK_N=32, num_warps=4)
            out = torch.empty(R, D, device=dev, dtype=q.dtype)
            _k_dense_combine[(R,)](self.pm, self.pl, self.pacc, out, self.FSPLIT, D=D,
                                   BLOCK_S=max(2, triton.next_power_of_2(self.FSPLIT)), num_warps=4)
        if getattr(self, "_spy", False):                 # self test: expose this call's selection
            F._last["exp"], F._last["sel"] = exp_idx.clone(), sel.clone()
        return out.view(B, HQ, 1, D)


G5.METHODS["ours2"] = OursG2


def selftest(dev):
    print("== OursG2 (v66) vs OursG (v65) on shared state")
    dec = G5.OursG.decode

    def spied(self, l, q, n):
        old = bl.ours_flash

        def spy(qr, k, v, sel, exp_idx, kb, vb, ln, nb=None):
            F._last["exp"], F._last["sel"] = exp_idx.clone(), sel.clone()
            return old(qr, k, v, sel, exp_idx, kb, vb, ln, nb)
        bl.ours_flash = spy
        try:
            return dec(self, l, q, n)
        finally:
            bl.ours_flash = old
    G5.OursG.decode = spied
    for rho in (0.01, 0.03):
        for B in (1, 8):
            a, Lmax, qp = F._synthetic(B, rho, dev, extra=160)
            n0 = qp.shape[2]
            ref, new = G5.OursG(a, B, Lmax, dev), OursG2(a, B, Lmax, dev)
            new._spy = True
            for b in range(B):
                ref.build(0, b, qp, n0)
            for nm in ("codes", "U", "scale", "W", "colmean", "cent", "asg", "nb", "Sk", "Sk2", "Sv"):
                getattr(new, nm)[0].copy_(getattr(ref, nm)[0])
            for b in range(B):
                ref._build_csr(0, b, n0)
                new._build_csr(0, b, n0)
            C.n0_dev = torch.tensor([n0], device=dev, dtype=torch.int32)
            n, bad_sel, bad_exp, rows, relmax = n0, 0, 0, 0, 0.0
            for it in range(40):                                 # enough steps for generated tokens to matter
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
                  f"{bad_exp}/{rows}, output rel err max {relmax:.2e}, max generated per cluster "
                  f"{int(new.reccnt[0].max())}")
            del ref, new
            torch.cuda.empty_cache()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        dev = torch.device(sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:1")
        torch.cuda.set_device(dev)
        selftest(dev)
    else:
        G5.main()
