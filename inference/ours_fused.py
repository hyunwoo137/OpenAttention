#!/usr/bin/env python3
"""V64: the ours decode step of methods_eager with its selection fused into a handful of Triton kernels.

methods_eager runs the deployed per-step pipeline as ~260 small PyTorch kernels per layer; at batch 1 the
host dispatch of those kernels is ~70% of the step, and at batch 8 the within-cluster top-q (a sort of the
whole cache per row) is the largest GPU stage.  Same algorithm, same state, same hyper-parameters; what
changes is only how each stage is executed:

  1  index update      one kernel: QGS code of the new key, nearest QGC centroid, pooled sums += (k, k^2, v)
  2  QGS scan + top-k  unchanged (methods_eager code)
  3  ErrorGate         one kernel per (row, cluster block): pooled sums minus the selected tokens (one-hot
                       fp32 dot), then the wl1 score of every cluster
  4  water-filling     one kernel per row: the sort-based water-filling of methods_eager.Ours.hier_quota with
                       the clusters in registers
  5  within-cluster    one kernel per (row, expanded cluster): exact top-q_b by the QGS score with a 4-pass
     top-q_b           8-bit radix select over that cluster's members only (CSR lists built with the
                       clusters at prefill, plus the tokens generated since), instead of sorting all n
  6  residual stats    one kernel per (row, cluster block): pooled sums minus every token read -> kbar', vbar',
                       log n'
  7  ours.flash        unchanged

    python ours_fused.py --selftest --device cuda:1            # equivalence + per-layer timing
    python ours_fused.py --method oursfast --rho 0.01 --batch 1 --device cuda:1 --out ...   # = methods_eager driver
"""
import os
import statistics
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
sys.path.insert(0, _HERE)
import methods_eager as bl                                                  # noqa: E402
from methods_eager import C, CTX_LEN, D, G, HKV, HQ, NEG, SC, sink_local    # noqa: E402



# ============================================================ 1. index update
@triton.jit
def _k_index_update(K, V, Codes, U, Scale, W, Colmean, Cent, Asg, Nb, Sk, Sk2, Sv,
                    pos, Lmax, CN,
                    G: tl.constexpr, D: tl.constexpr, RQ: tl.constexpr, RC: tl.constexpr,
                    BLOCK_C: tl.constexpr):
    r = tl.program_id(0)
    kvr = r // G
    offs_d = tl.arange(0, D)
    kn = tl.load(K + kvr * Lmax * D + pos * D + offs_d)
    vn = tl.load(V + kvr * Lmax * D + pos * D + offs_d).to(tl.float32)
    knf = kn.to(tl.float32)
    if r % G == 0:                                    # the QGS code belongs to the KV head
        offs_q = tl.arange(0, RQ)
        u = tl.load(U + kvr * D * RQ + offs_d[:, None] * RQ + offs_q[None, :])
        zc = tl.sum(knf[:, None] * u, axis=0)
        sc = tl.load(Scale + kvr * RQ + offs_q)
        code = tl.minimum(tl.maximum(libdevice.rint(zc / sc), -127.0), 127.0)
        tl.store(Codes + kvr * Lmax * RQ + pos * RQ + offs_q, code.to(Codes.dtype.element_ty))
    offs_c = tl.arange(0, RC)
    w = tl.load(W + r * D * RC + offs_d[:, None] * RC + offs_c[None, :]).to(tl.float32)
    zp = tl.sum(knf[:, None] * w, axis=0).to(tl.bfloat16).to(tl.float32)      # methods_eager: bf16 matmul
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


# ============================================================ 2. QGS scan + top-k candidates
@triton.jit
def _k_qgs_scan(Zq, Codes, S, CandV, CandI, n, ns, nl, Lmax, NT,
                G: tl.constexpr, RQ: tl.constexpr, BLOCK_N: tl.constexpr, KT: tl.constexpr):
    """scores of one KV head's G query heads over one tile of int8 codes (rounded to bf16 as the bf16
    matmul of methods_eager does), sink/local masked, written in fp32; plus the tile's top-KT per row, so
    the row top-k only has to look at NT*KT candidates (exact: every global top-k key is in the top-KT
    of its tile)."""
    kvr = tl.program_id(0)
    t = tl.program_id(1)
    offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
    vm = offs_n < n
    offs_r = tl.arange(0, RQ)
    codes = tl.load(Codes + kvr * Lmax * RQ + offs_n[:, None] * RQ + offs_r[None, :],
                    mask=vm[:, None], other=0).to(tl.float32)
    dead = (offs_n < ns) | (offs_n >= n - nl) | (~vm)
    for g in tl.static_range(G):
        r = kvr * G + g
        zq = tl.load(Zq + r * RQ + offs_r).to(tl.float32)
        sc = tl.sum(codes * zq[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
        sc = tl.where(dead, -float("inf"), sc)
        tl.store(S + r * n + offs_n, sc, mask=vm)
        v = sc
        for k in tl.static_range(KT):
            m = tl.max(v, axis=0)
            i = tl.min(tl.where(v == m, offs_n, 2147483647), axis=0)
            tl.store(CandV + (r * NT + t) * KT + k, m)
            tl.store(CandI + (r * NT + t) * KT + k, i)
            v = tl.where(offs_n == i, -float("inf"), v)


# ============================================================ 3. ErrorGate on the tail statistics
@triton.jit
def _k_errorgate(Q, K, Asg, Sel, Sk, Sk2, Nb, Score, Nt,
                 Lmax, CN, NS, sm_scale,
                 G: tl.constexpr, D: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr):
    r = tl.program_id(0)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = offs_c < CN
    offs_d = tl.arange(0, D)
    kvr = r // G
    q = tl.load(Q + r * D + offs_d).to(tl.float32)
    sp = r * CN * D + offs_c[:, None] * D + offs_d[None, :]
    sk = tl.load(Sk + sp, mask=cm[:, None], other=0.0)
    sk2 = tl.load(Sk2 + sp, mask=cm[:, None], other=0.0)
    nb = tl.load(Nb + r * CN + offs_c, mask=cm, other=0.0)
    for t0 in range(0, NS, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        tm = offs_t < NS
        tok = tl.load(Sel + r * NS + offs_t, mask=tm, other=0)
        cid = tl.load(Asg + r * Lmax + tok, mask=tm, other=-1)
        kt = tl.load(K + kvr * Lmax * D + tok[:, None] * D + offs_d[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        oh = ((cid[None, :] == offs_c[:, None]) & tm[None, :]).to(tl.float32)          # (BLOCK_C, BLOCK_T)
        sk -= tl.dot(oh, kt, input_precision="ieee")
        sk2 -= tl.dot(oh, kt * kt, input_precision="ieee")
        nb -= tl.sum(oh, axis=1)
    inv = 1.0 / tl.maximum(nb, 1.0)
    kbar = sk * inv[:, None]
    var = tl.maximum(sk2 * inv[:, None] - kbar * kbar, 0.0)
    zbar = sm_scale * tl.sum(q[None, :] * kbar, axis=1)
    s2 = (sm_scale * sm_scale) * tl.sum((q * q)[None, :] * var, axis=1)
    er = libdevice.erf(tl.sqrt(s2) * 0.7071067811865476)
    score = zbar + tl.log(tl.maximum(nb, 1.0)) + 0.5 * s2 + tl.log(tl.maximum(er, 1e-30))
    tl.store(Score + r * CN + offs_c, score, mask=cm)
    tl.store(Nt + r * CN + offs_c, nb, mask=cm)


# ============================================================ 4. hierarchical water-filling
@triton.jit
def _mono_desc_key(x, idx):
    """int64 key whose ascending order is x descending, ties by index ascending."""
    b = x.to(tl.int32, bitcast=True)
    mono = b ^ ((b >> 31) & 0x7FFFFFFF)                      # ascending with x
    return ((~mono).to(tl.int64) << 32) | idx.to(tl.int64)


@triton.jit
def _k_waterfill(Score, Nt, Quota, CN, E,
                 BLOCK: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < CN
    score = tl.load(Score + r * CN + offs, mask=m, other=-float("inf"))
    nt = tl.load(Nt + r * CN + offs, mask=m, other=0.0)
    live = (nt > 1.0) & m
    s = tl.where(live, score, -float("inf"))
    smax = tl.max(s, axis=0)
    w = tl.where(live, tl.exp(s - smax), 0.0)
    cap = tl.where(live, nt, 0.0)
    tot = tl.sum(cap, axis=0)
    ratio = tl.where(w > 0, cap / tl.maximum(w, 1e-30), float("inf"))
    # ascending ratio (ratio >= 0, so its float bits are monotone as int32); ties do not change lambda
    key = (ratio.to(tl.int32, bitcast=True).to(tl.int64) << 32) | offs.to(tl.int64)
    key = tl.sort(key)
    o = key.to(tl.int32)                                     # low 32 bits = cluster index
    r_s = (key >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    om = o < CN
    nt_s = tl.load(Nt + r * CN + o, mask=om, other=0.0)
    sc_s = tl.load(Score + r * CN + o, mask=om, other=-float("inf"))
    live_s = (nt_s > 1.0) & om
    c_s = tl.where(live_s, nt_s, 0.0)
    w_s = tl.where(live_s, tl.exp(sc_s - smax), 0.0)
    c_ex = tl.cumsum(c_s, axis=0) - c_s
    w_suf = tl.cumsum(w_s, axis=0, reverse=True)
    f_at = tl.where(w_suf > 0, c_ex + r_s * w_suf, float("inf"))
    jj = tl.min(tl.where(f_at >= E, offs, BLOCK), axis=0)
    jj = tl.where(jj == BLOCK, 0, jj)
    c_j = tl.sum(tl.where(offs == jj, c_ex, 0.0), axis=0)
    w_j = tl.sum(tl.where(offs == jj, w_suf, 0.0), axis=0)
    lam = (E - c_j) / tl.maximum(w_j, 1e-30)
    qb = tl.minimum(cap, tl.floor(lam * w))
    qb = tl.where(tot <= E, cap, qb)
    rem = tl.maximum(tl.minimum(tot, E) - tl.sum(qb, axis=0), 0.0)
    # leftover whole tokens: one each to the highest-scoring clusters with room
    tl.store(Quota + r * CN + offs, qb, mask=m)
    tl.debug_barrier()
    key2 = tl.sort(_mono_desc_key(s, offs))
    o2 = key2.to(tl.int32)
    om2 = o2 < CN
    qb_s = tl.load(Quota + r * CN + o2, mask=om2, other=0.0)
    nt2 = tl.load(Nt + r * CN + o2, mask=om2, other=0.0)
    cap_s = tl.where((nt2 > 1.0) & om2, nt2, 0.0)
    room = (qb_s < cap_s) & om2
    rk = tl.cumsum(room.to(tl.int32), axis=0) - 1
    add = room & (rk.to(tl.float32) < rem)
    tl.debug_barrier()
    tl.store(Quota + r * CN + o2, qb_s + add.to(tl.float32), mask=om2)


# ============================================================ 5. within-cluster top-q_b (radix select)
@triton.jit
def _ukey(s):
    """uint32 key, ascending with the float score (-inf lowest)."""
    b = s.to(tl.uint32, bitcast=True)
    neg = (b >> 31) == 1
    return tl.where(neg, ~b, b | 0x80000000)


@triton.jit
def _topq_regs(srow, Mem, Asg, Exp, r, c, qv, base, start, len0, n, n0, Lmax, E,
               BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr, SMALL_Q: tl.constexpr):
    """fast path: the cluster's members (<= BLOCK_M) and the generated tokens (<= BLOCK_R) are read
    once into registers; a small quota is taken by repeated max extraction, a larger one by the four
    radix passes.  Both order by score, then members before generated tokens, then position."""
    oj = tl.arange(0, BLOCK_M)
    vm = oj < len0
    tok = tl.load(Mem + r * Lmax + start + oj, mask=vm, other=0)
    u = _ukey(tl.load(srow + tok, mask=vm, other=0.0))
    lr = tl.arange(0, BLOCK_R)
    rt = n0 + lr
    rv = rt < n
    a = tl.load(Asg + r * Lmax + rt, mask=rv, other=-1)
    rv = rv & (a == c)
    ur = _ukey(tl.load(srow + rt, mask=rv, other=0.0))
    if qv <= SMALL_Q:
        um = tl.where(vm, u, 0)                                   # 0 = gone (every real key is > 0)
        uq = tl.where(rv, ur, 0)
        for k in range(qv):
            m = tl.maximum(tl.max(um, axis=0), tl.max(uq, axis=0))
            p = tl.minimum(tl.min(tl.where(um == m, oj, 2147483647), axis=0),
                           tl.min(tl.where(uq == m, BLOCK_M + lr, 2147483647), axis=0))
            t = tl.where(p < BLOCK_M, tl.sum(tl.where(oj == p, tok, 0), axis=0), n0 + p - BLOCK_M)
            tl.store(Exp + r * E + base + k, t)
            um = tl.where(oj == p, 0, um)
            uq = tl.where(BLOCK_M + lr == p, 0, uq)
        return
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
def _topq_loops(srow, Mem, Asg, Exp, r, c, qv, base, start, len0, n, n0, Lmax, E, BLOCK_T: tl.constexpr):
    """general path: clusters larger than the register tile; every pass re-reads the members."""
    prefix = tl.full([], 0, tl.uint32)
    pmask = tl.full([], 0, tl.uint32)
    k_rem = qv
    bins = tl.arange(0, 256)
    for i in tl.static_range(4):
        shift = 24 - 8 * i
        hist = tl.zeros([256], dtype=tl.int32)
        for j0 in range(0, len0, BLOCK_T):
            oj = j0 + tl.arange(0, BLOCK_T)
            vm = oj < len0
            tok = tl.load(Mem + r * Lmax + start + oj, mask=vm, other=0)
            u = _ukey(tl.load(srow + tok, mask=vm, other=0.0))
            mt = vm & ((u & pmask) == prefix)
            hist += tl.histogram(((u >> shift) & 0xFF).to(tl.int32), 256, mask=mt)
        for t0 in range(n0, n, BLOCK_T):
            tok = t0 + tl.arange(0, BLOCK_T)
            vm = tok < n
            a = tl.load(Asg + r * Lmax + tok, mask=vm, other=-1)
            vm = vm & (a == c)
            u = _ukey(tl.load(srow + tok, mask=vm, other=0.0))
            mt = vm & ((u & pmask) == prefix)
            hist += tl.histogram(((u >> shift) & 0xFF).to(tl.int32), 256, mask=mt)
        ge = tl.cumsum(hist, axis=0, reverse=True)               # members with digit >= d
        gt = ge - hist
        hit = (ge >= k_rem) & (gt < k_rem)
        d = tl.sum(tl.where(hit, bins, 0), axis=0)
        k_rem = k_rem - tl.sum(tl.where(hit, gt, 0), axis=0)
        prefix = prefix | (d.to(tl.uint32) << shift)
        pmask = pmask | (tl.full([], 0xFF, tl.uint32) << shift)
    # take every member above the threshold key, and the first k_rem members equal to it
    taken = tl.full([], 0, tl.int32)
    ties = tl.full([], 0, tl.int32)
    for j0 in range(0, len0, BLOCK_T):
        oj = j0 + tl.arange(0, BLOCK_T)
        vm = oj < len0
        tok = tl.load(Mem + r * Lmax + start + oj, mask=vm, other=0)
        u = _ukey(tl.load(srow + tok, mask=vm, other=0.0))
        eq = vm & (u == prefix)
        er = tl.cumsum(eq.to(tl.int32), axis=0) - 1 + ties
        take = (vm & (u > prefix)) | (eq & (er < k_rem))
        tl.store(Exp + r * E + base + taken + tl.cumsum(take.to(tl.int32), axis=0) - 1, tok, mask=take)
        taken += tl.sum(take.to(tl.int32), axis=0)
        ties += tl.sum(eq.to(tl.int32), axis=0)
    for t0 in range(n0, n, BLOCK_T):
        tok = t0 + tl.arange(0, BLOCK_T)
        vm = tok < n
        a = tl.load(Asg + r * Lmax + tok, mask=vm, other=-1)
        vm = vm & (a == c)
        u = _ukey(tl.load(srow + tok, mask=vm, other=0.0))
        eq = vm & (u == prefix)
        er = tl.cumsum(eq.to(tl.int32), axis=0) - 1 + ties
        take = (vm & (u > prefix)) | (eq & (er < k_rem))
        tl.store(Exp + r * E + base + taken + tl.cumsum(take.to(tl.int32), axis=0) - 1, tok, mask=take)
        taken += tl.sum(take.to(tl.int32), axis=0)
        ties += tl.sum(eq.to(tl.int32), axis=0)


@triton.jit
def _k_topq(S, Mem, Cstart, Clen, Asg, Quota, Exp,
            n, n0, Lmax, CN, E,
            BLOCK_T: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr,
            SMALL_Q: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1)
    qv = tl.load(Quota + r * CN + c).to(tl.int32)
    if qv > 0:
        offs_q = tl.arange(0, BLOCK_Q)
        qa = tl.load(Quota + r * CN + offs_q, mask=offs_q < c, other=0.0)
        base = tl.sum(qa, axis=0).to(tl.int32)                 # this cluster's slots in exp_idx
        start = tl.load(Cstart + r * CN + c)
        len0 = tl.load(Clen + r * CN + c)
        srow = S + r * n
        if (len0 <= BLOCK_M) & (n - n0 <= BLOCK_R):
            _topq_regs(srow, Mem, Asg, Exp, r, c, qv, base, start, len0, n, n0, Lmax, E, BLOCK_M, BLOCK_R, SMALL_Q)
        else:
            _topq_loops(srow, Mem, Asg, Exp, r, c, qv, base, start, len0, n, n0, Lmax, E, BLOCK_T)


# ============================================================ 6. residual statistics of the unread members
@triton.jit
def _k_resid_scatter(K, V, Asg, Sel, Exp, DK, DV, Lmax, CN, NS, E,
                     G: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr):
    """every token read (selected + expanded) adds its k and v to its cluster's delta, once"""
    r = tl.program_id(0)
    offs_t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    kvr = r // G
    in_s = offs_t < NS
    in_e = (offs_t >= NS) & (offs_t < NS + E)
    tok_s = tl.load(Sel + r * NS + offs_t, mask=in_s, other=-1)
    tok_e = tl.load(Exp + r * E + (offs_t - NS), mask=in_e, other=-1)
    tok = tl.where(in_s, tok_s, tok_e)
    tm = tok >= 0
    tok = tl.where(tm, tok, 0)
    cid = tl.load(Asg + r * Lmax + tok, mask=tm, other=0)
    kp = kvr * Lmax * D + tok[:, None] * D + offs_d[None, :]
    kt = tl.load(K + kp, mask=tm[:, None], other=0.0).to(tl.float32)
    vt = tl.load(V + kp, mask=tm[:, None], other=0.0).to(tl.float32)
    dp = r * CN * D + cid[:, None] * D + offs_d[None, :]
    tl.atomic_add(DK + dp, kt, mask=tm[:, None], sem="relaxed")
    tl.atomic_add(DV + dp, vt, mask=tm[:, None], sem="relaxed")


@triton.jit
def _k_resid_norm(Sk, Sv, DK, DV, Nt, Quota, KbarR, VbarR, LogN, CN,
                  D: tl.constexpr, BLOCK_C: tl.constexpr):
    """pooled sums minus every token read -> kbar', vbar' (bf16) and log n'"""
    r = tl.program_id(0)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = offs_c < CN
    offs_d = tl.arange(0, D)
    sp = r * CN * D + offs_c[:, None] * D + offs_d[None, :]
    sk = tl.load(Sk + sp, mask=cm[:, None], other=0.0) - tl.load(DK + sp, mask=cm[:, None], other=0.0)
    sv = tl.load(Sv + sp, mask=cm[:, None], other=0.0) - tl.load(DV + sp, mask=cm[:, None], other=0.0)
    nr = tl.load(Nt + r * CN + offs_c, mask=cm, other=0.0) - tl.load(Quota + r * CN + offs_c, mask=cm, other=0.0)
    inv = 1.0 / tl.maximum(nr, 1.0)
    tl.store(KbarR + sp, (sk * inv[:, None]).to(tl.bfloat16), mask=cm[:, None])
    tl.store(VbarR + sp, (sv * inv[:, None]).to(tl.bfloat16), mask=cm[:, None])
    ln = tl.where(nr > 0.5, tl.log(tl.maximum(nr, 1.0)), -float("inf"))
    tl.store(LogN + r * CN + offs_c, ln, mask=cm)


# ============================================================ the method
class OursFast(bl.Ours):
    name = "oursfast"
    BC, BT = 16, 32
    SMALL_Q = 8                  # quotas up to this are taken by repeated max extraction
    BN = 256                     # QGS scan tile (tokens)
    BM_MIN, BM_MAX = 128, 2048   # register tile of the within-cluster top-q: next pow2 of the MEAN cluster
                                 # size n0/C (fastest in a sweep: 512 at 1%, 256 at 3%); larger clusters
                                 # take the loop path

    def __init__(self, a, B, Lmax, dev):
        super().__init__(a, B, Lmax, dev)
        Cn = self.C
        self.mem = [torch.zeros(B, HQ, Lmax, device=dev, dtype=torch.int32) for _ in range(bl.NL)]
        self.cstart = [torch.zeros(B, HQ, Cn, device=dev, dtype=torch.int32) for _ in range(bl.NL)]
        self.clen = [torch.zeros(B, HQ, Cn, device=dev, dtype=torch.int32) for _ in range(bl.NL)]
        self.n0 = 0
        self.BQ = triton.next_power_of_2(Cn)
        self.BM = [self.BM_MIN] * bl.NL
        # QGS codes as int8 (methods_eager keeps the same int8 values in a bf16 buffer: twice the bytes per scan)
        self.codes8 = [torch.zeros(B, HKV, Lmax, self.rq, device=dev, dtype=torch.int8) for _ in range(bl.NL)]

    def build(self, l, b, q, T):
        super().build(l, b, q, T)
        self._build_csr(l, b, T)

    def _build_csr(self, l, b, T):
        self.codes8[l][b] = self.codes[l][b].to(torch.int8)
        # CSR member lists of the prefix, built with the clusters (charged to the prefill)
        asg = self.asg[l][b, :, :T].long()
        self.mem[l][b, :, :T] = torch.argsort(asg, dim=-1, stable=True).to(torch.int32)
        cnt = self.nb[l][b].round().to(torch.int32)
        self.cstart[l][b] = (cnt.cumsum(-1) - cnt).to(torch.int32)
        self.clen[l][b] = cnt
        self.n0 = T
        self.BM[l] = max(self.BM_MIN, min(self.BM_MAX, triton.next_power_of_2(-(-T // self.C))))

    def decode(self, l, q, n):
        B, R, Cn, E, dev, Lmax = self.B, self.R, self.C, self.E, self.dev, self.Lmax
        K, V = C.K[l], C.V[l]
        qr = q.reshape(R, D).contiguous()
        # 1. index update of the new token (one kernel)
        _k_index_update[(R,)](K, V, self.codes8[l], self.U[l], self.scale[l], self.W[l], self.colmean[l],
                              self.cent[l], self.asg[l], self.nb[l], self.Sk[l], self.Sk2[l], self.Sv[l],
                              n - 1, Lmax, Cn, G=G, D=D, RQ=self.rq, RC=self.rc, BLOCK_C=64, num_warps=4)
        # 2. QGS scan (int8 codes, one kernel) and top-k over the tiles' candidates
        zq = (torch.matmul(q.view(B, HKV, G, D).float(), self.U[l]) * self.scale[l]).to(torch.bfloat16)
        ns = nl = max(1, int(0.0001 * n))
        heavy = self.a.rho - 0.0002 - E / CTX_LEN - 1.5 * Cn / CTX_LEN
        k_top = max(1, int(heavy * n))
        NT = triton.cdiv(n, self.BN)
        s = torch.empty(R, n, device=dev)
        cv = torch.empty(R, NT * k_top, device=dev)
        ci = torch.empty(R, NT * k_top, device=dev, dtype=torch.int32)
        _k_qgs_scan[(B * HKV, NT)](zq, self.codes8[l], s, cv, ci, n, ns, nl, Lmax, NT,
                                   G=G, RQ=self.rq, BLOCK_N=self.BN, KT=k_top, num_warps=1)
        top = ci.gather(1, cv.topk(k_top, dim=-1).indices)
        sel = torch.cat([sink_local(n, ns, nl, R, dev).to(torch.int32), top], 1).contiguous()
        NS = sel.shape[1]
        # 3. ErrorGate (wl1) on the tail statistics
        nblk = triton.cdiv(Cn, self.BC)
        score = torch.empty(R, Cn, device=dev)
        n_t = torch.empty(R, Cn, device=dev)
        _k_errorgate[(R, nblk)](qr, K, self.asg[l], sel, self.Sk[l], self.Sk2[l], self.nb[l], score, n_t,
                                Lmax, Cn, NS, SC, G=G, D=D, BLOCK_C=self.BC, BLOCK_T=16, num_warps=4)
        # 4. hierarchical water-filling of E
        quota = torch.empty(R, Cn, device=dev)
        _k_waterfill[(R,)](score, n_t, quota, Cn, float(E), BLOCK=self.BQ, num_warps=4)
        self._quota = quota
        # 5. within-cluster top-q_b by the QGS score (selected tokens go last in their cluster)
        s.scatter_(1, sel.long(), NEG)
        exp_idx = torch.full((R, E), -1, device=dev, dtype=torch.int32)
        _k_topq[(R, Cn)](s, self.mem[l], self.cstart[l], self.clen[l], self.asg[l], quota, exp_idx,
                         n, self.n0, Lmax, Cn, E, BLOCK_T=128, BLOCK_Q=self.BQ, BLOCK_M=self.BM[l],
                         BLOCK_R=triton.next_power_of_2(max(Lmax - self.n0, 16)), SMALL_Q=self.SMALL_Q,
                         num_warps=4)
        # 6. residual statistics of the unread members
        kbar_r = torch.empty(R, Cn, D, device=dev, dtype=torch.bfloat16)
        vbar_r = torch.empty(R, Cn, D, device=dev, dtype=torch.bfloat16)
        logn = torch.empty(R, Cn, device=dev)
        dk = torch.zeros(R, Cn, D, device=dev)
        dv = torch.zeros(R, Cn, D, device=dev)
        _k_resid_scatter[(R, triton.cdiv(NS + E, self.BT))](K, V, self.asg[l], sel, exp_idx, dk, dv, Lmax, Cn, NS, E,
                                                           G=G, D=D, BLOCK_T=self.BT, num_warps=4)
        _k_resid_norm[(R, nblk)](self.Sk[l], self.Sv[l], dk, dv, n_t, quota, kbar_r, vbar_r, logn, Cn,
                                 D=D, BLOCK_C=self.BC, num_warps=4)
        self.log_read((2 * ns + k_top + E + 1.5 * Cn) / n)
        # 7. ours.flash
        bl.mark()
        out = bl.ours_flash(qr, K.view(-1, Lmax, D), V.view(-1, Lmax, D), sel, exp_idx, kbar_r, vbar_r, logn)
        return out.view(B, HQ, 1, D)


bl.METHODS["oursfast"] = OursFast


# ============================================================ equivalence test against methods_eager.Ours
class _RefRec(bl.Ours):
    def attend(self, l, q, sel, exp_idx, kbar_r, vbar_r, logn):
        self.rec = (sel, exp_idx, kbar_r, vbar_r, logn)
        return super().attend(l, q, sel, exp_idx, kbar_r, vbar_r, logn)


def _synthetic(B, rho, dev, n0=30012, extra=64, seed=0):
    torch.manual_seed(seed)
    bl.NL = 1
    Lmax = n0 + extra
    C.K = [torch.randn(B, HKV, Lmax, D, device=dev, dtype=torch.bfloat16)]
    C.V = [torch.randn(B, HKV, Lmax, D, device=dev, dtype=torch.bfloat16)]
    C.qlen = [40] * B
    a = types.SimpleNamespace(rho=rho)
    qp = torch.randn(1, HQ, n0, D, device=dev, dtype=torch.bfloat16)
    return a, Lmax, qp


def selftest(dev, steps=8):
    print("== equivalence: methods_eager Ours (reference) vs OursFast, synthetic data of the real shapes")
    for rho in (0.01, 0.03):
        for B in (1, 8):
            a, Lmax, qp = _synthetic(B, rho, dev)
            n0 = qp.shape[2]
            ref, fast = _RefRec(a, B, Lmax, dev), OursFast(a, B, Lmax, dev)
            # one build, copied: the reference build itself is not bitwise reproducible (scatter_add_
            # atomics in the Lloyd steps), so two builds would start the comparison from different clusters
            for b in range(B):
                ref.build(0, b, qp, n0)
            for nm in ("codes", "U", "scale", "W", "colmean", "cent", "asg", "nb", "Sk", "Sk2", "Sv"):
                getattr(fast, nm)[0].copy_(getattr(ref, nm)[0])
            for b in range(B):
                fast._build_csr(0, b, n0)
            n = n0
            worst = dict(rel_med=0.0, rel_max=0.0, exp_rows=0, rows=0, sel_rows=0, st=0.0, asg=0)
            for it in range(steps):
                n += 1
                q = torch.randn(B, HQ, 1, D, device=dev, dtype=torch.bfloat16)
                o_r = ref.decode(0, q, n).float()
                o_f = fast.decode(0, q, n).float()
                sel_r, exp_r, _, _, _ = ref.rec
                rel = (o_f - o_r).norm(dim=-1) / o_r.norm(dim=-1).clamp(min=1e-30)
                worst["rel_med"] = max(worst["rel_med"], rel.median().item())
                worst["rel_max"] = max(worst["rel_max"], rel.max().item())
                # exp sets: compare per row as sorted lists (order inside exp_idx does not matter)
                er = exp_r.sort(-1).values
                ef = fast_last_exp(fast).long().sort(-1).values
                badr = (er != ef).any(-1).nonzero().flatten().tolist()
                worst["exp_rows"] += len(badr)
                for r in badr[:2]:                          # diagnose: quota difference or selection difference?
                    asg_r = fast.asg[0].view(-1, Lmax)[r]
                    A, Bs = set(exp_r[r].tolist()), set(ef[r].tolist())
                    cl = sorted({int(asg_r[t]) for t in (A ^ Bs) if t >= 0})
                    qf = _last["quota"][r]
                    for c in cl:
                        nr = sum(1 for t in A if t >= 0 and int(asg_r[t]) == c)
                        nf = sum(1 for t in Bs if t >= 0 and int(asg_r[t]) == c)
                        print(f"    [diag] step {it} row {r} cluster {c}: ref takes {nr}, fast takes {nf}, fast quota {qf[c].item():.0f}")
                worst["rows"] += er.shape[0]
                worst["st"] = max(worst["st"], ((fast.Sk[0] - ref.Sk[0]).abs().max() / ref.Sk[0].abs().max()).item())
                worst["asg"] += int((fast.asg[0][..., :n] != ref.asg[0][..., :n]).sum())
                worst["codes"] = worst.get("codes", 0) + int((fast.codes8[0][..., :n, :].to(torch.bfloat16) != ref.codes[0][..., :n, :]).sum())
                worst["sel_rows"] += int((sel_r.sort(-1).values != fast_last_sel().long().sort(-1).values).any(-1).sum())
            print(f"  rho={rho} B={B}: output rel err median {worst['rel_med']:.2e} max {worst['rel_max']:.2e} | "
                  f"rows with a different expanded set {worst['exp_rows']}/{worst['rows']} | "
                  f"rows with a different top-k {worst['sel_rows']} | state: max rel |Sk diff| {worst['st']:.1e}, "
                  f"new-token cluster assignments differing {worst['asg']}/{steps * B * HQ}, code entries differing {worst['codes']}")
            del ref, fast
            torch.cuda.empty_cache()


_last = {}


def fast_last_exp(m):
    return _last["exp"]


def fast_last_sel():
    return _last["sel"]


_orig_topq = _k_topq


def _record_exp():
    """wrap OursFast.decode so the selftest can read the expanded set it produced"""
    dec = OursFast.decode

    def wrapped(self, l, q, n):
        old = bl.ours_flash

        def spy(qr, k, v, sel, exp_idx, kb, vb, ln, nb=None):
            _last["exp"] = exp_idx.clone()
            _last["quota"] = self._quota.clone()
            _last["sel"] = sel.clone()
            return old(qr, k, v, sel, exp_idx, kb, vb, ln, nb)
        bl.ours_flash = spy
        try:
            return dec(self, l, q, n)
        finally:
            bl.ours_flash = old
    OursFast.decode = wrapped
    return dec


def timing(dev, iters=40):
    """per-layer decode time (CUDA events around the whole decode call, no sync inside), ref vs fast"""
    print("== per-layer decode step (selection + attention), CUDA events, median of iters, no host sync")
    for B in (1, 8):
        for rho in (0.01, 0.03):
            row = {}
            for kind, cls in (("v62", bl.Ours), ("fast", OursFast)):
                a, Lmax, qp = _synthetic(B, rho, dev, extra=iters + 16)
                n0 = qp.shape[2]
                m = cls(a, B, Lmax, dev)
                for b in range(B):
                    m.build(0, b, qp, n0)
                n, ts = n0, []
                qs = [torch.randn(B, HQ, 1, D, device=dev, dtype=torch.bfloat16) for _ in range(iters)]
                torch.cuda.synchronize()
                for it in range(iters):
                    n += 1
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    m.decode(0, qs[it], n)
                    e1.record()
                    ts.append((e0, e1))
                torch.cuda.synchronize()
                row[kind] = statistics.median(e0.elapsed_time(e1) for e0, e1 in ts[5:])
                # GPU kernel time and kernel count of one call
                from torch.profiler import ProfilerActivity, profile
                from torch.autograd import DeviceType
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(4):
                        n += 1
                        m.decode(0, qs[0], n)
                    torch.cuda.synchronize()
                kern = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
                row[kind + "_gpu"] = sum(e.time_range.elapsed_us() for e in kern) / 4 / 1e3
                row[kind + "_nk"] = len(kern) / 4
                del m
                torch.cuda.empty_cache()
            print(f"  B={B} rho={rho}: v62 {row['v62']:.3f} ms/layer (GPU {row['v62_gpu']:.3f}, {row['v62_nk']:.0f} kernels)"
                  f"  ->  fast {row['fast']:.3f} ms/layer (GPU {row['fast_gpu']:.3f}, {row['fast_nk']:.0f} kernels)"
                  f"   x{row['v62'] / row['fast']:.2f}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        dev = torch.device(sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:1")
        torch.cuda.set_device(dev)
        _record_exp()
        selftest(dev)
        if "--equiv-only" not in sys.argv:
            timing(dev)
    else:
        bl.main()
