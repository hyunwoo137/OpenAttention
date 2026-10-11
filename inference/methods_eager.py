#!/usr/bin/env python3
"""V62: decode-attention latency and end-to-end latency of the final (paper v27) configurations.

One process = one (method, read, batch) on ONE idle GPU.  Llama-3.1-8B-Instruct, bf16, real
RULER-32K qa_1 prompts (batch 1: request #0, 30,012 tokens; batch 8: requests #0-#7 cut to the
shortest), HF eager decode loop.  Every method's decode attention is plugged into the real model
through the attention-function table, so the decode step, the prefill and the end-to-end time are
MEASURED, not composed.  The attention time is taken inside the same model with CUDA events around
every layer's attention call (selection | attention), summed over the 32 layers of a decode step.

The algorithms and hyper-parameters are those of the accuracy runs (run_multibin_hat_bench.py,
paper v27 App. fairsplit / baselines); what changes is only HOW they are executed.  The framework
the accuracy numbers come from simulates sparsity with dense masks (full QK every step), which says
nothing about speed, so each method runs here as a sparse GPU path in PyTorch -- scan, top-k,
gather, softmax over the gathered rows -- and ours additionally with its fused Triton kernel.
Everything a method needs per request (signatures, hash codes, clusters, statistics) is built
inside the prefill of every layer and charged to the prefill; everything it needs per step
(the new key's signature / code / cluster, the scan, the estimator) is charged to the step.

  dense  fused SDPA (default backend dispatch) over the whole cache
  hat    HashAttention, the released Llama-3.1-8B signature weights (3-layer MLP, 32 bits, per query
         head for queries AND keys, as in the hub); sink 0.1% + local 0.1% + top-(rho-0.2%) by
         signature match; gather + softmax.
  vatt   vAttention(HashAttention) as in the paper: sink/local 0.1% + HAT top-(rho/2-0.2%) + base
         sample of rho/2 of the range, std of the sampled weights, (eps,delta)=(0.3,0.3) budget
         (Phi^-1(0.7) std range / (eps den))^2 clamped to [base, range], a fresh sample of that
         size with weight range/budget.  The per-row budget is data dependent, so the sample is
         padded to the batch maximum, which needs one host round trip per layer.
  mpig   MagicPIG as shipped in the hub: key centring, Neyshabur-Srebro MIPS transform, K=9-bit
         SimHash x L tables, a key is sampled if ANY table collides (sink/local 0.1% on top),
         importance weight 1/p with p = 1-(1-(1-theta/pi)^K)^L.  L per read is calibrated on the
         prompt (--mp-L); the candidate list is ragged, so it is padded to the batch maximum
         (one host round trip per layer).
  ours   the deployed pipeline: QGS (r=32 int8 codes, one U per KV head, M = static (RoPE-window
         averaged) + question), QGC (r=16, 4 Lloyd iterations, C clusters per query head, cached
         n_b, sum k, sum k^2, sum v), sink/local 0.01%, top-k from the budget, ErrorGate wl1 on
         the tail statistics (pooled sums minus the selected tokens), hierarchical water-filling
         of E, within-cluster top-q_b by the QGS score, residual statistics (pooled sums minus
         every token read), and the ours.flash kernel (Algorithm 1: P1 selected, P2 expanded,
         P3 per-query-head residual proxies).  (C,E) = (64,216) at 1%, (192,679) at 3%.
  hatfused  (supplementary) HashAttention's selection with its attention on the same fused
         Triton gather kernel (P1 only), to separate the kernel from the method.

    python methods_eager.py --method ours --rho 0.01 --batch 1 --device cuda:1 \
        --out results_v62_latency/ours_r0.01_b1.json
    python methods_eager.py --selftest --device cuda:1
    python methods_eager.py --method mpig --rho 0.01 --mp-calibrate --device cuda:1 --out ...
"""
import argparse
import json
import math
import os
import pickle
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import os as _os
HUB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # repo root
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HUB)
sys.path.insert(0, _HERE)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# HashAttention signature weights (only the hat / hatfused / vatt baselines need them):
# point HAT_ARTIFACTS at a checkout of https://github.com/xAlg-ai/HashAttention-1.0 artifacts/
HAT_WEIGHTS = os.path.join(os.environ.get("HAT_ARTIFACTS", os.path.join(HUB, "hat_artifacts")),
                           "llama3.1-8b-patch.64K.v1.hat_weights.pkl")
STATIC_M = os.environ.get("OA_STATIC_M",
                          f"{HUB}/assets/llama-3.1-8b-instruct.query_metric.pt")
HQ, HKV, D, NL = 32, 8, 128, 32
G = HQ // HKV
SC = D ** -0.5
NEG = float("-inf")
DEPLOYED = {0.01: (64, 216), 0.03: (192, 679)}          # ours (C, E), Table deployedsplit
CTX_LEN = 32768                                          # the accounting context of the accuracy runs
VATT_EPS = VATT_DELTA = 0.3
VATT_PPF = 0.5244005127080407                            # Phi^-1(1 - 0.3)


# ============================================================ ours.flash (per-query-head proxies)
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _ours_flash_kernel(
    Q, K, V, Out, SelIdx, ExpIdx, KBar, VBar, LogN,
    stride_kb, stride_ks, stride_qb, stride_ob, stride_kbar_b, stride_kbar_n, stride_ell_b,
    G, N_SEL, N_EXP, NB, sm_scale,
    BLOCK_N: tl.constexpr, BLOCK_B: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    # triton_expand_decode._decode_expand_kernel with one change: the proxies (kbar, vbar, log n)
    # belong to the QUERY head (pid), because the statistics of the unread members depend on the
    # tokens that head read (Algorithm 1 inputs).  P1/P2 are unchanged.
    pid = tl.program_id(0)
    kvh = pid // G
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + pid * stride_qb + offs_d).to(tl.float32)
    qk_scale = sm_scale * LOG2E
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    k_base = kvh * stride_kb
    for start in range(0, N_SEL, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        valid = offs < N_SEL
        tok = tl.load(SelIdx + pid * N_SEL + offs, mask=valid, other=-1)
        valid = valid & (tok >= 0)
        tok = tl.where(valid, tok, 0)
        k = tl.load(K + k_base + tok[:, None] * stride_ks + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        z = tl.sum(q[None, :] * k, axis=1) * qk_scale
        z = tl.where(valid, z, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(z, axis=0))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(z - m_new)
        v = tl.load(V + k_base + tok[:, None] * stride_ks + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    for start in range(0, N_EXP, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        valid = offs < N_EXP
        tok = tl.load(ExpIdx + pid * N_EXP + offs, mask=valid, other=-1)
        valid = valid & (tok >= 0)
        tok = tl.where(valid, tok, 0)
        k = tl.load(K + k_base + tok[:, None] * stride_ks + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        z = tl.sum(q[None, :] * k, axis=1) * qk_scale
        z = tl.where(valid, z, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(z, axis=0))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(z - m_new)
        v = tl.load(V + k_base + tok[:, None] * stride_ks + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    kb_base = pid * stride_kbar_b
    for start in range(0, NB, BLOCK_B):
        offs = start + tl.arange(0, BLOCK_B)
        valid = offs < NB
        kbar = tl.load(KBar + kb_base + offs[:, None] * stride_kbar_n + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        logn = tl.load(LogN + pid * stride_ell_b + offs, mask=valid, other=-float("inf")).to(tl.float32)
        z = (tl.sum(q[None, :] * kbar, axis=1) * sm_scale + logn) * LOG2E
        z = tl.where(valid, z, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(z, axis=0))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(z - m_new)
        vbar = tl.load(VBar + kb_base + offs[:, None] * stride_kbar_n + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * vbar, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    safe = tl.where(l_i == 0.0, 1.0, l_i)
    tl.store(Out + pid * stride_ob + offs_d, (acc / safe).to(Out.dtype.element_ty))


def ours_flash(q, k, v, sel_idx, exp_idx, kbar, vbar, log_n, nb=None):
    """q:(R,D) k,v:(B*HKV,Lmax,D) sel:(R,Ns) exp:(R,Ne) int32 (-1 = pad); kbar,vbar:(R,C,D); log_n:(R,C)."""
    R = q.shape[0]
    out = torch.empty_like(q)
    NB = kbar.shape[1] if nb is None else nb
    _ours_flash_kernel[(R,)](
        q, k, v, out, sel_idx, exp_idx, kbar, vbar, log_n,
        k.stride(0), k.stride(1), q.stride(0), out.stride(0),
        kbar.stride(0), kbar.stride(1), log_n.stride(0),
        R // k.shape[0], sel_idx.shape[1], exp_idx.shape[1], NB, SC,
        BLOCK_N=64, BLOCK_B=64, HEAD_DIM=D, num_warps=4)
    return out


# ============================================================ run state and the attention hook
class Ctx:
    phase = "prefill"
    brow = 0            # batch row being prefilled
    n = 0               # valid cache length during decode (incl. the token just written)
    timing = False
    profiling = False
    events = []         # per layer call: (e0, e1, e2)
    e1 = None
    K = V = None        # per layer (B, HKV, Lmax, D)
    qlen = []           # question-chunk length per batch row
    method = None


C = Ctx()


def mark():
    if C.timing:
        C.e1 = torch.cuda.Event(enable_timing=True)
        C.e1.record()


def attn_hook(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
    l = module.layer_idx
    T = query.shape[2]
    if C.phase == "prefill":
        b = C.brow
        C.K[l][b:b + 1, :, :T] = key
        C.V[l][b:b + 1, :, :T] = value
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True, scale=SC)
        C.method.build(l, b, query, T)
        return out.transpose(1, 2).contiguous(), None
    n = C.n
    C.K[l][:, :, n - 1:n] = key
    C.V[l][:, :, n - 1:n] = value
    if C.timing:
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        C.e1 = None
    if C.profiling:
        with torch.profiler.record_function("v62_attn"):
            out = C.method.decode(l, query, n)
    else:
        out = C.method.decode(l, query, n)
    if C.timing:
        e2 = torch.cuda.Event(enable_timing=True)
        e2.record()
        C.events.append((e0, C.e1 or e0, e2))
    return out.transpose(1, 2).contiguous(), None


# ============================================================ shared helpers
def row_offsets(B, Lmax, dev):
    """flat-row offset of the KV head each query-head row reads: row r = b*HQ + h -> (r // G) * Lmax."""
    return (torch.arange(B * HQ, device=dev) // G * Lmax).view(-1, 1)


def gather_rows(flat, idx, off):
    """flat:(B*HKV*Lmax, D), idx:(R,m) token positions -> (R,m,D)."""
    R, m = idx.shape
    return flat.index_select(0, (idx + off).reshape(-1)).view(R, m, D)


def sink_local(n, ns, nl, R, dev):
    return torch.cat([torch.arange(ns, device=dev), torch.arange(n - nl, n, device=dev)]).view(1, -1).expand(R, -1)


class Method:
    name = "base"

    def __init__(self, a, B, Lmax, dev):
        self.a, self.B, self.Lmax, self.dev = a, B, Lmax, dev
        self.R = B * HQ
        self.off = row_offsets(B, Lmax, dev)
        self.read_sum = torch.zeros((), device=dev)      # realised read (fraction), summed over calls
        self.read_cnt = 0

    def flat(self, l):
        return C.K[l].view(-1, D), C.V[l].view(-1, D)

    def build(self, l, b, q, T):
        pass

    def log_read(self, frac):
        if not C.timing:
            self.read_sum += frac
            self.read_cnt += 1

    def realised_read(self):
        return float(self.read_sum) / max(self.read_cnt, 1)


class Dense(Method):
    name = "dense"

    def decode(self, l, q, n):
        mark()
        return F.scaled_dot_product_attention(q, C.K[l][:, :, :n], C.V[l][:, :, :n], enable_gqa=True, scale=SC)


# ============================================================ HashAttention
class HAT(Method):
    name = "hat"

    def __init__(self, a, B, Lmax, dev, heavy_frac=None):
        super().__init__(a, B, Lmax, dev)
        w = pickle.load(open(HAT_WEIGHTS, "rb"))
        self.qW = [[m.to(dev) for m in w[l]["query_matrix"]] for l in range(NL)]
        self.qB = [[m.to(dev) for m in w[l]["query_bias"]] for l in range(NL)]
        self.kW = [[m.to(dev) for m in w[l]["key_matrix"]] for l in range(NL)]
        self.kB = [[m.to(dev) for m in w[l]["key_bias"]] for l in range(NL)]
        self.sig = [torch.empty(B, HQ, Lmax, 32, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
        self.heavy_frac = a.rho - 0.002 if heavy_frac is None else heavy_frac

    @staticmethod
    def mlp(x, Ws, bs):
        for i, (W, bb) in enumerate(zip(Ws, bs)):
            x = torch.einsum("bhsd,hde->bhse", x, W) + bb[None, :, None, :]
            if i < len(Ws) - 1:
                x = F.silu(x)
        return torch.sign(x)

    def build(self, l, b, q, T):
        k = C.K[l][b:b + 1, :, :T].repeat_interleave(G, 1)                  # (1,HQ,T,D): key MLP is per query head
        for t0 in range(0, T, 8192):
            t1 = min(t0 + 8192, T)
            self.sig[l][b:b + 1, :, t0:t1] = self.mlp(k[:, :, t0:t1], self.kW[l], self.kB[l])

    def select(self, l, q, n, heavy_frac):
        kn = C.K[l][:, :, n - 1:n].repeat_interleave(G, 1)
        self.sig[l][:, :, n - 1:n] = self.mlp(kn, self.kW[l], self.kB[l])  # signature of the new key
        qs = self.mlp(q, self.qW[l], self.qB[l])                             # (B,HQ,1,32)
        s = torch.matmul(qs, self.sig[l][:, :, :n].transpose(-1, -2)).view(self.R, n)
        ns = nl = int(0.001 * n)
        h = int(heavy_frac * n)
        s[:, :ns] = NEG
        s[:, n - nl:] = NEG
        top = s.topk(h, dim=-1).indices
        return torch.cat([sink_local(n, ns, nl, self.R, self.dev), top], 1), ns + nl + h

    def decode(self, l, q, n):
        idx, m = self.select(l, q, n, self.heavy_frac)
        self.log_read(m / n)
        mark()
        kf, vf = self.flat(l)
        ks, vs = gather_rows(kf, idx, self.off), gather_rows(vf, idx, self.off)
        q2 = q.reshape(self.R, 1, D)
        z = torch.bmm(q2, ks.transpose(1, 2)).float() * SC
        p = torch.softmax(z, -1).to(vs.dtype)
        return torch.bmm(p, vs).view(self.B, HQ, 1, D)


class HATFused(HAT):
    name = "hatfused"

    def __init__(self, a, B, Lmax, dev):
        super().__init__(a, B, Lmax, dev)
        self.dummy_kbar = torch.zeros(self.R, 1, D, device=dev, dtype=torch.bfloat16)
        self.dummy_ell = torch.full((self.R, 1), NEG, device=dev)
        self.no_exp = torch.full((self.R, 1), -1, device=dev, dtype=torch.int32)

    def decode(self, l, q, n):
        idx, m = self.select(l, q, n, self.heavy_frac)
        self.log_read(m / n)
        idx = idx.int()
        mark()
        out = ours_flash(q.reshape(self.R, D), C.K[l].view(-1, self.Lmax, D), C.V[l].view(-1, self.Lmax, D),
                         idx, self.no_exp, self.dummy_kbar, self.dummy_kbar, self.dummy_ell, nb=0)
        return out.view(self.B, HQ, 1, D)


# ============================================================ vAttention (HashAttention top-k)
class VAtt(HAT):
    name = "vatt"

    def __init__(self, a, B, Lmax, dev):
        super().__init__(a, B, Lmax, dev, heavy_frac=round(a.rho - 0.002 - a.rho * 0.5, 6))
        self.base_frac = round(a.rho * 0.5, 6)

    def decode(self, l, q, n):
        idx, m1 = self.select(l, q, n, self.heavy_frac)
        R = self.R
        ns = nl = int(0.001 * n)
        start, end = ns, n - nl
        rng = end - start
        nbase = max(2, int(self.base_frac * rng))
        kf, vf = self.flat(l)
        q2 = q.reshape(R, 1, D)
        kt, vt = gather_rows(kf, idx, self.off), gather_rows(vf, idx, self.off)
        zt = torch.bmm(q2, kt.transpose(1, 2)).float() * SC                  # (R,1,m1)
        mx = zt.amax(-1, keepdim=True)
        static_den = torch.exp(zt - mx).sum(-1)                              # (R,1)
        bidx = torch.randint(start, end, (R, nbase), device=self.dev)        # base sample: std estimate
        kb = gather_rows(kf, bidx, self.off)
        eb = torch.exp(torch.bmm(q2, kb.transpose(1, 2)).float() * SC - mx)  # (R,1,nbase)
        std = eb.std(-1).clamp(min=1e-8)
        est = static_den + eb.sum(-1) * (rng / nbase)
        budget = ((VATT_PPF * std * rng) / (VATT_EPS * est).clamp(min=1e-8)) ** 2
        budget = budget.clamp(min=nbase, max=rng).floor()                    # (R,1)
        bmax = int(budget.max())                                             # data dependent: host round trip
        sidx = torch.randint(start, end, (R, bmax), device=self.dev)
        valid = torch.arange(bmax, device=self.dev).view(1, -1) < budget
        self.log_read((m1 + budget.mean()) / n)
        mark()
        ks, vs = gather_rows(kf, sidx, self.off), gather_rows(vf, sidx, self.off)
        zs = torch.bmm(q2, ks.transpose(1, 2)).float() * SC + torch.log(rng / budget).unsqueeze(-1)
        zs = zs.masked_fill(~valid.unsqueeze(1), NEG)
        p = torch.softmax(torch.cat([zt, zs], -1), -1).to(vt.dtype)
        return torch.bmm(p, torch.cat([vt, vs], 1)).view(self.B, HQ, 1, D)


# ============================================================ MagicPIG (hub masker, sparse execution)
class MPig(Method):
    name = "mpig"

    def __init__(self, a, B, Lmax, dev):
        super().__init__(a, B, Lmax, dev)
        self.Kb, self.L = a.mp_K, a.mp_L
        self.W = []
        for l in range(NL):
            g = torch.Generator(device=dev)
            g.manual_seed(42 + l)
            self.W.append(torch.randn(D + 1, self.Kb * self.L, device=dev, generator=g).to(torch.bfloat16))
        self.codes = [torch.empty(B, HKV, Lmax, self.L, device=dev, dtype=torch.int16) for _ in range(NL)]
        self.mu = [torch.zeros(B, HKV, 1, D, device=dev) for _ in range(NL)]
        self.Mx = [torch.ones(B, 1, 1, 1, device=dev) for _ in range(NL)]
        self.pw = (2 ** torch.arange(self.Kb, device=dev, dtype=torch.int32))
        self.calib = None                                      # (L,) candidate-fraction sums (--mp-calibrate)

    def hash(self, x, W):
        bits = (torch.matmul(x, W) > 0).view(*x.shape[:-1], self.L, self.Kb)
        return (bits.to(torch.int32) * self.pw).sum(-1).to(torch.int16)

    def build(self, l, b, q, T):
        k = C.K[l][b:b + 1, :, :T].float()
        mu = k.mean(2, keepdim=True)
        kc = k - mu
        nrm = kc.norm(dim=-1, keepdim=True)
        M = nrm.amax()
        self.mu[l][b:b + 1] = mu
        self.Mx[l][b:b + 1] = M
        for t0 in range(0, T, 8192):
            t1 = min(t0 + 8192, T)
            kc_, nr_ = kc[:, :, t0:t1], nrm[:, :, t0:t1]
            kt = torch.cat([kc_ / M, (M * M - nr_ * nr_).clamp(min=0).sqrt() / M], -1).to(torch.bfloat16)
            self.codes[l][b:b + 1, :, t0:t1] = self.hash(kt, self.W[l])

    def decode(self, l, q, n):
        R, B = self.R, self.B
        mu = self.mu[l]
        kn = C.K[l][:, :, n - 1:n].float() - mu                              # new key: centre, MIPS, hash
        nr = kn.norm(dim=-1, keepdim=True)
        M = torch.maximum(self.Mx[l], nr.amax(dim=(1, 2, 3), keepdim=True))
        self.Mx[l] = M
        kt = torch.cat([kn / M, (M * M - nr * nr).clamp(min=0).sqrt() / M], -1).to(torch.bfloat16)
        self.codes[l][:, :, n - 1:n] = self.hash(kt, self.W[l])
        qc = q.float() - mu.repeat_interleave(G, 1)                          # (B,HQ,1,D)
        qn = qc.norm(dim=-1, keepdim=True)
        qt = torch.cat([qc / qn, torch.zeros_like(qn)], -1).to(torch.bfloat16)
        qcode = self.hash(qt, self.W[l]).view(B, HKV, G, 1, self.L)
        eq = self.codes[l][:, :, :n].unsqueeze(2) == qcode                    # (B,HKV,G,n,L)
        if self.calib is not None:
            ns_ = int(0.001 * n)
            hit = eq.view(R, n, self.L)[:, ns_:n - ns_].to(torch.uint8).cummax(-1).values.float().mean(1)
            self.calib += hit.mean(0)                                          # any over the first L' tables
        match = eq.any(-1).view(R, n)
        ns = nl = int(0.001 * n)
        match[:, :ns] = False
        match[:, n - nl:] = False
        cnt = match.sum(-1, keepdim=True)
        cmax = max(1, int(cnt.max()))                                        # ragged: host round trip
        cand = match.to(torch.float16).topk(cmax, dim=-1).indices
        valid = torch.arange(cmax, device=self.dev).view(1, -1) < cnt
        self.log_read((ns + nl + cnt.float().mean()) / n)
        mark()
        kf, vf = self.flat(l)
        idx = torch.cat([sink_local(n, ns, nl, R, self.dev), cand], 1)
        ks, vs = gather_rows(kf, idx, self.off), gather_rows(vf, idx, self.off)
        q2 = q.reshape(R, 1, D)
        z = torch.bmm(q2, ks.transpose(1, 2)).float() * SC                   # (R,1,m)
        kcand = ks[:, ns + nl:].float() - mu.repeat_interleave(G, 1).view(R, 1, D)
        Mr = M.expand(B, HQ, 1, 1).reshape(R, 1, 1)
        cos = torch.bmm(qc.view(R, 1, D), kcand.transpose(1, 2)) / (qn.view(R, 1, 1) * Mr)
        p1 = (1 - torch.acos(cos.clamp(-1, 1)) / math.pi) ** self.Kb
        prob = 1 - (1 - p1) ** self.L
        bias = (-torch.log(prob.clamp(min=1e-30))).masked_fill(~valid.unsqueeze(1), NEG)
        z = torch.cat([z[:, :, :ns + nl], z[:, :, ns + nl:] + bias], -1)
        p = torch.softmax(z, -1).to(vs.dtype)
        return torch.bmm(p, vs).view(B, HQ, 1, D)


# ============================================================ ours (deployed pipeline + ours.flash)
class Ours(Method):
    name = "ours"

    def __init__(self, a, B, Lmax, dev):
        super().__init__(a, B, Lmax, dev)
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.static_top_k import StaticTopKMasker
        self.STK = StaticTopKMasker
        if not os.path.exists(STATIC_M):                 # restore from the packed copy on first use
            sys.path.insert(0, os.path.join(HUB, "assets"))
            from restore_query_metric import unpack
            packed = os.path.join(HUB, "assets", "llama-3.1-8b-instruct.query_metric.triu.pt")
            torch.save(unpack(torch.load(packed, map_location="cpu")), STATIC_M)
        st = torch.load(STATIC_M, map_location="cpu")
        self.Mpre = st["M"].float().to(dev)                                  # (L,HQ,D,D) pre-RoPE
        self._inv_freq = st["inv_freq"].float().to(dev)
        self._rope_scaling = float(st.get("attention_scaling", 1.0))
        self.C, self.E = DEPLOYED[round(a.rho, 4)]
        self.rq, self.rc, self.iters, self.win, self.P = 32, 16, 4, 32, 64
        Cn = self.C
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=dev, dtype=dt)   # noqa: E731
        self.codes = [z(B, HKV, Lmax, self.rq, dt=torch.bfloat16) for _ in range(NL)]   # int8 values held in bf16
        self.U = [z(B, HKV, D, self.rq) for _ in range(NL)]
        self.scale = [z(B, HKV, 1, self.rq) for _ in range(NL)]
        self.W = [z(B, HQ, D, self.rc, dt=torch.bfloat16) for _ in range(NL)]
        self.colmean = [z(B, HQ, 1, self.rc) for _ in range(NL)]
        self.cent = [z(B, HQ, Cn, self.rc) for _ in range(NL)]
        self.asg = [torch.zeros(B, HQ, Lmax, device=dev, dtype=torch.int32) for _ in range(NL)]
        self.nb = [z(B, HQ, Cn) for _ in range(NL)]
        self.Sk = [z(B, HQ, Cn, D) for _ in range(NL)]
        self.Sk2 = [z(B, HQ, Cn, D) for _ in range(NL)]
        self.Sv = [z(B, HQ, Cn, D) for _ in range(NL)]
        self.ar = torch.arange(Lmax, device=dev)

    # ---------------------------------------------------------------- per request (prefill)
    def _nt(self, x):
        return x / torch.diagonal(x, dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12)[..., None, None]

    def build(self, l, b, q, T):
        dev, Cn = self.dev, self.C
        Qn = C.qlen[b]
        k = C.K[l][b, :, :T]                                                 # (HKV,T,D)
        v = C.V[l][b, :, :T]
        # metric: static M rotated to the decode position (window 32, closed form) + question rows
        Mst = self.STK._rot_avg(self, self.Mpre[l], T, self.win)             # (HQ,D,D)
        P = min(self.P, max(Qn - 2, 1))
        pi = torch.linspace(0, max(Qn - 3, 0), P, device=dev).long().unique()
        qsel = q[0, :, T - Qn:, :].index_select(1, pi).float()               # (HQ,P,D)
        cos, sin = self.STK._rope_cs(self._inv_freq, self._rope_scaling, (Qn - pi).to(dev))
        qrot = self.STK._rope_apply(qsel, cos, sin)
        Mq = torch.matmul(qrot.transpose(-1, -2), qrot) / qrot.shape[1]
        M = 0.5 * self._nt(Mst) + 0.5 * self._nt(Mq)                          # (HQ,D,D)
        # QGS: one U per KV head, r=32, int8 codes with a per-coordinate scale
        _, Ukv = torch.linalg.eigh(M.view(HKV, G, D, D).mean(1))
        U = Ukv[..., -self.rq:].contiguous()                                 # (HKV,D,r)
        zc = torch.matmul(k.float(), U)                                      # (HKV,T,r)
        scale = (zc.abs().amax(1, keepdim=True) / 127.0).clamp(min=1e-12)
        self.codes[l][b, :, :T] = (zc / scale).round().clamp(-127, 127).to(torch.bfloat16)
        self.U[l][b], self.scale[l][b] = U, scale
        # QGC: per query head, phi = Lambda^1/2 U^T k (r=16), 4 Lloyd iterations in bf16
        ev, Uq = torch.linalg.eigh(M)
        W = (Uq[..., -self.rc:] * ev[..., -self.rc:].clamp(min=0).sqrt().unsqueeze(-2)).to(torch.bfloat16)   # (HQ,D,16)
        krep = k.repeat_interleave(G, 0)                                     # (HQ,T,D) bf16
        zP = torch.matmul(krep, W).float()                                   # (HQ,T,16)
        colmean = zP.mean(1, keepdim=True)
        zP = zP - colmean
        g = torch.Generator(device="cpu").manual_seed(31 * (l + 1))
        init = torch.randperm(T, generator=g)[:Cn].to(dev)
        cent = zP.index_select(1, init).clone()
        zb = zP.to(torch.bfloat16)
        asg = torch.empty(HQ, T, device=dev, dtype=torch.long)
        for _ in range(self.iters):
            half = (0.5 * (cent * cent).sum(-1)).to(torch.bfloat16).unsqueeze(1)
            cb = cent.to(torch.bfloat16).transpose(1, 2)
            for h0 in range(0, HQ, 8):
                asg[h0:h0 + 8] = (torch.matmul(zb[h0:h0 + 8], cb[h0:h0 + 8]) - half[h0:h0 + 8]).argmax(-1)
            acc = torch.zeros(HQ, Cn, self.rc, device=dev).scatter_add_(1, asg.unsqueeze(-1).expand(-1, -1, self.rc), zP)
            cnt = torch.zeros(HQ, Cn, device=dev).scatter_add_(1, asg, torch.ones_like(zP[..., 0]))
            cent = torch.where(cnt.unsqueeze(-1) > 0, acc / cnt.clamp(min=1).unsqueeze(-1), zP.index_select(1, init))
        self.asg[l][b, :, :T] = asg.to(torch.int32)
        self.W[l][b], self.colmean[l][b], self.cent[l][b] = W, colmean, cent
        # cached cluster statistics: n_b, sum k, sum k^2 (diag variance), sum v
        idx = asg.unsqueeze(-1).expand(-1, -1, D)
        kf = krep.float()
        vf = v.repeat_interleave(G, 0).float()
        self.nb[l][b] = torch.zeros(HQ, Cn, device=dev).scatter_add_(1, asg, torch.ones_like(kf[..., 0]))
        self.Sk[l][b] = torch.zeros(HQ, Cn, D, device=dev).scatter_add_(1, idx, kf)
        self.Sk2[l][b] = torch.zeros(HQ, Cn, D, device=dev).scatter_add_(1, idx, kf * kf)
        self.Sv[l][b] = torch.zeros(HQ, Cn, D, device=dev).scatter_add_(1, idx, vf)

    # ---------------------------------------------------------------- per step
    @staticmethod
    def hier_quota(score, cap, live, E):
        """Water-filling q_b = min(n_b, lam w_b), w_b = exp(score_b - max), sum q_b = E exactly
        (leftover whole tokens one each to the highest-scoring clusters with room)."""
        s = score.masked_fill(~live, NEG)
        w = torch.exp(s - s.amax(-1, keepdim=True)).masked_fill(~live, 0.0)
        cap = cap.masked_fill(~live, 0.0)
        tot = cap.sum(-1, keepdim=True)
        ratio = torch.where(w > 0, cap / w.clamp(min=1e-30), torch.full_like(w, float("inf")))
        o = ratio.argsort(-1)
        r_s, c_s, w_s = ratio.gather(-1, o), cap.gather(-1, o), w.gather(-1, o)
        c_ex = c_s.cumsum(-1) - c_s
        w_suf = w_s.flip(-1).cumsum(-1).flip(-1)
        f_at = torch.where(w_suf > 0, c_ex + r_s * w_suf, torch.full_like(w_suf, float("inf")))
        j = (f_at >= E).float().argmax(-1, keepdim=True)
        lam = (E - c_ex.gather(-1, j)) / w_suf.gather(-1, j).clamp(min=1e-30)
        qb = torch.minimum(cap, (lam * w).floor())
        qb = torch.where(tot <= E, cap, qb)
        rem = (tot.clamp(max=E) - qb.sum(-1, keepdim=True)).clamp(min=0)
        order = s.argsort(-1, descending=True)
        room = (qb < cap).gather(-1, order)
        rk = room.long().cumsum(-1) - 1
        add = torch.zeros_like(room).scatter(-1, order, room & (rk < rem.long()))
        return qb + add.float()

    def decode(self, l, q, n):
        B, R, Cn, E, dev = self.B, self.R, self.C, self.E, self.dev
        kf, vf = self.flat(l)
        # 1. the new token joins the index: QGS code, nearest QGC centroid, pooled sums
        kn = C.K[l][:, :, n - 1:n]                                           # (B,HKV,1,D)
        vn = C.V[l][:, :, n - 1:n]
        zc = torch.matmul(kn.float(), self.U[l])
        self.codes[l][:, :, n - 1:n] = (zc / self.scale[l]).round().clamp(-127, 127).to(torch.bfloat16)
        knr = kn.repeat_interleave(G, 1)                                     # (B,HQ,1,D)
        zp = torch.matmul(knr, self.W[l]).float() - self.colmean[l]          # (B,HQ,1,16)
        cent = self.cent[l]
        cid = (torch.matmul(zp, cent.transpose(-1, -2)) - 0.5 * (cent * cent).sum(-1).unsqueeze(-2)).argmax(-1)   # (B,HQ,1)
        self.asg[l][:, :, n - 1:n] = cid.to(torch.int32)
        kd = knr.float()
        ci = cid.unsqueeze(-1).expand(-1, -1, -1, D)
        self.nb[l].scatter_add_(2, cid, torch.ones_like(cid, dtype=torch.float32))
        self.Sk[l].scatter_add_(2, ci, kd)
        self.Sk2[l].scatter_add_(2, ci, kd * kd)
        self.Sv[l].scatter_add_(2, ci, vn.repeat_interleave(G, 1).float())
        # 2. QGS scan and top-k (sink/local at their 0.01% floor)
        zq = (torch.matmul(q.view(B, HKV, G, D).float(), self.U[l]) * self.scale[l]).to(torch.bfloat16)
        s = torch.matmul(zq, self.codes[l][:, :, :n].transpose(-1, -2)).view(R, n).float()
        ns = nl = max(1, int(0.0001 * n))
        heavy = self.a.rho - 0.0002 - E / CTX_LEN - 1.5 * Cn / CTX_LEN
        k_top = max(1, int(heavy * n))
        s[:, :ns] = NEG
        s[:, n - nl:] = NEG
        top = s.topk(k_top, dim=-1).indices
        sel = torch.cat([sink_local(n, ns, nl, R, dev), top], 1)             # (R,Ns)
        # 3. ErrorGate (wl1) on the tail statistics: pooled sums minus the selected tokens
        asg = self.asg[l].view(R, self.Lmax)
        cid_s = asg.gather(1, sel).long()
        k_s = gather_rows(kf, sel, self.off).float()                         # (R,Ns,D)
        ci_s = cid_s.unsqueeze(-1).expand(-1, -1, D)
        Sk_t = self.Sk[l].view(R, Cn, D).scatter_add(1, ci_s, -k_s)
        Sk2_t = self.Sk2[l].view(R, Cn, D).scatter_add(1, ci_s, -k_s * k_s)
        n_t = self.nb[l].view(R, Cn).scatter_add(1, cid_s, -torch.ones_like(k_s[..., 0]))
        inv = 1.0 / n_t.clamp(min=1).unsqueeze(-1)
        kbar = Sk_t * inv
        var = (Sk2_t * inv - kbar * kbar).clamp(min=0)
        qf = q.reshape(R, 1, D).float()
        zbar = SC * torch.bmm(qf, kbar.transpose(1, 2)).squeeze(1)           # (R,C)
        s2 = (SC * SC) * torch.bmm(qf * qf, var.transpose(1, 2)).squeeze(1)
        score = zbar + torch.log(n_t.clamp(min=1)) + 0.5 * s2 + torch.log(
            torch.erf(s2.sqrt() * 0.7071067811865476).clamp(min=1e-30))
        quota = self.hier_quota(score, n_t, n_t > 1, float(E))              # (R,C), sums to E
        # 4. within-cluster top-q_b by the QGS score: one segmented sort on (cluster, -score)
        s = s.scatter(1, sel, NEG)                                           # selected tokens go last in their cluster
        bits = (-s).view(torch.int32)
        mono = bits ^ ((bits >> 31) & 0x7FFFFFFF)
        key = (asg[:, :n] << 23) | ((mono >> 9) + (1 << 22))
        perm = key.argsort(-1)
        cs = (key.gather(1, perm) >> 23).long()
        start = (self.nb[l].view(R, Cn).cumsum(-1) - self.nb[l].view(R, Cn)).long()
        take = (self.ar[:n].view(1, -1) - start.gather(1, cs)) < quota.long().gather(1, cs)
        pos = torch.where(take, (take.long().cumsum(-1) - 1).clamp(max=E), torch.full_like(perm, E))
        exp_idx = torch.full((R, E + 1), -1, device=dev, dtype=torch.long).scatter_(1, pos, perm)[:, :E]
        # 5. residual statistics of the unread members (pooled sums minus every token read)
        ex = exp_idx.clamp(min=0)
        cid_e = asg.gather(1, ex).long()
        k_e = gather_rows(kf, ex, self.off).float()
        v_r = gather_rows(vf, torch.cat([sel, ex], 1), self.off).float()
        Sk_r = Sk_t.scatter_add(1, cid_e.unsqueeze(-1).expand(-1, -1, D), -k_e)
        Sv_r = self.Sv[l].view(R, Cn, D).scatter_add(
            1, torch.cat([cid_s, cid_e], 1).unsqueeze(-1).expand(-1, -1, D), -v_r)
        n_r = n_t - quota
        inv_r = 1.0 / n_r.clamp(min=1).unsqueeze(-1)
        kbar_r = (Sk_r * inv_r).to(torch.bfloat16)
        vbar_r = (Sv_r * inv_r).to(torch.bfloat16)
        logn = torch.where(n_r > 0.5, torch.log(n_r.clamp(min=1)), torch.full_like(n_r, NEG))
        self.log_read((2 * ns + k_top + E + 1.5 * Cn) / n)
        return self.attend(l, q, sel, exp_idx, kbar_r, vbar_r, logn)

    def attend(self, l, q, sel, exp_idx, kbar_r, vbar_r, logn):
        # 6. ours.flash: P1 selected, P2 expanded, P3 residual proxies in one online-softmax pass
        sel_i, exp_i = sel.int(), exp_idx.int()
        mark()
        out = ours_flash(q.reshape(self.R, D), C.K[l].view(-1, self.Lmax, D), C.V[l].view(-1, self.Lmax, D),
                         sel_i, exp_i, kbar_r, vbar_r, logn)
        return out.view(self.B, HQ, 1, D)


class OursUnfused(Ours):
    """Same selection, statistics and estimator as `ours`; the attention stage is the unfused PyTorch
    path (the paper's "ours, unfused gather" row): gather the selected + expanded rows, one softmax over
    their logits and the C proxies (q.kbar' + log n'), then the two value products."""
    name = "oursunfused"

    def attend(self, l, q, sel, exp_idx, kbar_r, vbar_r, logn):
        mark()
        R = self.R
        kf, vf = self.flat(l)
        idx = torch.cat([sel, exp_idx], 1)
        ok = idx >= 0
        idx = idx.clamp(min=0)
        ks, vs = gather_rows(kf, idx, self.off), gather_rows(vf, idx, self.off)
        q2 = q.reshape(R, 1, D)
        z = (torch.bmm(q2, ks.transpose(1, 2)).float() * SC).masked_fill(~ok.unsqueeze(1), NEG)
        zb = torch.bmm(q2, kbar_r.transpose(1, 2)).float() * SC + logn.unsqueeze(1)
        p = torch.softmax(torch.cat([z, zb], -1), -1).to(vs.dtype)
        m = idx.shape[1]
        out = torch.bmm(p[:, :, :m], vs) + torch.bmm(p[:, :, m:], vbar_r)
        return out.view(self.B, HQ, 1, D)


METHODS = {"dense": Dense, "hat": HAT, "hatfused": HATFused, "vatt": VAtt, "mpig": MPig, "ours": Ours,
           "oursunfused": OursUnfused}


# ============================================================ self test of the kernel variant
def selftest(dev):
    torch.manual_seed(0)
    B, Lmax, Cn, Ns, Ne = 2, 4096, 64, 14, 216
    R = B * HQ
    k = torch.randn(B * HKV, Lmax, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B * HKV, Lmax, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(R, D, device=dev, dtype=torch.bfloat16)
    perm = torch.stack([torch.randperm(Lmax, device=dev) for _ in range(R)])
    sel, ex = perm[:, :Ns], perm[:, Ns:Ns + Ne].clone()
    ex[:, -7:] = -1
    kb = torch.randn(R, Cn, D, device=dev, dtype=torch.bfloat16)
    vb = torch.randn(R, Cn, D, device=dev, dtype=torch.bfloat16)
    ln = torch.rand(R, Cn, device=dev) * 5
    ln[:, :5] = NEG
    out = ours_flash(q, k, v, sel.int(), ex.int(), kb, vb, ln).float()
    off = (torch.arange(R, device=dev) // G * Lmax).view(-1, 1)
    kf, vf = k.view(-1, D).float(), v.view(-1, D).float()
    idx = torch.cat([sel, ex], 1)
    ok = idx >= 0
    ks = kf[(idx.clamp(min=0) + off)]
    vs = vf[(idx.clamp(min=0) + off)]
    z = (ks @ q.float().unsqueeze(-1)).squeeze(-1) * SC
    z = z.masked_fill(~ok, NEG)
    zb = (kb.float() @ q.float().unsqueeze(-1)).squeeze(-1) * SC + ln
    zz = torch.cat([z, zb], 1)
    p = torch.softmax(zz, -1)
    ref = (p[:, :idx.shape[1], None] * vs).sum(1) + (p[:, idx.shape[1]:, None] * vb.float()).sum(1)
    rel = ((out - ref).norm(dim=-1) / ref.norm(dim=-1))
    print(f"ours_flash vs fp32 reference: median rel err {rel.median():.2e}, max {rel.max():.2e}")
    # water-filling: exact sum and cap
    sc_ = torch.randn(R, Cn, device=dev) * 3
    cap = torch.randint(1, 900, (R, Cn), device=dev).float()
    qb = Ours.hier_quota(sc_, cap, cap > 1, 216.0)
    print(f"hier quota: sum==E {bool((qb.sum(-1) == 216).all())}, <=cap {bool((qb <= cap).all())}, "
          f"max quota {int(qb.max())}")
    assert rel.max() < 1e-2


# ============================================================ driver
def gpu_state():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception as e:                                                    # noqa: BLE001
        return str(e)


def load_prompts(tok, B):
    import ruler_data as RD
    rows = RD.load_rows(["qa_1"], max_requests=B)
    pr = [RD.build_prompt_ids(tok, r) for r in rows[:B]]
    L0 = min(len(c) + len(qq) for c, qq in pr)
    ids, qlen = [], []
    for c, qq in pr:
        ids.append(c[:L0 - len(qq)] + qq)
        qlen.append(len(qq))
    return torch.tensor(ids), qlen, [r["answer"] for r in rows[:B]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=list(METHODS), default="dense")
    p.add_argument("--rho", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--gen", type=int, default=128, help="decode steps timed per request")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--attn-steps", type=int, default=32)
    p.add_argument("--gpu-time-steps", type=int, default=8,
                   help="decode steps profiled for GPU kernel time (0 = skip)")
    p.add_argument("--mp-K", type=int, default=9)
    p.add_argument("--mp-L", type=int, default=25)
    p.add_argument("--mp-calibrate", action="store_true")
    p.add_argument("--read-only", action="store_true", help="stop after the E2E reps (realised read)")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", default="")
    a = p.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    if a.selftest:
        selftest(dev)
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    state_before = gpu_state()
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    ids, qlen, refs = load_prompts(tok, a.batch)
    B, L0 = ids.shape
    ids = ids.to(dev)
    steps_max = max(a.gen, a.attn_steps, 24)
    Lmax = L0 + steps_max + 8
    C.K = [torch.zeros(B, HKV, Lmax, D, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
    C.V = [torch.zeros(B, HKV, Lmax, D, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
    C.qlen = qlen
    if a.mp_calibrate:
        a.mp_L = 64
    C.method = METHODS[a.method](a, B, Lmax, dev)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = attn_hook
    pos0 = torch.arange(L0, device=dev).view(1, -1)

    @torch.inference_mode()
    def prefill():
        C.phase = "prefill"
        nx = []
        for b in range(B):
            C.brow = b
            out = model(input_ids=ids[b:b + 1], position_ids=pos0, use_cache=False, logits_to_keep=1)
            nx.append(out.logits[:, -1].argmax(-1))
        return torch.stack(nx, 0)                                             # (B,1)

    @torch.inference_mode()
    def decode(nxt, steps, keep=False):
        C.phase = "decode"
        toks = [nxt] if keep else None
        for s in range(steps):
            C.n = L0 + s + 1
            out = model(input_ids=nxt, position_ids=torch.full((B, 1), L0 + s, device=dev), use_cache=False)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            if keep:
                toks.append(nxt)
        return toks

    def sync():
        torch.cuda.synchronize(dev)

    res = {"method": a.method, "rho": a.rho, "batch": B, "prompt_tokens": L0, "gen": a.gen,
           "gpu": torch.cuda.get_device_name(dev), "device": str(dev), "torch": torch.__version__,
           "gpu_state_before": state_before}
    if a.method == "mpig":
        res["mp"] = {"K": a.mp_K, "L": a.mp_L}
    if a.method == "ours":
        res["ours"] = {"C": C.method.C, "E": C.method.E}

    # warm-up + sanity answer (greedy, 24 tokens, row 0)
    toks = decode(prefill(), 24, keep=True)
    sync()
    gen0 = torch.cat(toks, 1)[0].tolist()
    res["answer_row0"] = tok.decode(gen0, skip_special_tokens=True)
    res["reference_row0"] = refs[0]
    print(f"[{a.method} rho={a.rho} B={B} L0={L0}] answer: {res['answer_row0']!r}  ref {refs[0]}", flush=True)

    if a.mp_calibrate:
        # candidate fraction (sink/local excluded) of the first L' of 64 tables, every layer and
        # head, over the same --gen decode steps the timed runs use
        C.method.calib = torch.zeros(64, device=dev)
        decode(prefill(), a.gen)
        sync()
        frac = (C.method.calib / (a.gen * NL)).tolist()
        res["calib_candidate_frac_by_L"] = frac
        for Lc in (4, 6, 8, 9, 10, 12, 16, 20, 25, 30, 40, 64):
            print(f"  L={Lc:3d}  candidates {100 * frac[Lc - 1]:6.3f}%  read (+0.2% sink/local) {100 * frac[Lc - 1] + 0.2:6.3f}%")
        if a.out:
            json.dump(res, open(a.out, "w"), indent=2)
        return

    # end-to-end: prefill (+ per-request build) + gen decode steps, no per-step host sync
    C.method.read_sum.zero_()
    C.method.read_cnt = 0
    e2e = []
    for r in range(a.reps):
        sync()
        t0 = time.perf_counter()
        nxt = prefill()
        sync()
        t1 = time.perf_counter()
        decode(nxt, a.gen)
        sync()
        t2 = time.perf_counter()
        e2e.append({"prefill_ms": (t1 - t0) * 1e3, "decode_ms": (t2 - t1) * 1e3, "e2e_ms": (t2 - t0) * 1e3,
                    "tpot_ms": (t2 - t1) * 1e3 / a.gen})
        print(f"  rep{r}: prefill {e2e[-1]['prefill_ms']:8.1f} ms  decode {e2e[-1]['decode_ms']:8.1f} ms "
              f"({e2e[-1]['tpot_ms']:.2f} ms/step)  E2E {e2e[-1]['e2e_ms']:8.1f} ms", flush=True)
    res["e2e_reps"] = e2e
    for k in ("prefill_ms", "decode_ms", "e2e_ms", "tpot_ms"):
        res[k] = statistics.median(x[k] for x in e2e)
    res["realised_read"] = C.method.realised_read()
    if a.read_only:
        print(f"[{a.method} rho={a.rho} B={B} L={a.mp_L}] realised read {100 * res['realised_read']:.3f}%", flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=2)
        return

    # decode attention time inside the model: CUDA events around every layer's attention call
    nxt = prefill()
    sync()
    C.timing, C.events = True, []
    decode(nxt, a.attn_steps)
    sync()
    C.timing = False
    per_step = []
    for s in range(a.attn_steps):
        evs = C.events[s * NL:(s + 1) * NL]
        tot = sum(e0.elapsed_time(e2) for e0, _, e2 in evs)
        sel = sum(e0.elapsed_time(e1) for e0, e1, _ in evs)
        per_step.append((tot, sel, tot - sel))
    per_step = per_step[2:]
    res["attn_step_ms"] = statistics.median(x[0] for x in per_step)
    res["attn_select_step_ms"] = statistics.median(x[1] for x in per_step)
    res["attn_compute_step_ms"] = statistics.median(x[2] for x in per_step)
    res["attn_layer_ms"] = res["attn_step_ms"] / NL
    if a.gpu_time_steps:
        # GPU kernel time (no host dispatch gaps): what a CUDA-graph / fused-launch execution of the
        # same kernels could reach at best; host round trips of vatt/mpig are NOT in these numbers
        from torch.profiler import ProfilerActivity, profile
        nxt = prefill()
        sync()
        C.profiling = True
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            decode(nxt, a.gpu_time_steps)
            sync()
        C.profiling = False
        ka = prof.key_averages()
        # two entries carry this name: the CPU range (device time = its kernels) and the GPU annotation
        # span (idle gaps included); the kernel sum is the smaller one
        att = [e.device_time_total for e in ka if e.key == "v62_attn"]
        res["gpu_attn_step_ms"] = (min(att) / a.gpu_time_steps / 1e3) if att else None
        from torch.autograd import DeviceType
        kern = [e for e in prof.events() if e.device_type == DeviceType.CUDA and e.name != "v62_attn"]   # kernels/copies, not the annotation span
        res["gpu_step_ms"] = sum(e.time_range.elapsed_us() for e in kern) / a.gpu_time_steps / 1e3
        res["gpu_kernels_per_step"] = len(kern) / a.gpu_time_steps
        print(f"  GPU kernel time per step: all {res['gpu_step_ms']:.2f} ms ({res['gpu_kernels_per_step']:.0f} kernels), "
              f"attention {res['gpu_attn_step_ms']:.2f} ms", flush=True)
    res["peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    res["wall_s"] = time.time() - t_start
    res["gpu_state_after"] = gpu_state()
    print(f"[{a.method} rho={a.rho} B={B}] read {100 * res['realised_read']:.2f}%  prefill {res['prefill_ms']:.1f} ms  "
          f"TPOT {res['tpot_ms']:.2f} ms  E2E {res['e2e_ms']:.1f} ms  attention/step {res['attn_step_ms']:.2f} ms "
          f"(select {res['attn_select_step_ms']:.2f} + attend {res['attn_compute_step_ms']:.2f})  "
          f"peak {res['peak_mem_gb']:.1f} GB", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
