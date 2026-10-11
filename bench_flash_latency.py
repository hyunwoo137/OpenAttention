#!/usr/bin/env python3
"""Decode attention-stage latency per layer (Table 9 and Fig. 1(c) of the paper).

Synthetic tensors with Llama-3.1-8B shapes (32 query / 8 KV heads, d = 128, bf16), K = 32768 keys,
one GPU, median of CUDA-event timings.  Every path reads the same total budget rho of KV rows per
query head; the selector stage (index scan + top-k), which every sparse method shares, is NOT
timed here.

  dense        fused SDPA on the GQA tensors (torch.nn.functional.scaled_dot_product_attention)
  ours_triton  OpenAttention.flash: the fused Triton kernel of triton_expand_decode.py; the exact
               rows and the C cluster proxies (mean key, mean value, +log n_b on the logit) are
               merged in one online-softmax pass
  ours_torch   the same estimator as an unfused PyTorch gather path
  hat          HashAttention's attention stage as implemented in the framework: gather of the
               rho*K selected rows, then attention over them
  vatt         vAttention, LOWER BOUND: rho/2 selected rows + rho/2 uniform samples with
               importance weights and the statistics pass of its budget rule; the re-sampling
               loop and the per-layer host synchronisation are not charged
  mpig         MagicPIG, LOWER BOUND: gather of rho*K sampled rows with importance weights as a
               logit bias; LSH probing and the CPU residency of the released system not charged

Budget: the baselines read rho*K exact rows; OpenAttention reads (rho*K - 1.5*C) exact rows plus
the C proxies (1.5 rows each).  With --E, OpenAttention.flash splits its exact rows as the deployed
method does: E tokens through the ErrorGate expansion stream X and the rest (sink, local, top-k)
through the selection stream S.  --E 0 puts every exact row in S (the split used for Table 9).

    python bench_flash_latency.py --device cuda:0 --rho 0.01 --C 64  --E 216 --out latency_1pct.json
    python bench_flash_latency.py --device cuda:0 --rho 0.03 --C 192 --E 679 --out latency_3pct.json
"""
import argparse, json, math, statistics, sys
import torch
import torch.nn.functional as F

from triton_expand_decode import decode_expand_attention  # noqa: E402


def med(fn, dev, w=25, it=150):
    for _ in range(w):
        fn()
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(it):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(dev)
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def build(B, Hq, Hkv, K, D, rho, C, dev, dt, E=0, seed=0):
    torch.manual_seed(seed)
    g, BH, BQ = Hq // Hkv, B * Hkv, B * Hq
    sc = D ** -0.5
    n_base = int(round(K * rho))                       # baselines: exact rows incl. sink/local
    n_ours = int(round(K * rho - 1.5 * C))             # ours: proxies charged at 1.5 rows/bin
    n_half = n_base // 2                               # vAttention: 50/50 split
    q = torch.randn(BQ, D, device=dev, dtype=dt)
    k = torch.randn(BH, K, D, device=dev, dtype=dt)
    v = torch.randn(BH, K, D, device=dev, dtype=dt)
    perm = torch.stack([torch.randperm(K, device=dev) for _ in range(BQ)])
    sel_base = perm[:, :n_base].int().contiguous()
    sel_ours = perm[:, :n_ours].int().contiguous()     # every exact row (the PyTorch path reads them as one set)
    n_exp = min(E, n_ours)                             # OpenAttention.flash: S = sink+local+top-k, X = E expanded tokens
    sel_s = perm[:, :n_ours - n_exp].int().contiguous()
    sel_top = perm[:, :n_half].int().contiguous()
    samp = perm[:, n_half:2 * n_half].int().contiguous()
    kbar = torch.randn(BH, C, D, device=dev, dtype=dt)
    vbar = torch.randn(BH, C, D, device=dev, dtype=dt)
    ell = torch.full((BQ, C), math.log((K - n_ours) / C), device=dev, dtype=torch.float32)
    exp_x = (perm[:, n_ours - n_exp:n_ours].int().contiguous() if n_exp > 0      # X: ErrorGate expansion stream
             else torch.full((BQ, 64), -1, device=dev, dtype=torch.int32))       # E=0: one fully masked tile
    return dict(B=B, g=g, BH=BH, BQ=BQ, sc=sc, q=q, k=k, v=v, sel_base=sel_base, sel_ours=sel_ours,
                sel_top=sel_top, samp=samp, kbar=kbar, vbar=vbar, ell=ell, sel_s=sel_s, exp_x=exp_x,
                n_base=n_base, n_ours=n_ours, n_exp=n_exp, n_half=n_half, K=K, D=D, dt=dt, dev=dev)


def make_paths(t):
    q, k, v, dt, dev = t["q"], t["k"], t["v"], t["dt"], t["dev"]
    g, BH, BQ, sc, K, D = t["g"], t["BH"], t["BQ"], t["sc"], t["K"], t["D"]
    Bsz, Hkv_ = t["B"], BH // t["B"]
    kflat, vflat = k.reshape(BH * K, D), v.reshape(BH * K, D)
    off = (torch.arange(BH, device=dev) * K).view(BH, 1)

    def rows(idx):
        n = idx.shape[1]
        return (idx.view(BH, g * n).long() + off).reshape(-1), n

    sb_r, nb_ = rows(t["sel_base"]); so_r, no_ = rows(t["sel_ours"])
    st_r, nt_ = rows(t["sel_top"]); sm_r, nm_ = rows(t["samp"])
    qf = q.view(BQ, 1, D); qg = q.view(BH, g, D)
    n_T = float(K - nt_)                       # tail size seen by vAttention's sampler
    w_samp = n_T / nm_

    # dense: fused SDPA on GQA tensors.  q:(B,Hq,1,D) k,v:(B,Hkv,K,D)
    qd = q.view(Bsz, Hkv_ * g, 1, D); kd = k.view(Bsz, Hkv_, K, D); vd = v.view(Bsz, Hkv_, K, D)

    def dense():
        return F.scaled_dot_product_attention(qd, kd, vd, enable_gqa=True)

    def gather_attn(idx_r, n):
        ks = kflat.index_select(0, idx_r).view(BQ, n, D)
        vs = vflat.index_select(0, idx_r).view(BQ, n, D)
        z = torch.bmm(qf, ks.transpose(1, 2)) * sc
        p = torch.softmax(z.float(), -1).to(dt)
        return torch.bmm(p, vs)

    def hat():
        return gather_attn(sb_r, nb_)

    def vatt():
        # heavy hitters (rho/2) + uniform sample (rho/2), importance-weighted, one softmax;
        # plus the statistics pass of the budget rule on the sampled scores (lower bound)
        ks = kflat.index_select(0, st_r).view(BQ, nt_, D); vs = vflat.index_select(0, st_r).view(BQ, nt_, D)
        km = kflat.index_select(0, sm_r).view(BQ, nm_, D); vm = vflat.index_select(0, sm_r).view(BQ, nm_, D)
        z = torch.bmm(qf, ks.transpose(1, 2)) * sc
        zs = torch.bmm(qf, km.transpose(1, 2)) * sc
        rm = torch.maximum(z.amax(-1, True), zs.amax(-1, True))
        e = torch.exp((z - rm).float()); es = torch.exp((zs - rm).float())
        mu = es.mean(-1, True); var = es.var(-1, unbiased=False, keepdim=True)
        # budget rule (eps, delta): needed samples ~ z_{1-delta}^2 var / (eps mu)^2  (host sync not charged)
        _ = (var / (mu * mu).clamp(min=1e-30))
        num = torch.bmm(e.to(dt), vs) + w_samp * torch.bmm(es.to(dt), vm)
        return num / (e.sum(-1, True) + w_samp * es.sum(-1, True)).to(dt)

    kbar, vbar, ell, C = t["kbar"], t["vbar"], t["ell"], t["kbar"].shape[1]

    def ours_torch():
        ks = kflat.index_select(0, so_r).view(BQ, no_, D); vs = vflat.index_select(0, so_r).view(BQ, no_, D)
        z = torch.bmm(qf, ks.transpose(1, 2)) * sc
        zb = torch.bmm(qg, kbar.transpose(1, 2)).view(BQ, 1, C) * sc + ell.view(BQ, 1, C)
        rm = torch.maximum(z.amax(-1, True), zb.amax(-1, True))
        e = torch.exp((z - rm).float()).to(dt); w = torch.exp((zb - rm).float()).to(dt)
        num = torch.bmm(e, vs) + torch.bmm(w.view(BH, g, C), vbar).view(BQ, 1, D)
        return num / (e.sum(-1, True) + w.sum(-1, True))

    def ours_triton():
        return decode_expand_attention(q, k, v, t["sel_s"], t["exp_x"], kbar, vbar, ell, sm_scale=sc)

    # MagicPIG lower bound: the sample IS the read (rho*K rows); importance weights
    # log(1/p_i) enter the softmax as a per-row bias.  LSH probing not charged.
    bias = torch.randn(BQ, 1, nb_, device=dev, dtype=torch.float32).abs()

    def mpig():
        ks = kflat.index_select(0, sb_r).view(BQ, nb_, D)
        vs = vflat.index_select(0, sb_r).view(BQ, nb_, D)
        z = torch.bmm(qf, ks.transpose(1, 2)) * sc
        p = torch.softmax(z.float() + bias, -1).to(dt)
        return torch.bmm(p, vs)

    return {"dense": dense, "hat": hat, "vatt": vatt, "mpig": mpig,
            "ours_torch": ours_torch, "ours_triton": ours_triton}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batches", default="1,8,32")
    p.add_argument("--rho", type=float, default=0.03)
    p.add_argument("--C", type=int, default=64)
    p.add_argument("--E", type=int, default=0,
                   help="ErrorGate expansion tokens: OpenAttention.flash reads E of its exact rows through the "
                        "expansion stream X and the rest through the selection stream S (deployed: 216 at 1%%, 679 at 3%%)")
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--out", default="latency.json")
    a = p.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    Hq, Hkv, D = 32, 8, 128
    res = {"config": vars(a), "gpu": torch.cuda.get_device_name(dev), "ms": {}}
    for B in [int(x) for x in a.batches.split(",")]:
        t = build(B, Hq, Hkv, a.seq_len, D, a.rho, a.C, dev, torch.bfloat16, E=a.E)
        paths = make_paths(t)
        # correctness of the two ours paths against each other
        o_t = paths["ours_triton"]().float(); o_p = paths["ours_torch"]().view(-1, D).float()
        rel = float(((o_t - o_p).norm(dim=-1) / o_p.norm(dim=-1)).median())
        res["ms"][str(B)] = {}
        print(f"\nB={B}  K={a.seq_len}  rho={a.rho:.0%}  C={a.C}  exact rows: baselines {t['n_base']}, ours {t['n_ours']} (S {t['n_ours'] - t['n_exp']} + X {t['n_exp']})  "
              f"(triton vs torch rel err {rel:.2e})")
        for name, fn in paths.items():
            ms = med(fn, dev, it=a.iters); res["ms"][str(B)][name] = ms
            print(f"  {name:12s} {ms:8.3f} ms")
        del t, paths; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
