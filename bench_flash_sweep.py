#!/usr/bin/env python3
"""Decode attention time per layer vs batch size and context length (Fig. 11 of the paper).

Synthetic tensors with Llama-3.1-8B shapes (32 query / 8 KV heads, d = 128, bf16), one GPU, median
of CUDA-event timings, on the 5% + 256-token-expansion configuration: n_sel = round(K*(0.05-0.002))
exact rows, 8 expanded runs of 32 tokens, (K - n_sel)/32 cluster proxies stored once per KV head
(no variance stream).  Paths: fused dense SDPA, the unfused PyTorch gather path of the estimator,
and OpenAttention.flash (triton_expand_decode.py).

    python bench_flash_sweep.py --device cuda:0 --sweep batch --json sweep_batch.json   # (a) B = 1..32 at K = 32K
    python bench_flash_sweep.py --device cuda:0 --sweep ctx   --json sweep_ctx.json     # (b) K = 8K..64K at B = 8

Output: {B or K: {"sdpa": ms, "torch": ms, "triton": ms}}.
"""
import argparse, json, math, statistics

import torch
import torch.nn.functional as F

from triton_expand_decode import decode_expand_attention


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


def run_one(B, K, a, dev, with_torch=True):
    torch.manual_seed(0)
    dt = torch.bfloat16
    Hq, Hkv, D = a.heads, a.kv_heads, a.head_dim
    g, BH, BQ = Hq // Hkv, B * Hkv, B * Hq
    sc = D ** -0.5
    n_sel = int(round(K * (a.density - 0.002)))
    nb = (K - n_sel) // a.bin_size
    ne = a.expand_m * a.bin_size

    q = torch.randn(BQ, D, device=dev, dtype=dt)
    k = torch.randn(BH, K, D, device=dev, dtype=dt)
    v = torch.randn(BH, K, D, device=dev, dtype=dt)
    sel = torch.stack([torch.randperm(K, device=dev)[:n_sel] for _ in range(BQ)]).int()
    exp = torch.stack([torch.randperm(K - a.bin_size, device=dev)[:a.expand_m] for _ in range(BQ)])
    exp = (exp.unsqueeze(-1) + torch.arange(a.bin_size, device=dev)).reshape(BQ, ne).int()
    kbar = torch.randn(BH, nb, D, device=dev, dtype=dt)
    vbar = torch.randn(BH, nb, D, device=dev, dtype=dt)
    ell = torch.full((BQ, nb), math.log(float(a.bin_size)), device=dev, dtype=torch.float32)

    qd = q.view(B, Hq, 1, D)
    kd, vd = k.view(B, Hkv, K, D), v.view(B, Hkv, K, D)

    def dense():
        return F.scaled_dot_product_attention(qd, kd, vd, enable_gqa=True)

    qg = q.view(BH, g, D)
    kflat, vflat = k.reshape(BH * K, D), v.reshape(BH * K, D)
    off = (torch.arange(BH, device=dev) * K).view(BH, 1)
    sel_r = (sel.view(BH, g * n_sel).long() + off).reshape(-1)
    exp_r = (exp.view(BH, g * ne).long() + off).reshape(-1)

    def torch_gather_path():
        ks = kflat.index_select(0, sel_r).view(BQ, n_sel, D)
        vs = vflat.index_select(0, sel_r).view(BQ, n_sel, D)
        ke = kflat.index_select(0, exp_r).view(BQ, ne, D)
        ve = vflat.index_select(0, exp_r).view(BQ, ne, D)
        qf = q.view(BQ, 1, D)
        z = torch.bmm(qf, ks.transpose(1, 2)) * sc
        ze = torch.bmm(qf, ke.transpose(1, 2)) * sc
        zb = torch.bmm(qg, kbar.transpose(1, 2)).view(BQ, 1, nb) * sc + ell.view(BQ, 1, nb)
        rm = torch.maximum(torch.maximum(z.amax(-1, True), ze.amax(-1, True)), zb.amax(-1, True))
        e, ee, w = [torch.exp(x.float() - rm.float()).to(dt) for x in (z, ze, zb)]
        num = (torch.bmm(e, vs) + torch.bmm(ee, ve)
               + torch.bmm(w.view(BH, g, nb), vbar).view(BQ, 1, D))
        return num / (e.sum(-1, True) + ee.sum(-1, True) + w.sum(-1, True))

    def triton_path():
        return decode_expand_attention(q, k, v, sel, exp, kbar, vbar, ell, sm_scale=sc,
                                       block_n=a.block_n, block_b=a.block_b, num_warps=a.warps)

    ref = torch_gather_path().view(BQ, D).float()
    rel = ((triton_path().float() - ref).norm(dim=-1) / ref.norm(dim=-1)).median().item()
    res = {"sdpa": med(dense, dev, it=a.iters)}
    if with_torch:
        res["torch"] = med(torch_gather_path, dev, it=a.iters)
    res["triton"] = med(triton_path, dev, it=a.iters)
    print(f"B={B:3d} K={K:6d} n_sel={n_sel} clusters={nb} expand={ne}  "
          + "  ".join(f"{k}={v:.3f}ms" for k, v in res.items())
          + f"  (flash vs torch rel.err {rel:.1e})", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sweep", choices=["batch", "ctx"], required=True)
    p.add_argument("--batches", default="1,4,8,16,32")
    p.add_argument("--ctx-lens", default="8192,16384,32768,65536")
    p.add_argument("--ctx-batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--expand-m", type=int, default=8)
    p.add_argument("--bin-size", type=int, default=32)
    p.add_argument("--block-n", type=int, default=64)
    p.add_argument("--block-b", type=int, default=64)
    p.add_argument("--warps", type=int, default=4)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    dev = torch.device(a.device); torch.cuda.set_device(dev)
    print(f"device={torch.cuda.get_device_name(dev)}")
    out = {}
    if a.sweep == "batch":
        for B in map(int, a.batches.split(",")):
            out[str(B)] = run_one(B, a.seq_len, a, dev)
    else:
        for K in map(int, a.ctx_lens.split(",")):
            out[str(K)] = run_one(a.ctx_batch, K, a, dev, with_torch=False)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()
