#!/usr/bin/env python3
"""V65 (graph_runner methods, eager execution; kernel times are what a graph replays):\nV64: where does a decode step of the final configuration spend its time?  Real Llama-3.1-8B, real
RULER-32K qa_1 prompt(s), the methods_eager decode loop with ours_fused oursfast (or any methods_eager method).

Per decode step it reports
  wall          synchronised wall time of the step (= TPOT)
  host          CPU time the Python/PyTorch side spends issuing the step (model forward call), and the
                part of it spent inside the attention method (selection + attention), no syncs inside
  GPU           kernel time on the GPU, split at the attention calls by two empty marker kernels the
                method's decode is wrapped in (single stream, so everything between the markers is the
                attention's), and by kernel name into the pipeline stages

    python profile_v64_real.py --method oursfast --rho 0.01 --batch 1 --device cuda:1 --out ...json
"""
import argparse
import collections
import json
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

import os as _os
HUB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # repo root
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HUB)
sys.argv, _argv = [sys.argv[0]], sys.argv
import graph_runner as G5                                                      # noqa: E402
import ours_graph                                                            # noqa: E402,F401  (registers ours2)
import methods_eager as bl                                                    # noqa: E402
from methods_eager import C, D, HKV, NL                                       # noqa: E402
sys.argv = _argv


@triton.jit
def _mark_attn_begin(X):
    pass


@triton.jit
def _mark_attn_end(X):
    pass


STAGES = [  # (label, name substrings) -- first match wins; applied to kernels inside the attention markers
    ("1 index update", ("_k_index_update",)),
    ("2 QGS scan", ("_k_qgs_scan",)),
    ("2 top-k (tiles -> keys) + sel", ("_k_select",)),
    ("7 ours.flash", ("_k_ours_flash_split", "_k_dense_combine")),
    ("2 QGS query projection (matmul)", ("gemm", "gemv", "sm90", "cutlass", "ampere", "sgemm", "Kernel2")),
    ("2 top-k", ("topk", "sort", "radix", "bitonic", "TopK")),
    ("3 ErrorGate", ("_k_errorgate",)),
    ("4 water-filling", ("_k_waterfill",)),
    ("5 within-cluster top-q", ("_k_topq",)),
    ("6 residual stats", ("_k_resid",)),
    ("7 ours.flash", ("_ours_flash_kernel",)),
    ("buffers (fill/zero)", ("fill", "Fill", "memset", "zero")),
]
OUTSIDE = [
    ("linear layers (GEMM)", ("gemm", "gemv", "sm90", "cutlass", "ampere", "sgemm", "Kernel2", "nvjet")),
    ("RMSNorm / reductions", ("reduce", "Reduce", "norm", "rsqrt", "pow")),
    ("KV write / copies", ("copy", "Copy", "cat", "Cat")),
]


def classify(name, table, default):
    for lab, keys in table:
        if any(k in name for k in keys):
            return lab
    return default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="ours")
    p.add_argument("--rho", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--prof-steps", type=int, default=8)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--sdpa-backend", default="", choices=["", "flash", "efficient", "cudnn"],
                   help="restrict the dense SDPA backend (default dispatch picks cuDNN, which builds a new graph "
                        "for every new sequence length)")
    p.add_argument("--out", default="")
    a = p.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    sys.path.insert(0, os.path.join(bl.BASE, "common"))
    tok = AutoTokenizer.from_pretrained(bl.MODEL)
    model = AutoModelForCausalLM.from_pretrained(bl.MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    ids, qlen, _ = bl.load_prompts(tok, a.batch)
    B, L0 = ids.shape
    ids = ids.to(dev)
    Lmax = L0 + max(a.steps + a.prof_steps + 24, 24) + 8
    C.K = [torch.zeros(B, HKV, Lmax, D, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
    C.V = [torch.zeros(B, HKV, Lmax, D, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
    C.qlen = qlen
    ns = argparse.Namespace(rho=a.rho, mp_K=9, mp_L=25)
    C.method = G5.METHODS[a.method](ns, B, Lmax, dev)
    C.n_dev = torch.zeros(1, device=dev, dtype=torch.int32)
    C.n0_dev = torch.full((1,), L0, device=dev, dtype=torch.int32)
    C.kv_idx = torch.zeros(1, device=dev, dtype=torch.long)
    C.timing = False
    ALL_ATTENTION_FUNCTIONS["sdpa"] = G5.attn_hook
    dummy = torch.zeros(1, device=dev)
    host_attn = [0.0]
    orig = C.method.decode

    from torch.nn.attention import SDPBackend, sdpa_kernel
    BK = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
          "cudnn": SDPBackend.CUDNN_ATTENTION}

    def wrapped(l, q, n):
        _mark_attn_begin[(1,)](dummy)
        t0 = time.perf_counter()
        if a.sdpa_backend:
            with sdpa_kernel(BK[a.sdpa_backend]):
                out = orig(l, q, n)
        else:
            out = orig(l, q, n)
        host_attn[0] += time.perf_counter() - t0
        _mark_attn_end[(1,)](dummy)
        return out
    C.method.decode = wrapped
    pos0 = torch.arange(L0, device=dev).view(1, -1)

    @torch.inference_mode()
    def prefill():
        C.phase = "prefill"
        nx = []
        for b in range(B):
            C.brow = b
            out = model(input_ids=ids[b:b + 1], position_ids=pos0, use_cache=False, logits_to_keep=1)
            nx.append(out.logits[:, -1].argmax(-1))
        return torch.stack(nx, 0)

    @torch.inference_mode()
    def step(nxt, s):
        C.phase = "decode"
        C.n = L0 + s + 1
        C.n_dev.fill_(C.n)
        C.kv_idx.fill_(C.n - 1)
        t0 = time.perf_counter()
        out = model(input_ids=nxt, position_ids=torch.full((B, 1), L0 + s, device=dev), use_cache=False)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        return nxt, time.perf_counter() - t0

    nxt = prefill()
    for s in range(8):                                                     # warm-up (compiles, allocator)
        nxt, _ = step(nxt, s)
    torch.cuda.synchronize(dev)
    walls, hosts, hattn = [], [], []
    for s in range(8, 8 + a.steps):
        host_attn[0] = 0.0
        t0 = time.perf_counter()
        nxt, h = step(nxt, s)
        torch.cuda.synchronize(dev)
        walls.append(time.perf_counter() - t0)
        hosts.append(h)
        hattn.append(host_attn[0])
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    s0 = 8 + a.steps
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for s in range(s0, s0 + a.prof_steps):
            nxt, _ = step(nxt, s)
        torch.cuda.synchronize(dev)
    kern = sorted((e for e in prof.events() if e.device_type == DeviceType.CUDA), key=lambda e: e.time_range.start)
    inside, outside = collections.Counter(), collections.Counter()
    n_in = n_out = 0
    in_attn = False
    for e in kern:
        nm, dt = e.name, e.time_range.elapsed_us() / 1e3
        if "_mark_attn_begin" in nm:
            in_attn = True
            continue
        if "_mark_attn_end" in nm:
            in_attn = False
            continue
        if in_attn:
            inside[classify(nm, STAGES, "misc torch ops (gather/cat/scatter/cast)")] += dt
            n_in += 1
        else:
            outside[classify(nm, OUTSIDE, "other (elementwise, RoPE, sampling)")] += dt
            n_out += 1
    P = a.prof_steps
    res = {"method": a.method, "rho": a.rho, "batch": B, "prompt_tokens": L0, "sdpa_backend": a.sdpa_backend or "default",
           "wall_ms": 1e3 * statistics.median(walls), "host_ms": 1e3 * statistics.median(hosts),
           "host_attn_ms": 1e3 * statistics.median(hattn),
           "gpu_attn_ms": sum(inside.values()) / P, "gpu_other_ms": sum(outside.values()) / P,
           "kernels_attn_per_step": n_in / P, "kernels_other_per_step": n_out / P,
           "attn_stages_ms": {k: v / P for k, v in inside.most_common()},
           "other_ms": {k: v / P for k, v in outside.most_common()}}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
