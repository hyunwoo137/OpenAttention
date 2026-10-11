#!/usr/bin/env python3
"""Record a streaming-generation run for the demo video: dense attention vs OpenAttention.

Feeds Llama-3.1-8B-Instruct a real 30,012-token RULER document (qa_1 request #0) with a custom
question, then decodes greedily while logging a wall-clock timestamp for every generated token.
`dense` is HF's out-of-the-box eager path (SDPA default backend dispatch); `ours` is the deployed
OpenAttention decode (1% read, fused Triton selection + CUDA-graph step).

    python demo/record_demo.py --method dense --device cuda:0 --out demo/runs/dense.json
    python demo/record_demo.py --method ours  --device cuda:0 --out demo/runs/ours.json
    python demo/render_demo.py --dense demo/runs/dense.json --ours demo/runs/ours.json \
        --mp4 docs/static/demo.mp4 --gif assets/demo.gif
"""
import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "inference"))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import methods_eager as bl                                   # noqa: E402
import graph_runner as G5                                    # noqa: E402
import ours_fastbuild                                        # noqa: E402,F401  (registers ours3)
import ruler_data as RD                                      # noqa: E402
sys.argv = _argv

QUESTION = ("Based on the documents above, write a short summary of three bullet points, "
            "then answer in one sentence: of what country was Saint Denis once the capital?")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["dense", "ours"], required=True)
    p.add_argument("--rho", type=float, default=0.01)
    p.add_argument("--gen", type=int, default=200, help="decode steps recorded (stops early on EOS)")
    p.add_argument("--question", default=QUESTION)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    key, mode = ("densesdpadef", "eager") if a.method == "dense" else ("ours3", "graph")
    tok = AutoTokenizer.from_pretrained(bl.MODEL)
    model = AutoModelForCausalLM.from_pretrained(bl.MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(dev).eval()
    row = dict(RD.load_rows(["qa_1"], 1)[0], question=a.question, answer_prefix="")
    ctx, q = RD.build_prompt_ids(tok, row)
    ids = torch.tensor([ctx + q], device=dev)
    L0 = ids.shape[1]
    eos = set(RD.eos_ids(model))
    W = 3
    C = bl.C
    Lmax = L0 + a.gen + W + 8
    C.K = [torch.zeros(1, bl.HKV, Lmax, bl.D, device=dev, dtype=torch.bfloat16) for _ in range(bl.NL)]
    C.V = [torch.zeros(1, bl.HKV, Lmax, bl.D, device=dev, dtype=torch.bfloat16) for _ in range(bl.NL)]
    C.qlen = [len(q)]
    C.n_dev = torch.zeros(1, device=dev, dtype=torch.int32)
    C.n0_dev = torch.full((1,), L0, device=dev, dtype=torch.int32)
    C.kv_idx = torch.zeros(1, device=dev, dtype=torch.long)
    C.timing = False
    import types
    C.method = G5.METHODS[key](types.SimpleNamespace(rho=a.rho, mp_K=9, mp_L=25), 1, Lmax, dev)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = G5.attn_hook
    pos0 = torch.arange(L0, device=dev).view(1, -1)
    tok_buf = torch.zeros(1, 1, device=dev, dtype=torch.long)
    pos_buf = torch.zeros(1, 1, device=dev, dtype=torch.long)

    @torch.inference_mode()
    def prefill():
        C.phase, C.brow = "prefill", 0
        out = model(input_ids=ids, position_ids=pos0, use_cache=False, logits_to_keep=1)
        return out.logits[:, -1].argmax(-1)

    def set_step(s):
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

    step = graph_step if mode == "graph" else eager_step
    # warm-up: one prefill, W eager steps (Triton/cuBLAS warm), capture for the graph path
    nxt = prefill()
    warm = [int(nxt)]
    for s in range(W):
        nxt = eager_step(nxt, s)
        warm.append(int(nxt))
    if mode == "graph":
        capture(W)
    torch.cuda.synchronize()
    # recorded run: fresh prefill, then streamed decoding with a timestamp per token
    t0 = time.perf_counter()
    nxt = prefill()
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    toks, stamps = [int(nxt)], [0.0]
    td = time.perf_counter()
    for s in range(1, a.gen):
        nxt = step(nxt, s - 1)
        t = int(nxt)                                     # host sync: this is what streaming costs
        toks.append(t)
        stamps.append(time.perf_counter() - td)
        if t in eos:
            break
    text = tok.decode(toks, skip_special_tokens=True)
    pieces = [tok.decode(toks[:i + 1], skip_special_tokens=True) for i in range(len(toks))]
    res = {"method": a.method, "label": ("Dense attention (HF default)" if a.method == "dense"
                                         else f"OpenAttention ({a.rho:.0%} read)"),
           "rho": None if a.method == "dense" else a.rho,
           "prompt_tokens": L0, "prefill_s": t_prefill,
           "stamps_s": stamps, "pieces": pieces, "text": text,
           "tpot_ms": 1e3 * stamps[-1] / max(len(stamps) - 1, 1),
           "e2e_s": t_prefill + stamps[-1],
           "gpu": torch.cuda.get_device_name(dev)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(f"[{a.method}] prefill {t_prefill:.2f} s  {len(toks)} tokens  "
          f"{res['tpot_ms']:.1f} ms/token  e2e {res['e2e_s']:.2f} s")
    print("text:", text[:400].replace("\n", " | "))


if __name__ == "__main__":
    main()
