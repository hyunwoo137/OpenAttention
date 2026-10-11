#!/usr/bin/env python3
"""V67: RULER-32K accuracy of the FAST decode path (ours_graph OursG2 under CUDA graphs), request by request.

Prompts, decoding and metric are the hub's (baselines/common/ruler_data.py reproduces them: chat template,
context and question tokenised separately and truncated as the hub does, greedy, max_new_tokens =
min(120, row), stop on EOS, string_match).  One protocol difference to the paper's hub runs is kept on
purpose and reported: the whole prompt (context + question chunk) is prefilled DENSELY and sparse attention
applies to generation only -- the latency benchmarks' protocol; the hub also routes the question chunk
through the sparse method.

methods
  ours2     ours_graph OursG2 (fast kernels), CUDA graph replay per step (graphs cached per step constants)
  ours_v62  methods_eager Ours (the original unfused port), eager -- the same selection, for a per-sample check
  dense     split-K dense decode (graph_runner DenseG), CUDA graph -- the ceiling of this protocol
options   --rq R (QGS code dimensions, default 32)   --qbits 8|4 (QGS code precision, default 8)

    python eval_accuracy.py --method ours2 --rho 0.01 --device cuda:1 --out results_v67_acc/ours2_r0.01.jsonl
"""
import argparse
import json
import os
import sys
import time
import types

import torch

import os as _os
HUB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # repo root
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HUB)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import methods_eager as bl                                               # noqa: E402
import graph_runner as G5                                                 # noqa: E402
import ours_graph as G6                                                 # noqa: E402
from methods_eager import C, CTX_LEN, D, HKV, HQ, NL                     # noqa: E402
import ruler_data as RD                                                      # noqa: E402
sys.argv = _argv


class OursQ(G6.OursG2):
    """OursG2 with the QGS code dimension r and precision as parameters (deployed: r=32, int8)"""

    def __init__(self, a, B, Lmax, dev):
        super().__init__(a, B, Lmax, dev)
        rq, bits = int(getattr(a, "rq", 32)), int(getattr(a, "qbits", 8))
        self.QMAX = float(2 ** (bits - 1) - 1)
        if rq != self.rq:
            self.rq = rq
            z = lambda *s, dt=torch.float32: torch.zeros(*s, device=dev, dtype=dt)   # noqa: E731
            self.codes = [z(B, HKV, Lmax, rq, dt=torch.bfloat16) for _ in range(bl.NL)]
            self.U = [z(B, HKV, D, rq) for _ in range(bl.NL)]
            self.scale = [z(B, HKV, 1, rq) for _ in range(bl.NL)]
            self.codes8 = [torch.zeros(B, HKV, Lmax, rq, device=dev, dtype=torch.int8) for _ in range(bl.NL)]

    def _build_csr(self, l, b, T):
        if self.QMAX != 127.0:                       # re-quantise the prefix codes to the smaller range
            zc = torch.matmul(C.K[l][b, :, :T].float(), self.U[l][b])
            scale = (zc.abs().amax(1, keepdim=True) / self.QMAX).clamp(min=1e-12)
            self.codes[l][b, :, :T] = (zc / scale).round().clamp(-self.QMAX, self.QMAX).to(torch.bfloat16)
            self.scale[l][b] = scale
        super()._build_csr(l, b, T)

    def graph_key(self, n):
        heavy = self.a.rho - 0.0002 - self.E / CTX_LEN - 1.5 * self.C / CTX_LEN
        return (max(1, int(0.0001 * n)), max(1, int(heavy * n)))


class DenseQ(G5.DenseG):
    def graph_key(self, n):
        return ()


class OursV62(bl.Ours):
    graph_key = None                                 # eager only


METHODS = {"ours2": OursQ, "dense": DenseQ, "ours_v62": OursV62}


def score(task, pred, refs):
    import re
    p = re.sub(r"[\x00-\x1f]", "", pred.strip()).strip().lower()
    hit = [1.0 if r.lower() in p else 0.0 for r in refs]
    return 100.0 * (max(hit) if task.startswith("qa") else sum(hit) / len(hit))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=list(METHODS), default="ours2")
    p.add_argument("--rho", type=float, default=0.01)
    p.add_argument("--rq", type=int, default=32)
    p.add_argument("--qbits", type=int, default=8)
    p.add_argument("--subsets", default=",".join(RD.SUBSETS))
    p.add_argument("--max-requests", type=int, default=200)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    tok = AutoTokenizer.from_pretrained(bl.MODEL)
    model = AutoModelForCausalLM.from_pretrained(bl.MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    eos = set(RD.eos_ids(model))
    rows = RD.load_rows(a.subsets.split(","), max_requests=a.max_requests)
    done = {}
    if os.path.exists(a.out):
        for line in open(a.out):
            x = json.loads(line)
            done[(x["task"], x["index"])] = x
    prompts = [RD.build_prompt_ids(tok, r) for r in rows]
    Lmax = max(len(c) + len(q) for c, q in prompts) + RD.MAX_NEW_TOKENS_CAP + 16
    C.K = [torch.zeros(1, HKV, Lmax, D, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
    C.V = [torch.zeros(1, HKV, Lmax, D, device=dev, dtype=torch.bfloat16) for _ in range(NL)]
    C.n_dev = torch.zeros(1, device=dev, dtype=torch.int32)
    C.n0_dev = torch.zeros(1, device=dev, dtype=torch.int32)
    C.kv_idx = torch.zeros(1, device=dev, dtype=torch.long)
    C.timing = False
    m = METHODS[a.method](types.SimpleNamespace(rho=a.rho, rq=a.rq, qbits=a.qbits), 1, Lmax, dev)
    C.method = m
    ALL_ATTENTION_FUNCTIONS["sdpa"] = G5.attn_hook
    tok_buf = torch.zeros(1, 1, device=dev, dtype=torch.long)
    pos_buf = torch.zeros(1, 1, device=dev, dtype=torch.long)
    graphs = {}
    use_graph = m.graph_key is not None

    def set_step(L0, s):
        C.n = L0 + s + 1
        C.n_dev.fill_(C.n)
        C.kv_idx.fill_(C.n - 1)
        pos_buf.fill_(L0 + s)

    @torch.inference_mode()
    def eager_step():
        C.phase = "decode"
        out = model(input_ids=tok_buf, position_ids=pos_buf, use_cache=False)
        return out.logits[:, -1].argmax(-1, keepdim=True)

    @torch.inference_mode()
    def capture(key):
        C.phase = "decode"
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(input_ids=tok_buf, position_ids=pos_buf, use_cache=False)
            nb = out.logits[:, -1].argmax(-1, keepdim=True)
        graphs[key] = (g, nb)

    f_out = open(a.out, "a")
    t_start, n_new, tot_steps = time.time(), 0, 0
    for row, (ctx, q) in zip(rows, prompts):
        if (row["task"], row["index"]) in done:
            continue
        ids = torch.tensor([ctx + q], device=dev)
        L0 = ids.shape[1]
        C.qlen = [len(q)]
        C.n0_dev.fill_(L0)
        with torch.inference_mode():
            C.phase, C.brow = "prefill", 0
            out = model(input_ids=ids, position_ids=torch.arange(L0, device=dev).view(1, -1),
                        use_cache=False, logits_to_keep=1)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        gen = [int(nxt)]
        for s in range(row["max_new_tokens"] - 1):     # as the hub: the first token is not checked for EOS
            set_step(L0, s)
            tok_buf.copy_(nxt)
            key = m.graph_key(C.n) if use_graph else None
            if use_graph and key in graphs:
                graphs[key][0].replay()
                nxt = graphs[key][1]
            else:
                nxt = eager_step().clone()             # executes this step (and compiles on first use)
                if use_graph:
                    capture(key)                       # recorded, not executed; replayed from the next step
            gen.append(int(nxt))
            tot_steps += 1
            if gen[-1] in eos:
                break
        pred = tok.decode(gen, skip_special_tokens=True)
        rec = {"task": row["task"], "index": row["index"], "pred": pred, "answer": row["answer"],
               "score": score(row["task"], pred, row["answer"]), "prompt_tokens": L0, "gen_tokens": len(gen)}
        f_out.write(json.dumps(rec) + "\n")
        f_out.flush()
        n_new += 1
        if n_new % 50 == 0:
            el = time.time() - t_start
            print(f"[{a.method} rho={a.rho} r={a.rq} b={a.qbits}] {n_new} requests, {el / n_new:.2f} s/request, "
                  f"{tot_steps / max(n_new, 1):.1f} steps/request, graphs {len(graphs)}", flush=True)
    f_out.close()
    # summary
    recs = [json.loads(line) for line in open(a.out)]
    by = {}
    for r in recs:
        by.setdefault(r["task"], []).append(r["score"])
    tasks = [t for t in RD.SUBSETS if t in by]
    avg = sum(sum(by[t]) / len(by[t]) for t in tasks) / len(tasks)
    print(f"[{a.method} rho={a.rho} r={a.rq} bits={a.qbits}] " +
          " ".join(f"{t}:{sum(by[t]) / len(by[t]):.2f}(n={len(by[t])})" for t in tasks) + f" | avg {avg:.2f}", flush=True)


if __name__ == "__main__":
    main()
