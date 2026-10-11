#!/usr/bin/env python3
"""Collect the run_latency_table.sh results into one markdown table (LATENCY.md)."""
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "results_latency"
ROWS = [("dense (SDPA default backend, eager)", "densesdpadef", "eager", False),
        ("dense (SDPA FlashAttention backend, eager)", "densesdpa", "eager", False),
        ("OpenAttention, reference port (eager)", "ours_ref", "eager", True),
        ("OpenAttention (fused selection + fast build, CUDA graph)", "ours3", "graph", True)]


def ld(m, r, b, mode):
    p = f"{D}/{m}_r{r}_b{b}_{mode}.json"
    return json.load(open(p)) if os.path.exists(p) else None


L = []
for title, key, fmt in (("End to end (s): prefill incl. per-request build + 128 decode steps", "e2e_ms_rep0", lambda x: f"{x / 1e3:.2f}"),
                        ("Time per output token (ms)", "tpot_ms_rep0", lambda x: f"{x:.1f}"),
                        ("Prefill incl. build (s)", "prefill_ms_rep0", lambda x: f"{x / 1e3:.2f}")):
    L += [f"### {title}", "", "| | 1% B=1 | 1% B=8 | 3% B=1 | 3% B=8 |", "|---|---|---|---|---|"]
    for name, m, mode, sparse in ROWS:
        cells = []
        for r in ("0.01", "0.03"):
            for b in (1, 8):
                d = ld(m, r if sparse else "0.01", b, mode)
                cells.append("--" if d is None else fmt(d[key]))
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")
L.append("Row-0 answers (should all agree):")
for name, m, mode, sparse in ROWS:
    for r in (("0.01", "0.03") if sparse else ("0.01",)):
        for b in (1, 8):
            d = ld(m, r, b, mode)
            if d:
                L.append(f"- {m} rho={r} B={b}: {d['answer_row0'][:24]!r}")
open(f"{D}/LATENCY.md", "w").write("\n".join(L) + "\n")
print("\n".join(L[:30]))
print(f"-> {D}/LATENCY.md")
