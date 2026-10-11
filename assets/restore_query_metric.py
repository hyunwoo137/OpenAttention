#!/usr/bin/env python3
"""Restore the static query metric M_st (Sec. 3.2) from its packed upper-triangular form.

M_st is symmetric (32 layers x 32 query heads x 128 x 128, fp32, pre-RoPE), so it is shipped as
its upper triangle; the restored tensor is bit-identical to the one used for the paper.  QGC reads
it through --profile-static-file and QGS through --static-file (run_openattention.sh passes both
and calls this script on first use):

    python assets/restore_query_metric.py \
        --packed assets/llama-3.1-8b-instruct.query_metric.triu.pt --out assets/llama-3.1-8b-instruct.query_metric.pt
"""
import argparse
from pathlib import Path

import torch


def unpack(packed: dict) -> dict:
    d = packed["dim"]
    iu = torch.triu_indices(d, d)
    tri = packed["M_triu"]                               # (L, H, d(d+1)/2) float32
    M = torch.zeros(*tri.shape[:-1], d, d, dtype=tri.dtype)
    M[..., iu[0], iu[1]] = tri
    M[..., iu[1], iu[0]] = tri                           # mirror (diagonal written twice, same value)
    out = {k: v for k, v in packed.items() if k not in ("M_triu", "dim")}
    out["M"] = M
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default="assets/llama-3.1-8b-instruct.query_metric.triu.pt")
    ap.add_argument("--out", default="assets/llama-3.1-8b-instruct.query_metric.pt")
    a = ap.parse_args()
    d = unpack(torch.load(a.packed, map_location="cpu", weights_only=False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, a.out)
    print(f"wrote {a.out}: M {tuple(d['M'].shape)} ({d.get('source')}, {d.get('n_requests')} requests)")


if __name__ == "__main__":
    main()
