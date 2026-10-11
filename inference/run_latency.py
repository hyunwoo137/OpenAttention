#!/usr/bin/env python3
"""V71: one E2E latency table over every implementation, one driver (graph_runner main), one GPU, V68 protocol.

  densesdpadef  dense, SDPA default dispatch (cuDNN here), eager          -- "dense"
  densesdpa     dense, SDPA restricted to its FlashAttention backend, eager -- "dense.flash"
  dense         dense, split-K FlashDecoding Triton, CUDA graph           -- reference for the graph rows
  ours_v62      ours as first ported (methods_eager Ours: PyTorch selection + ours.flash kernel), eager -- "ours"
  ours2         ours_graph OursG2: fused Triton selection, CUDA graph      -- "ours2"
  ours3         ours_fastbuild OursG3: ours2 + fast per-request build          -- "ours3"

    python run_latency.py --method ours_v62 --rho 0.01 --batch 1 --mode eager --fresh-lengths --device cuda:0 --out ...
"""
import sys

import os as _os
HUB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # repo root
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HUB)
sys.path.insert(0, _HERE)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import methods_eager as bl                     # noqa: E402
import graph_runner as G5                       # noqa: E402
import ours_fastbuild                         # noqa: E402,F401  (registers ours2 via ours_graph, and ours3)
sys.argv = _argv
G5.METHODS["ours_ref"] = bl.Ours

if __name__ == "__main__":
    G5.main()
