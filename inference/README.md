# Deployed decode path

This directory runs OpenAttention as it would be served: the same algorithm as the accuracy path
(QGC + QGS + ErrorGate + count-aware proxies + OpenAttention.flash), executed as sparse GPU kernels
with a fast per-request index build and **one CUDA graph per decode step**. The accuracy path at the
repo root simulates sparsity with dense masks, which is exact for scores but says nothing about
speed; this path is what the end-to-end latency claims are measured on.

## Files

```
methods_eager.py     reference ports of every method (dense / HashAttention / vAttention /
                     MagicPIG / OpenAttention) as eager sparse GPU pipelines, one attention hook
ours_fused.py        OpenAttention selection fused into Triton kernels (index update, QGS scan,
                     ErrorGate, water-filling, within-cluster top-q, residual statistics)
graph_runner.py      graph-safe method variants (device-side lengths) + the benchmark driver
ours_graph.py        OpenAttention decode step, fully graph-replayable ("ours2")
ours_fastbuild.py    + the fast per-request build: batched eigh, Triton k-means and cluster sums
                     ("ours3" = the deployed configuration)
eval_accuracy.py     RULER-32K accuracy of this path, request by request (resumable jsonl)
profile_step.py      per-stage GPU timing of one decode step (CUDA events + profiler)
run_latency.py       entry point for the end-to-end latency table
run_latency_table.sh one-command reproduction of the README latency table
collect_latency.py   collects the table -> LATENCY.md
ruler_data.py        RULER-32K prompts, decoding rule and string-match metric (hub-identical)
```

Method keys: `ours3` (deployed), `ours2` (same decode, reference build), `ours_ref` (first eager
port), `dense` (split-K Triton + graph), `densesdpa` (SDPA FlashAttention backend),
`densesdpadef` (SDPA default dispatch = what stock Hugging Face runs), `hat`, `vatt`, `mpig`.

## Usage

```bash
# the latency table (one idle GPU; ~25 min)
bash run_latency_table.sh 0                  # -> results_latency/LATENCY.md

# one cell
python run_latency.py --method ours3 --rho 0.01 --batch 1 --mode graph \
    --fresh-lengths --device cuda:0 --out ours3_1pct_b1.json

# accuracy of the deployed path (resumable; ~2.3 s/request at 1%)
python eval_accuracy.py --method ours3 --rho 0.01 --device cuda:0 --out results_acc/ours3_1pct.jsonl
python eval_accuracy.py --method dense --rho 0.01 --device cuda:0 --out results_acc/dense.jsonl

# where does a decode step spend its time?
python profile_step.py --method ours3 --rho 0.01 --batch 8 --device cuda:0

# numerical self-tests (no model download needed)
python ours_fused.py --selftest --device cuda:0        # fused selection == reference, bitwise sets
python ours_graph.py --selftest --device cuda:0        # graph kernels   == fused selection
python ours_fastbuild.py --selftest --device cuda:0    # fast build      == reference build
```

## Protocol notes

- Latency protocol: real RULER qa_1 prompts (B=1: request #0 with 30,012 tokens; B=8: requests
  #0–7 cut to the shortest), prefill including the per-request index build, then 128 decode steps;
  `--fresh-lengths` gives every timed repetition decode lengths no earlier step used, so no
  per-length kernel cache can hit; the first repetition is reported.
- Accuracy protocol of `eval_accuracy.py`: the **whole prompt** (context + question chunk) is
  prefilled densely and sparse attention applies to generation. The paper's accuracy path also
  routes the question chunk through sparse attention. Under this protocol: dense 87.07,
  OpenAttention 86.32 at 1% and 86.65 at 3% (7 tasks × 200 requests).
- The deployed per-budget configuration is `(C, E)` = (64, 216) at 1% and (192, 679) at 3%, as in
  the accuracy path. `E` and `C` are sized against a 32,768-token budget, so on a 30,012-token
  prompt the realised read is 1.09% / 3.27%.
- HashAttention / vAttention baselines here need `HAT_ARTIFACTS` (see the root README); dense and
  OpenAttention need nothing beyond the model and `assets/`.
