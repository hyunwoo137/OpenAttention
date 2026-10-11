#!/bin/sh
# Baselines on RULER-32K HARD with Llama-3.1-8B-Instruct, at the configurations of the paper
# (App. "Baseline configurations", "MagicPIG configuration" and "vAttention protocol").
#
#   usage: sh run_baselines.sh DEVICE METHOD BUDGET [TASK] [NREQ] [OUTDIR]
#     DEVICE  cuda:0, cuda:1, ...
#     METHOD  hashattention | vattention | magicpig | dense
#     BUDGET  1pct | 3pct | 5pct | 10pct       (ignored for dense)
#     TASK    one of qa_1 qa_2 vt fwe niah_multikey_2 niah_multikey_3 niah_multivalue, or "all" (default)
#     NREQ    requests per task (default 200, as in the paper)
#     OUTDIR  output directory (default ./results/n<NREQ>)
#
#   examples
#     sh run_baselines.sh cuda:0 hashattention 1pct qa_1 5    # smoke test
#     sh run_baselines.sh cuda:0 magicpig 3pct                # paper run -> results/n200/
#     sh run_baselines.sh cuda:0 dense -                      # dense reference (87.04)
#
# All methods share the protocol of run_openattention.sh: dense prefill of the context, sparse
# attention on the question chunk and generation, greedy decoding, string_match scoring.
# HashAttention and vAttention share a sink and a local window of 0.1% each (the driver default)
# and need the public HashAttention signature weights:
#     git clone https://github.com/xAlg-ai/HashAttention-1.0
#     export HAT_ARTIFACTS=/path/to/HashAttention-1.0/artifacts
# MagicPIG needs no weights.  Its realised read depends on the hash configuration, so the paper
# calibrates L at K=9 to the target read on the benchmark itself; the values below realise
# 1.09 / 2.88 / 5.13 / 9.97% at the 1 / 3 / 5 / 10% targets.
# vAttention runs under its authors' per-budget protocol: a 50/50 heavy/sampling split with
# eps = delta = 0.3 at 1-5%, and the vAttention paper's own 10% grid point (heavy 7.5%,
# sampling 2.5%) at 10%.
set -e
DEV=$1; METHOD=$2; BUDGET=$3; TASK=${4:-all}; NREQ=${5:-200}; OUT=${6:-./results/n${5:-200}}
if [ -z "$DEV" ] || [ -z "$METHOD" ]; then
  echo "usage: sh run_baselines.sh DEVICE METHOD BUDGET [TASK] [NREQ] [OUTDIR]"; exit 2
fi
PY=${PYTHON:-python}
cd "$(dirname "$0")"

case $BUDGET in
  1pct)  RHO=0.01; MPL=8   ;;
  3pct)  RHO=0.03; MPL=25  ;;
  5pct)  RHO=0.05; MPL=46  ;;
  10pct) RHO=0.10; MPL=100 ;;
  *) if [ "$METHOD" != "dense" ]; then echo "unknown budget $BUDGET (use 1pct | 3pct | 5pct | 10pct)"; exit 2; fi ;;
esac
if [ "$TASK" = "all" ]; then
  SUBSETS="qa_1,qa_2,vt,fwe,niah_multikey_2,niah_multikey_3,niah_multivalue"
else
  SUBSETS=$TASK
fi

mkdir -p "$OUT"
case $METHOD in
  dense)
    set -x
    $PY run_multibin_hat_bench.py --device "$DEV" --output-dir "$OUT" \
      --methods dense --densities 1.0 --subsets "$SUBSETS" --max-requests "$NREQ"
    ;;
  hashattention)
    set -x
    $PY run_multibin_hat_bench.py --device "$DEV" --output-dir "$OUT" \
      --methods HAT --densities $RHO --subsets "$SUBSETS" --max-requests "$NREQ"
    ;;
  vattention)
    SPLIT=0.5
    [ "$BUDGET" = "10pct" ] && SPLIT=0.25
    set -x
    $PY run_multibin_hat_bench.py --device "$DEV" --output-dir "$OUT" \
      --methods vAttention --densities $RHO --vatt-split $SPLIT --eps 0.3 --delta 0.3 \
      --subsets "$SUBSETS" --max-requests "$NREQ"
    ;;
  magicpig)
    set -x
    $PY run_multibin_hat_bench.py --device "$DEV" --output-dir "$OUT" \
      --methods magicpig --densities $RHO --lsh-k 9 --lsh-l $MPL \
      --subsets "$SUBSETS" --max-requests "$NREQ"
    ;;
  *) echo "unknown method $METHOD (use hashattention | vattention | magicpig | dense)"; exit 2 ;;
esac
