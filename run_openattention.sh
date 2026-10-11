#!/bin/sh
# OpenAttention on RULER-32K HARD with Llama-3.1-8B-Instruct, at the deployed configuration of
# each total read budget (accuracy path, PyTorch estimator).
#
#   usage: sh run_openattention.sh DEVICE BUDGET [TASK] [NREQ] [OUTDIR]
#     DEVICE  cuda:0, cuda:1, ...
#     BUDGET  1pct | 3pct | 5pct | 10pct       (total per-query read, proxy stream included)
#     TASK    one of qa_1 qa_2 vt fwe niah_multikey_2 niah_multikey_3 niah_multivalue, or "all" (default)
#     NREQ    requests per task (default 200, as in the paper)
#     OUTDIR  output directory (default ./results/n<NREQ>, so runs with different NREQ never share it)
#
#   examples
#     sh run_openattention.sh cuda:0 1pct qa_1 5           # smoke test  -> results/n5/
#     sh run_openattention.sh cuda:0 1pct                  # paper run   -> results/n200/
#
# Per budget: C clusters (QGC), E tokens of ErrorGate expansion, sink and local at their
# few-token floor (0.01% each).  The accounting charges every stream the query reads:
#   rho = sink + local + top-k + E/n + 1.5C/n     (1.5 KV rows per cluster = mean + variance token)
set -e
DEV=$1; BUDGET=$2; TASK=${3:-all}; NREQ=${4:-200}; OUT=${5:-./results/n${4:-200}}
if [ -z "$DEV" ] || [ -z "$BUDGET" ]; then
  echo "usage: sh run_openattention.sh DEVICE BUDGET [TASK] [NREQ] [OUTDIR]"; exit 2
fi
PY=${PYTHON:-python}
cd "$(dirname "$0")"

case $BUDGET in
  1pct)  RHO=0.01; C=64;  E=216  ;;
  3pct)  RHO=0.03; C=192; E=679  ;;
  5pct)  RHO=0.05; C=192; E=1335 ;;
  10pct) RHO=0.10; C=256; E=2877 ;;
  *) echo "unknown budget $BUDGET (use 1pct | 3pct | 5pct | 10pct)"; exit 2 ;;
esac
if [ "$TASK" = "all" ]; then
  SUBSETS="qa_1,qa_2,vt,fwe,niah_multikey_2,niah_multikey_3,niah_multivalue"
else
  SUBSETS=$TASK
fi

# Static query metric M_st (Sec. 3.2), shipped packed; restored once on first use.
M=${STATIC_M:-assets/llama-3.1-8b-instruct.query_metric.pt}
if [ ! -f "$M" ]; then
  $PY assets/restore_query_metric.py --packed assets/llama-3.1-8b-instruct.query_metric.triu.pt --out "$M"
fi

# QGS (Sec. 3.2.2): rank-32 query eigenbasis, int8 key codes, one basis per KV head.
SEL="--selector static --static-file $M --static-rank 32 --static-window 32 --static-source static+question --static-share-kv --static-bits 8"
# QGC (Sec. 3.2.1): cluster the keys projected onto the rank-16 eigenvectors of M = 1/2 M_st + 1/2 M_q.
PART="--profile-dtype bf16 --profile-source static+question --profile-static-file $M --profile-rank 16 --profile-rot-window 32"
# ErrorGate (Sec. 3.3): rank clusters by the estimated weight error, hierarchical (water-filling) quotas.
GATE="--expand-rank wl1 --residual-stats full --expand-alloc hier --expand-alloc-gamma 1"

mkdir -p "$OUT"
set -x
$PY run_multibin_hat_bench.py --device "$DEV" --output-dir "$OUT" \
  --methods UTA+fair --densities $RHO $SEL $PART $GATE \
  --sink 0.0001 --local 0.0001 \
  --profile-C $C --expand-tokens $E --expand-bin-cap 0 --expand-c 3.0 \
  --subsets "$SUBSETS" --max-requests "$NREQ"
