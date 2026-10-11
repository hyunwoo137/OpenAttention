#!/bin/bash
# End-to-end latency table: dense (SDPA default backend / SDPA FlashAttention backend) vs
# OpenAttention (reference port, and the deployed CUDA-graph path), on ONE idle GPU.
#
#   usage: bash inference/run_latency_table.sh [GPU_INDEX] [OUTDIR]
#
# Protocol: real RULER qa_1 prompts (B=1: request #0, 30,012 tokens; B=8: requests #0-7 cut to the
# shortest), prefill incl. the per-request index build + 128 decode steps, fresh decode lengths in
# every timed repetition, first repetition reported.  One process per cell; the script waits until
# the GPU is empty before each run.
set -e
G=${1:-0}
cd "$(dirname "$0")"
O=${2:-results_latency}
PY=${PYTHON:-python}
mkdir -p $O
wait_free() { while : ; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)
    [ "$used" -lt 1024 ] && break; sleep 5; done; sleep 5; }
JOBS="densesdpadef:0.01:1:eager densesdpa:0.01:1:eager
      ours_ref:0.01:1:eager ours3:0.01:1:graph ours_ref:0.03:1:eager ours3:0.03:1:graph
      densesdpadef:0.01:8:eager densesdpa:0.01:8:eager
      ours_ref:0.01:8:eager ours3:0.01:8:graph ours_ref:0.03:8:eager ours3:0.03:8:graph"
for job in $JOBS; do
  IFS=: read m r b mode <<< "$job"
  tag=${m}_r${r}_b${b}_${mode}
  wait_free
  echo "start $tag $(date +%T)" >> $O/chain.log
  $PY run_latency.py --method $m --rho $r --batch $b --mode $mode \
      --fresh-lengths --device cuda:$G --out $O/$tag.json > $O/log_$tag.txt 2>&1
  echo "done  $tag rc=$? $(date +%T)" >> $O/chain.log
done
echo ALLDONE >> $O/chain.log
$PY collect_latency.py $O
