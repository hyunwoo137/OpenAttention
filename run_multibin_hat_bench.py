#!/usr/bin/env python3
"""
Experiment 3: multi-bin UTA on top of a HashAttention (HAT) selector.

Why HAT rather than oracle-top-k
--------------------------------
oracle-top-k needs the full QK product to pick its indices, so nothing built on
it can ever be fast.  HAT produces approximate top-k indices from learned hash
signatures -- it is the selector vAttention actually deploys.  Everything after
selection (the tail) is then genuinely free budget, which is where multi-bin UTA
lives.  HAT's selection is also *imperfect*: some high-logit tokens leak into the
tail, which raises the tail's logit variance.  Prediction: multi-bin should help
MORE under HAT than under oracle-top-k, because it isolates those leaked tokens
into their own bins instead of averaging them away.

Methods (all at matched total density rho)
------------------------------------------
  HAT              sink + local + HAT top-k                 (no tail recovery)
  UTA-HAT          + single mean-pooled tail proxy          (base UTA)
  MultiBin-HAT     + per-bin proxies, bin_size sweep        (this work)
  vAttention-HAT   HAT top-k at rho/2 + adaptive sampling   (baseline to beat)

Density accounting: sink 0.1% + local 0.1% + heavy (+ sampling for vAttention).
The tail proxy itself costs no *tokens*, so all methods read the same number of
KV entries; the multi-bin overhead is bin metadata, accounted separately.

Usage
-----
    python run_multibin_hat_bench.py --device cuda:2 --densities 0.03
    python run_multibin_hat_bench.py --device cuda:2 --summarize-only
"""

import argparse
import json
import os
import sys
from typing import Any, Dict
from pathlib import Path

import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from benchmark.ruler32k import Ruler32K  # noqa: E402
from sparse_attention_hub.adapters import ModelAdapterHF  # noqa: E402
from sparse_attention_hub.sparse_attention.research_attention import (  # noqa: E402
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (  # noqa: E402
    HashAttentionTopKMaskerConfig,
    LocalMaskerConfig,
    OracleTopKConfig,
    SnapKVMaskerConfig,
    StaticTopKMaskerConfig,
    SinkMaskerConfig,
    LSHTopKMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (  # noqa: E402
    AdaptiveSamplingMaskerConfig,
    MagicPigConfig,
)
from sparse_attention_hub.sparse_attention.uta_attention import (  # noqa: E402
    UTAAttentionConfig,
    UTAMultiBinConfig,
    UTAMultiBinSketchConfig,
)

SUBSETS = [
    "qa_1", "qa_2", "vt", "fwe",
    "niah_multikey_2", "niah_multikey_3", "niah_multivalue",
]

# HashAttention signature weights, only needed for the `--selector hat` baseline path (the
# proposed method uses the QGS and needs none of these).  Point HAT_ARTIFACTS at a checkout of
# the HashAttention authors' released artifacts directory.
_HAT_DIR = os.environ.get("HAT_ARTIFACTS", "./hat_artifacts").rstrip("/") + "/"
HAT_WEIGHT_FILE = _HAT_DIR + "llama3.1-8b-patch.64K.v1.hat_weights.pkl"
HAT_KW = dict(hat_bits=32, hat_mlp_layers=3, hat_mlp_hidden_size=128,
              hat_mlp_activation="silu", hat_weight_file=HAT_WEIGHT_FILE)
# The released signature weights for the other two models.
# DeepSeek: 2-layer MLP, 64-bit signatures; Mistral: 3-layer MLP, 32-bit (converted from the
# USA checkpoint with create_hat_weights_file_from_usa, num_mlp_layers=3).
MODEL_HAT = {
    "meta-llama/Llama-3.1-8B-Instruct": (HAT_WEIGHT_FILE, 32, 3),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": (
        _HAT_DIR + "DeepSeek-R1-Distill-Llama-8B-patch-layers2-dim64-max-context-24K_hat_weights.pkl", 64, 2),
    "mistralai/Mistral-7B-Instruct-v0.3": (_HAT_DIR + "Mistral-7B-Instruct-v0.3.24K.20.500.hat_weights.pkl", 32, 3),
}

SINK = 0.001
LOCAL = 0.001

# Context length the density fractions are resolved against.  The expansion budget is
# a token COUNT (expand_m * bin_size), so turning it into a density needs this; RULER32K
# is fixed at 32768, and the density-matched variant is only defined for that benchmark.
CTX_LEN = 32768
EXPAND_M_TABLE = "results_v9_decomp/expand_m_table_mean8.json"

# bin_mode="profile" build parameters (V22); overridden from the CLI.  Defaults
# reproduce the V20 run exactly (P=64, 4 Lloyd iterations, fp32, k-means).
MP_KW: Dict[str, Any] = {"lsh_k": 10, "lsh_l": 150}   # MagicPIG LSH (K bits/table, L tables)
# V39: PyramidKV/SnapKV under the protocol of their own papers -- the budget is an
# ABSOLUTE average KV cache size per layer (Cai et al. 2024 report 64 .. 2048), the
# observation window is the last alpha=8 tokens, and the prompt is encoded with full
# attention (only the generation reads the compressed cache).
KV_BUDGET: int = 0
# V54: KV-cache *press* (CentroidKV) applied to the prefilled context cache before the
# question is seen (kvpress protocol).  Composes with any method: `dense` gives the
# compressor alone, `full` the same through the research path (needed for the paper's
# log N term), `UTA+fair` BinAttention reading the merged cache.
PRESS_KW: Dict[str, Any] = dict(mode="none", ratio=0.75, sink=16, local=64, chunk=256,
                                merge=0.8, decay=0.2, logcount=False, when="context")


def _make_press():
    if PRESS_KW["mode"] == "none":
        return None
    if PRESS_KW["mode"] in ("snapkv", "pyramidkv", "streamingllm"):
        # V54c: the eviction compressors under the kvpress split (context only)
        from sparse_attention_hub.sparse_attention.utils.evict_press import EvictPress
        return EvictPress(mode=str(PRESS_KW["mode"]), budget=int(PRESS_KW.get("budget", 8000)),
                          kernel=7, beta=20, num_layers=32, when="context")
    if PRESS_KW["mode"] != "centroidkv":
        raise ValueError(f"unknown --kv-press {PRESS_KW['mode']!r}")
    from sparse_attention_hub.sparse_attention.utils.centroidkv import CentroidKVPress
    return CentroidKVPress(compression_ratio=float(PRESS_KW["ratio"]), sink_budget=int(PRESS_KW["sink"]),
                           local_budget=int(PRESS_KW["local"]), chunk_size=int(PRESS_KW["chunk"]),
                           merge_ratio=float(PRESS_KW["merge"]), decrease_rate=float(PRESS_KW["decay"]),
                           logcount=bool(PRESS_KW["logcount"]), when=str(PRESS_KW.get("when", "context")))


def _press_tag() -> str:
    p = _make_press()
    return "" if p is None else "-" + p.tag()
KV_ALPHA: int = 8

PROFILE_KW: Dict[str, Any] = dict(profile_P=64, profile_iters=4,
                                  profile_dtype="fp32", profile_method="kmeans",
                                  profile_source="question", profile_static_file=None,
                                  profile_rank=8, profile_rot_window=0, profile_rank_q=8,
                                  profile_C=1024, profile_basis="eig")
# V26: TOTAL-traffic-matched configuration ("UTA+fair").  proxy_rows = (k,v)-row
# equivalents charged per bin and per query: k-mean + v-mean + diagonal key variance
# = 3 vectors = 1.5 rows.  expand_tokens is the EXACT expansion budget (crossing bin
# truncated by selector score, see UTAMultiBinConfig.expand_truncate).
FAIR_KW: Dict[str, Any] = dict(expand_tokens=0, bin_cap=0, proxy_rows=1.5, truncate=True,
                               bin_mode="profile", rank="bound", var_mode="diag", var_rank=4,
                               two_point=False, rep_k=1, kappa="none", rep_pick="whiten",
                               kappa_samples=8, kappa_scale=1.0,
                               # V38: compose with KV-cache compression.  The eviction pass
                               # keeps `evict_budget` of the context resident; rho is then a
                               # fraction of the FULL context as always, so a 50% budget with
                               # rho=0.05 reads 10% of what survives.
                               evict_mode="none", evict_budget=0.0, evict_tokens=0)
# V28 additions to UTA+fair (all still charged with the same accounting):
#   bin_mode  "profile" (ours) | "score" (oracle: equal-width bins over the TRUE tail logit,
#             query-dependent, only meaningful under --selector oracle) | "equalcount"
#             (positional adjacency, bin_size = N / C)
#   rank      "bound" (ours, zbar + c sigma) | "mass" (zbar + log n: the centroid-mass rule of
#             Double-P / RetroInfer / ClusterKV)
#   var_mode  "diag" (ours) | "exact" (= full covariance, by linearity of the logit in k) |
#             "lowrank" (diag + var_rank factors per bin)


def _profile_tag() -> str:
    """Job-name suffix; empty for the V20 defaults so old dirs stay resumable."""
    d = PROFILE_KW
    if (d["profile_P"], d["profile_iters"], d["profile_dtype"],
            d["profile_method"], d.get("profile_basis", "eig")) == (64, 4, "fp32", "kmeans", "eig"):
        return ""
    m = {"qsort": "qs", "grid": "gr", "geo": "ge", "geo2": "g2", "keykm": "kk",
         "retro": f"rt{d['profile_iters']}", "centroidkv": "ck"}.get(d["profile_method"], f"i{d['profile_iters']}")
    base = f",P{d['profile_P']}{m}{d['profile_dtype']}"
    src = d.get("profile_source", "question")
    if src != "question":
        # V23: the metric source replaces the question rows -- name carries source, rank, window
        stag = {"static": "st", "question_rot": "qr", "static+question": "sq", "static+rows": "sr",
                "static+qrot": "uq"}[src]
        base += f",S{stag}r{d['profile_rank']}" + ("rnd" if d.get("profile_basis", "eig") == "random" else "") \
            + (f"q{d['profile_rank_q']}" if src == "static+qrot" else "") \
            + (f"w{d['profile_rot_window']}" if d.get("profile_rot_window", 0) else "")
        if src != "question_rot" and d.get("profile_static_file"):
            base += "-" + Path(d["profile_static_file"]).stem.replace("Mstatic_", "").replace("_pre", "")
    return base


# V25: static-metric selector (replaces HashAttention).  Overridden from the CLI.
STATIC_KW: Dict[str, Any] = dict(static_file=os.environ.get("STATIC_M", "assets/llama-3.1-8b-instruct.query_metric.pt"), rank=16, rot_window=32,
                                 source="static+question", profile_P=64, share_kv=False, code_bits=16,
                                 rebuild_every=0, basis="eig")


def _static_tag() -> str:
    d = STATIC_KW
    return (f"ST{'sq' if d['source'] == 'static+question' else 'st'}r{d['rank']}w{d['rot_window']}"
            + ("kv" if d["share_kv"] else "") + (f"i{d['code_bits']}" if d["code_bits"] != 16 else "")
            + (f"P{d['profile_P']}" if d["profile_P"] != 64 else "")
            + (f"rb{d['rebuild_every']}" if d["rebuild_every"] else "")
            + ("rnd" if d.get("basis", "eig") == "random" else ""))


def _abs_heavy(heavy: float):
    """V54: with --sink-local-tokens the selector's top-k is pinned to the token count the
    fraction yields on the uncompressed context (about 9 at the deployed 1--10% splits),
    instead of a fraction of a pressed cache (2 on 8,000 entries, 0 on PyramidKV's last
    layer)."""
    if SINK_LOCAL_TOKENS <= 0 or not isinstance(heavy, float):
        return heavy
    return max(1, int(round(heavy * CTX_LEN)))


def _selector(heavy: float, selector: str):
    """Top-k selector.  'hat' is the deployable one; 'oracle' needs the full QK;
    'static' is the V25 static-metric selector (rank-r key codes, no learned weights)."""
    h = _abs_heavy(heavy)
    h = round(h, 6) if isinstance(h, float) else int(h)
    if selector == "hat":
        return HashAttentionTopKMaskerConfig(heavy_size=h, **HAT_KW)
    if selector == "oracle":
        return OracleTopKConfig(heavy_size=h)
    if selector == "static":
        return StaticTopKMaskerConfig(heavy_size=h, **STATIC_KW)
    if selector == "lsh":
        # V49: MagicPIG hashing (K bits x L tables) used as a deterministic top-k index
        return LSHTopKMaskerConfig(heavy_size=h, lsh_k=MP_KW["lsh_k"], lsh_l=MP_KW["lsh_l"])
    raise ValueError(f"unknown selector: {selector}")


SINK_LOCAL_TOKENS: int = 0     # V54: >0 = sink/local as ABSOLUTE token counts (see --sink-local-tokens)


def _abs_or_frac(frac: float):
    """Sink/local size for the maskers.  Fractions are of the *cache length*, which a KV
    press shrinks (0.01% of 8,000 merged entries is 0 tokens where the same setting gives
    3 tokens on a 32K cache); with --sink-local-tokens the count is pinned to what the
    fraction yields on the uncompressed context, so pairs stay comparable."""
    if SINK_LOCAL_TOKENS <= 0:
        return frac
    return max(int(SINK_LOCAL_TOKENS), int(frac * CTX_LEN))


def _prefix(heavy: float, selector: str = "hat"):
    base = [SinkMaskerConfig(sink_size=_abs_or_frac(SINK)),
            LocalMaskerConfig(window_size=_abs_or_frac(LOCAL))]
    if selector == "none":
        # V28: no heavy-hitter stage at all (sink + local only); everything else is tail.
        # Used for the Double-P-style port, whose only exact reads are the expanded clusters.
        return base
    return base + [_selector(heavy, selector)]


# Default: bin the tail by whatever score the selector already computed, which costs
# nothing extra.  MEASURED CAVEAT (run_block_tail_diag, 5%): HAT's approximate score
# is useless for this -- within-bin logit variance ratio 0.72 vs 0.47 for plain
# position blocks and 0.97 for a random partition.  Its sign-vector signatures are
# trained to surface the top tokens, not to rank the bulk.  So under a HAT selector
# pass --bin-mode explicitly: "score" for the accuracy ceiling (needs the full QK, so
# not cost-consistent) or "fixed" for the cost-consistent position-block point.
BIN_MODE_FOR = {"hat": "hat", "oracle": "score", "static": "fixed", "none": "fixed", "lsh": "fixed"}
POSITION_MODES = ("fixed", "equalcount")


def build_config(method: str, rho: float, num_bins: int, kappa: str,
                 vatt_split: float = 0.5, eps: float = 0.4, delta: float = 0.4,
                 selector: str = "hat", bin_mode: str = "auto", var_mode: str = "diag",
                 expand_c: float = 3.0):
    heavy = round(rho - SINK - LOCAL, 6)
    _prefix_ = lambda h: _prefix(h, selector)  # noqa: E731
    mode = BIN_MODE_FOR[selector] if bin_mode == "auto" else bin_mode

    # ---- ablation ladder (all at the same density, same selector) ------------
    # dense -> no sparsity at all;  selector -> top-k only, tail discarded;
    # UTA -> one mean-pooled tail proxy;  +jensen -> that proxy gets log(1+s2/2);
    # +multibin -> proxy per run of adjacent tail indices;  ours -> both.
    if method == "dense":
        return None

    if method == "full":
        # V54: dense attention through the research path (a local window covering
        # every key), so that a KV press's log-count bias reaches the attention.
        return ResearchAttentionConfig(masker_configs=[LocalMaskerConfig(window_size=1.0)])

    if method == "selector":
        return ResearchAttentionConfig(masker_configs=_prefix_(heavy))

    if method == "UTA":
        return UTAAttentionConfig(masker_configs=_prefix_(heavy))

    if method == "UTA+jensen":
        # one bin spanning the whole tail == base UTA, plus the correction
        return UTAMultiBinConfig(masker_configs=_prefix_(heavy), bin_mode="equalcount",
                                 bin_size=10 ** 9, kappa_mode="j2", var_mode="diag")

    if method == "UTA+multibin":
        return UTAMultiBinConfig(masker_configs=_prefix_(heavy), bin_mode="equalcount",
                                 bin_size=num_bins, kappa_mode="none", var_mode="diag")

    if method == "UTA+expand+matched":
        # Density-MATCHED expansion: the m*bin_size tokens the second stage reads are
        # taken OUT of the selector's budget instead of being added on top, so the
        # total tokens touched equal vAttention's and HAT's exactly.  Without this the
        # comparison gives us 0.78% more density than the baselines.
        exp_frac = num_bins * 32 / CTX_LEN
        heavy_m = round(rho - SINK - LOCAL - exp_frac, 6)
        if heavy_m <= 0:
            raise ValueError(f"rho={rho} too small for m={num_bins} expansion")
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy_m), bin_mode="equalcount", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins, expand_c=expand_c,
            expand_rank="bound")

    if method == "UTA+expand+triton":
        # identical configuration to UTA+expand; the decode step is routed through the
        # fused Triton kernel so the benchmark verifies end-to-end equivalence
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy), bin_mode="equalcount", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins, expand_c=expand_c,
            expand_rank="bound", use_triton_decode=True)

    if method == "UTA+expandhead":
        # Non-uniform but STATIC per-(layer,head) budget with the same MEAN m, so the
        # token count matches UTA+expand exactly on average and the run-time cost is
        # identical (no per-step decision, no ragged launch).
        import json as _json
        tbl = _json.load(open(EXPAND_M_TABLE))
        table = {int(l): {int(h): int(m) for h, m in hh.items()}
                 for l, hh in tbl["table"].items()}
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy), bin_mode="equalcount", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins, expand_c=expand_c,
            expand_rank="bound", expand_m_table=table)

    if method in ("expand+noagg", "expand+agg1"):
        # The aggregate-term ablation.  Selector, partition, ranking and expansion are
        # identical to UTA+expand; only what happens to the UNEXPANDED bins changes:
        #   noagg -> they contribute nothing (tail discarded)
        #   agg1  -> they collapse into ONE proxy (base-UTA treatment of the residual)
        # `num_bins` is the expansion count m.
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy), bin_mode="equalcount", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins, expand_c=expand_c,
            expand_rank="bound",
            aggregate_mode="none" if method == "expand+noagg" else "single")

    if method in ("UTA+expandmass", "UTA+expandmassj2"):
        # The centroid-mass ranking every prior cluster/page method uses: pick the
        # bins with the largest estimated mass, zbar_b + log n_b.  Same partition,
        # same budget, same merge -- ONLY the ranking statistic differs from
        # UTA+expand, so the gap isolates the second moment's contribution.
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy), bin_mode="equalcount", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins, expand_c=expand_c,
            expand_rank="mass" if method == "UTA+expandmass" else "massj2")

    if method.startswith("UTA+sketch"):
        # V17: the surviving proxies carry a rank-r sketch of their block's keys
        # (and values, for "+val").  bin_mode must be "fixed" -- the sketch basis
        # is cached with the KV pages, so block membership has to be
        # query-independent.  `num_bins` is the expansion count m, as for
        # UTA+expand; the rank comes from the method name.
        rank = int(method.rsplit("r", 1)[1].rstrip("v"))
        return UTAMultiBinSketchConfig(
            masker_configs=_prefix_(heavy), bin_mode="fixed", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins,
            expand_c=expand_c, expand_rank="bound",
            sketch_rank=rank, sketch_values=method.endswith("v"))

    if method == "UTA+expand+nosel":
        # V12: NO sink / local / selector at all -- the WHOLE context is tail.
        # Adjacency bins of 32 over everything, top `num_bins` bins by the key-side
        # bound computed exactly, decode through the fused Triton kernel.  The only
        # exact-token budget is the expansion itself, so total density is
        # num_bins*32/CTX_LEN; `rho` enters the job name only (pick num_bins to match).
        return UTAMultiBinConfig(
            masker_configs=[], bin_mode="equalcount", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins, expand_c=expand_c,
            expand_rank="bound", use_triton_decode=True)

    if method == "UTA+expandpfm":
        # V24: density-MATCHED profile expansion.  The num_bins*32 exact tokens the
        # expansion reads are taken OUT of the selector budget, so the total exact
        # read (sink + local + HAT + expansion) equals rho -- the same accounting
        # as HAT / vAttention / MagicPIG in the density sweep.
        exp_frac = num_bins * 32 / CTX_LEN
        heavy_m = round(rho - SINK - LOCAL - exp_frac, 6)
        if heavy_m <= 0:
            raise ValueError(f"rho={rho} too small for m={num_bins} expansion")
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy_m), bin_mode="profile", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins,
            expand_c=expand_c, expand_rank="bound",
            expand_budget_tokens=num_bins * 32, **PROFILE_KW)

    if method == "UTA+fair":
        # V26: TOTAL-traffic-matched BinAttention.  Every stream read per query is
        # charged in (k,v)-row equivalents against rho:
        #   sink + local + selector top-k + expansion tokens (exactly E, the crossing
        #   bin is truncated by selector score) + proxy stream (C bins x proxy_rows).
        # The selector gets whatever is left.  C and E come from the CLI
        # (--profile-C, --expand-tokens), so the split can follow rho.
        C = int(PROFILE_KW["profile_C"]); E = int(FAIR_KW["expand_tokens"])
        proxy_frac = FAIR_KW["proxy_rows"] * C / CTX_LEN
        heavy_m = round(rho - SINK - LOCAL - E / CTX_LEN - proxy_frac, 6)
        if selector == "none":
            # no selector: the budget must be spent by C and E exactly (within one row)
            if abs(heavy_m) > 1.5 / CTX_LEN:
                raise ValueError(f"selector=none: rho={rho} is not exhausted by C={C}, E={E} "
                                 f"(unspent {heavy_m:.5f}); pick E = (rho-0.002-1.5C/N)*N")
        elif heavy_m <= 0:
            raise ValueError(f"rho={rho} too small for C={C} (proxy {proxy_frac:.4f}) + E={E}")
        common = dict(masker_configs=_prefix_(heavy_m),
                      kappa_mode=str(FAIR_KW["kappa"]),
                      two_point=bool(FAIR_KW["two_point"]), rep_k=int(FAIR_KW["rep_k"]),
                      rep_pick=str(FAIR_KW.get("rep_pick", "whiten")),
                      kappa_samples=int(FAIR_KW.get("kappa_samples", 8)),
                      kappa_scale=float(FAIR_KW.get("kappa_scale", 1.0)),
                      var_mode=FAIR_KW["var_mode"], var_rank=int(FAIR_KW["var_rank"]),
                      expand_m=1 if E > 0 else 0, expand_c=expand_c,
                      expand_rank=FAIR_KW["rank"], expand_budget_tokens=E,
                      expand_truncate=bool(FAIR_KW["truncate"]),
                      expand_bin_cap=int(FAIR_KW["bin_cap"]),
                      residual_stats=str(FAIR_KW.get("residual_stats", "none")),
                      expand_alloc=str(FAIR_KW.get("expand_alloc", "uniform")),
                      expand_alloc_gamma=float(FAIR_KW.get("alloc_gamma", 1.0)),
                      evict_mode=str(FAIR_KW["evict_mode"]),
                      evict_budget=float(FAIR_KW["evict_budget"]),
                      evict_tokens=int(FAIR_KW.get("evict_tokens", 0)),
                      evict_alpha=KV_ALPHA,
                      evict_sink=SINK, evict_local=LOCAL)
        bm = FAIR_KW["bin_mode"]
        if bm == "profile":
            return UTAMultiBinConfig(bin_mode="profile", bin_size=32, **common, **PROFILE_KW)
        if bm == "score":
            return UTAMultiBinConfig(bin_mode="score", num_bins=C, **common)
        if bm == "equalcount":
            return UTAMultiBinConfig(bin_mode="equalcount", bin_size=max(1, CTX_LEN // C), **common)
        raise ValueError(f"unknown fair bin mode {bm}")

    if method == "magicpig":
        # MagicPIG (Chen et al. 2024) as shipped in sparse-attention-hub: sink + local
        # + LSH sampling with importance weights.  Density is set by (K, L); the
        # realised density must be measured (--log-density), rho is only the label.
        return ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=SINK), LocalMaskerConfig(window_size=LOCAL),
            MagicPigConfig(lsh_l=MP_KW["lsh_l"], lsh_k=MP_KW["lsh_k"], center=True,
                           packing="int64", seed=42)])

    if method in ("UTA+expandpf", "UTA+expandpfmass"):
        # V19/V20: PROFILE partition -- the tail is clustered by logit profiles
        # against P=64 query rows sampled from the question chunk (cached per
        # request, extended at decode).  Expansion is TOKEN-budget equalised
        # (num_bins*32 tokens) because profile clusters have unequal sizes; the
        # "mass" variant swaps the ranking statistic for the centroid rule.
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy), bin_mode="profile", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins,
            expand_c=expand_c,
            expand_rank="mass" if method.endswith("mass") else "bound",
            expand_budget_tokens=num_bins * 32, **PROFILE_KW)

    if method == "UTA+qmeanexp":
        # V19/V20: STATIC expansion -- the top num_bins*32 tail tokens by cached
        # qmean (mean logit against the profile queries) are promoted into the
        # exact set once per request; the rest of the tail keeps profile-cluster
        # proxies.  No per-step expansion ranking at all.
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy), bin_mode="profile", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=0, expand_c=expand_c,
            profile_promote=num_bins * 32, **PROFILE_KW)

    if method in ("UTA+expand", "UTA+expandrand"):
        # E1/E2: adjacency bins of 32, then the top `num_bins` bins by the key-side
        # bound are computed exactly.  `num_bins` is the EXPANSION COUNT m here.
        return UTAMultiBinConfig(
            masker_configs=_prefix_(heavy), bin_mode="equalcount", bin_size=32,
            kappa_mode="none", var_mode="diag", expand_m=num_bins, expand_c=expand_c,
            expand_rank="random" if method == "UTA+expandrand" else "bound")

    if method == "UTA+randbin":
        # V4: partition the tail (= everything sink/local/HAT did NOT take) uniformly
        # at random into `num_bins` bins.  num_bins is a BIN COUNT here, not a size.
        return UTAMultiBinConfig(masker_configs=_prefix_(heavy), bin_mode="random",
                                 num_bins=num_bins, kappa_mode="none", var_mode="diag")

    if method in ("ours", "UTA+multibin+jensen"):
        if method != "ours":
            return UTAMultiBinConfig(masker_configs=_prefix_(heavy), bin_mode="equalcount",
                                     bin_size=num_bins, kappa_mode="j2", var_mode="diag")
        # position/adjacency modes are parameterised by tokens-per-bin, score modes by
        # bin count
        cfg: Dict[str, Any] = ({"bin_size": num_bins} if mode in POSITION_MODES
                               else {"num_bins": num_bins})
        return UTAMultiBinConfig(masker_configs=_prefix_(heavy), bin_mode=mode,
                                 kappa_mode=kappa, var_mode=var_mode, **cfg)

    # ---- legacy names kept so already-completed 1%/3% runs stay resumable ----
    if method == "HAT":
        return ResearchAttentionConfig(masker_configs=_prefix(heavy, "hat"))

    if method == "UTA-HAT":
        return UTAAttentionConfig(masker_configs=_prefix(heavy, "hat"))

    if method == "MultiBin-HAT":
        return UTAMultiBinConfig(masker_configs=_prefix(heavy, "hat"),
                                 bin_size=num_bins, bin_mode="fixed", kappa_mode=kappa)

    # ---- V35: KV-cache compression baselines (paper v10, Table `main` block) ----
    if method == "streamingllm":
        # StreamingLLM read pattern: sink + recent window, nothing else; the whole
        # budget is sink 0.1% + local (rho - 0.1%).
        return ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=SINK),
            LocalMaskerConfig(window_size=round(rho - SINK, 6))])

    if method in ("snapkv-paper", "pyramidkv-paper", "streamingllm-paper"):
        # V40: the KV-cache compression baselines exactly as KVCache-Factory (the
        # PyramidKV codebase) runs them -- absolute per-layer budget, maxpool kernel 7,
        # window 8 (snapkv/pyramidkv) or budget-4 (streamingllm, = 4 attention sinks),
        # and the prompt encoded with full attention.  No sink/local maskers: the
        # retained window is the method's only positional guarantee.
        mode = method[: -len("-paper")]
        return ResearchAttentionConfig(masker_configs=[
            SnapKVMaskerConfig(heavy_size=int(KV_BUDGET), obs_window=KV_ALPHA,
                               pool_kernel=7, pyramid=(mode == "pyramidkv"),
                               pyramid_beta=20, num_layers=32,
                               paper_mode=True, alpha=KV_ALPHA, mode=mode)])

    if method in ("snapkv", "pyramidkv"):
        # SnapKV / PyramidKV: sink + local as everywhere else, plus a one-shot
        # observation-window selection frozen for the rest of the request.
        return ResearchAttentionConfig(masker_configs=[
            SinkMaskerConfig(sink_size=SINK), LocalMaskerConfig(window_size=LOCAL),
            SnapKVMaskerConfig(heavy_size=heavy, obs_window=32, pool_kernel=7,
                               pyramid=(method == "pyramidkv"), pyramid_beta=20,
                               num_layers=32)])

    if method in ("vAttention", "vAttention-HAT"):
        # `vatt_split` is the fraction of the total density spent on random sampling.
        # 0.5 reproduces the paper's 50/50 setting; lower values hand budget back to
        # the HAT selector, which matters at low rho where a starved selector — not
        # the sampling estimator — is what breaks retrieval.
        sampling = round(rho * vatt_split, 6)
        h = round(rho - SINK - LOCAL - sampling, 6)
        if h <= 0:
            raise ValueError(f"rho={rho} too small for vatt_split={vatt_split}")
        # (eps, delta) drive the adaptive budget as ~(ppf(1-delta)/eps)^2, so 0.4/0.4
        # allocates ~18x fewer samples than 0.25/0.25.  Any comparison against the
        # oracle-top-k density sweep must use the same pair (0.25/0.25).
        return ResearchAttentionConfig(masker_configs=_prefix_(h) + [
            AdaptiveSamplingMaskerConfig(base_rate_sampling=sampling, epsilon=eps, delta=delta,
                                         init_offset=SINK, local_offset=LOCAL),
        ])

    raise ValueError(f"unknown method: {method}")


def job_name(*a, **kw) -> str:
    """Job name; V54 appends the KV-press tag (e.g. `-ckv0.75`) before the density."""
    name = _job_name_base(*a, **kw)
    pt = _press_tag()
    return name.replace("@", f"{pt}@", 1) if pt else name


def _job_name_base(method: str, rho: float, num_bins: int, kappa: str,
                   vatt_split: float = 0.5, eps: float = 0.4, delta: float = 0.4,
                   selector: str = "hat", bin_mode: str = "auto",
                   expand_c: float = 3.0) -> str:
    tag = f"{rho * 100:g}pct"
    sel = {"hat": "HAT", "oracle": "oracle", "static": _static_tag(), "none": "nosel",
           "lsh": f"LSHk{MP_KW['lsh_k']}l{MP_KW['lsh_l']}"}[selector]
    if selector != "hat" and method not in ("selector", "UTA", "ours", "vAttention"):
        tag = f"{tag}-{sel}"          # non-HAT selector is part of the job name
    mode = BIN_MODE_FOR[selector] if bin_mode == "auto" else bin_mode

    # --- new selector-aware names ---
    if method == "ours":
        suffix = "" if kappa == "j2" else f"_{kappa}"
        # name the bin criterion whenever it is not the selector's free default
        tail = "" if mode == BIN_MODE_FOR[selector] else f"-{mode}"
        return f"ours-b{num_bins}{tail}{suffix}({sel})@{tag}"
    if method in ("selector", "UTA"):
        label = sel if method == "selector" else f"UTA({sel})"
        return f"{label}@{tag}"
    if method in ("UTA+multibin", "UTA+multibin+jensen"):
        return f"{method}(b{num_bins})@{tag}"
    if method == "UTA+randbin":
        return f"UTA+randbin(n{num_bins})@{tag}"
    if method == "UTA+expand":
        return f"UTA+expand(m{num_bins}c{expand_c:g})@{tag}"
    if method == "UTA+expand+triton":
        return f"UTA+expand-triton(m{num_bins}c{expand_c:g})@{tag}"
    if method == "UTA+expand+nosel":
        return f"UTA+expand-nosel-triton(m{num_bins}c{expand_c:g})@{tag}"
    if method.startswith("UTA+sketch"):
        return f"{method}(m{num_bins}c{expand_c:g})@{tag}"
    if method == "UTA+expand+matched":
        return f"UTA+expand-matched(m{num_bins}c{expand_c:g})@{tag}"
    if method == "UTA+expandrand":
        return f"UTA+expandrand(m{num_bins})@{tag}"
    if method == "UTA+expandpfm":
        return f"UTA+expand-pfm{PROFILE_KW['profile_C']}(m{num_bins}c{expand_c:g}{_profile_tag()})@{tag}"
    if method == "UTA+fair":
        C = int(PROFILE_KW["profile_C"]); E = int(FAIR_KW["expand_tokens"])
        bc = f"t{int(FAIR_KW['bin_cap'])}" if int(FAIR_KW.get("bin_cap", 0)) > 0 else ""
        tr = "" if FAIR_KW["truncate"] else "-notrunc"
        pr = "" if FAIR_KW["proxy_rows"] == 1.5 else f"-pr{FAIR_KW['proxy_rows']:g}"
        if not (SINK == 0.001 and LOCAL == 0.001):
            pr += f"-sl{SINK * 1000:g}+{LOCAL * 1000:g}"
        if SINK_LOCAL_TOKENS > 0:
            hm = round(rho - SINK - LOCAL - E / CTX_LEN - FAIR_KW["proxy_rows"] * C / CTX_LEN, 6)
            pr += f"-sl{_abs_or_frac(SINK)}tokh{_abs_heavy(hm)}"
        bm = FAIR_KW["bin_mode"]
        ptag = _profile_tag() if bm == "profile" else {"score": ",score", "equalcount": ",adj"}[bm]
        rk = "" if FAIR_KW["rank"] == "bound" else f"-{FAIR_KW['rank']}"
        vm = {"diag": "", "exact": "-varexact", "lowrank": f"-varlr{int(FAIR_KW['var_rank'])}"}[FAIR_KW["var_mode"]]
        tp = f"-2pt{FAIR_KW['rep_k']}" if FAIR_KW["two_point"] else ""
        tp += "" if FAIR_KW["kappa"] == "none" else f"-{FAIR_KW['kappa']}"
        if str(FAIR_KW["kappa"]).startswith("jsamp"):
            tp += str(FAIR_KW.get("kappa_samples", 8))
        if float(FAIR_KW.get("kappa_scale", 1.0)) != 1.0:
            tp += f"-ks{float(FAIR_KW['kappa_scale']):g}"
        tp += "" if FAIR_KW.get("rep_pick", "whiten") == "whiten" else "-selpick"
        if str(FAIR_KW.get("residual_stats", "none")) != "none":
            tp += "-resid" + str(FAIR_KW["residual_stats"])
        if str(FAIR_KW.get("expand_alloc", "uniform")) != "uniform":
            tp += f"-{FAIR_KW['expand_alloc']}g{float(FAIR_KW.get('alloc_gamma', 1.0)):g}"
        if FAIR_KW.get("evict_mode", "none") != "none":
            et = int(FAIR_KW.get("evict_tokens", 0))
            tp += (f"-{FAIR_KW['evict_mode']}t{et}" if et > 0
                   else f"-{FAIR_KW['evict_mode']}{float(FAIR_KW['evict_budget']) * 100:g}")
        return f"OursFair(C{C}e{E}{bc}c{expand_c:g}{ptag}){tr}{pr}{rk}{vm}{tp}@{tag}"
    if method == "magicpig":
        return f"MagicPIG(K{MP_KW['lsh_k']}L{MP_KW['lsh_l']})@{tag}"
    if method in ("UTA+expandpf", "UTA+expandpfmass"):
        rule = "mass" if method.endswith("mass") else f"c{expand_c:g}"
        return f"UTA+expand-pf1024(m{num_bins}{rule}{_profile_tag()})@{tag}"
    if method == "UTA+qmeanexp":
        return f"UTA+qmeanexp(t{num_bins * 32}{_profile_tag()})@{tag}"
    if method in ("expand+noagg", "expand+agg1"):
        return f"{method}(m{num_bins}c{expand_c:g})@{tag}"
    if method == "UTA+expandhead":
        return f"UTA+expand-perhead(mbar{num_bins})@{tag}"
    if method in ("UTA+expandmass", "UTA+expandmassj2"):
        rule = "mass" if method == "UTA+expandmass" else "massj2"
        return f"UTA+expand-{rule}(m{num_bins})@{tag}"
    if method == "vAttention":
        parts = "" if (eps, delta) == (0.25, 0.25) else f"-e{eps:g}d{delta:g}"
        if vatt_split != 0.5:
            parts += f"-s{vatt_split:g}"
        return f"vAttention({sel}){parts}@{tag}"
    if method == "streamingllm":
        return f"StreamingLLM@{tag}"
    if method == "snapkv":
        return f"SnapKV@{tag}"
    if method == "pyramidkv":
        return f"PyramidKV@{tag}"
    if method == "streamingllm-paper":
        return f"StreamingLLMpaper{KV_BUDGET}@{tag}"
    if method == "snapkv-paper":
        return f"SnapKVpaper{KV_BUDGET}@{tag}"
    if method == "pyramidkv-paper":
        return f"PyramidKVpaper{KV_BUDGET}@{tag}"

    # --- legacy names: keep byte-identical so finished 1%/3% dirs still resume ---
    if method == "MultiBin-HAT":
        suffix = "" if kappa == "none" else f"_{kappa}"
        return f"MultiBin{num_bins}{suffix}-HAT@{tag}"
    if method == "vAttention-HAT":
        parts = ""
        if vatt_split != 0.5:
            parts += f"-s{vatt_split:g}"
        if (eps, delta) != (0.4, 0.4):
            parts += f"-e{eps:g}d{delta:g}"
        return f"vAttention-HAT{parts}@{tag}"
    return f"{method}@{tag}"


def _bench(subset: str):
    """V32: RULER at the context length CTX_LEN (hub datasets at 16K/32K, locally generated parquet otherwise)."""
    if CTX_LEN == 32768:
        return Ruler32K(subsets_to_run=[subset])
    if CTX_LEN == 16384:
        from benchmark.ruler16k import Ruler16K
        return Ruler16K(subsets_to_run=[subset])
    from benchmark.rulerlocal import make_ruler_local
    return make_ruler_local(CTX_LEN, subsets_to_run=[subset])


def _finished(task_dir: Path, max_requests: int) -> bool:
    """A task counts as finished only if its metrics exist AND its config.json records the same
    --max-requests.  Whenever a task directory already carries a config.json or metrics.json for
    a different (or unverifiable) request count -- e.g. a smoke test in the same output
    directory -- the run stops: those files are never reused and never overwritten."""
    cfg, met = task_dir / "config.json", task_dir / "metrics.json"
    if not cfg.exists() and not met.exists():
        return False
    try:
        done = json.loads(cfg.read_text())["request_kwargs"]["max_requests"]
    except (OSError, ValueError, KeyError, TypeError):
        done = None
    if done != max_requests:
        sys.exit(f"[ERROR] {task_dir} already holds results for max_requests={done}, but this run "
                 f"asks for max_requests={max_requests}. Results for a different request count are "
                 f"never reused; choose another --output-dir or remove that directory.")
    return met.exists()


def run_job(name: str, config, args) -> None:
    method_dir = Path(args.output_dir) / name.replace("@", "_at_")
    want = ([s.strip() for s in args.subsets.split(",")]
            if getattr(args, "subsets", None) else SUBSETS)
    pending = [s for s in want if not _finished(method_dir / s, args.max_requests)]
    if not pending:
        print(f"[SKIP] {name} — complete")
        return

    print(f"\n{'=' * 72}\n[{args.device}] {name} — {len(pending)} subset(s)\n{'=' * 72}", flush=True)
    if getattr(args, "log_density", False) or getattr(args, "log_error", False):
        from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
        method_dir.mkdir(parents=True, exist_ok=True)
        # V30: --log-error adds the dense-vs-sparse output error (vAttention-style parameter search)
        MicroMetricLogger().configure_logging(
            log_path=str(method_dir),
            enabled_metrics=["research_attention_density", "uta_expand_frac", "uta_proxy_bins"]
                            + (["research_attention_output_error"] if getattr(args, "log_error", False) else []),
            sampling_factor=args.density_sampling)
    adapter = ModelAdapterHF(
        model_name=args.model,
        sparse_attention_config=config,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=args.device,
    )
    adapter.kv_press = _make_press()          # V54: None unless --kv-press is set
    try:
        for subset in pending:
            d = method_dir / subset
            d.mkdir(parents=True, exist_ok=True)
            print(f"  -> {name}/{subset}", flush=True)
            _bench(subset).run_benchmark(
                adapter, str(d),
                request_kwargs={"max_requests": args.max_requests, "max_context_length": CTX_LEN - 768},
                generation_kwargs={"max_new_tokens": 120},
            )
            mf = d / "metrics.json"
            if mf.exists():
                s = json.loads(mf.read_text()).get("task_scores", {}).get(subset, {}).get("string_match", 0.0)
                print(f"     {subset}: {s:.2f}", flush=True)
    finally:
        if getattr(args, "log_density", False) or getattr(args, "log_error", False):
            from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
            MicroMetricLogger().flush()
        del adapter
        torch.cuda.empty_cache()


# Finished runs elsewhere in the repo that use the SAME protocol (sink/local 0.001,
# eps=delta=0.25, 200 samples) and can therefore be quoted next to the new ones.
REUSABLE = {
    "vAttention(oracle)@5pct": "results_covuta/vAttention_at_5pct",
    "vAttention(oracle)@10pct": "results_covuta/vAttention_at_10pct",
    "UTA(oracle)@5pct": "results_covuta/UTA_at_5pct",
    "UTA(oracle)@10pct": "results_covuta/UTA_at_10pct",
    "oracle@5pct": "results_covuta/oracle_top_k_at_5pct",
    "oracle@10pct": "results_covuta/oracle_top_k_at_10pct",
}


def summarize(output_dir: Path, include_reusable: bool = True) -> None:
    dirs = [(d.name.replace("_at_", "@"), d) for d in sorted(output_dir.glob("*_at_*"))]
    if include_reusable:
        dirs += [(label, Path(p)) for label, p in REUSABLE.items() if Path(p).is_dir()]

    rows = []
    for name, d in sorted(dirs):
        row = {"Method": name}
        scores = []
        for subset in SUBSETS:
            mf = d / subset / "metrics.json"
            if mf.exists():
                s = json.loads(mf.read_text()).get("task_scores", {}).get(subset, {}).get("string_match", 0.0)
                row[subset] = round(s, 2)
                scores.append(s)
            else:
                row[subset] = None
        row["Average"] = round(sum(scores) / len(scores), 2) if len(scores) == len(SUBSETS) else None
        rows.append(row)

    if not rows:
        print("no results yet")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[["Method", "Average"] + SUBSETS]
    print("\n" + "=" * 104)
    print("MULTI-BIN UTA on HAT — RULER32K")
    print("=" * 104)
    print(df.to_string(index=False))
    out = output_dir / "multibin_hat_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-bin UTA with a HAT selector")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--output-dir", default="./results_multibin_hat")
    p.add_argument("--max-requests", type=int, default=200)
    p.add_argument("--densities", default="0.05,0.10")
    p.add_argument("--bin-mode", default="auto",
                   choices=["auto", "score", "hat", "fixed", "equalcount"],
                   help="tail bin criterion; 'auto' uses the selector's free score "
                        "(see BIN_MODE_FOR). Under a HAT selector 'hat' is measurably "
                        "bad -- prefer 'score' (accuracy ceiling) or 'fixed'.")
    p.add_argument("--selector", default="hat", choices=["hat", "oracle", "static", "none", "lsh"],
                   help="top-k selector; also picks the tail bin criterion "
                        "(hat -> HAT approx scores, oracle -> true logits)")
    p.add_argument("--methods", default="ours,vAttention",
                   help="ours | vAttention | UTA | selector (legacy: MultiBin-HAT, ...)")
    p.add_argument("--bin-sizes", default="16,32",
                   help="comma-separated num_bins for 'ours' (bin_size for legacy MultiBin-HAT)")
    p.add_argument("--no-reusable", action="store_true",
                   help="omit already-finished protocol-compatible runs from the summary")
    p.add_argument("--kappa", default="j2", choices=["none", "j2"],
                   help="per-bin Jensen correction on the bin logit (denominator side)")
    p.add_argument("--eps", type=float, default=0.4,
                   help="vAttention adaptive-sampling error bound (0.25 matches the "
                        "oracle-top-k density sweep; 0.4 matches reproduce_table1)")
    p.add_argument("--delta", type=float, default=0.4,
                   help="vAttention adaptive-sampling confidence bound")
    p.add_argument("--vatt-split", type=float, default=0.5,
                   help="fraction of total density vAttention spends on random sampling "
                        "(0.5 = paper setting; lower gives the HAT selector more budget)")
    p.add_argument("--expand-c", type=float, default=3.0,
                   help="bins are ranked by zbar_b + c*sigma_b for the expansion stage")
    p.add_argument("--subsets", default=None,
                   help="comma-separated subset list (default: all 7); lets two "
                        "GPUs split one job's subsets across processes")
    p.add_argument("--profile-P", type=int, default=64)
    p.add_argument("--profile-iters", type=int, default=4)
    p.add_argument("--profile-dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--profile-method", default="kmeans", choices=["kmeans", "qsort", "grid", "geo", "geo2", "keykm", "retro", "centroidkv"])
    p.add_argument("--profile-source", default="question",
                   choices=["question", "question_rot", "static", "static+question", "static+rows", "static+qrot"],
                   help="V23: where the query metric behind the profile partition comes from")
    p.add_argument("--profile-static-file", default=None,
                   help="pre-RoPE per-(layer,head) M file from analyse_v23_qgeom.py (also carries RoPE params)")
    p.add_argument("--profile-rank", type=int, default=8)
    p.add_argument("--profile-rot-window", type=int, default=0)
    p.add_argument("--profile-rank-q", type=int, default=8, help="question-block rank for static+qrot")
    p.add_argument("--profile-C", type=int, default=1024, help="number of profile clusters (bins)")
    p.add_argument("--profile-basis", default="eig", choices=["eig", "random"],
                   help="V57: 'random' clusters the keys under a random orthonormal frame of rank profile-rank "
                        "instead of the query eigenvectors (Table clustinput)")
    p.add_argument("--expand-tokens", type=int, default=0,
                   help="V26 UTA+fair: exact expansion budget in tokens (0 = no expansion stage)")
    p.add_argument("--kv-budget-tokens", type=int, default=0,
                   help="V39: absolute average KV cache size per layer for the "
                        "snapkv-paper / pyramidkv-paper methods and for --evict-tokens")
    p.add_argument("--residual-stats", default="none",
                   choices=["none", "value", "full", "fullvar", "wexact"],
                   help="V42: rebuild the proxy's moments over the tokens still proxied "
                        "after the capped expansion. 'value' fixes the centroid only, "
                        "'full' also the mass; 'none' is the deployed behaviour")
    p.add_argument("--expand-alloc", default="uniform", choices=["uniform", "hier"],
                   help="V47: split of the expansion budget over ranked bins. 'uniform' "
                        "gives each taken bin min(n_b, t); 'hier' water-fills "
                        "q_b = min(n_b, lam * exp(gamma * (score_b - max score)))")
    p.add_argument("--expand-alloc-gamma", type=float, default=1.0,
                   help="V47: sharpness of the hierarchical split (1 = error-proportional)")
    p.add_argument("--evict-tokens", type=int, default=0,
                   help="V39: same, for composing BinAttention with the compressed cache "
                        "(overrides --evict-budget when > 0)")
    p.add_argument("--evict-mode", default="none", choices=["none", "snapkv", "pyramidkv", "streamingllm"],
                   help="V38 UTA+fair: compose with KV-cache compression; the eviction pass "
                        "runs once per request and its rows are hidden from every stage")
    p.add_argument("--evict-budget", type=float, default=0.0,
                   help="V38: fraction of the context the compression keeps resident (0.5 = 50%%)")
    p.add_argument("--kv-press", default="none",
                   choices=["none", "centroidkv", "snapkv", "pyramidkv", "streamingllm"],
                   help="V54: rewrite the prefilled context cache with a KV press before the "
                        "question is seen (kvpress protocol); composes with any --methods. "
                        "V54c: snapkv/pyramidkv/streamingllm = the eviction compressors under "
                        "that split (absolute --kv-press-budget entries per layer)")
    p.add_argument("--kv-press-budget", type=int, default=8000,
                   help="V54c: per-layer entry budget of an eviction press")
    p.add_argument("--kv-press-ratio", type=float, default=0.75,
                   help="V54: CentroidKV compression_ratio (fraction of entries removed; 0.75 keeps 25%%)")
    p.add_argument("--kv-press-sink", type=int, default=16)
    p.add_argument("--kv-press-local", type=int, default=64)
    p.add_argument("--kv-press-chunk", type=int, default=256)
    p.add_argument("--kv-press-merge", type=float, default=0.8)
    p.add_argument("--kv-press-decay", type=float, default=0.2)
    p.add_argument("--sink-local-tokens", type=int, default=0,
                   help="V54: pin the sink and local maskers to max(N, frac*CTX_LEN) TOKENS instead "
                        "of a fraction of the (possibly pressed) cache length")
    p.add_argument("--kv-press-when", default="context", choices=["context", "prompt"],
                   help="V54b: 'context' = kvpress protocol (context merged before the question is "
                        "seen); 'prompt' = the table's eviction protocol (whole prompt encoded "
                        "densely, the stored cache merged, only generation reads it)")
    p.add_argument("--kv-press-logcount", action="store_true",
                   help="V54: attend to centroids with the paper's log N term (Eq. 3); the "
                        "release drops the counts, which is the default here")
    p.add_argument("--expand-bin-cap", type=int, default=0,
                   help="V35d UTA+fair: spread the expansion budget over bins, at most this "
                        "many tokens (top by selector score) per bin (0 = whole-bin expansion)")
    p.add_argument("--proxy-rows", type=float, default=1.5,
                   help="V26 UTA+fair: (k,v)-row equivalents charged per bin per query")
    p.add_argument("--fair-kappa",
                   choices=["none", "j1", "j2", "jmax", "jtopm", "jsamp", "jsampm", "jsampmq"],
                   default="none",
                   help="V26 UTA+fair: per-bin Jensen correction on the tail "
                        "residual. jmax sizes it from the bin maximum the rep_k "
                        "round already scored, which needs --two-point")
    p.add_argument("--two-point", action="store_true",
                   help="represent each surviving bin by (representative, residual) "
                        "instead of one value mean; still ONE virtual token in the "
                        "merge, so proxy_rows accounting is unchanged")
    p.add_argument("--rep-k", type=int, default=1,
                   help="with --two-point: pick the representative from the top rep_k "
                        "query-blind candidates by whitened key norm, resolving the tie "
                        "with rep_k real dot products at decode (rep_k=1 is the plain "
                        "argmax and is bit-identical to the single-candidate path)")
    p.add_argument("--kappa-samples", type=int, default=8,
                   help="jsamp/jsampm: keys sampled per bin at a deterministic "
                        "uniform stride (JSAMPM_METHOD.md recommends 4)")
    p.add_argument("--kappa-scale", type=float, default=1.0,
                   help="damping on the kappa correction (ell = zbar + log n + "
                        "scale*kappa); 1.0 is bit-identical to no damping")
    p.add_argument("--rep-pick", choices=["whiten", "selector"], default="whiten",
                   help="with --two-point: rank the rep_k candidates by the whitened "
                        "key norm (old behaviour) or by the selector's per-token score "
                        "(logit-aware; needs a score-publishing selector, e.g. static)")
    p.add_argument("--no-expand-truncate", action="store_true",
                   help="V26 UTA+fair: take the crossing bin whole (legacy overshoot) instead of truncating")
    p.add_argument("--fair-bin-mode", default="profile", choices=["profile", "score", "equalcount"],
                   help="V28 UTA+fair partition: profile (ours) | score (oracle logit bins) | equalcount (adjacency)")
    p.add_argument("--ctx-len", type=int, default=32768,
                   help="V32: RULER context length (16384 hub / 32768 hub / 65536, 98304 local); sets the "
                        "density accounting length and the benchmark class")
    p.add_argument("--expand-rank", default="bound",
                   choices=["bound", "mass", "massj2", "ucblogn", "boundlogn", "random",
                            "var", "m2raw", "m2", "expvar", "expvarn", "jgap", "wl1",
                            "jgapv", "errmix", "errv", "outerr"],
                   help="V28/V36 UTA+fair expansion ranking: bound (zbar+c sigma, ours) | "
                        "mass (zbar+log n, Double-P style) | var (sigma^2 only) | m2raw (uncentred "
                        "diagonal second moment) | m2 (zbar^2+sigma^2) | expvar/expvarn (weight-domain "
                        "sd, per token / per bin) | jgap (log missing mass) | wl1 (softmax weight-vector "
                        "error of the proxy, n_b e^zbar D_b) | errmix/errv/outerr "
                        "(V36b: both error channels, with 0 / mass-weighted / fully weighted value factors)")
    p.add_argument("--var-mode", default="diag", choices=["diag", "exact", "lowrank"],
                   help="V28 UTA+fair sigma_b^2: diag | exact (= full covariance) | lowrank (+var-rank factors)")
    p.add_argument("--var-rank", type=int, default=4)
    p.add_argument("--static-file", default=STATIC_KW["static_file"],
                   help="static query metric M_st used by the QGS selector (restored by assets/restore_query_metric.py)")
    p.add_argument("--static-rank", type=int, default=16)
    p.add_argument("--static-window", type=int, default=32)
    p.add_argument("--static-source", default="static+question", choices=["static", "static+question"])
    p.add_argument("--static-P", type=int, default=64)
    p.add_argument("--static-share-kv", action="store_true", help="one U per KV head (codes stored per KV head)")
    p.add_argument("--static-bits", type=int, default=16, choices=[16, 8])
    p.add_argument("--static-rebuild", type=int, default=0, help="rebuild U every N generated tokens (0 = never)")
    p.add_argument("--static-basis", default="eig", choices=["eig", "random"],
                   help="V49: 'random' replaces the query eigenvectors by a random orthonormal frame")
    p.add_argument("--lsh-k", type=int, default=10, help="MagicPIG bits per LSH table")
    p.add_argument("--lsh-l", type=int, default=150, help="MagicPIG number of LSH tables")
    p.add_argument("--log-density", action="store_true",
                   help="record realised per-layer density (research_attention_density) into "
                        "<method_dir>/micro_metrics.jsonl -- use to calibrate MagicPIG (K, L)")
    p.add_argument("--density-sampling", type=float, default=0.05,
                   help="sampling factor for --log-density")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--log-error", action="store_true",
                   help="V30: also log research_attention_output_error (dense-vs-sparse output error per layer call)")
    p.add_argument("--run-name", default=None,
                   help="V30: override the job name (single job only), e.g. 'vAttention(HAT)@10pct'")
    p.add_argument("--sink", type=float, default=0.001,
                   help="V34: sink share of the context (default 0.1%%); charged against rho")
    p.add_argument("--local", type=float, default=0.001,
                   help="V34: local-window share of the context (default 0.1%%); charged against rho")
    args = p.parse_args()
    global SINK, LOCAL, SINK_LOCAL_TOKENS
    SINK = float(args.sink)
    LOCAL = float(args.local)
    SINK_LOCAL_TOKENS = int(args.sink_local_tokens)
    PROFILE_KW.update(profile_P=args.profile_P, profile_iters=args.profile_iters,
                      profile_dtype=args.profile_dtype,
                      profile_method=args.profile_method,
                      profile_source=args.profile_source,
                      profile_static_file=args.profile_static_file,
                      profile_rank=args.profile_rank,
                      profile_rot_window=args.profile_rot_window,
                      profile_rank_q=args.profile_rank_q, profile_C=args.profile_C,
                      profile_basis=args.profile_basis)
    global KV_BUDGET
    KV_BUDGET = int(args.kv_budget_tokens)
    PRESS_KW.update(mode=str(args.kv_press), ratio=float(args.kv_press_ratio),
                    sink=int(args.kv_press_sink), local=int(args.kv_press_local),
                    chunk=int(args.kv_press_chunk), merge=float(args.kv_press_merge),
                    decay=float(args.kv_press_decay), logcount=bool(args.kv_press_logcount),
                    when=str(args.kv_press_when), budget=int(args.kv_press_budget))
    FAIR_KW.update(expand_tokens=args.expand_tokens, bin_cap=args.expand_bin_cap,
                   proxy_rows=args.proxy_rows,
                   residual_stats=str(args.residual_stats),
                   expand_alloc=str(args.expand_alloc),
                   alloc_gamma=float(args.expand_alloc_gamma),
                   evict_mode=args.evict_mode, evict_budget=args.evict_budget,
                   evict_tokens=int(args.evict_tokens),
                   truncate=not args.no_expand_truncate, bin_mode=args.fair_bin_mode,
                   rank=args.expand_rank, var_mode=args.var_mode, var_rank=args.var_rank,
                   two_point=bool(args.two_point), rep_k=int(args.rep_k),
                   kappa=str(args.fair_kappa), rep_pick=str(args.rep_pick),
                   kappa_samples=int(args.kappa_samples),
                   kappa_scale=float(args.kappa_scale))
    if args.model in MODEL_HAT:
        wf, bits, layers = MODEL_HAT[args.model]
        HAT_KW.update(hat_weight_file=wf, hat_bits=bits, hat_mlp_layers=layers)
    elif args.selector == "hat":
        raise ValueError(f"no HashAttention weights registered for {args.model} (MODEL_HAT)")

    MP_KW.update(lsh_k=args.lsh_k, lsh_l=args.lsh_l)
    global CTX_LEN
    CTX_LEN = int(args.ctx_len)
    STATIC_KW.update(static_file=args.static_file, rank=args.static_rank, rot_window=args.static_window,
                     source=args.static_source, profile_P=args.static_P, share_kv=args.static_share_kv,
                     code_bits=args.static_bits, rebuild_every=args.static_rebuild,
                     basis=args.static_basis)

    output_dir = Path(args.output_dir)
    if args.summarize_only:
        summarize(output_dir, include_reusable=not args.no_reusable)
        return

    densities = [float(x) for x in args.densities.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    bin_sizes = [int(x) for x in args.bin_sizes.split(",")]

    if args.selector == "hat" and methods != ["dense"] and not Path(HAT_KW["hat_weight_file"]).exists():
        raise FileNotFoundError(f"HAT weights not found: {HAT_KW['hat_weight_file']}")

    jobs = []
    for rho in densities:
        for m in methods:
            sizes = bin_sizes if (m.startswith("UTA+sketch") or m in (
                                       "ours", "MultiBin-HAT", "UTA+multibin",
                                       "UTA+multibin+jensen", "UTA+randbin",
                                       "UTA+expand", "UTA+expandrand", "UTA+expandmass", "UTA+expandmassj2", "UTA+expandhead", "expand+noagg", "expand+agg1",
                                       "UTA+expandpf", "UTA+expandpfmass", "UTA+expandpfm", "UTA+qmeanexp",
                                       "UTA+expand+triton", "UTA+expand+nosel",
                                       "UTA+expand+matched")) else [0]
            for bs in sizes:
                # NOTE: expand_c must be a KEYWORD here.  Passed positionally it
                # lands on build_config's var_mode slot (job_name has no var_mode
                # param), so the label says c=X while the run silently uses c=3.
                common = (m, rho, bs, args.kappa, args.vatt_split,
                          args.eps, args.delta, args.selector, args.bin_mode)
                jobs.append((job_name(*common, expand_c=args.expand_c),
                             build_config(*common, expand_c=args.expand_c)))

    if getattr(args, "run_name", None):
        assert len(jobs) == 1, "--run-name needs exactly one job"
        jobs = [(args.run_name, jobs[0][1])]
    print(f"[{args.device}] {len(jobs)} job(s):")
    for n, _ in jobs:
        print(f"   - {n}")

    for name, cfg in jobs:
        run_job(name, cfg, args)

    summarize(output_dir, include_reusable=not args.no_reusable)


if __name__ == "__main__":
    main()
