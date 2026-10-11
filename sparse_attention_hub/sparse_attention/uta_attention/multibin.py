"""UTA with multi-bin tail aggregation.

Base UTA collapses the whole tail into ONE proxy:

    S_T ~= N * exp(s * q.k_mean),      mu_T ~= v_mean

which is exact only if every tail logit is identical.  Its error is driven by the
within-tail logit variance.  Multi-bin UTA instead partitions the tail into bins
of adjacent key positions and gives each bin its own proxy:

    logit_b = s * (q . k_bar_b) + log(n_b)          <-- n_b is the bin's OWN count
    S_T     ~= sum_b exp(logit_b)
    num_T   ~= sum_b exp(logit_b) * v_bar_b

Key properties
--------------
* n_b, not N.  Each bin carries its own log-count; sum_b n_b == N is preserved.
* One global softmax.  Bin logits join the heavy-token logits in a single
  numerator/denominator merge -- never a per-bin softmax.
* Bins with n_b == 1 are EXACT (k_bar_b = k_i, v_bar_b = v_i, log 1 = 0), so bin
  size is a clean accuracy dial whose limit is exact attention.
* Bins with n_b == 0 contribute nothing (skipped, no log(0)).
* Residual Jensen bias per bin is ~sigma_b^2/2, which shrinks with bin size --
  partitioning attacks the bias at its source rather than correcting it.
  `kappa_mode="j2"` additionally adds log(1 + sigma_b^2/2) (moment form, bounded);
  it is OFF by default.

Prototype vs deployable
-----------------------
Under the default var_mode="diag" the bin statistics are computed the deployable
way: from the bin's own key moments, never from the score matrix.

    z_bar_b   = s * q . k_bar_b          <-- identical to the mean of the true logits
                                             over the bin's tail members, since
                                             mean_i s*q.k_i = s*q.mean_i k_i
    sigma_b^2 = s^2 * sum_d q_d^2 Var_b(k_d)

k_bar_b and Var_b(k) are query-independent, so a kernel gets them from per-block
sums cached once with the KV pages,
    k_bar_b = (sum_k[b] - sum_{i in b, selected} k_i) / n_b
at O(n_bins*d + k*d) per query.  The math in this file is the math that kernel
implements, and now so is the data flow.

The score matrix is still materialised in this file, but only for the parts a
sparse kernel computes anyway: the logits of the SELECTED tokens, which enter the
softmax exactly.  Restricting that to a gather over the selected indices is a
kernel change, not a math change.

var_mode="exact" is the exception and is deliberately circular: the true
within-bin logit variance cannot be recovered from key-side statistics, so that
mode reads the full score matrix.  It exists as an accuracy ceiling, not as a
deployable configuration.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from ..base import SparseAttentionConfig
from ..research_attention.maskers.base import ResearchMasker
from ..utils.kv_compress import default_window, select_resident
from ..utils.kv_utils import _get_num_key_value_groups, repeat_kv
from ..utils.mask import Mask
from .base import UTAAttention, UTAAttentionConfig

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger

# V26: realised per-query cost of the tail stage (fraction of K expanded exactly;
# number of live proxies), for total-traffic accounting next to research_attention_density.
MicroMetricLogger.register_metric("uta_expand_frac", float)
MicroMetricLogger.register_metric("uta_proxy_bins", float)
from sparse_attention_hub.metric_logging.stage_timer import stage

NEG = -1e4

# How the tail is partitioned.  "fixed"/"equalcount" are parameterised by `bin_size`
# (tokens per bin); "score"/"hat"/"random" are parameterised by `num_bins` (bins per row).
# "profile" (V19) clusters tokens by their logit profile against P sampled query rows
# of the question chunk -- query-DISTRIBUTION aware but query-independent per request:
# built once at the question-chunk call, cached per layer, extended incrementally at
# decode by assigning each new key to its nearest cached centroid.
BIN_MODES = ("fixed", "equalcount", "score", "hat", "topblock", "random", "profile")

# How sigma_b^2 is obtained.  This is the (2) axis: the true within-bin logit variance
# needs every tail logit (circular), whereas
#     sigma_b^2 = s^2 * q^T Cov_b(K) q
# needs only key-side statistics, which are query-independent and cacheable.
#   "exact" : variance of the true logits            -- prototype ceiling, circular
#   "diag"  : s^2 * sum_d q_d^2 Var_b(k_d)           -- drops cross-dim covariance,
#             O(d) per bin, storage d per bin (same footprint as the bin mean)
#   "lowrank": diag + top-r eigen-directions of the bin's key covariance (V28):
#             sigma_b^2 = s^2 [ sum_d q_d^2 (Var_b(k_d) - Doff_b,d)^+ + sum_j (q . L_bj)^2 ],
#             L_b = U_r Lambda_r^{1/2} of Cov_b(K), Doff = diag(L L^T); r extra d-vectors per
#             bin, built once per request from the partition (profile mode only).
# Full covariance is deliberately absent as a DEPLOYABLE mode: B*d^2 per query is ~10x
# vAttention's sampling cost at B=256, d=128, and B*d^2 floats per head is far past any
# KV-cache budget.  Note that "exact" IS the full-covariance answer: logits are linear in
# k, so Var_b(z) = s^2 q^T Cov_b(K) q exactly -- the exact mode is what a full covariance
# would reproduce, at the cost of reading every tail logit.
VAR_MODES = ("exact", "diag", "lowrank")

# Jensen correction K_b(1) = log E[exp(z - zbar_b)]:
#   "none" : 0                      -- always under-estimates the tail mass
#   "j1"   : sigma^2/2              -- exact iff z is Gaussian; unbounded in sigma
#   "j2"   : log(1 + sigma^2/2)     -- moment expansion; agrees with j1 to O(sigma^2),
#                                      grows only logarithmically afterwards
KAPPA_MODES = ("none", "j1", "j2", "jmax", "jtopm", "jsamp", "jsampm", "jsampmq")
PROFILE_METHODS = ("kmeans", "qsort", "grid", "geo", "geo2", "keykm", "retro", "centroidkv")
PROFILE_SOURCES = ("question", "question_rot", "static", "static+question", "static+rows", "static+qrot")


# V50 tuning probe (env UTA_ERRLOG=<path.npz>): per-row relative output error vs
# dense attention ("dec" decode rows, "qc" question-chunk rows) and the sorted-bin
# error curve ("curve"), dumped once at interpreter exit.
_ERRLOG: Dict[str, list] = {}


def _dump_errlog() -> None:
    path = os.environ.get("UTA_ERRLOG")
    if not path or not _ERRLOG:
        return
    import numpy as _np
    out = {}
    for k_, v_ in _ERRLOG.items():
        if k_ == "curve":
            w = max(a.shape[-1] for a in v_)
            out[k_] = _np.concatenate([_np.pad(a.astype(_np.float32), ((0, 0), (0, w - a.shape[-1])),
                                               constant_values=_np.nan) for a in v_])
        else:
            out[k_] = _np.concatenate(v_)
    _np.savez_compressed(path, **out)


import atexit as _atexit
_atexit.register(_dump_errlog)


@dataclass
class UTAMultiBinConfig(UTAAttentionConfig):
    """Config for multi-bin UTA.

    Attributes:
        bin_size: target number of key positions per bin.  Smaller = more proxies
            = more accurate = more per-query work.  bin_size=1 is exact attention.
        num_bins: number of bins per row for the "score"/"hat" modes.
        bin_mode: how the tail is partitioned.
            "fixed"      contiguous key-position blocks intersected with the tail.
                         Boundaries are query-independent, so bin statistics are
                         cacheable alongside the KV pages.
            "equalcount" runs of exactly bin_size consecutive tail tokens.
            "score"      equal-width bins over the true tail logit.  Directly
                         minimises within-bin logit variance -- the quantity that
                         drives UTA's error.  Needs every tail logit, so it is only
                         cost-neutral under a selector that already computes them
                         (oracle-top-k).
            "hat"        same, but binned by HashAttention's *approximate* score,
                         which the HAT masker already computes for selection -- so
                         it is cost-neutral under a HAT selector.
        kappa_mode: "none" (default) or "j2" = add log(1 + sigma_b^2/2) per bin.
    """

    bin_size: int = 32
    num_bins: int = 16
    bin_mode: str = "fixed"
    kappa_mode: str = "none"
    var_mode: str = "exact"
    var_rank: int = 4                  # V28: rank r of the low-rank covariance term (var_mode="lowrank")
    refine_blocks: int = 0
    random_seed: int = 1234
    # Second stage (E1/E2): compute the top `expand_m` bins EXACTLY instead of
    # representing them by their mean.  Bins are ranked by zbar_b + expand_c*sigma_b,
    # both key-side, so the ranking needs no logits.  expand_rank="random" is the
    # control that spends the same budget without the ranking.
    #
    # V36 asked what the ranking looks like WITHOUT the free constant c, i.e. as one
    # second-moment number rather than a sum of two quantities in different units:
    #   "var"      sigma_b^2 alone (centred, so mass-blind)          -- the control
    #   "m2raw"    s^2 sum_d q_d^2 E_b[k_d^2] -- the SAME reduction with the centring
    #              dropped, so the mean is inside the number (sign-blind, see below)
    #   "m2"       E_b[z^2] = zbar_b^2 + sigma_b^2 (sign-blind likewise)
    #   "expvar"   log sd_b(e^z): the second moment in the WEIGHT domain, where the
    #              mass enters multiplicatively; "expvarn" is its bin total (x n_b)
    #   "wl1"      V56: the proxy's error in the SOFTMAX WEIGHT VECTOR, token by token,
    #              n_b e^{zbar_b} D_b with D_b = e^{s^2/2} erf(s/sqrt2) ~ sqrt(2/pi) s.
    #              Same cached statistics as "jgap"; first order in sigma, because the
    #              absolute deviations cannot cancel the way the signed Jensen gap does.
    #   "jgap"     log of the mass the mean-proxy actually misses,
    #              n_b e^{zbar_b}(e^{sigma_b^2/2} - 1) -- the error being removed
    # V36b then noticed that "jgap" and "expvarn" are the two channels of ONE first-order
    # output-error expression, and added the rules that keep both:
    #   "errmix"   both channels, value factors set to 1 (no extra state at all)
    #   "errv"     + ||vbar_b - ohat|| weighting the mass channel (no extra state)
    #   "outerr"   + sqrt(tr Cov_b(v)) weighting the shape channel (1 float per bin)
    expand_m: int = 0
    expand_c: float = 3.0
    expand_rank: str = "bound"
    # Drop the proxy term entirely: attend only to {selected} u {expanded}, with no
    # aggregate contribution from the remaining bins.  This is the ablation that
    # isolates what the aggregate is worth -- everything else (selector, partition,
    # ranking, expansion) is bit-identical.
    use_aggregate: bool = True
    # How the UNEXPANDED bins are represented:
    #   "multi"  one proxy per bin (the shipped method)
    #   "single" all surviving bins collapsed into ONE proxy, i.e. base UTA applied to
    #            whatever expansion left behind.  NOTE: this must NOT be expressed as
    #            bin_size=1e9 -- that leaves a single bin, which expansion then expands
    #            in full, silently producing dense attention.
    #   "none"   no proxy at all (equivalent to use_aggregate=False)
    aggregate_mode: str = "multi"
    # Per-(layer, head) expansion budget, calibrated offline: {layer: {head: m}}.
    # 65% of the per-row error variance sits BETWEEN heads rather than within them,
    # so the head is the unit at which a non-uniform budget actually pays.  Being
    # static, it costs nothing at run time -- no per-step decision, no ragged launch --
    # unlike a per-row rule, which makes the kernel pad every row to the batch maximum.
    expand_m_table: Optional[Dict[int, Dict[int, int]]] = None
    # Route the DECODE step through the fused Triton kernel.  Prefill keeps the torch
    # path (the kernel is a one-query-per-program decode kernel).  Numerically this is
    # the same computation; the point of running it end-to-end is to prove that.
    use_triton_decode: bool = False
    # --- bin_mode="profile" (V19) ---------------------------------------------
    # P query rows sampled from the question chunk (excluding its last two rows),
    # C clusters, Lloyd iterations.  Matches the V19 diagnostic configuration.
    profile_P: int = 64
    profile_C: int = 1024
    profile_iters: int = 4
    # STATIC token promotion: the top `profile_promote` tail tokens by cached qmean
    # (mean logit against the profile queries) are moved into the exact set before
    # any bin statistic is taken.  Per-request static -- no per-step ranking.
    profile_promote: int = 0
    # When > 0, the expansion stage is TOKEN-budget equalised instead of top-m bins:
    # bins are taken in rank order while the cumulative token count stays below this
    # budget (the crossing bin is included).  Needed for unequal-size partitions
    # ("profile"); expand_m must still be > 0 to enable the stage.
    expand_budget_tokens: int = 0
    # V26: EXACT token budget.  With expand_budget_tokens > 0 the bin that crosses the
    # budget is not taken whole: only its highest-scoring tokens (by the selector's own
    # per-token score -- static_scores / hat_scores, never the true logits) fill the
    # remaining budget, and the rest of that bin stays a proxy with its count reduced.
    # Realised expansion is then exactly min(budget, tail) tokens, so the total
    # exact-read density is matched to the row (no crossing-bin overshoot).
    expand_truncate: bool = False
    # V35d: per-bin cap on the expansion.  With expand_truncate and a cap t > 0 the
    # budget is SPREAD over bins instead of concentrated: bins are taken in rank order
    # at an effective cost min(n_b, t) each, and every taken bin contributes only its
    # top-min(n_b, t) tokens by the selector's per-token score (the crossing bin its
    # top-remaining).  The rest of each taken bin stays a proxy with its count
    # reduced.  Realised expansion is still exactly min(budget, sum min(n_b, t)).
    expand_bin_cap: int = 0
    # V47: how the capped path splits the expansion budget E over the ranked bins.
    #   "uniform"  V35d: every taken bin gets the same min(n_b, t), in rank order
    #   "hier"     hierarchical: the budget follows the ranking score itself.  With
    #              w_b = exp(gamma * (score_b - max_b score_b)) -- for an error-type
    #              rank (outerr, jgap) score_b is the log of the bin's estimated error,
    #              so w_b is its error relative to the worst bin -- each bin gets
    #              q_b = min(n_b, lambda * w_b) tokens by water-filling, lambda set so
    #              sum_b q_b = E exactly (leftover whole tokens go one each to the
    #              highest-ranked bins with room).  t is not used.  gamma = 1 is
    #              error-proportional, gamma -> 0 flattens toward an equal split.
    # The per-bin counts differ, so every proxy's log n_b is its OWN residual count
    # (n_b - q_b): the merge already reads the reduced n_b, and a bin read out in full
    # leaves the proxy set.
    expand_alloc: str = "uniform"
    expand_alloc_gamma: float = 1.0
    # V42: after the capped expansion has made a bin's top-t tokens exact, the bin's
    # remaining tokens stay a proxy.  Its COUNT is reduced (n_b - kept), but zbar_b and
    # vbar_b are still the statistics of the WHOLE bin, including the tokens just made
    # exact -- and those are the bin's highest-scoring ones, so the residual proxy
    # contributes n_res e^{zbar_full} > sum_{i in res} e^{z_i} and re-counts mass the
    # exact term has already added.  With this on, both are rebuilt for the residual by
    # subtracting the expanded tokens' own sums, which is exact (sums are additive) and
    # costs one scatter over the expanded tokens only.  s2_b is deliberately left alone:
    # it feeds the ranking, which is decided BEFORE the expansion, and under
    # kappa_mode="none" it does not enter the aggregate at all.
    #   "none"   deployed behaviour: only the COUNT is reduced
    #   "value"  also rebuild vbar_b, so a token read exactly does not also sit in the
    #            proxy centroid that stands for the tokens still unread
    #   "full"   additionally rebuild zbar_b (the mass channel)
    #   "wexact" ORACLE: the residual centroid WEIGHTED by the true e^{z_i}, which is
    #            the quantity the proxy actually stands for.  Reads the exact tail
    #            logits, so it is not deployable -- it measures whether the plain mean's
    #            bias towards the bin's top tokens is what the contamination was buying.
    residual_stats: str = "none"
    # How the profile partition is formed and computed (V22):
    #   profile_method "kmeans" -- Lloyd iterations in the P-dim profile space (V20)
    #                  "qsort"  -- NO iterations: sort tokens by qmean (mean logit
    #                              against the profile queries) and cut into C
    #                              equal-count bins; new tokens are bucketised
    #                              against the cached bin edges.
    #                  "grid"   -- NO iterations: project the centred profile on its
    #                              top-2 principal directions (64x64 Gram eigh) and
    #                              cut an equal-count 32x32 grid (C=1024).  Keeps
    #                              the second profile direction that 1-D qsort
    #                              discards (V22 psweep: P=1 -> P=2 is a 3x drop).
    #   profile_dtype  "fp32" | "bf16" -- dtype of the two large matmuls (profile
    #                  logits and centroid similarities); accumulations stay fp32.
    profile_method: str = "kmeans"
    profile_dtype: str = "fp32"
    # --- V23: where the query metric M = E[q q^T] behind the profile comes from ---
    #   "question"        P sampled question-chunk rows (V19/V20; query rows ARE the
    #                     profile queries)
    #   "question_rot"    the same rows, RoPE-normalised: every row is rotated to the
    #                     decode start position before M is formed (rank-r eigen-
    #                     directions of M become r virtual profile queries)
    #   "static"          M from an OFFLINE calibration file (pre-RoPE, per layer/head),
    #                     rotated to the decode start position -- question-independent
    #   "static+question" trace-normalised 0.5/0.5 shrinkage of the two
    #   "static+rows"     UNION: the P question rows are kept as they are and the r
    #                     static directions are appended as extra virtual queries
    #                     (request-specific directions are never averaged away)
    # The virtual queries are W^T with W = U_r Lambda_r^{1/2}, so that z_i = W^T k_i
    # and k-means on z is k-means on keys under the metric M (V23 0).
    #   "static+qrot"     UNION of eigen-directions: top-profile_rank_q directions of the
    #                     RoPE-normalised question rows + top-profile_rank static
    #                     directions (V23 E2 winner: median -21%, tail q90 -15% vs V20)
    profile_source: str = "question"
    profile_static_file: Optional[str] = None
    profile_rank: int = 8
    profile_rank_q: int = 8
    profile_rot_window: int = 0        # >0: average the rotated M over this many decode positions
    # V57 (paper v22, Table clustinput): "random" replaces the query eigenvectors of M by a
    # random orthonormal frame of the same rank (same Lloyd loop, same units, no query geometry)
    profile_basis: str = "eig"

    # the same computation; the point of running it end-to-end is to prove that.
    use_triton_decode: bool = False
    # Represent each surviving bin by TWO points -- its centroid and the token
    # furthest from that centroid in the whitened key metric -- instead of one.
    # The extremum is chosen from key geometry alone, so it is query-independent
    # and caches beside kbar_b.  See `_two_point_merge`.
    two_point: bool = False
    # With two_point, apply kappa to the residual using the RESIDUAL's own
    # variance instead of the whole bin's.  The bin's sigma^2 is inflated by the
    # representative that two_point just removed, so reusing it over-corrects the
    # leftover mass -- see `_two_point_merge`.  Off by default so runs that
    # predate the fix stay reproducible.
    residual_sigma: bool = False
    # With two_point, generate the top-rep_k tokens by whitened norm per bin
    # (still query-blind) instead of just the argmax, and break the tie with
    # rep_k real dot products at decode time -- see `_two_point_merge`.  The
    # merge stays a strict 2-point-per-bin structure regardless of rep_k; only
    # the representative-selection accuracy changes.  rep_k=1 is bit-identical
    # to the original single-candidate behaviour.
    rep_k: int = 1
    # With two_point, rank the rep_k candidates by the SELECTOR's per-token
    # score (the rank-r static metric the top-k masker already computed for
    # every token) instead of the whitened norm.  The whitened norm is a
    # direction criterion; the mass estimate wants the bin's top-LOGIT tokens,
    # which the selector score approximates far better (P1/P2 probes,
    # queue_v31: mass error 0.560 -> 0.210 at 3%, 0.762 -> 0.274 at 0.5%).
    # "whiten" reproduces the old behaviour bit-for-bit.  Only the scatter
    # (non-equalcount) path honours "selector"; the compacted equalcount path
    # ignores it and keeps whitening.
    rep_pick: str = "whiten"
    # kappa_mode "jsamp"/"jsampm" (JSAMPM_METHOD.md): per bin, take
    # kappa_samples keys at a deterministic uniform stride, score them with real
    # dot products, and apply the Jensen-gap definition to that sample.
    # "jsampm" then pools the per-bin gaps into ONE scalar per (b,h,q) -- the
    # measured win lives in the mean shift, and per-bin sample noise inflates
    # the cross-bin deviation enough to erase it (doc §2-3, §4).
    kappa_samples: int = 8
    # global damping on the kappa correction: ell_b = zbar + log n + scale*kap.
    # 1.0 (default) leaves every mode bit-identical; 0 disables the correction
    # without touching the sampling reads.  Motivated by the 0.5% split where
    # full kappa helps needle tasks but costs aggregate fwe (REPORT_V31 §6).
    kappa_scale: float = 1.0
    # V38: compose with KV-cache COMPRESSION.  A SnapKV/PyramidKV pass at the first
    # sparse call of the request marks `evict_budget` of the context resident and the
    # rest evicted; the evicted rows are then invisible to every stage below --- the
    # selector cannot propose them, they never enter a bin, and no proxy stands for
    # them --- so the two mechanisms compose instead of competing for the same budget.
    # `evict_budget` is a fraction of the context (0.5 = the 50% compression budget);
    # the densities are then read against the surviving cache.
    evict_mode: str = "none"          # "none" | "snapkv" | "pyramidkv"
    evict_budget: float = 0.0
    evict_tokens: int = 0             # V39: absolute average cache size per layer
    evict_alpha: int = 8              # V39: PyramidKV's alpha instruction tokens
    evict_window: int = 0             # V40: retained window; 0 = the method's default
    evict_obs: int = 32
    evict_pool: int = 7
    evict_beta: int = 20
    evict_layers: int = 32
    evict_sink: float = 0.0           # positions always kept resident (sink share)
    evict_local: float = 0.0          # positions always kept resident (local share)

    def __post_init__(self) -> None:
        if self.profile_source not in PROFILE_SOURCES:
            raise ValueError(f"profile_source must be one of {sorted(PROFILE_SOURCES)}, "
                             f"got {self.profile_source}")
        if self.profile_source != "question" and not self.profile_static_file:
            raise ValueError("profile_source != 'question' needs profile_static_file "
                             "(it carries the RoPE parameters and, for static sources, M)")
        if self.profile_method not in PROFILE_METHODS:
            raise ValueError(f"profile_method must be one of {sorted(PROFILE_METHODS)}, got {self.profile_method}")
        if self.bin_size < 1:
            raise ValueError(f"bin_size must be >= 1, got {self.bin_size}")
        if self.num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {self.num_bins}")
        if self.bin_mode not in BIN_MODES:
            raise ValueError(f"bin_mode must be one of {sorted(BIN_MODES)}, got {self.bin_mode}")
        if self.kappa_mode not in KAPPA_MODES:
            raise ValueError(f"kappa_mode must be one of {sorted(KAPPA_MODES)}, "
                             f"got {self.kappa_mode}")
        if self.rep_k < 1:
            raise ValueError(f"rep_k must be >= 1, got {self.rep_k}")
        if self.rep_pick not in ("whiten", "selector"):
            raise ValueError(f"rep_pick must be 'whiten' or 'selector', "
                             f"got {self.rep_pick}")
        if self.kappa_samples < 1:
            raise ValueError(f"kappa_samples must be >= 1, got {self.kappa_samples}")
        if self.kappa_scale < 0.0:
            raise ValueError(f"kappa_scale must be >= 0, got {self.kappa_scale}")
        if self.var_mode not in VAR_MODES:
            raise ValueError(f"var_mode must be one of {sorted(VAR_MODES)}, got {self.var_mode}")
        if self.bin_mode == "topblock" and self.refine_blocks < 1:
            raise ValueError("bin_mode='topblock' needs refine_blocks >= 1")


def _log_expm1(x: torch.Tensor) -> torch.Tensor:
    """log(exp(x) - 1), stable at both ends.

    Small x: log(expm1(x)) -> log x, the limit the weight-variance and Jensen-gap
    rankings below reduce to.  Large x: exp(x) overflows float32 past ~88, so use
    log(e^x - 1) = x + log1p(-e^-x).  Both branches are evaluated, hence the clamps.
    """
    lo = torch.log(torch.expm1(x.clamp(min=1e-12, max=20.0)))
    hi = x + torch.log1p(-torch.exp(-x.clamp(min=20.0)))
    return torch.where(x > 20.0, hi, lo)


class UTAMultiBinAttention(UTAAttention):
    """UTA whose tail is represented by many local proxies instead of one."""

    def __init__(self, config: UTAMultiBinConfig, maskers: List[ResearchMasker]) -> None:
        super().__init__(config, maskers)
        self.bin_size = int(config.bin_size)
        self.num_bins = int(config.num_bins)
        self.bin_mode = config.bin_mode
        self.kappa_mode = config.kappa_mode
        self.var_mode = getattr(config, "var_mode", "exact")
        self.var_rank = int(getattr(config, "var_rank", 4))
        self.refine_blocks = int(getattr(config, "refine_blocks", 0))
        self.random_seed = int(getattr(config, "random_seed", 1234))
        self.expand_m = int(getattr(config, "expand_m", 0))
        self.expand_c = float(getattr(config, "expand_c", 3.0))
        self.expand_rank = getattr(config, "expand_rank", "bound")
        self._want_m2raw = self.expand_rank == "m2raw"
        self._m2raw: Optional[torch.Tensor] = None   # set by _moments_from_key_stats
        self._want_vspread = self.expand_rank == "outerr"
        self._vspread: Optional[torch.Tensor] = None  # set by _bins_scatter_keyside
        self.use_triton_decode = bool(getattr(config, "use_triton_decode", False))
        self.two_point = bool(getattr(config, "two_point", False))
        self.residual_sigma = bool(getattr(config, "residual_sigma", False))
        self.rep_k = int(getattr(config, "rep_k", 1))
        self.rep_pick = getattr(config, "rep_pick", "whiten")
        self.kappa_samples = int(getattr(config, "kappa_samples", 8))
        self.kappa_scale = float(getattr(config, "kappa_scale", 1.0))
        self.use_aggregate = bool(getattr(config, "use_aggregate", True))
        self.aggregate_mode = getattr(config, "aggregate_mode", "multi")
        self.expand_m_table = getattr(config, "expand_m_table", None)
        self._mtab: Dict[Any, torch.Tensor] = {}     # per-layer (H,) budget, cached
        self._cur_layer: int = -1
        self._rand_bid: Optional[torch.Tensor] = None       # cache, see _bin_ids_random
        self._exp_rnd: Optional[torch.Tensor] = None        # cache, random-rank control
        self.profile_P = int(getattr(config, "profile_P", 64))
        self.profile_C = int(getattr(config, "profile_C", 1024))
        self.profile_iters = int(getattr(config, "profile_iters", 4))
        self.profile_promote = int(getattr(config, "profile_promote", 0))
        self.expand_budget_tokens = int(getattr(config, "expand_budget_tokens", 0))
        self.expand_truncate = bool(getattr(config, "expand_truncate", False))
        self.expand_bin_cap = int(getattr(config, "expand_bin_cap", 0))
        self.expand_alloc = str(getattr(config, "expand_alloc", "uniform") or "uniform")
        if self.expand_alloc not in ("uniform", "hier"):
            raise ValueError(f"expand_alloc must be uniform|hier, got {self.expand_alloc!r}")
        self.expand_alloc_gamma = float(getattr(config, "expand_alloc_gamma", 1.0))
        self.residual_stats = str(getattr(config, "residual_stats", "none") or "none")
        if self.residual_stats not in ("none", "value", "full", "fullvar", "wexact"):
            raise ValueError(f"residual_stats must be none|value|full, got {self.residual_stats!r}")
        self.profile_method = getattr(config, "profile_method", "kmeans")
        self.profile_dtype = getattr(config, "profile_dtype", "fp32")
        self.profile_source = getattr(config, "profile_source", "question")
        self.profile_static_file = getattr(config, "profile_static_file", None)
        self.profile_rank = int(getattr(config, "profile_rank", 8))
        self.profile_rank_q = int(getattr(config, "profile_rank_q", 8))
        self.profile_rot_window = int(getattr(config, "profile_rot_window", 0))
        self.profile_basis = str(getattr(config, "profile_basis", "eig") or "eig")
        if self.profile_basis not in ("eig", "random"):
            raise ValueError(f"profile_basis must be eig|random, got {self.profile_basis!r}")
        self._static: Optional[Dict[str, Any]] = None        # loaded lazily
        self._prof_cache: Dict[int, Dict[str, Any]] = {}     # per-layer, per-request
        self.evict_mode = getattr(config, "evict_mode", "none")
        self.evict_budget = float(getattr(config, "evict_budget", 0.0))
        self.evict_tokens = int(getattr(config, "evict_tokens", 0))
        self.evict_alpha = int(getattr(config, "evict_alpha", 8))
        self.evict_window = int(getattr(config, "evict_window", 0))
        self.evict_obs = int(getattr(config, "evict_obs", 32))
        self.evict_pool = int(getattr(config, "evict_pool", 7))
        self.evict_beta = int(getattr(config, "evict_beta", 20))
        self.evict_layers = int(getattr(config, "evict_layers", 32))
        self.evict_sink = float(getattr(config, "evict_sink", 0.0))
        self.evict_local = float(getattr(config, "evict_local", 0.0))

    def _head_budget(self, n_heads: int, device: torch.device) -> Optional[torch.Tensor]:
        """(H,) expansion budget for the current layer, or None for a uniform budget."""
        if self.expand_m_table is None:
            return None
        key = (self._cur_layer, n_heads, device)
        if key not in self._mtab:
            per_head = self.expand_m_table.get(self._cur_layer, {})
            vals = [int(per_head.get(h, self.expand_m)) for h in range(n_heads)]
            self._mtab[key] = torch.tensor(vals, dtype=torch.long, device=device)
        return self._mtab[key]

    def _kappa(self, s2: torch.Tensor) -> torch.Tensor:
        if self.kappa_mode == "j1":
            return s2 / 2
        if self.kappa_mode == "j2":
            return torch.log1p(s2 / 2)
        return torch.zeros_like(s2)

    def _kappa_sampled(self, q: torch.Tensor, k: torch.Tensor, bid: torch.Tensor,
                       nbins: int, n_b: torch.Tensor, zbar_b: torch.Tensor,
                       scaling: float, m: int, pool: bool,
                       pool_q: bool = False) -> torch.Tensor:
        """Jensen-gap correction read off the bin itself (JSAMPM_METHOD.md §5).

        Per bin, score `m` keys at a deterministic uniform stride with real dot
        products and apply the gap's definition to the sample:
            kappa_b = log((1/m) sum_j exp(z_j - zbar_b))
        With `pool`, average over the valid bins of each (b, h, q) into one
        scalar -- the closed forms fail because the cached diagonal variance is
        inflated by a NON-constant factor, and what the output error actually
        rewards is the mean shift, not per-bin detail (doc §2-2/§2-3).  Cost is
        m*nbins dot products per query (0.8% of K at m=4, C=64); no storage,
        no fitted constants, proxy_rows unchanged.
        """
        B, H, Q, K = bid.shape
        D = k.shape[-1]
        dev = k.device
        # 1) query-independent bin assignment: non-tail slots are padded with
        # `nbins`, which exceeds every real bin id, so the min over queries
        # recovers the key's bin; keys in no query's tail stay `nbins`.
        asg = bid.amin(dim=2)                                          # (B,H,K)
        counts = torch.zeros(B, H, nbins + 1, dtype=torch.long, device=dev)
        counts.scatter_add_(-1, asg, torch.ones_like(asg))
        nmax = int(counts[..., :nbins].max())
        if nmax == 0 or m < 1:
            return torch.zeros_like(zbar_b)
        # 2) sort keys by bin into a padded (nbins, nmax) batch; sentinel K
        order = torch.argsort(asg, dim=-1, stable=True)
        asg_s = torch.gather(asg, -1, order)
        offs = torch.cumsum(counts, -1) - counts
        rank = (torch.arange(K, device=dev).view(1, 1, K)
                - torch.gather(offs, -1, asg_s))
        trash = nbins * nmax
        flat = torch.where(asg_s < nbins, asg_s * nmax + rank,
                           torch.full_like(asg_s, trash))
        pidx = torch.full((B, H, trash + 1), K, dtype=torch.long, device=dev)
        pidx.scatter_(-1, flat, order)
        pidx = pidx[..., :trash].view(B, H, nbins, nmax)
        # 3) m deterministic stride samples per bin: slot_j = floor(cnt*(j+.5)/m)
        cb = counts[..., :nbins].clamp(min=1).unsqueeze(-1)            # (B,H,nbins,1)
        jj = (torch.arange(m, device=dev, dtype=torch.float32) + 0.5) / m
        slot = (cb.float() * jj).long().clamp(max=nmax - 1)            # (B,H,nbins,m)
        sel = torch.gather(pidx, -1, slot)
        live_s = slot < cb
        sel = sel.clamp(max=K - 1)
        # 4) real logits, gap definition on the sample
        ks = torch.gather(
            k.float(), 2,
            sel.reshape(B, H, nbins * m, 1).expand(-1, -1, -1, D),
        ).view(B, H, nbins, m, D)
        zs = (scaling * torch.einsum("bhqd,bhcmd->bhqcm", q.float(), ks)
              - zbar_b.float().unsqueeze(-1))
        zs = zs.masked_fill(~live_s.unsqueeze(2), float("-inf"))
        cnt = live_s.sum(-1).clamp(min=1).unsqueeze(2)                 # (B,H,1,nbins)
        kap = torch.logsumexp(zs, dim=-1) - torch.log(cnt.float())
        # the true gap is >= 0 by Jensen; negative values are sample noise
        kap = torch.where(torch.isfinite(kap) & (n_b > 1.5),
                          kap.clamp(min=0.0), torch.zeros_like(kap))
        if pool:
            # unweighted mean over valid bins -- one scalar per (b,h,q)
            w = (n_b > 1.5).float()
            if pool_q:
                # "jsampmq": also pool over the chunk's queries -- one scalar
                # per (b,h).  The pooled estimate's variance is what leaks into
                # fwe-style aggregation tasks, and Q more queries cut it by ~Q
                # at zero extra reads; decode (Q=1) is unchanged by definition.
                kap = ((kap * w).sum((-2, -1), keepdim=True)
                       / w.sum((-2, -1), keepdim=True).clamp(min=1.0)
                       ).expand_as(kap)
            else:
                kap = ((kap * w).sum(-1, keepdim=True)
                       / w.sum(-1, keepdim=True).clamp(min=1.0)).expand_as(kap)
        return kap.to(zbar_b.dtype)

    def _blockwise(self, x: torch.Tensor, tb: torch.Tensor, nb: int, m: int,
                   den: torch.Tensor) -> torch.Tensor:
        """Per-block tail mean of a key/value-side tensor. x: (B,H,K,D) -> (B,H,Q,nb,D)."""
        xv = x.reshape(x.shape[:-2] + (nb, m, x.shape[-1]))
        return torch.einsum("bhqnm,bhnmd->bhqnd", tb, xv) / den.unsqueeze(-1)

    def _moments_from_key_stats(self, q: torch.Tensor, k_mean: torch.Tensor,
                                k_sq: torch.Tensor,
                                scaling: float,
                                lr: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(zbar_b, sigma_b^2) from a bin's first two key moments.

            zbar_b    = s * q . k_bar_b
            sigma_b^2 = s^2 * sum_d q_d^2 Var_b(k_d)

        zbar_b here is *algebraically identical* to the mean of the true logits over the
        bin's tail members -- mean_i s*q.k_i = s*q.mean_i k_i -- so reading it off the
        full score matrix, as the prototype used to, computes the same number the
        expensive way.  Going through k_bar_b instead is what makes the quantity
        query-independent and therefore cacheable with the KV pages.

        sigma_b^2 drops cross-dimension covariance, which is what keeps it O(d) per bin
        instead of O(d^2).
        """
        k_var = (k_sq - k_mean * k_mean).clamp(min=0)
        qe = q.unsqueeze(-2)                                              # (B,H,Q,1,D)
        if self._want_m2raw:
            # V36 expand_rank="m2raw": the same reduction with the centring left OUT,
            # s^2 sum_d q_d^2 E_b[k_d^2] = sigma_b^2 + s^2 sum_d q_d^2 kbar_bd^2.  A bin
            # caches either Var_b(k_d) or E_b[k_d^2] -- one is the other minus the mean
            # it already stores -- so the proxy stream costs exactly what it costs now.
            self._m2raw = (scaling ** 2) * ((q * q).unsqueeze(-2) * k_sq).sum(-1)
        zbar_b = scaling * (qe * k_mean).sum(-1)                          # (B,H,Q,nb)
        if lr is not None:
            # V28 low-rank correction: the diagonal keeps only what the r factors do
            # not already explain; the factors add the off-diagonal energy along
            # their directions.  Doff (B,H,C,D), L (B,H,C,D,r); bins == clusters here.
            Doff, L = lr
            k_var = (k_var - Doff.unsqueeze(2)).clamp(min=0)
            proj = torch.einsum("bhqd,bhcdr->bhqcr", q, L)
            s2_b = (scaling ** 2) * (((q * q).unsqueeze(-2) * k_var).sum(-1)
                                     + (proj * proj).sum(-1))
            return zbar_b, s2_b
        s2_b = (scaling ** 2) * ((q * q).unsqueeze(-2) * k_var).sum(-1)
        return zbar_b, s2_b

    def _bin_moments_fixed(self, q: torch.Tensor, k: torch.Tensor, tb: torch.Tensor,
                           nb: int, m: int, den: torch.Tensor,
                           scaling: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """(zbar_b, sigma_b^2) for position blocks, from key-side statistics only.

        Var_b(k_d) can be cached with the KV pages (d floats per block, the same
        footprint as the block mean), so nothing here ever needs the logits.
        """
        with stage("mb/T4a_bin_k_mean"):
            k_mean = self._blockwise(k, tb, nb, m, den)                   # (B,H,Q,nb,D)
        with stage("mb/T4b_bin_k_sq_mean"):
            k_sq = self._blockwise(k * k, tb, nb, m, den)
        with stage("mb/T4c_zbar_and_sigma"):
            return self._moments_from_key_stats(q, k_mean, k_sq, scaling)

    def _diag_var_fixed(self, q: torch.Tensor, k: torch.Tensor, tb: torch.Tensor,
                        nb: int, m: int, den: torch.Tensor, scaling: float) -> torch.Tensor:
        """sigma_b^2 only, for callers that already have zbar_b (the topblock path)."""
        return self._bin_moments_fixed(q, k, tb, nb, m, den, scaling)[1]

    # -------------------------------------------------- key-side bin statistics --
    # The two functions below produce the same (n_b, zbar_b, s2_b, vbar_b) as
    # `_bins_fixed` / `_aggregate_by_bin_id`, but WITHOUT reading the full score
    # matrix.  Every quantity is a reduction over K, V and the tail mask, i.e. over
    # the tokens the selector left behind -- which is the form a deployable kernel
    # can serve from per-block sums cached with the KV pages.  Used whenever
    # var_mode="diag"; var_mode="exact" still needs the true logits by construction.

    def _bins_fixed_keyside(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        tail_f: torch.Tensor, scaling: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`fixed` position blocks, statistics from K/V only."""
        m = self.bin_size
        K = tail_f.shape[-1]
        pad = (-K) % m
        if pad:
            tail_f = torch.nn.functional.pad(tail_f, (0, pad), value=0.0)
            v = torch.nn.functional.pad(v, (0, 0, 0, pad), value=0.0)
            k = torch.nn.functional.pad(k, (0, 0, 0, pad), value=0.0)
        nb = (K + pad) // m

        with stage("mb/T3_bin_stats"):
            tb = tail_f.reshape(tail_f.shape[:-1] + (nb, m))
            n_b = tb.sum(-1)
            den = n_b.clamp(min=1)
            vbar_b = self._blockwise(v, tb, nb, m, den)
        with stage("mb/T4_bin_moments"):
            zbar_b, s2_b = self._bin_moments_fixed(q, k, tb, nb, m, den, scaling)
        return n_b, zbar_b, s2_b, vbar_b

    def _bins_scatter_keyside(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        tail_f: torch.Tensor, bid: torch.Tensor, nbins: int, scaling: float,
        lr: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_key_stats: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Irregular (adjacency / score) bins, statistics from K/V only.

        return_key_stats=True additionally returns (k_mean, k_sq): the same per-bin
        key mean / second moment zbar_b and s2_b are reduced from.  `_two_point_merge`
        needs exactly those to whiten its per-token gather, and recomputing them there
        is a second reduction over the same tokens for the same answer.  Only ask for
        them when holding a (B,H,Q,nbins,D) tensor is cheap -- the Q==1 decode gate in
        `_multibin_output`.  Unavailable under var_mode='lowrank', whose moments do not
        come from a plain key mean/second moment pair.
        """
        with stage("mb/T3_bin_stats"):
            shp = tail_f.shape[:-1] + (nbins + 1,)
            n_b = torch.zeros(shp, dtype=tail_f.dtype, device=tail_f.device)
            n_b = n_b.scatter_add_(-1, bid, tail_f)[..., :nbins]
            den = n_b.clamp(min=1)
            vbar_b = self._scatter_mean(v, tail_f, bid, nbins, den)
            if self._want_vspread:
                # V36b: sqrt(tr Cov_b(v)), ONE scalar per bin.  Cauchy-Schwarz bounds the
                # shape error ||vtilde_b - vbar_b|| by CV_b(e^z) times this, so it is the
                # only new quantity expand_rank="outerr" has to cache -- 1 float on top of
                # the three d-vectors a bin already stores (+0.3% of the proxy stream).
                vsq = self._scatter_mean((v * v).sum(-1, keepdim=True),
                                         tail_f, bid, nbins, den)[..., 0]
                self._vspread = (vsq - (vbar_b * vbar_b).sum(-1)).clamp(min=0).sqrt()
        with stage("mb/T4_bin_moments"):
            if return_key_stats:
                if lr is not None:
                    raise ValueError("return_key_stats is incompatible with lowrank")
                k_mean = self._scatter_mean(k, tail_f, bid, nbins, den)
                k_sq = self._scatter_mean(k * k, tail_f, bid, nbins, den)
                zbar_b, s2_b = self._moments_from_key_stats(q, k_mean, k_sq, scaling)
                s2_b = torch.where(n_b >= 2, s2_b, torch.zeros_like(s2_b))
                return n_b, zbar_b, s2_b, vbar_b, k_mean, k_sq
            zbar_b, s2_b = self._bin_moments_scatter(
                q, k, tail_f, bid, nbins, den, scaling, lr=lr)
            s2_b = torch.where(n_b >= 2, s2_b, torch.zeros_like(s2_b))
        return n_b, zbar_b, s2_b, vbar_b


    def _whiten_topm(
        self, k: torch.Tensor, kbar_b: torch.Tensor, inv_sd: torch.Tensor,
        bid: torch.Tensor, tail: torch.Tensor, nbins: int, m: int,
    ) -> torch.Tensor:
        """Whitened key norms, exact at each bin's top-`m` and -inf elsewhere."""
        B, H, Q, K = tail.shape
        D = k.shape[-1]
        dev = k.device
        # The bin assignment is query-blind; `bid` only hides it behind the pad
        # index nbins on rows that are not tail.  nbins exceeds every real bin,
        # so the per-query minimum recovers it for any key that is tail at least
        # once.  Keys that are tail for no query keep nbins and are dropped --
        # their norm is masked out for every query regardless.
        asg = bid.amin(dim=2)
        counts = torch.zeros(B, H, nbins + 1, device=dev, dtype=torch.float32)
        counts = counts.scatter_add_(
            -1, asg, torch.ones(B, H, K, device=dev, dtype=torch.float32)).long()
        nmax = int(counts[..., :nbins].max().item())
        if nmax == 0:
            return torch.full((B, H, Q, K), float("-inf"), device=dev,
                              dtype=torch.float32)
        order = torch.argsort(asg, dim=-1, stable=True)
        asg_s = torch.gather(asg, -1, order)
        offs = torch.cumsum(counts, -1) - counts
        rank = (torch.arange(K, device=dev).view(1, 1, K)
                - torch.gather(offs, -1, asg_s))
        trash = nbins * nmax
        flat = torch.where(asg_s < nbins, asg_s * nmax + rank,
                           torch.full_like(rank, trash))
        pidx = torch.full((B, H, trash + 1), K, dtype=torch.long, device=dev)
        pidx = pidx.scatter(-1, flat, order)[..., :trash].view(B, H, nbins, nmax)
        valid = (torch.arange(nmax, device=dev).view(1, 1, 1, nmax)
                 < counts[..., :nbins].unsqueeze(-1))
        del asg, counts, order, asg_s, offs, rank, flat

        aa = inv_sd * inv_sd
        bb = aa * kbar_b
        cc = (bb * kbar_b).sum(-1)
        mm = min(m, nmax)
        gp = pidx.clamp(max=K - 1)
        wn = torch.full((B, H, Q, K + 1), float("-inf"), device=dev,
                        dtype=torch.float32)
        # The GEMM operand is (nbins, nmax, D) floats per head; tile the heads to
        # a fixed byte budget, as the dense loop tiles its rows.
        hstep = max(1, min(H, (256 * 1024 * 1024)
                           // max(1, nbins * nmax * D * 4)))
        for h0 in range(0, H, hstep):
            h1 = min(H, h0 + hstep)
            hn = h1 - h0
            kp = torch.gather(
                k[:, h0:h1], 2,
                gp[:, h0:h1].reshape(B, hn, trash, 1).expand(-1, -1, -1, D)
            ).reshape(B * hn * nbins, nmax, D).float()
            A = aa[:, h0:h1].permute(0, 1, 3, 4, 2).reshape(B * hn * nbins, D, Q)
            Bm = bb[:, h0:h1].permute(0, 1, 3, 4, 2).reshape(B * hn * nbins, D, Q)
            Cv = cc[:, h0:h1].permute(0, 1, 3, 2).reshape(B * hn * nbins, 1, Q)
            g = torch.bmm(kp * kp, A) - 2.0 * torch.bmm(kp, Bm) + Cv
            del kp, A, Bm, Cv
            g = g.view(B, hn, nbins, nmax, Q).permute(0, 1, 4, 2, 3)
            tp = torch.gather(
                tail[:, h0:h1], 3,
                gp[:, h0:h1].reshape(B, hn, 1, trash).expand(-1, -1, Q, -1)
            ).view(B, hn, Q, nbins, nmax)
            g = g.masked_fill(~(valid[:, h0:h1].unsqueeze(2) & tp), float("-inf"))
            del tp
            cand = g.topk(mm, dim=-1).indices
            alive = torch.isfinite(g.gather(-1, cand))
            del g
            rp = torch.gather(
                pidx[:, h0:h1].unsqueeze(2).expand(B, hn, Q, nbins, nmax),
                -1, cand)
            del cand
            # Re-score the shortlist exactly as the dense loop would have.
            kc = torch.gather(
                k[:, h0:h1].unsqueeze(2).expand(B, hn, Q, K, D), 3,
                rp.clamp(max=K - 1).reshape(B, hn, Q, nbins * mm, 1)
                  .expand(-1, -1, -1, -1, D)
            ).view(B, hn, Q, nbins, mm, D)
            dv = ((kc - kbar_b[:, h0:h1].unsqueeze(-2))
                  * inv_sd[:, h0:h1].unsqueeze(-2))
            del kc
            wn[:, h0:h1].scatter_(
                3, torch.where(alive, rp, torch.full_like(rp, K))
                        .reshape(B, hn, Q, nbins * mm),
                dv.square().sum(-1).reshape(B, hn, Q, nbins * mm))
            del dv, rp, alive
        return wn[..., :K]

    def _two_point_scatter(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        tail_f: torch.Tensor, bid: torch.Tensor, nbins: int, n_b: torch.Tensor,
        zbar_b: torch.Tensor, vbar_b: torch.Tensor, scaling: float,
        live: torch.Tensor, s2_b: Optional[torch.Tensor] = None,
        rep_k: int = 1, k_mean: Optional[torch.Tensor] = None,
        k_sq: Optional[torch.Tensor] = None,
        sel_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`_two_point_merge` for partitions whose bins are not position runs.

        Same construction -- whiten by the bin's own mean/variance, pull out the
        furthest token, fold (exact point, residual mean) back into one
        (ell, vbar) pair -- but every reduction is a scatter over `bid`, so bins
        may be scattered through the key axis and unequal in size.  Slower than
        the compacted path (it touches all K per bin statistic rather than the
        bin's own m rows), which is why equalcount keeps its own route.
        """
        den_b = n_b.clamp(min=1)
        use_sel = (getattr(self, "rep_pick", "whiten") == "selector"
                   and sel_scores is not None
                   and sel_scores.shape == tail_f.shape)
        # `_bins_scatter_keyside` reduced these very quantities for zbar_b/s2_b.
        # Recomputing them is a second pass over every tail token for the same
        # numbers -- the compacted path has taken them as arguments since the
        # key-stat reuse landed; this route simply was never wired up.
        if use_sel:
            kbar_b = inv_sd = None      # whitening skipped entirely
        elif k_mean is not None:
            kbar_b = k_mean
            inv_sd = (k_sq - kbar_b * kbar_b).clamp(min=1e-6).rsqrt()
        else:
            kbar_b = self._scatter_mean(k, tail_f, bid, nbins, den_b)
            ksq_b = self._scatter_mean(k * k, tail_f, bid, nbins, den_b)
            inv_sd = (ksq_b - kbar_b * kbar_b).clamp(min=1e-6).rsqrt()
            del ksq_b

        tail = tail_f > 0
        B, H, Q, K = tail_f.shape
        D = k.shape[-1]
        if use_sel:
            # The selector already scored every token with its rank-r metric --
            # a logit proxy.  Ranking the rep_k candidates by it hands the mass
            # estimate the bin's (approximate) top logits, which is what z_ext
            # and the exact top-m sum need; the whitened norm is a direction
            # criterion and misses them.  No new reads: the scores were paid
            # for by the selector, and the rounds below still rescore every
            # candidate with a real dot product before using it.
            wn = sel_scores.float().masked_fill(~tail, float("-inf"))
        elif Q > 1:
            # A shortlist four times the rounds the loop will run.  The
            # measurement needed only two to never lose a winner, and the
            # shortlist costs linearly in its width, so this is two
            # doublings of margin for very little.
            wn = self._whiten_topm(k, kbar_b, inv_sd, bid, tail, nbins,
                                   4 * max(1, rep_k))
            del kbar_b, inv_sd
        else:
            wn = torch.zeros((B, H, Q, K), device=k.device, dtype=torch.float32)
            # One (head, query) row costs K*D floats in the difference tensor below, so
            # the tile is sized in BYTES: splitting heads alone (hstep = 8 // Q) keeps a
            # constant row COUNT, which at the question chunk (Q large, hstep floored to
            # 1) is a whole Q's worth of rows -- gigabytes per step, thirty-two times.
            # `_two_point_merge` tiles its query rows for the same reason.
            rows = max(1, (256 * 1024 * 1024) // max(1, K * D * 4))
            hstep = max(1, min(H, rows // max(1, Q)))
            qstep = Q if hstep * Q <= rows else max(1, rows)
            # bid routes non-tail tokens to the pad index nbins, one past the per-bin
            # tensors, so the gather index is clamped; those rows are masked out below.
            bid_g = bid.clamp(max=nbins - 1)
            for h0 in range(0, H, hstep):
                h1 = min(H, h0 + hstep)
                for q0 in range(0, Q, qstep):
                    q1 = min(Q, q0 + qstep)
                    gi = (bid_g[:, h0:h1, q0:q1].unsqueeze(-1)
                          .expand(-1, -1, -1, -1, D))
                    dv = ((k[:, h0:h1].unsqueeze(2)
                           - kbar_b[:, h0:h1, q0:q1].gather(3, gi))
                          * inv_sd[:, h0:h1, q0:q1].gather(3, gi))
                    wn[:, h0:h1, q0:q1] = dv.square().sum(-1)
                    del gi, dv
            del kbar_b, inv_sd
            wn = wn.masked_fill(~tail, float("-inf"))

        shp = tail_f.shape[:-1] + (nbins + 1,)
        rk = max(1, rep_k)
        work = wn
        ar = torch.arange(K, device=k.device).view(1, 1, 1, K).expand(B, H, Q, K)
        K_full = torch.full((), K, dtype=torch.long, device=k.device)
        z_ext = torch.full_like(zbar_b, float("-inf"))
        vext = torch.zeros_like(vbar_b)
        zs_rounds = []
        for _ in range(rk):
            # per-bin argmax of the whitened norm, as a token mask
            cmax = torch.full(shp, float("-inf"), device=wn.device, dtype=wn.dtype)
            cmax = cmax.scatter_reduce(-1, bid, work, reduce="amax",
                                       include_self=True)
            pick = (work == cmax.gather(-1, bid)) & tail & torch.isfinite(work)
            pf = pick.to(tail_f.dtype)
            cnt = torch.zeros(shp, device=pf.device, dtype=pf.dtype)
            cnt = cnt.scatter_add_(-1, bid, pf)[..., :nbins]
            # A bin with fewer than rk members runs dry in a later round.  Its
            # scatter-mean is then 0/0 -> the zero vector, whose logit is 0 --
            # which would BEAT a genuinely negative representative and hand the
            # bin a token that does not exist.  `drew` gates those rounds out.
            drew = cnt > 0.5
            # One token per bin is picked, so a scatter over the whole key axis
            # reads K/nbins (~512x) more than the nbins rows it returns.  Resolve
            # the winner's INDEX with a scalar reduction, then gather that row.
            # Ties differ: the scatter averaged them, this takes the lowest index.
            # `uta_2pt_tie_bins` counts the bins where that can bite.
            pos = torch.where(pick, ar, K_full)
            widx = torch.full(shp, K, dtype=torch.long, device=pick.device)
            widx = widx.scatter_reduce(-1, bid, pos, reduce="amin",
                                       include_self=True)[..., :nbins]
            gsel = widx.clamp(max=K - 1).unsqueeze(-1).expand(-1, -1, -1, -1, D)
            kexp = k.unsqueeze(2).expand(-1, -1, Q, -1, -1)
            vexp = v.unsqueeze(2).expand(-1, -1, Q, -1, -1)
            kc = kexp.gather(3, gsel).to(tail_f.dtype)
            vc = vexp.gather(3, gsel).to(tail_f.dtype)
            if MicroMetricLogger().is_metric_enabled("uta_2pt_tie_bins"):
                MicroMetricLogger().log(
                    "uta_2pt_tie_bins", float((cnt > 1.5).sum().item()))
            del pos, widx, gsel, kexp, vexp
            zc = scaling * (q.unsqueeze(-2) * kc).sum(-1)
            # keep the candidate with the highest TRUE logit, exactly as the
            # compacted path does; losers fall back into the residual mean
            better = drew & (zc > z_ext)
            z_ext = torch.where(better, zc, z_ext)
            vext = torch.where(better.unsqueeze(-1), vc, vext)
            if self.kappa_mode == "jtopm":
                zs_rounds.append(torch.where(
                    drew, zc.to(z_ext.dtype),
                    torch.full_like(z_ext, float("-inf"))))
            if rk > 1:
                work = work.masked_fill(pick, float("-inf"))
            del cmax, pick, pf, cnt, drew, kc, vc, zc, better
        z_ext = torch.where(torch.isfinite(z_ext), z_ext, zbar_b)
        del wn, work

        n_rest = (n_b - 1).clamp(min=0)
        rd = n_rest.clamp(min=1)
        has_rest = n_rest > 0
        zbar_r = (n_b * zbar_b - z_ext) / rd
        if self.kappa_mode in ("jmax", "jtopm"):
            # s^2/2 is the gap of a Gaussian with infinitely many draws; a bin
            # holds a few hundred keys, so its sum never reaches the tail that
            # s^2 implies and the correction overshoots (measured 2.08 against a
            # true gap of 0.72, which is worse than not correcting at all).  The
            # bin's own maximum says how far its upper tail actually reaches:
            # for n draws from N(zbar, s^2) the maximum sits near zbar +
            # s*sqrt(2 ln n), so inverting that gives the sigma the sample really
            # realises.  z_ext is the maximum the rep_k round already scored with
            # a real dot product, so this reads nothing new and fits nothing.
            ln_n = torch.log(n_b.clamp(min=2.0))
            s_eff = ((z_ext - zbar_b) / (2.0 * ln_n).sqrt()).clamp(min=0.0)
            kap = 0.5 * s_eff * s_eff
        elif self.kappa_mode in ("jsamp", "jsampm", "jsampmq"):
            # sample-read gap for the residual term (JSAMPM_METHOD.md §5-3)
            kap = self._kappa_sampled(q, k, bid, nbins, n_b, zbar_b, scaling,
                                      int(getattr(self, "kappa_samples", 8)),
                                      pool=(self.kappa_mode != "jsamp"),
                                      pool_q=(self.kappa_mode == "jsampmq"))
        elif s2_b is None:
            kap = torch.zeros_like(z_ext)
        elif self.residual_sigma:
            sum_sq = n_b * (s2_b + zbar_b * zbar_b) - z_ext * z_ext
            kap = self._kappa((sum_sq / rd - zbar_r * zbar_r).clamp(min=0.0))
        else:
            kap = self._kappa(s2_b)
        ks = float(getattr(self, "kappa_scale", 1.0))
        if ks != 1.0:
            kap = kap * ks
        lb = torch.where(has_rest, zbar_r + torch.log(rd) + kap,
                         torch.full_like(z_ext, float("-inf")))
        lse = torch.logaddexp(z_ext, lb)
        if self.kappa_mode == "jtopm" and zs_rounds:
            # Every candidate the rep_k rounds scored carries an EXACT logit;
            # keeping only the maximum re-buries the other rep_k-1 exponentials
            # inside the mean term.  Sum the scored ones exactly and estimate
            # only the unscored residual, under the same s_eff kappa -- the
            # candidates were already paid for, so no new reads.
            zs = torch.stack(zs_rounds, dim=0)
            fin = torch.isfinite(zs)
            m_b = fin.sum(dim=0).to(n_b.dtype)
            top_lse = torch.logsumexp(zs, dim=0)
            sum_z = torch.where(fin, zs, torch.zeros_like(zs)).sum(dim=0)
            n_rm = (n_b - m_b).clamp(min=0.0)
            rd_m = n_rm.clamp(min=1.0)
            zbar_rm = (n_b * zbar_b - sum_z) / rd_m
            lb_m = torch.where(n_rm > 0, zbar_rm + torch.log(rd_m) + kap,
                               torch.full_like(z_ext, float("-inf")))
            lse = torch.logaddexp(top_lse, lb_m)
            lse = torch.where(torch.isfinite(lse), lse, zbar_b)
        wa = torch.exp(z_ext - lse)
        one_m = 1.0 - wa
        alpha = torch.where(has_rest, wa - one_m / rd, torch.ones_like(wa))
        beta = torch.where(has_rest, one_m * n_b / rd, torch.zeros_like(wa))
        vbar_new = vbar_b * beta.unsqueeze(-1) + vext * alpha.unsqueeze(-1)
        return lse.masked_fill(~live, float("-inf")), vbar_new

    def _two_point_merge(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        tail_f: torch.Tensor, bid: torch.Tensor, nbins: int, n_b: torch.Tensor,
        zbar_b: torch.Tensor, vbar_b: torch.Tensor, scaling: float,
        live: torch.Tensor, s2_b: Optional[torch.Tensor] = None,
        k_mean: Optional[torch.Tensor] = None, k_sq: Optional[torch.Tensor] = None,
        rep_k: int = 1, sel_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Represent each surviving bin by TWO points instead of one.

        The second point is the bin's token furthest from its centroid in the
        SAME diagonally whitened metric the UCB already uses,
        u_i = (k_i - kbar_b)/sqrt(Var_b(k)) -- no new statistic, only its argmax.
        Chosen from key geometry alone, so it is query-independent and caches
        beside kbar_b (measured: it is the bin's true max-logit token 60.4% of
        the time, where the selector's own scores manage 9.4%).

        rep_k > 1: generate the top-`rep_k` tokens by ||u_i|| per bin (still
        query-blind, still cacheable) instead of just the argmax, then break
        the tie at decode time with `rep_k` real dot products and keep only
        the winner as the bin's exact point -- the loser(s) fall back into the
        residual mean, so the merge below is untouched and stays a strict
        2-point-per-bin structure regardless of rep_k.  Whenever the bin's
        true needle is among the rep_k candidates it is guaranteed to win
        (it has the bin's highest true logit by construction), so this
        recovers exactly the membership rate already measured as
        `k{rep_k}_needle_id` (58.06% -> 71.69% -> 83.38% for rep_k=1/2/4) at
        the cost of rep_k-1 extra cached key vectors and rep_k-1 extra
        decode-time dot products -- no change to value fetches or the number
        of softmax terms.

        Why a second point: the tail's participation ratio is 3.65 and one token
        holds 52% of its mass (V9 §3.3), so a bin's mass IS one token.  A single
        centroid dilutes it 1:31 and no mass correction can undo that (V13).
        The bin then contributes  e^{z*}v* + (n-1)e^{z_r}v_r  instead of
        n*e^{zbar}vbar, folded back into one (ell, vbar) pair so every consumer
        of the merge stays unchanged.

        Runs in query-row slices: the per-bin key tensors are (B,H,q,nbins,D) and
        would otherwise peak at several GB during the question prefill.

        k_mean/k_sq, when given, are the bin mean/second moment already computed
        by `_bins_scatter_keyside` for zbar_b/s2_b -- algebraically the same
        kbar/kvar this function would otherwise re-derive from a per-token
        gather below.  Passing them in skips that re-derivation; the caller is
        responsible for only doing so when holding onto a (B,H,Q,nbins,D)
        tensor this long is cheap (decode, Q==1 -- see `_multibin_output`).
        """
        # The compaction below turns bin b into tail ranks [m*b, m*b+m), which is
        # what an equalcount bin IS and what no other partition is.  Anything whose
        # membership follows `bid` rather than the tail's position order -- a
        # k-means clustering, a score partition, a random one -- has to go through
        # the scatter form instead.  (This method used to accept `bid` and never
        # read it, so those modes silently got the wrong representative.)
        if self.bin_mode != "equalcount":
            return self._two_point_scatter(
                q, k, v, tail_f, bid, nbins, n_b, zbar_b, vbar_b, scaling,
                live, s2_b=s2_b, rep_k=rep_k, k_mean=k_mean, k_sq=k_sq,
                sel_scores=sel_scores)

        B, H, Q, K = tail_f.shape
        D = k.shape[-1]
        m = self.bin_size
        NB = nbins * m
        ell_out = torch.empty_like(zbar_b)
        kf, vf = k.reshape(B * H, K, D), v.reshape(B * H, K, D)
        reuse_stats = k_mean is not None
        if reuse_stats:
            k_mean_f = k_mean.reshape(B * H, Q, nbins, D)
            k_sq_f = k_sq.reshape(B * H, Q, nbins, D)
        # Query rows are independent here (each has its own tail, hence its own
        # bins), and batching them only inflates the live footprint of the
        # compaction gather.  Measured at Q=32, K=32K (H100): qc=2 costs 311 ms
        # against 466 ms for the byte-budget width of 11, so narrow wins.
        # Re-measured on this RTX 4090 (same shape): qc=1 166ms, qc=2 162ms,
        # qc=4 284ms, qc>=8 ~318ms -- same qualitative shape (narrow wins,
        # cliff past qc=2), so the default carries over unchanged despite the
        # very different absolute numbers.  `_tp_qc` is here to re-measure
        # that on other shapes/hardware.
        qc = int(getattr(self, "_tp_qc", 2)) or 2
        qc = max(1, min(Q, qc))
        tile = max(1, (256 * 1024 * 1024) // max(1, NB * D * 4))
        for q0 in range(0, Q, qc):
            q1 = min(Q, q0 + qc)
            qn = q1 - q0
            tf_c = tail_f[:, :, q0:q1]
            n_c = n_b[:, :, q0:q1]
            tail_c = tf_c > 0

            # An equalcount bin is a run of `m` CONSECUTIVE tail tokens, so
            # compacting the tail makes bin b exactly rows [m*b, m*b+m).  Every
            # per-token statistic below is then a broadcast over that reshaped
            # view instead of a per-token gather of its bin's d-vectors -- the
            # gather form re-reads the same nbins vectors `m` times over.
            rank = (tf_c.cumsum(-1) - 1).clamp(min=0).long()
            slot = torch.where(tail_c, rank.clamp(max=NB - 1),
                               torch.full_like(rank, NB))
            pos = torch.full((B, H, qn, NB + 1), -1, dtype=torch.long,
                             device=k.device)
            pos.scatter_(-1, slot, torch.arange(K, device=k.device)
                         .view(1, 1, 1, K).expand(B, H, qn, K))
            pos = pos[..., :NB]
            del rank, slot

            rk = max(1, min(rep_k, m))
            ext_pos = torch.zeros((B * H, qn, nbins, rk), dtype=torch.long,
                                  device=k.device)
            # A bin with fewer than rk live members pads its extra candidate
            # slots with an arbitrary (masked-out) index -- ext_valid says
            # which of the rk slots are a real member, so the decode-time
            # winner pick below never considers a padding slot.  Slot 0 is
            # always valid for any live bin (it's today's single argmax).
            ext_valid = torch.zeros((B * H, qn, nbins, rk), dtype=torch.bool,
                                    device=k.device)
            posv = pos.reshape(B * H, qn, NB)
            for bh in range(B * H):
                krow = kf[bh].unsqueeze(0)          # (1,K,D) view, never copied
                for t0 in range(0, qn, tile):
                    t1 = min(qn, t0 + tile)
                    pt = posv[bh, t0:t1]                            # (t,NB)
                    okt = pt >= 0
                    gi = pt.clamp(min=0).unsqueeze(-1).expand(-1, -1, D)
                    kc = torch.gather(krow.expand(t1 - t0, -1, -1), 1, gi)
                    kc = kc.to(torch.float32).view(-1, nbins, m, D)
                    if reuse_stats:
                        # Same reduction _bins_scatter_keyside already did for
                        # zbar_b/s2_b (scatter-add over K instead of this
                        # gather-then-sum over m) -- redoing it here would be
                        # pure waste for no different answer.
                        #
                        # Widening this loop over several `bh` at once keeps
                        # coming up as an idea and does not pay.  In ISOLATION
                        # it looks like it must: shrinking the per-iteration
                        # data to 3% leaves the wall clock unchanged (5.46 ->
                        # 5.58 ms), i.e. standalone the loop is pure launch
                        # overhead, and batching it two-wide measures 1.52x.
                        # In SITU it measures 1.004x (rk=1) / 1.021x (rk=8) --
                        # the launches hide behind the GPU work the earlier
                        # stages have already queued -- while 4-wide and above
                        # are 12% SLOWER.  Cheaper math does not pay either:
                        # an algebraic expansion of the whitened norm below is
                        # 2.4x slower and a fused reduction only 1.18x faster,
                        # on a term that is ~0% of the stage.
                        kbar = k_mean_f[bh, q0 + t0: q0 + t1]
                        kvar = (k_sq_f[bh, q0 + t0: q0 + t1]
                               - kbar * kbar).clamp(min=1e-6)
                    else:
                        vm = okt.view(-1, nbins, m, 1).to(torch.float32)
                        cnt = vm.sum(-2).clamp(min=1.0)
                        kbar = (kc * vm).sum(-2) / cnt
                        kvar = ((kc * kc * vm).sum(-2) / cnt - kbar * kbar).clamp(min=1e-6)
                        del vm, cnt
                    wn = (((kc - kbar.unsqueeze(-2)) * kvar.rsqrt().unsqueeze(-2))
                          .square().sum(-1))                        # (t,nbins,m)
                    wn = wn.masked_fill(~okt.view(-1, nbins, m), float("-inf"))
                    # Rank-0 candidate is always plain argmax -- bit-identical
                    # to the rk=1 path even on a two-way tie in wn (which a
                    # 2-member bin hits structurally: with only two points,
                    # both sit exactly equidistant from their shared mean, so
                    # argmax and topk(1) can legally disagree on which one
                    # "wins").  rk>1 candidates are topk's next picks with
                    # that rank-0 index excluded, so they're never a repeat.
                    sib0 = wn.argmax(-1, keepdim=True)              # (t,nbins,1)
                    if rk > 1:
                        wn_rest = wn.scatter(-1, sib0, float("-inf"))
                        val_rest, idx_rest = wn_rest.topk(rk - 1, dim=-1)
                        sib = torch.cat([sib0, idx_rest], dim=-1)   # (t,nbins,rk)
                        sib_val = torch.cat([wn.gather(-1, sib0), val_rest], dim=-1)
                        del wn_rest, val_rest, idx_rest
                    else:
                        sib = sib0
                        sib_val = wn.gather(-1, sib0)
                    ext_pos[bh, t0:t1] = (pt.view(-1, nbins, m)
                                          .gather(-1, sib)
                                          .clamp(min=0))
                    ext_valid[bh, t0:t1] = torch.isfinite(sib_val)
                    del pt, okt, gi, kc, kbar, kvar, wn, sib, sib_val, sib0
            del pos, posv

            # only nbins*rk rows per (head, query) are touched here, not K
            gi2 = ext_pos.reshape(B * H, qn * nbins * rk, 1).expand(-1, -1, D)
            kext_all = torch.gather(kf, 1, gi2).reshape(B, H, qn, nbins, rk, D)
            del gi2

            z_ext_all = scaling * (
                q[:, :, q0:q1].unsqueeze(-2).unsqueeze(-2) * kext_all).sum(-1)
            del kext_all
            if rk > 1:
                # Pick the winner with a real query dot product -- rk-1 extra
                # scalars per bin, no extra value fetch or softmax term: the
                # loser(s) are simply never pulled out of the residual mean.
                z_ext_all = z_ext_all.masked_fill(
                    ~ext_valid.reshape(B, H, qn, nbins, rk), float("-inf"))
                widx = z_ext_all.argmax(-1, keepdim=True)
                z_ext = z_ext_all.gather(-1, widx).squeeze(-1)
                # Fetch ONLY the winner's value.  Gathering all rk and then
                # selecting one discards rk-1 of them, i.e. (rk-1)/rk of the
                # value traffic -- which is what the "no extra value fetch"
                # promise above is supposed to mean.  Resolve the winner in
                # INDEX space first, then make a single (nbins,D) gather.
                win_pos = ext_pos.gather(-1, widx.reshape(B * H, qn, nbins, 1))
                del widx
            else:
                z_ext = z_ext_all.squeeze(-1)
                win_pos = ext_pos
            gi3 = win_pos.reshape(B * H, qn * nbins, 1).expand(-1, -1, D)
            vext = torch.gather(vf, 1, gi3).reshape(B, H, qn, nbins, D)
            del z_ext_all, ext_valid, ext_pos, win_pos, gi3
            n_rest = (n_c - 1).clamp(min=0)
            rd = n_rest.clamp(min=1)
            has_rest = n_rest > 0
            # residual bin = the bin minus its representative.  kappa is applied
            # HERE rather than to the raw bin: once the dominant token has been
            # pulled out and made exact, the remaining tokens are homogeneous, so
            # j2's moderate-sigma premise finally holds and its old failure mode
            # (amplifying a wrong mean direction) is gone.
            zbar_r = (n_c * zbar_b[:, :, q0:q1] - z_ext) / rd
            if s2_b is None:
                kap = torch.zeros_like(z_ext)
            elif self.residual_sigma:
                # sigma_b^2 is inflated by the very token that was just removed,
                # so feeding it to kappa over-corrects the residual's mass -- and
                # that mass sits on the residual's MEAN value, i.e. the direction
                # the two-point split was meant to stop trusting.  Recover the
                # residual's own variance from the second-moment identity
                #     sum z^2 = n (sigma_b^2 + zbar_b^2)
                # minus the representative's contribution.  All terms are already
                # in hand, so this costs a handful of elementwise ops.
                s2c, zbc = s2_b[:, :, q0:q1], zbar_b[:, :, q0:q1]
                sum_sq = n_c * (s2c + zbc * zbc) - z_ext * z_ext
                kap = self._kappa((sum_sq / rd - zbar_r * zbar_r).clamp(min=0.0))
            else:
                kap = self._kappa(s2_b[:, :, q0:q1])
            lb = torch.where(has_rest, zbar_r + torch.log(rd) + kap,
                             torch.full_like(z_ext, float("-inf")))
            lse = torch.logaddexp(z_ext, lb)
            wa = torch.exp(z_ext - lse)
            # vbar_new = wa*v* + (1-wa)*(n*vbar - v*)/(n-1), expanded so only two
            # full-size tensors are touched instead of five
            one_m = 1.0 - wa
            alpha = torch.where(has_rest, wa - one_m / rd, torch.ones_like(wa))
            beta = torch.where(has_rest, one_m * n_c / rd, torch.zeros_like(wa))
            vbar_b[:, :, q0:q1] = (vbar_b[:, :, q0:q1] * beta.unsqueeze(-1)
                                   + vext * alpha.unsqueeze(-1))
            ell_out[:, :, q0:q1] = lse
            del vext, z_ext, lb, lse, wa, one_m, alpha, beta
        return ell_out.masked_fill(~live, float("-inf")), vbar_b

    # ------------------------------------------- V28: low-rank covariance factors --
    @staticmethod
    def topr_factors(cov: torch.Tensor, r: int, iters: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-r eigenpairs of a batch of symmetric PSD matrices (..., D, D) by subspace
        iteration + Rayleigh-Ritz.  Returns (U (..., D, r), lam (..., r)) with lam descending.
        Avoids a batched eigh over tens of thousands of D x D matrices."""
        D = cov.shape[-1]
        r = min(r, D)
        g = torch.Generator(device="cpu").manual_seed(4242)
        Q = torch.randn(cov.shape[:-2] + (D, r), generator=g).to(cov.device, cov.dtype)
        eye = torch.eye(r, device=cov.device, dtype=cov.dtype)

        def orth(Y):
            # Orthonormalise the r columns through the r x r Gram matrix (batched QR is a
            # per-matrix loop on GPU; Cholesky fails on rank-deficient bins, e.g. n_b <= r
            # or empty bins with cov = 0).  eigh of the tiny Gram matrix handles both:
            # degenerate directions get a bounded scale and are annihilated by the next
            # multiplication with cov.
            G = torch.matmul(Y.transpose(-1, -2), Y)
            G = 0.5 * (G + G.transpose(-1, -2))
            ev, V = torch.linalg.eigh(G)
            floor = (ev[..., -1:].clamp(min=0) * 1e-6).clamp(min=1e-30)
            inv_sqrt = torch.where(ev > floor, ev.clamp(min=1e-30).rsqrt(), torch.zeros_like(ev))
            return torch.matmul(Y, V * inv_sqrt.unsqueeze(-2))
        Q = orth(Q)
        for _ in range(iters):
            Q = orth(torch.matmul(cov, Q))
        T = torch.matmul(Q.transpose(-1, -2), torch.matmul(cov, Q))         # (..., r, r)
        T = 0.5 * (T + T.transpose(-1, -2))
        lam, W = torch.linalg.eigh(T)                                        # ascending
        lam, W = lam.flip(-1), W.flip(-1)
        return torch.matmul(Q, W), lam.clamp(min=0)

    @classmethod
    def bin_cov_factors(cls, k: torch.Tensor, asg: torch.Tensor, C: int, r: int,
                        chunk: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-bin (D, r) factors of the key covariance.  k (B,H,K,D) fp32, asg (B,H,K)
        bin ids in [0, C).  Returns (Doff (B,H,C,D), L (B,H,C,D,r)) with
        Cov_b ~= diag(Var_b - Doff) + L L^T.  Query-independent: built once per request."""
        B, H, K, D = k.shape
        dev = k.device
        n = torch.zeros((B, H, C), dtype=torch.float32, device=dev)
        S1 = torch.zeros((B, H, C, D), dtype=torch.float32, device=dev)
        S2 = torch.zeros((B, H, C, D * D), dtype=torch.float32, device=dev)
        for k0 in range(0, K, chunk):
            k1 = min(K, k0 + chunk)
            kc = k[:, :, k0:k1].float()
            a = asg[:, :, k0:k1]
            n.scatter_add_(2, a, torch.ones_like(a, dtype=torch.float32))
            S1.scatter_add_(2, a.unsqueeze(-1).expand(-1, -1, -1, D), kc)
            kk = (kc.unsqueeze(-1) * kc.unsqueeze(-2)).reshape(B, H, k1 - k0, D * D)
            S2.scatter_add_(2, a.unsqueeze(-1).expand(-1, -1, -1, D * D), kk)
            del kc, kk
        den = n.clamp(min=1)
        mean = S1 / den.unsqueeze(-1)
        cov = S2.reshape(B, H, C, D, D) / den[..., None, None] - mean.unsqueeze(-1) * mean.unsqueeze(-2)
        cov = torch.where((n >= 2)[..., None, None], cov, torch.zeros_like(cov))
        del S2
        U, lam = cls.topr_factors(cov, r)
        del cov
        L = U * lam.sqrt().unsqueeze(-2)                                    # (B,H,C,D,r)
        Doff = (L * L).sum(-1)                                               # (B,H,C,D)
        return Doff, L

    def _lowrank_factors(self, ent: Dict[str, Any], k: torch.Tensor,
                         C: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cached per-request factors for the profile partition (built at the question
        chunk; NOT refreshed for keys appended during decoding, which join their nearest
        cluster but do not update its factors -- the same staleness the centroids have)."""
        if ent.get("lr_L") is None or ent["lr_L"].shape[2] != C or ent["lr_L"].shape[-1] != self.var_rank:
            K0 = min(k.shape[-2], ent["asg"].shape[-1])
            Doff, L = self.bin_cov_factors(k[:, :, :K0], ent["asg"][:, :, :K0], C, self.var_rank)
            ent["lr_Doff"], ent["lr_L"] = Doff, L
        return ent["lr_Doff"], ent["lr_L"]

    # ------------------------------------------------------------------ bins --
    def _bins_fixed(
        self, raw: torch.Tensor, tail_f: torch.Tensor, v: torch.Tensor,
        key_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Contiguous position blocks intersected with the tail.

        Returns (n_b, zbar_b, s2_b, vbar_b) with bin axis last-but-one.
        """
        m = self.bin_size
        K = raw.shape[-1]
        pad = (-K) % m
        if pad:
            raw = torch.nn.functional.pad(raw, (0, pad), value=NEG - 1.0)
            tail_f = torch.nn.functional.pad(tail_f, (0, pad), value=0.0)
            v = torch.nn.functional.pad(v, (0, 0, 0, pad), value=0.0)
        nb = (K + pad) // m

        shp = raw.shape[:-1] + (nb, m)                       # (B,H,Q,nb,m)
        rb, tb = raw.reshape(shp), tail_f.reshape(shp)
        n_b = tb.sum(-1)                                     # (B,H,Q,nb)
        den = n_b.clamp(min=1)
        zbar_b = (rb * tb).sum(-1) / den
        s2_b = ((rb * rb * tb).sum(-1) / den - zbar_b * zbar_b).clamp(min=0)
        s2_b = torch.where(n_b >= 2, s2_b, torch.zeros_like(s2_b))

        v_view = v.reshape(v.shape[:-2] + (nb, m, v.shape[-1]))   # (B,H,nb,m,D)
        vbar_b = torch.einsum("bhqnm,bhnmd->bhqnd", tb, v_view) / den.unsqueeze(-1)
        return n_b, zbar_b, s2_b, vbar_b

    @staticmethod
    def _hier_quota_impl(score_b: torch.Tensor, n_b: torch.Tensor, live0: torch.Tensor,
                         budget: float, order: torch.Tensor, gamma: float) -> torch.Tensor:
        """V47 water-filling: q_b = min(n_b, lam * w_b), w_b = exp(gamma (s_b - max s)).

        lam is found per row by bisection in log space so that sum_b q_b = budget
        (or every bin is read out when the tail is smaller than the budget).  Quotas
        are floored to whole tokens and the leftover goes one token each to the
        highest-ranked bins that still have room, so the realised expansion is exactly
        min(budget, sum n_b) -- the same accounting as the uniform split.
        """
        ok = live0 & (n_b > 0)
        s = score_b.double().masked_fill(~ok, float("-inf"))
        smax = s.amax(-1, keepdim=True)
        smax = torch.where(torch.isfinite(smax), smax, torch.zeros_like(smax))
        w = torch.exp(gamma * (s - smax)).masked_fill(~ok, 0.0)
        cap = n_b.double().masked_fill(~ok, 0.0)
        tot = cap.sum(-1, keepdim=True)
        B = float(budget)
        # f(lam) = sum min(cap, lam w) is nondecreasing; f(1e-3) < 1 <= B, and at
        # lam = max cap/w every weighted bin is at its cap
        ratio = torch.where(w > 0, cap / w.clamp(min=1e-300), torch.zeros_like(w))
        lo = torch.full_like(tot, -3.0 * 2.302585092994046)
        hi = torch.log(ratio.amax(-1, keepdim=True).clamp(min=1e-3)) + 1.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            f = torch.minimum(cap, torch.exp(mid) * w).sum(-1, keepdim=True)
            big = f > B
            hi = torch.where(big, mid, hi)
            lo = torch.where(big, lo, mid)
        q = torch.minimum(cap, torch.exp(lo) * w).floor()
        q = torch.where(tot <= B, cap, q)
        r = (torch.clamp(tot, max=B) - q.sum(-1, keepdim=True)).clamp(min=0).round().long()
        room_s = (q < cap).gather(-1, order)
        rk = room_s.long().cumsum(-1) - 1
        add_s = room_s & (rk < r)
        add = torch.zeros_like(room_s).scatter_(-1, order, add_s)
        q = q + add.double()
        return q.to(n_b.dtype)

    def _hier_quota(self, score_b, n_b, live0, budget, order):
        return self._hier_quota_impl(score_b, n_b, live0, budget, order,
                                     self.expand_alloc_gamma)

    def _scatter_mean(self, x: torch.Tensor, tail_f: torch.Tensor, bid: torch.Tensor,
                      nbins: int, den: torch.Tensor) -> torch.Tensor:
        """Per-bin tail mean of a key/value-side tensor, for irregular bin ids."""
        B, H, Q, _ = tail_f.shape
        D = x.shape[-1]
        acc = torch.zeros((B, H, Q, nbins + 1, D), dtype=tail_f.dtype, device=tail_f.device)
        hstep = max(1, 8 // max(1, Q))
        for h0 in range(0, H, hstep):
            h1 = min(H, h0 + hstep)
            src = x[:, h0:h1].unsqueeze(2) * tail_f[:, h0:h1].unsqueeze(-1)
            idx = bid[:, h0:h1].unsqueeze(-1).expand(-1, -1, -1, -1, D)
            acc[:, h0:h1].scatter_add_(3, idx, src)
            del src, idx
        return acc[..., :nbins, :] / den.unsqueeze(-1)

    def _bin_moments_scatter(self, q: torch.Tensor, k: torch.Tensor,
                             tail_f: torch.Tensor, bid: torch.Tensor, nbins: int,
                             den: torch.Tensor,
                             scaling: float,
                             lr: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(zbar_b, sigma_b^2) for irregular (adjacency) bins, key-side only."""
        with stage("mb/T4a_bin_k_mean"):
            k_mean = self._scatter_mean(k, tail_f, bid, nbins, den)
        with stage("mb/T4b_bin_k_sq_mean"):
            k_sq = self._scatter_mean(k * k, tail_f, bid, nbins, den)
        with stage("mb/T4c_zbar_and_sigma"):
            return self._moments_from_key_stats(q, k_mean, k_sq, scaling, lr=lr)

    def _diag_var_scatter(self, q: torch.Tensor, k: torch.Tensor, tail_f: torch.Tensor,
                          bid: torch.Tensor, nbins: int, den: torch.Tensor,
                          scaling: float) -> torch.Tensor:
        """sigma_b^2 only, for callers that already have zbar_b."""
        return self._bin_moments_scatter(q, k, tail_f, bid, nbins, den, scaling)[1]

    def _aggregate_by_bin_id(
        self, raw: torch.Tensor, tail_f: torch.Tensor, v: torch.Tensor,
        bid: torch.Tensor, nbins: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scatter-reduce tail statistics into `nbins` bins given per-token bin ids.

        `bid` must already route every non-tail position to index `nbins`, a discard
        bin that is dropped on the way out.

        Statistics are always taken over the TRUE logits in `raw`, never over whatever
        quantity produced `bid`.  That separation is what lets the "hat" mode group by
        an *approximate* score without letting the approximation into the bin logit.

        Prototype only: materialises a (B,h_chunk,Q,K,D) buffer, so it is chunked over
        heads and is considerably slower than "fixed".
        """
        B, H, Q, _ = raw.shape
        D = v.shape[-1]

        shp = (B, H, Q, nbins + 1)
        z = torch.zeros(shp, dtype=raw.dtype, device=raw.device)
        n_b = z.clone().scatter_add_(-1, bid, tail_f)
        zsum = z.clone().scatter_add_(-1, bid, raw * tail_f)
        z2sum = z.clone().scatter_add_(-1, bid, raw * raw * tail_f)

        vsum = torch.zeros((B, H, Q, nbins + 1, D), dtype=raw.dtype, device=raw.device)
        hstep = max(1, 8 // max(1, Q))          # keep the temp buffer ~O(100MB)
        for h0 in range(0, H, hstep):
            h1 = min(H, h0 + hstep)
            src = v[:, h0:h1].unsqueeze(2) * tail_f[:, h0:h1].unsqueeze(-1)   # (B,h,Q,K,D)
            idx = bid[:, h0:h1].unsqueeze(-1).expand(-1, -1, -1, -1, D)
            vsum[:, h0:h1].scatter_add_(3, idx, src)
            del src, idx

        n_b, zsum = n_b[..., :nbins], zsum[..., :nbins]
        z2sum, vsum = z2sum[..., :nbins], vsum[..., :nbins, :]
        den = n_b.clamp(min=1)
        zbar_b = zsum / den
        s2_b = (z2sum / den - zbar_b * zbar_b).clamp(min=0)
        s2_b = torch.where(n_b >= 2, s2_b, torch.zeros_like(s2_b))
        return n_b, zbar_b, s2_b, vsum / den.unsqueeze(-1)

    def _bin_ids_equalcount(self, tail_f: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Runs of exactly `bin_size` ADJACENT tail indices (the complement of top-k).

        Membership depends only on positions, never on logits, so it escapes the
        circularity: a deployable kernel gets each run's sums from cached block prefix
        sums minus the selected tokens it is already loading.
        """
        m = self.bin_size
        K = tail_f.shape[-1]
        nbins = (K + m - 1) // m
        rank = (tail_f.cumsum(-1) - 1).clamp(min=0).long()
        bid = (rank // m).clamp(max=nbins - 1)
        return torch.where(tail_f > 0, bid, torch.full_like(bid, nbins)), nbins

    def _bins_equalcount(
        self, raw: torch.Tensor, tail_f: torch.Tensor, v: torch.Tensor,
        key_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bid, nbins = self._bin_ids_equalcount(tail_f)
        return self._aggregate_by_bin_id(raw, tail_f, v, bid, nbins)

    def _bin_ids_random(self, tail_f: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Uniformly random assignment of tail tokens to `num_bins` bins.

        Membership ignores position AND logits.  The assignment is drawn once per key
        POSITION and shared by every query row, which keeps the two properties that
        make a bin statistic deployable: it needs no logits (no circularity), and
        k_bar_b / Cov_b(K) stay query-independent, so one read serves the whole batch.

        What this isolates: every bin is an unbiased sample of the same tail, so the
        within-bin logit variance stays at the global tail variance for any bin count.
        Only the spread of the bin MEANS grows -- Var(zbar_b) ~ sigma^2 * B / n -- and
        that is the whole mechanism by which random binning recovers Jensen mass.
        """
        nbins = self.num_bins
        K = tail_f.shape[-1]
        # Cache across layers/steps: the partition must not be re-drawn per call, or
        # the tail proxy would jitter between decode steps for no reason.
        if (self._rand_bid is None or self._rand_bid.numel() < K
                or self._rand_bid.device != tail_f.device):
            gen = torch.Generator(device="cpu").manual_seed(self.random_seed)
            self._rand_bid = torch.randint(
                nbins, (max(K, 1),), generator=gen, dtype=torch.long).to(tail_f.device)
        bid = self._rand_bid[:K].expand(tail_f.shape).contiguous()
        return torch.where(tail_f > 0, bid, torch.full_like(bid, nbins)), nbins

    def _bins_random(
        self, raw: torch.Tensor, tail_f: torch.Tensor, v: torch.Tensor,
        key_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bid, nbins = self._bin_ids_random(tail_f)
        return self._aggregate_by_bin_id(raw, tail_f, v, bid, nbins)

    def _bins_score(
        self, raw: torch.Tensor, tail_f: torch.Tensor, v: torch.Tensor,
        key_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Equal-width bins over `key_scores`, taken across the tail's observed range.

        Grouping by score is what actually attacks UTA's error term: within-bin logit
        variance falls roughly as (range/num_bins)^2, whereas position blocks bottom
        out around half the global tail variance no matter how small they get.
        """
        bid, nbins = self._bin_ids_score(raw, tail_f, key_scores)
        return self._aggregate_by_bin_id(raw, tail_f, v, bid, nbins)

    def _bin_ids_score(self, raw: torch.Tensor, tail_f: torch.Tensor,
                       key_scores: Optional[torch.Tensor]) -> Tuple[torch.Tensor, int]:
        if key_scores is None:
            key_scores = raw
        nbins = self.num_bins
        tail = tail_f > 0
        s = key_scores.to(raw.dtype)

        lo = s.masked_fill(~tail, float("inf")).amin(-1, keepdim=True)
        hi = s.masked_fill(~tail, float("-inf")).amax(-1, keepdim=True)
        width = (hi - lo).clamp(min=1e-6)
        bid = ((s - lo) / width * nbins).floor().long().clamp(0, nbins - 1)
        return torch.where(tail, bid, torch.full_like(bid, nbins)), nbins

    # ------------------------------------------------- profile partition (V19) --
    @torch.no_grad()
    # ------------------------------------------------------ V23 profile sources --
    def _static_entry(self, device: torch.device) -> Dict[str, Any]:
        if self._static is None:
            d = torch.load(self.profile_static_file, map_location="cpu")
            self._static = {"M": d.get("M"), "inv_freq": d["inv_freq"].float().to(device),
                            "scaling": float(d.get("attention_scaling", 1.0))}
        return self._static

    @staticmethod
    def _rope_cs(inv_freq: torch.Tensor, scaling: float, positions: torch.Tensor):
        freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], -1)
        return emb.cos() * scaling, emb.sin() * scaling

    @staticmethod
    def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin

    @staticmethod
    def _rope_matrix(inv_freq: torch.Tensor, scaling: float, pos: int) -> torch.Tensor:
        D = 2 * inv_freq.numel()
        cos, sin = UTAMultiBinAttention._rope_cs(inv_freq, scaling,
                                                 torch.tensor([pos], device=inv_freq.device))
        J = torch.zeros(D, D, device=inv_freq.device)
        h = D // 2
        J[:h, h:] = -torch.eye(h, device=inv_freq.device)
        J[h:, :h] = torch.eye(h, device=inv_freq.device)
        return torch.diag(cos[0]) + torch.diag(sin[0]) @ J

    @torch.no_grad()
    def _profile_queries(self, q: torch.Tensor, pi: torch.Tensor, K: int, scaling: float,
                         layer_idx: int) -> torch.Tensor:
        """(B,H,P',D) profile queries for the chosen source (see UTAMultiBinConfig)."""
        qsel = q[:, :, pi, :].contiguous()                         # post-RoPE rows
        src = self.profile_source
        if src == "question":
            return qsel
        if src == "static+qrot":
            st = self._static_entry(q.device)
            inv, rsc = st["inv_freq"], st["scaling"]
            B, H, Q, D = q.shape
            # question block: rows rotated to the decode start K, top-r_q eigen-directions
            delta = (Q - pi).to(q.device)
            cos, sin = self._rope_cs(inv, rsc, delta)
            qrot = self._rope_apply(qsel.float(), cos, sin)
            Mq = torch.matmul(qrot.transpose(-1, -2), qrot) / max(qrot.shape[2], 1)   # (B,H,D,D)
            eq, vq = torch.linalg.eigh(Mq)
            rq = min(self.profile_rank_q, D)
            Wq = vq[..., -rq:] * eq[..., -rq:].clamp(min=0).sqrt().unsqueeze(-2)     # (B,H,D,rq)
            # static block: rotation-averaged calibration M, top-r, rescaled to the rows' RMS norm
            Mpre = st["M"][layer_idx].to(q.device).float()
            w = max(self.profile_rot_window, 1)
            Mst = torch.zeros_like(Mpre)
            for t in range(w):
                R = self._rope_matrix(inv, rsc, K + t)
                Mst = Mst + R @ Mpre @ R.transpose(-1, -2)
            Mst = Mst / w
            es, vs = torch.linalg.eigh(Mst)
            r = min(self.profile_rank, D)
            Ws = vs[..., -r:] * es[..., -r:].clamp(min=0).sqrt().unsqueeze(-2)        # (H,D,r)
            qrms = qsel.float().pow(2).sum(-1).mean(-1).sqrt()                          # (B,H)
            trs = torch.diagonal(Mst, dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12).sqrt()
            Ws = Ws.unsqueeze(0) * (qrms / trs.unsqueeze(0))[..., None, None]           # (B,H,D,r)
            return torch.cat([Wq.transpose(-1, -2), Ws.transpose(-1, -2)], 2).contiguous().to(q.dtype)
        if src == "static+rows":
            st = self._static_entry(q.device)
            inv, rsc = st["inv_freq"], st["scaling"]
            B, H, Q, D = q.shape
            Mpre = st["M"][layer_idx].to(q.device).float()
            w = max(self.profile_rot_window, 1)
            Mst = torch.zeros_like(Mpre)
            for t in range(w):
                R = self._rope_matrix(inv, rsc, K + t)
                Mst = Mst + R @ Mpre @ R.transpose(-1, -2)
            evals, evecs = torch.linalg.eigh(Mst / w)
            r = min(self.profile_rank, D)
            W = evecs[..., -r:] * evals[..., -r:].clamp(min=0).sqrt().unsqueeze(-2)   # (H,D,r)
            # The calibration file stores M trace-normalised, so its eigenvalues carry no
            # magnitude.  Rescale the virtual queries u_i*sqrt(lambda_i) to the RMS norm of
            # this request's real rows: then s*W^T k sits in the same logit units as the
            # row block and neither block dominates the k-means distance.
            qrms = qsel.float().pow(2).sum(-1).mean(-1).sqrt()                  # (B,H)
            trs = torch.diagonal(Mst / w, dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12).sqrt()   # (H,)
            W = W.unsqueeze(0) * (qrms / trs.unsqueeze(0))[..., None, None]     # (B,H,D,r)
            virt = W.transpose(-1, -2).to(q.dtype)
            return torch.cat([qsel, virt], 2).contiguous()          # (B,H,P+r,D)
        st = self._static_entry(q.device)
        inv, rsc = st["inv_freq"], st["scaling"]
        B, H, Q, D = q.shape
        M = None
        if src in ("question_rot", "static+question"):
            # row i sits at position K-Q+i; advance it by Q-i so every row is at K
            delta = (Q - pi).to(q.device)
            cos, sin = self._rope_cs(inv, rsc, delta)
            qrot = self._rope_apply(qsel.float(), cos, sin)
            Mq = torch.matmul(qrot.transpose(-1, -2), qrot) / max(qrot.shape[2], 1)
            M = Mq
        if src in ("static", "static+question"):
            Mpre = st["M"][layer_idx].to(q.device).float()         # (H,D,D) pre-RoPE
            w = max(self.profile_rot_window, 1)
            Mst = torch.zeros_like(Mpre)
            for t in range(w):
                R = self._rope_matrix(inv, rsc, K + t)
                Mst = Mst + R @ Mpre @ R.transpose(-1, -2)
            Mst = (Mst / w).unsqueeze(0).expand(B, -1, -1, -1)
            if M is None:
                M = Mst
            else:
                def _nt(x):
                    return x / torch.diagonal(x, dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12)[..., None, None]
                M = 0.5 * _nt(Mst) + 0.5 * _nt(M)
        evals, evecs = torch.linalg.eigh(M)
        r = min(self.profile_rank, D)
        U = evecs[..., -r:]
        lam = evals[..., -r:].clamp(min=0).sqrt()
        if self.profile_basis == "random":
            # V57: a random orthonormal D x r frame (seeded per layer) in place of the eigenvectors,
            # every direction weighted by the same RMS eigenvalue so the units are unchanged.
            g = torch.Generator(device="cpu").manual_seed(self.random_seed + 977 * (layer_idx + 1))
            G = torch.randn(tuple(M.shape[:-2]) + (D, r), generator=g).to(M.device, M.dtype)
            U, _ = torch.linalg.qr(G)
            lam = (torch.diagonal(M, dim1=-2, dim2=-1).sum(-1) / r).clamp(min=0).sqrt()[..., None].expand_as(lam)
        W = U * lam.unsqueeze(-2)                                  # (B,H,D,r)
        # z = k @ qprof^T * scaling  must equal  k @ W  -> qprof = W^T / scaling
        return (W.transpose(-1, -2) / scaling).contiguous().to(q.dtype)

    def _profile_build(self, q: torch.Tensor, k: torch.Tensor, scaling: float,
                       layer_idx: int, qprof: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Partition every key by its logit profile against P sampled query rows.

        The profile of token i is (s * q_p . k_i) for P rows sampled from the
        question chunk, columns centred over tokens.  Tokens whose logits MOVE
        TOGETHER across queries share a cluster -- the partition objective V9
        4.2(N2) identified (alignment with logit order), from quantities that are
        available once per request at prefill end.  Assignment covers ALL keys;
        per-row tail restriction happens at bin-id construction time.

        profile_method="kmeans": Lloyd iterations (profile_iters) in R^P.
        profile_method="qsort" : no iterations -- sort by the 1-D statistic qmean
                                 and cut into C equal-count bins (V22).
        profile_dtype="bf16"   : the two large matmuls run on tensor cores.
        """
        B, H, Q, D = q.shape
        K = k.shape[-2]
        mm_dt = torch.bfloat16 if self.profile_dtype == "bf16" else torch.float32
        P = min(self.profile_P, max(Q - 2, 1))
        pi = torch.linspace(0, max(Q - 3, 0), P, device=q.device).long().unique()
        # V54: after a KV press the true decode position is K + offset (see custom_attention);
        # V54b: a rebuild over a rewritten cache reuses the profile queries (`qprof`).
        if qprof is None:
            qprof = self._profile_queries(q, pi, K + int(getattr(self, "_pos_offset", 0)),
                                          scaling, layer_idx)   # (B,H,P',D)
        zP = torch.matmul(k.to(mm_dt), qprof.to(mm_dt).transpose(-1, -2)).float() \
            * scaling                                              # (B,H,K,P')
        qmean = zP.mean(-1)                                        # (B,H,K)
        C = min(self.profile_C, K)
        ent: Dict[str, Any] = {"K": K, "C": C, "qprof": qprof, "qmean": qmean,
                               "method": self.profile_method}

        if self.profile_method == "keykm":
            # RetroInfer / ClusterKV style: spherical k-means over the KEY vectors,
            # query-agnostic (the profile queries are ignored).  Same C, same
            # Lloyd iteration count as the profile k-means, so the two rows of the
            # benchmark differ ONLY in the clustering space.
            kn = torch.nn.functional.normalize(k, dim=-1)              # (B,H,K,D)
            g = torch.Generator(device="cpu").manual_seed(self.random_seed + 31 * (layer_idx + 1))
            init = torch.randperm(K, generator=g)[:C].to(q.device)
            cent = kn.index_select(2, init).clone()                    # (B,H,C,D)
            asg = torch.zeros((B, H, K), dtype=torch.long, device=q.device)
            kn_mm = kn.to(mm_dt)
            hstep = max(1, min(H, int(512e6 / max(1, B * K * C * 4))))
            for _ in range(self.profile_iters):
                cent_mm = cent.to(mm_dt)
                for h0 in range(0, H, hstep):
                    h1 = min(H, h0 + hstep)
                    sim = torch.matmul(kn_mm[:, h0:h1], cent_mm[:, h0:h1].transpose(-1, -2))
                    asg[:, h0:h1] = sim.argmax(-1)
                    del sim
                acc = torch.zeros((B, H, C, D), dtype=kn.dtype, device=kn.device)
                cnt = torch.zeros((B, H, C, 1), dtype=kn.dtype, device=kn.device)
                acc.scatter_add_(2, asg.unsqueeze(-1).expand(-1, -1, -1, D), kn)
                cnt.scatter_add_(2, asg.unsqueeze(-1), torch.ones_like(kn[..., :1]))
                cent = torch.nn.functional.normalize(acc / cnt.clamp(min=1), dim=-1)
                cent = torch.where(cnt > 0, cent, kn.index_select(2, init))
                del acc, cnt
            ent.update({"asg": asg, "kcent": cent})
            return ent

        if self.profile_method == "retro":
            # RetroInfer (Microsoft RetrievalAttention, cache_hub/kmeans.py
            # `segment_k_means`) partition, ported faithfully: keys are MEAN-
            # CENTRED per (batch, head); centroids are initialised by uniform
            # POSITION stride; the first (iters-1) Lloyd rounds are SEGMENT-LOCAL
            # (one contiguous positional segment per 8192 tokens, a token may only
            # join a centroid of its own segment) with centroids L2-renormalised
            # after each update (spherical k-means, max-inner-product assign);
            # the final round is GLOBAL over all C centroids and leaves the
            # centroids as plain means.  Only the assignment is consumed here.
            ent.update(self._retro_partition(k, C, layer_idx, mm_dt))
            return ent

        if self.profile_method == "centroidkv":
            # CentroidKV (Hu et al. 2025, centroidkv_press.py) grouping, ported
            # faithfully: iterative ToMe-style BIPARTITE SOFT MATCHING on cosine
            # similarity of (count-weighted mean) keys inside contiguous chunks
            # of 256 slots, the top-r edges merged globally per round with the
            # decaying ratio schedule 0.8/0.6/0.4/0.2, until C groups remain.
            # Each final merge group is one bin.
            ent.update(self._centroidkv_partition(k, C, mm_dt))
            return ent

        if self.profile_method == "qsort":
            # equal-count bins along qmean; edges cached for decode bucketing
            order = qmean.argsort(-1)
            rank = torch.empty_like(order)
            rank.scatter_(-1, order, torch.arange(K, device=q.device).expand_as(order))
            asg = (rank * C) // K
            srt = qmean.gather(-1, order)                          # ascending
            cut = (torch.arange(1, C, device=q.device) * K) // C   # first index of bin c
            ent["edges"] = srt[..., cut].contiguous()              # (B,H,C-1)
            ent["asg"] = asg
            return ent

        if self.profile_method in ("geo", "geo2"):
            # NO iterations, VARIABLE-size bins: sort by qmean (hottest first) and
            # cut with a geometric size schedule -- the top tokens get singleton
            # bins (exact proxies), the cold bulk gets wide bins.  Mimics what
            # k-means does for outliers (tiny clusters) with one sort.  Decode
            # tokens are bucketised against the cached bin-start qmean values.
            order = qmean.argsort(-1, descending=True)
            rank = torch.empty_like(order)
            rank.scatter_(-1, order, torch.arange(K, device=q.device).expand_as(order))
            sizes: List[int] = []
            if self.profile_method == "geo2":
                # 8 tiers x C/8 bins, sizes 1,2,4,...,128: total C/8*255 ~ K at C=1024
                per = max(1, C // 8)
                for t in range(7):
                    sizes += [2 ** t] * per
                nb_left = C - len(sizes)
            else:
                nb_left, t = C, 0
                while nb_left > 1 and t < 5:
                    nb_t = nb_left // 2
                    sizes += [4 ** t] * nb_t
                    nb_left -= nb_t
                    t += 1
            used = sum(sizes)
            rem = max(K - used, 0)
            sizes += [rem // nb_left + (1 if i < rem % nb_left else 0)
                      for i in range(nb_left)]
            sz = torch.tensor(sizes, device=q.device)
            starts = sz.cumsum(0) - sz                              # (C,)
            asg = (torch.searchsorted(starts, rank.reshape(-1).contiguous(), right=True)
                   - 1).clamp(0, C - 1).view_as(rank)
            srt_desc = qmean.gather(-1, order)                     # (B,H,K) descending
            st_idx = starts.clamp(max=K - 1)
            edges_desc = srt_desc[..., st_idx]                     # (B,H,C) qmean at bin starts
            ent["edges_neg"] = (-edges_desc).contiguous()          # ascending
            ent["asg"] = asg
            return ent

        if self.profile_method == "grid":
            G1 = int(round(C ** 0.5))
            G2 = C // G1
            colmean = zP.mean(dim=2, keepdim=True)
            zc = zP - colmean                                      # (B,H,K,P)
            gram = torch.matmul(zc.transpose(-1, -2), zc)          # (B,H,P,P)
            evals, evecs = torch.linalg.eigh(gram)                 # ascending
            V2 = evecs[..., -2:].flip(-1).contiguous()             # (B,H,P,2) top-2
            sc2 = torch.matmul(zc, V2)                             # (B,H,K,2)
            s1, s2 = sc2[..., 0], sc2[..., 1]
            o1 = s1.argsort(-1)
            r1 = torch.empty_like(o1)
            r1.scatter_(-1, o1, torch.arange(K, device=q.device).expand_as(o1))
            g1 = (r1 * G1) // K                                    # (B,H,K) in [0,G1)
            e1 = s1.gather(-1, o1)[..., (torch.arange(1, G1, device=q.device) * K) // G1]
            # rank of s2 WITHIN each g1 group: sort by (g1, s2)
            o2 = s2.argsort(-1)
            r2 = torch.empty_like(o2)
            r2.scatter_(-1, o2, torch.arange(K, device=q.device).expand_as(o2))
            comp = g1 * K + r2
            oc = comp.argsort(-1)
            rc = torch.empty_like(oc)
            rc.scatter_(-1, oc, torch.arange(K, device=q.device).expand_as(oc))
            cnt1 = torch.zeros(B, H, G1, device=q.device, dtype=torch.long)
            cnt1.scatter_add_(-1, g1, torch.ones_like(g1))
            start1 = cnt1.cumsum(-1) - cnt1                        # (B,H,G1)
            rin = rc - start1.gather(-1, g1)                       # rank within group
            size1 = cnt1.gather(-1, g1).clamp(min=1)
            g2 = ((rin * G2) // size1).clamp(max=G2 - 1)
            asg = g1 * G2 + g2
            # per-group s2 edges for decode bucketing: s2 sorted within group
            s2_sorted = s2.gather(-1, oc)                          # grouped by g1, asc s2
            e2 = torch.empty(B, H, G1, G2 - 1, device=q.device, dtype=s2.dtype)
            for gi in range(G1):
                st = start1[..., gi]                               # (B,H)
                sz = cnt1[..., gi].clamp(min=1)
                pos = st.unsqueeze(-1) + (torch.arange(1, G2, device=q.device) * sz.unsqueeze(-1)) // G2
                e2[..., gi, :] = s2_sorted.gather(-1, pos.clamp(max=K - 1))
            ent.update({"asg": asg, "colmean": colmean, "V2": V2, "e1": e1.contiguous(),
                        "e2": e2.contiguous(), "G1": G1, "G2": G2})
            return ent

        colmean = zP.mean(dim=2, keepdim=True)
        zc = zP - colmean
        Pn = zc.shape[-1]
        g = torch.Generator(device="cpu").manual_seed(self.random_seed + 31 * (layer_idx + 1))
        init = torch.randperm(K, generator=g)[:C].to(q.device)
        cent = zc.index_select(2, init).clone()                    # (B,H,C,P) fp32
        asg = torch.zeros((B, H, K), dtype=torch.long, device=q.device)
        zc_mm = zc.to(mm_dt)
        hstep = max(1, min(H, int(512e6 / max(1, B * K * C * 4))))
        for _ in range(self.profile_iters):
            cent_mm = cent.to(mm_dt)
            half = (0.5 * (cent ** 2).sum(-1)).to(mm_dt).unsqueeze(2)   # (B,H,1,C)
            for h0 in range(0, H, hstep):
                h1 = min(H, h0 + hstep)
                sim = torch.matmul(zc_mm[:, h0:h1], cent_mm[:, h0:h1].transpose(-1, -2))
                sim = sim - half[:, h0:h1]
                asg[:, h0:h1] = sim.argmax(-1)
                del sim
            acc = torch.zeros((B, H, C, Pn), dtype=zc.dtype, device=zc.device)
            cnt = torch.zeros((B, H, C, 1), dtype=zc.dtype, device=zc.device)
            acc.scatter_add_(2, asg.unsqueeze(-1).expand(-1, -1, -1, Pn), zc)
            cnt.scatter_add_(2, asg.unsqueeze(-1), torch.ones_like(zc[..., :1]))
            cent = torch.where(cnt > 0, acc / cnt.clamp(min=1), zc.index_select(2, init))
            del acc, cnt
        ent.update({"asg": asg, "cent": cent, "colmean": colmean})
        return ent

    @torch.no_grad()
    def _retro_partition(self, k: torch.Tensor, C: int, layer_idx: int,
                         mm_dt: torch.dtype) -> Dict[str, Any]:
        """RetroInfer `segment_k_means` over the keys of every head.  Returns
        {"asg","kcent","kmean","C_used"}; `kcent` are L2-normalised centroids of the
        mean-centred keys (used for decode-time nearest-centroid extension)."""
        B, H, K, D = k.shape
        n_iters = max(2, int(self.profile_iters))
        S = max(int(round(K / 8192)), 1)
        S = min(S, C)
        Cs = max(C // S, 1)
        C = Cs * S                                   # RetroInfer requires C % S == 0
        kmean = k.mean(dim=2, keepdim=True)          # (B,H,1,D)
        kc = k - kmean
        # uniform positional init (deterministic, as in the original)
        ci = (torch.arange(C, device=k.device, dtype=torch.float32) * (K / C)
              + K / C / 2).long().clamp(max=K - 1)
        cent = kc.index_select(2, ci).clone()        # (B,H,C,D)
        Ks = K // S
        asg = torch.zeros((B, H, K), dtype=torch.long, device=k.device)

        def _assign(data, cents, out):
            # data (B,H',S',N,D) cents (B,H',S',M,D) -> argmax inner product
            hs = max(1, min(H, int(512e6 / max(1, B * data.shape[2] * data.shape[3] * cents.shape[3] * 4))))
            for h0 in range(0, H, hs):
                h1 = min(H, h0 + hs)
                sim = torch.matmul(data[:, h0:h1].to(mm_dt),
                                   cents[:, h0:h1].to(mm_dt).transpose(-1, -2))
                out[:, h0:h1] = sim.argmax(-1)
                del sim

        def _update(data, idx, M, normalize):
            # data (B,H,N,D) idx (B,H,N) in [0,M) -> mean per centroid, fp32
            acc = torch.zeros((B, H, M, D), dtype=torch.float32, device=k.device)
            cnt = torch.zeros((B, H, M, 1), dtype=torch.float32, device=k.device)
            acc.scatter_add_(2, idx.unsqueeze(-1).expand(-1, -1, -1, D), data.float())
            cnt.scatter_add_(2, idx.unsqueeze(-1), torch.ones_like(data[..., :1], dtype=torch.float32))
            new = acc / cnt.clamp(min=1)
            if normalize:
                new = torch.nn.functional.normalize(new, dim=-1)
            return new, cnt

        # ---- segment-local rounds (centroids renormalised) ----
        data_s = kc[:, :, :Ks * S].reshape(B, H, S, Ks, D)
        loc = torch.zeros((B, H, S, Ks), dtype=torch.long, device=k.device)
        cent_n = torch.nn.functional.normalize(cent, dim=-1)
        for _ in range(n_iters - 1):
            _assign(data_s, cent_n.view(B, H, S, Cs, D), loc)
            gidx = (loc + (torch.arange(S, device=k.device) * Cs).view(1, 1, S, 1)).reshape(B, H, Ks * S)
            new, cnt = _update(kc[:, :, :Ks * S], gidx, C, normalize=True)
            cent_n = torch.where(cnt > 0, new, cent_n)   # empty clusters keep the old centroid
        # ---- final global round (centroids left as means) ----
        _assign(kc.unsqueeze(2), cent_n.unsqueeze(2), asg.unsqueeze(2))
        cent_mean, cnt = _update(kc, asg, C, normalize=False)
        kcent = torch.nn.functional.normalize(torch.where(cnt > 0, cent_mean, cent_n), dim=-1)
        return {"asg": asg, "kcent": kcent, "kmean": kmean, "C": C}

    @torch.no_grad()
    def _centroidkv_partition(self, k: torch.Tensor, C: int,
                              mm_dt: torch.dtype) -> Dict[str, Any]:
        """CentroidKV bipartite-soft-matching merge, run until C groups remain.
        Returns {"asg","kcent"}; kcent = L2-normalised mean key per group."""
        B, H, K, D = k.shape
        dev = k.device
        cs = 256                                     # CentroidKV chunk_size
        ratio0, dec = 0.8, 0.2                       # merge_ratio / decrease_rate
        key = k.float().clone()                      # (B,H,n,D) slot keys (weighted means)
        cnt = torch.ones((B, H, K, 1), dtype=torch.float32, device=dev)
        slab = torch.arange(K, device=dev).view(1, 1, K).expand(B, H, K).clone()  # slot label
        label = slab.clone()                         # token -> representative token
        n = K
        step = 0
        while n > C and step < 200:
            ratio = max(ratio0 - step * dec, dec)
            odd = n % 2
            reg = n - odd
            nc = reg // cs
            nex = reg - nc * cs                      # even
            nA = reg // 2
            r = min(n - C, int(reg * ratio / 2))
            if r <= 0:
                break
            kn = torch.nn.functional.normalize(key[:, :, :reg], dim=-1).to(mm_dt)
            a_base = torch.empty(nA, dtype=torch.long, device=dev)
            parts_max, parts_idx = [], []
            if nc > 0:
                kcn = kn[:, :, :nc * cs].reshape(B, H, nc, cs, D)
                sc = torch.matmul(kcn[:, :, :, ::2], kcn[:, :, :, 1::2].transpose(-1, -2)).float()
                mx, ix = sc.max(-1)                  # (B,H,nc,cs/2)
                parts_max.append(mx.reshape(B, H, -1))
                parts_idx.append(ix.reshape(B, H, -1))
                a_base[: nc * cs // 2] = (torch.arange(nc, device=dev) * cs).repeat_interleave(cs // 2)
                del sc, kcn
            if nex > 0:
                ke = kn[:, :, nc * cs:reg]
                se = torch.matmul(ke[..., ::2, :], ke[..., 1::2, :].transpose(-1, -2)).float()
                mx, ix = se.max(-1)
                parts_max.append(mx)
                parts_idx.append(ix)
                a_base[nc * cs // 2:] = nc * cs
                del se, ke
            node_max = torch.cat(parts_max, -1)      # (B,H,nA)
            node_idx = torch.cat(parts_idx, -1)      # (B,H,nA) local b index
            del kn
            edge_idx = node_max.argsort(-1, descending=True)
            src_a = edge_idx[..., :r]                # a-side indices to merge
            unm_a = edge_idx[..., r:]                # a-side indices preserved
            src_slot = a_base[src_a] + 2 * (src_a - a_base[src_a] // 2)
            unm_slot = a_base[unm_a] + 2 * (unm_a - a_base[unm_a] // 2)
            dst_b = a_base[src_a] // 2 + node_idx.gather(-1, src_a)   # b-side index
            # b side = odd slots of the region
            key_b = key[:, :, :reg][:, :, 1::2]
            cnt_b = cnt[:, :, :reg][:, :, 1::2]
            lab_b = slab[:, :, :reg][:, :, 1::2]
            key_s = key.gather(2, src_slot.unsqueeze(-1).expand(-1, -1, -1, D))
            cnt_s = cnt.gather(2, src_slot.unsqueeze(-1))
            lab_s = slab.gather(2, src_slot)
            wk = torch.zeros_like(key_b).scatter_add_(2, dst_b.unsqueeze(-1).expand(-1, -1, -1, D), key_s * cnt_s)
            wc = torch.zeros_like(cnt_b).scatter_add_(2, dst_b.unsqueeze(-1), cnt_s)
            new_cnt_b = cnt_b + wc
            new_key_b = (key_b * cnt_b + wk) / new_cnt_b
            # relabel every token of a merged source to its destination representative
            lab_d = lab_b.gather(2, dst_b)
            remap = torch.arange(K, device=dev).view(1, 1, K).expand(B, H, K).clone()
            remap.scatter_(2, lab_s, lab_d)
            label = remap.gather(2, label)
            # preserved a-side slots + updated b-side slots (+ odd leftover)
            key_u = key.gather(2, unm_slot.unsqueeze(-1).expand(-1, -1, -1, D))
            cnt_u = cnt.gather(2, unm_slot.unsqueeze(-1))
            lab_u = slab.gather(2, unm_slot)
            parts_k, parts_c, parts_l = [key_u, new_key_b], [cnt_u, new_cnt_b], [lab_u, lab_b]
            if odd:
                parts_k.append(key[:, :, reg:n]); parts_c.append(cnt[:, :, reg:n]); parts_l.append(slab[:, :, reg:n])
            key = torch.cat(parts_k, 2); cnt = torch.cat(parts_c, 2); slab = torch.cat(parts_l, 2)
            n = key.shape[2]
            step += 1
        # compact: slot j <- label; asg[token] = slot index of its representative
        inv = torch.zeros((B, H, K), dtype=torch.long, device=dev)
        inv.scatter_(2, slab, torch.arange(n, device=dev).view(1, 1, n).expand(B, H, n))
        asg = inv.gather(2, label)
        kcent = torch.nn.functional.normalize(key, dim=-1)   # (B,H,n,D)
        return {"asg": asg, "kcent": kcent, "C": n}

    @torch.no_grad()
    def _profile_extend(self, ent: Dict[str, Any], k: torch.Tensor,
                        scaling: float) -> Dict[str, Any]:
        """Assign keys appended since prefill: nearest cached centroid (kmeans) or
        bucket against the cached qmean edges (qsort)."""
        K0, K = ent["K"], k.shape[-2]
        knew = k[:, :, K0:, :]
        mm_dt = torch.bfloat16 if self.profile_dtype == "bf16" else torch.float32
        zn = torch.matmul(knew.to(mm_dt), ent["qprof"].to(mm_dt).transpose(-1, -2)) \
            .float() * scaling                                     # (B,H,dK,P)
        qn = zn.mean(-1)
        if ent.get("method") in ("keykm", "centroidkv"):
            knn = torch.nn.functional.normalize(knew, dim=-1)
            new_asg = torch.matmul(knn, ent["kcent"].transpose(-1, -2)).argmax(-1)
        elif ent.get("method") == "retro":
            new_asg = torch.matmul(knew - ent["kmean"], ent["kcent"].transpose(-1, -2)).argmax(-1)
        elif ent.get("method") == "qsort":
            new_asg = torch.searchsorted(ent["edges"].contiguous(), qn.contiguous())
        elif ent.get("method") in ("geo", "geo2"):
            new_asg = (torch.searchsorted(ent["edges_neg"], (-qn).contiguous(), right=True)
                       - 1).clamp(0, ent["C"] - 1)
        elif ent.get("method") == "grid":
            sc2 = torch.matmul(zn - ent["colmean"], ent["V2"])     # (B,H,dK,2)
            g1 = torch.searchsorted(ent["e1"], sc2[..., 0].contiguous())
            e2g = ent["e2"].gather(-2, g1.unsqueeze(-1).expand(-1, -1, -1, ent["G2"] - 1))
            g2 = (sc2[..., 1].unsqueeze(-1) > e2g).sum(-1)
            new_asg = g1 * ent["G2"] + g2
        else:
            zc = zn - ent["colmean"]
            sim = torch.matmul(zc, ent["cent"].transpose(-1, -2)) \
                - 0.5 * (ent["cent"] ** 2).sum(-1).unsqueeze(2)
            new_asg = sim.argmax(-1)
        ent["asg"] = torch.cat([ent["asg"], new_asg], -1)
        ent["qmean"] = torch.cat([ent["qmean"], qn], -1)
        ent["K"] = K
        return ent

    def _profile_entry(self, q: torch.Tensor, k: torch.Tensor, scaling: float,
                       layer_idx: int) -> Dict[str, Any]:
        """Per-layer cache: (re)built at every question-chunk call (Q >= 8), which a
        new request always issues before its decode steps; extended at decode."""
        ent = self._prof_cache.get(layer_idx)
        Q, K = q.shape[2], k.shape[-2]
        if Q >= 8 or ent is None or not (0 <= K - ent["K"] <= 4096):
            if Q < 8 and ent is not None and K < ent["K"] and ent.get("qprof") is not None:
                # V54b: the cache was rewritten by a prompt-time KV press (CentroidKV):
                # the partition is rebuilt over the new keys from the profile queries the
                # question chunk produced, which do not depend on the keys.
                ent = self._profile_build(q, k, scaling, layer_idx, qprof=ent["qprof"])
                self._prof_cache[layer_idx] = ent
                return ent
            if Q < 8:
                raise ValueError(
                    "bin_mode='profile' saw a decode-sized call with no cached "
                    f"partition for layer {layer_idx} (or the key count jumped)")
            ent = self._profile_build(q, k, scaling, layer_idx)
            self._prof_cache[layer_idx] = ent
        elif K != ent["K"]:
            ent = self._profile_extend(ent, k, scaling)
        return ent

    # ------------------------------------------------------- two-stage path --
    @torch.no_grad()
    def _topblock_output(self, q, k, v, raw, sel, tail_f, scaling) -> torch.Tensor:
        """Cached block proxies everywhere + exact logits for the top-J blocks only.

        This is the deployable answer to the circularity: binning the tail by its own
        logits needs the full QK we are trying to avoid, but ranking BLOCKS needs only
        zbar_b = s * q . kbar_b, which is O(d) per block off cached means.  We then spend
        a bounded refinement budget on the few blocks that ranking says carry the mass,
        computing their tokens' true logits exactly -- the same kind of budget vAttention
        spends, except aimed rather than random.

        Cost per query/head:  3*B*d  (means, diagonal variance, value accumulation)
                            + J*m*d  (refinement)
        vs vAttention's      2*m_s*d  with m_s random gathers.
        """
        m, J = self.bin_size, self.refine_blocks
        K = raw.shape[-1]
        pad = (-K) % m
        if pad:
            rawp = torch.nn.functional.pad(raw, (0, pad), value=NEG - 1.0)
            tfp = torch.nn.functional.pad(tail_f, (0, pad), value=0.0)
            vp = torch.nn.functional.pad(v, (0, 0, 0, pad), value=0.0)
            kp = torch.nn.functional.pad(k, (0, 0, 0, pad), value=0.0)
        else:
            rawp, tfp, vp, kp = raw, tail_f, v, k
        nb = (K + pad) // m

        shp = rawp.shape[:-1] + (nb, m)
        rb, tb = rawp.reshape(shp), tfp.reshape(shp)
        n_b = tb.sum(-1)
        den = n_b.clamp(min=1)
        zbar_b = (rb * tb).sum(-1) / den                     # = s * q . kbar_b
        vbar_b = self._blockwise(vp, tb, nb, m, den)

        if self.var_mode == "diag":
            s2_b = self._diag_var_fixed(q, kp, tb, nb, m, den, scaling)
        else:
            s2_b = ((rb * rb * tb).sum(-1) / den - zbar_b * zbar_b).clamp(min=0)
        s2_b = torch.where(n_b >= 2, s2_b, torch.zeros_like(s2_b))

        live = n_b > 0
        ell_b = (zbar_b + torch.log(den) + self._kappa(s2_b)).masked_fill(~live, float("-inf"))

        # rank blocks by the mass their cached statistics predict, refine the top J
        j = min(J, nb)
        top = ell_b.topk(j, dim=-1).indices                                  # (B,H,Q,j)
        refine_blk = torch.zeros_like(n_b, dtype=torch.bool).scatter_(-1, top, True)
        promoted = (refine_blk.unsqueeze(-1) & (tb > 0)).reshape(rawp.shape)[..., :K]

        exact = sel | promoted                       # tokens whose logit we truly compute
        ell_b = ell_b.masked_fill(refine_blk, float("-inf"))   # their blocks drop out

        row_max = torch.maximum(
            raw.masked_fill(~exact, float("-inf")).amax(-1, keepdim=True),
            ell_b.amax(-1, keepdim=True))
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

        e = torch.exp(raw - row_max) * exact
        num, den_ = torch.matmul(e, v), e.sum(-1, keepdim=True)

        w_b = torch.exp(ell_b - row_max)
        w_b = torch.where(torch.isfinite(w_b), w_b, torch.zeros_like(w_b))
        num = num + torch.einsum("bhqn,bhqnd->bhqd", w_b, vbar_b)
        den_ = den_ + w_b.sum(-1, keepdim=True)

        if MicroMetricLogger().is_metric_enabled("uta_refined_tokens"):
            MicroMetricLogger().log("uta_refined_tokens",
                                    float(promoted.sum(-1).float().mean().item()))

        return (num / den_.clamp(min=1e-30)).to(q.dtype).transpose(1, 2).contiguous()

    # ----------------------------------------------------------------- merge --
    def _multibin_output(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        sparse_mask: Mask,
        key_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with stage("mb/T2_scores_fullQK"):
            ngroups = _get_num_key_value_groups(queries, keys)
            k = repeat_kv(keys, ngroups).to(torch.float32)
            v = repeat_kv(values, ngroups).to(torch.float32)
            q = queries.to(torch.float32)

            raw = torch.matmul(q, k.transpose(2, 3)) * scaling        # (B,H,Q,K)
            if attention_mask is not None:
                raw = raw + attention_mask[:, :, :, : k.shape[-2]].to(torch.float32)

        with stage("mb/T1_tail_mask"):
            valid = raw > NEG
            sel = (sparse_mask.get_dense_mask() != 0) & valid
            tail_f = ((~sel) & valid).to(torch.float32)

        if self.bin_mode == "topblock":
            return self._topblock_output(q, k, v, raw, sel, tail_f, scaling)

        # var_mode="diag": every bin statistic is a reduction over K/V and the tail
        # mask, so the score matrix is never read here -- zbar_b comes from the bin's
        # own key mean.  var_mode="exact" wants the true within-bin logit variance,
        # which no key-side statistic can reproduce, so it keeps the raw path.
        bid = None
        k_mean_c = k_sq_c = None
        lr = None
        if self.var_mode == "lowrank" and self.bin_mode != "profile":
            raise ValueError("var_mode='lowrank' needs bin_mode='profile' (factors are cached per cluster)")
        if self.bin_mode == "fixed":
            if self.var_mode == "diag":
                n_b, zbar_b, s2_b, vbar_b = self._bins_fixed_keyside(
                    q, k, v, tail_f, scaling)
            else:
                with stage("mb/T3_bin_stats"):
                    n_b, zbar_b, s2_b, vbar_b = self._bins_fixed(raw, tail_f, v)
        else:
            with stage("mb/T2b_bin_ids"):
                if self.bin_mode == "equalcount":
                    bid, nbins = self._bin_ids_equalcount(tail_f)
                elif self.bin_mode == "random":
                    bid, nbins = self._bin_ids_random(tail_f)
                elif self.bin_mode == "profile":
                    ent = self._profile_entry(q, k, scaling, self._cur_layer)
                    nbins = ent["C"]
                    if self.profile_promote > 0:
                        # static promotion: per-request top tokens by cached qmean
                        # join the exact set BEFORE any bin statistic is taken
                        qm = ent["qmean"].unsqueeze(2).expand_as(tail_f)
                        qm = qm.masked_fill(tail_f <= 0, float("-inf"))
                        t = min(self.profile_promote, qm.shape[-1])
                        pidx = qm.topk(t, dim=-1).indices
                        promoted = torch.zeros_like(sel).scatter_(-1, pidx, True)
                        promoted = promoted & (tail_f > 0)
                        sel = sel | promoted
                        tail_f = tail_f * (~promoted).to(tail_f.dtype)
                    asgQ = ent["asg"].unsqueeze(2).expand(
                        -1, -1, tail_f.shape[2], -1)
                    bid = torch.where(tail_f > 0, asgQ,
                                      torch.full_like(asgQ, nbins))
                    if self.var_mode == "lowrank":
                        lr = self._lowrank_factors(ent, k, nbins)
                else:
                    bid, nbins = self._bin_ids_score(raw, tail_f, key_scores)
            if self.var_mode in ("diag", "lowrank"):
                # (B,H,Q,nbins,D) floats, twice (mean and second moment).
                _reuse_mb = (q.shape[0] * q.shape[1] * q.shape[2] * nbins
                             * k.shape[-1] * 4 * 2) / 1e6
                if (self.two_point and bid is not None and _reuse_mb <= 512.0
                        and self.var_mode == "diag"):
                    (n_b, zbar_b, s2_b, vbar_b, k_mean_c, k_sq_c
                     ) = self._bins_scatter_keyside(
                        q, k, v, tail_f, bid, nbins, scaling, lr=lr,
                        return_key_stats=True)
                else:
                    n_b, zbar_b, s2_b, vbar_b = self._bins_scatter_keyside(
                        q, k, v, tail_f, bid, nbins, scaling, lr=lr)
            else:
                with stage("mb/T3_bin_stats"):
                    n_b, zbar_b, s2_b, vbar_b = self._aggregate_by_bin_id(
                        raw, tail_f, v, bid, nbins)

        # --- second stage: pull the hottest bins out and compute them exactly -----
        # E0 established that what the tail loses is a few huge-logit tokens, which no
        # proxy SHAPE can carry.  So instead of correcting a bin's mass, remove the bin
        # from the proxy set and evaluate its tokens directly.  Ranking uses only
        # zbar_b = s*q.kbar_b and sigma_b, both key-side, so no logits are consulted.
        expand_tok: Optional[torch.Tensor] = None
        if (self.expand_m > 0 or self.expand_budget_tokens > 0) and bid is not None:
            with stage("mb/T4b_expand_rank"):
                # bins holding a single token are already exact as proxies, so
                # expanding them buys nothing -- exclude them from the ranking
                live0 = n_b > 1
                if self.expand_rank == "random":
                    if self._exp_rnd is None or self._exp_rnd.numel() < nbins:
                        g = torch.Generator(device="cpu").manual_seed(self.random_seed + 7)
                        self._exp_rnd = torch.rand(max(nbins, 1), generator=g).to(n_b.device)
                    score_b = self._exp_rnd[:nbins].expand_as(n_b)
                elif self.expand_rank == "mass":
                    # The criterion every centroid method ranks by: the bin's own
                    # estimated attention mass, size-weighted centroid logit
                    # (Double-P eq.5, RetroInfer's cluster estimate, ClusterKV's
                    # centroid inner product).  It is a FIRST-moment statistic, so a
                    # bin holding one outlier among many ordinary tokens looks
                    # ordinary -- which is exactly the bin worth expanding.
                    score_b = zbar_b + torch.log(n_b.clamp(min=1))
                elif self.expand_rank == "massj2":
                    # ...the same, with the bounded Jensen correction folded in, so
                    # the comparison is not merely "they forgot the second moment".
                    score_b = (zbar_b + torch.log(n_b.clamp(min=1))
                               + torch.log1p(s2_b.clamp(min=0) / 2))
                elif self.expand_rank == "ucblogn":
                    # V32: the sub-Gaussian constant made bin-adaptive, c_b = sqrt(2 log n_b),
                    # so that large bins (which can hide a larger maximum) get a larger margin.
                    c_b = (2.0 * torch.log(n_b.clamp(min=2))).sqrt()
                    score_b = zbar_b + c_b * s2_b.clamp(min=0).sqrt()
                elif self.expand_rank == "boundlogn":
                    # V32: the full observable part of the greedy bound rule (App. bound):
                    # log n_b + zbar_b + sqrt(2 log n_b) sigma_b -- mass-aware AND adaptive.
                    c_b = (2.0 * torch.log(n_b.clamp(min=2))).sqrt()
                    score_b = zbar_b + torch.log(n_b.clamp(min=1)) + c_b * s2_b.clamp(min=0).sqrt()
                elif self.expand_rank == "var":
                    # V36 control: the second moment ALONE, first moment dropped.
                    # sigma_b^2 is CENTRED, so it is invariant to the bin's own mean
                    # logit -- a bin the query never looks at ranks as high as a hot one
                    # of the same spread.  This is the "variance only" hypothesis in its
                    # literal form.
                    score_b = s2_b.clamp(min=0)
                elif self.expand_rank == "m2raw":
                    # V36: the same statistic with the normalisation (the centring)
                    # removed, so the bin's first moment is carried INSIDE the
                    # second-moment number instead of being added to it with a free
                    # constant.  Caveat: the mean enters as a per-dimension square, so
                    # the rule is blind to the SIGN of q.kbar_b -- a bin whose mean key
                    # is strongly anti-aligned with the query scores like an aligned one.
                    if self._m2raw is None:
                        raise RuntimeError("expand_rank='m2raw' needs var_mode='diag'")
                    score_b = self._m2raw
                elif self.expand_rank == "m2":
                    # V36: the uncentred second moment of the LOGIT itself,
                    # E_b[z^2] = zbar_b^2 + sigma_b^2.  Same idea as m2raw but with the
                    # true first moment (q.kbar_b)^2 in place of the diagonal form
                    # sum_d q_d^2 kbar_bd^2.  Still even in zbar_b, so still sign-blind.
                    score_b = zbar_b * zbar_b + s2_b.clamp(min=0)
                elif self.expand_rank in ("expvar", "expvarn"):
                    # V36: the second moment taken in the WEIGHT domain rather than the
                    # logit domain.  What the proxy replaces is exp(z_i), not z_i, and
                    # for z ~ N(zbar_b, sigma_b^2),
                    #   sd_b(e^z) = exp(zbar_b + sigma_b^2/2) sqrt(exp(sigma_b^2) - 1),
                    # in which the bin's mass appears as a MULTIPLICATIVE factor.  So a
                    # single second-moment number already carries the mean, with no free
                    # constant and no addition of two quantities in different units --
                    # the un-normalised variance the centred sigma_b^2 is missing.
                    # "expvarn" scores the whole bin (x n_b) instead of one token of it.
                    s2 = s2_b.clamp(min=0)
                    score_b = zbar_b + 0.5 * s2 + 0.5 * _log_expm1(s2)
                    if self.expand_rank == "expvarn":
                        score_b = score_b + torch.log(n_b.clamp(min=1))
                elif self.expand_rank == "jgap":
                    # V36: rank by the error that expanding the bin actually removes.
                    # The proxy contributes n_b e^{zbar_b}; the truth is
                    # sum_i e^{z_i} = n_b e^{zbar_b} E_b[e^{z-zbar}] ~ n_b e^{zbar_b+sigma^2/2},
                    # so the missing mass is n_b e^{zbar_b} (e^{sigma_b^2/2} - 1) and its
                    # log is the score below.  Mass x relative error, multiplicative
                    # again, and with no constant to tune.
                    score_b = (zbar_b + torch.log(n_b.clamp(min=1))
                               + _log_expm1(0.5 * s2_b.clamp(min=0)))
                elif self.expand_rank == "wl1":
                    # V56: rank by the error the bin's proxy leaves in the SOFTMAX WEIGHT
                    # VECTOR, token by token.  The proxy gives every member of b the same
                    # unnormalised weight e^{zbar_b} where the truth is e^{z_i}, so the
                    # error it leaves is
                    #     sum_{i in b} |e^{z_i} - e^{zbar_b}| = n_b e^{zbar_b} D_b,
                    #     D_b = (1/n_b) sum_i |e^{delta_i} - 1|,   delta_i = z_i - zbar_b.
                    # Unlike "jgap", which is the SIGNED sum and lets a token above the
                    # mean cancel one below it, this counts every token's deviation, so it
                    # is FIRST order in sigma_b where the Jensen gap is second order.
                    # Under the within-bin Gaussian model the closed form is exact,
                    #     D_b = e^{sigma^2/2} (2 Phi(sigma) - 1) = e^{sigma^2/2} erf(sigma/sqrt2)
                    #         ~ sqrt(2/pi) sigma_b   (sigma << 1),
                    # so the score reads the SAME two cached numbers as every other rule
                    # (n_b e^{zbar_b} and sigma_b^2) and costs one erf on a (B,H,1,C)
                    # tensor.  No free constant, and the value side never enters.
                    sig = s2_b.clamp(min=0).sqrt()
                    log_d = 0.5 * s2_b.clamp(min=0) + torch.log(
                        torch.erf(sig * 0.7071067811865476).clamp(min=1e-30))
                    score_b = zbar_b + torch.log(n_b.clamp(min=1)) + log_d
                elif self.expand_rank in ("jgapv", "errmix", "errv", "outerr"):
                    # V36b: rank by the first-order OUTPUT error the bin's proxy leaves
                    # behind, rather than by one of its two channels.  With
                    # Dhat_b = n_b e^{zbar_b} the proxy mass and vtilde_b the bin's true
                    # weight-averaged value, the error the softmax inherits from bin b is
                    # EXACTLY (no model yet)
                    #   (D_b - Dhat_b)(vtilde_b - o)  +  Dhat_b (vtilde_b - vbar_b),
                    # a MASS channel -- what "jgap" ranks on its own -- plus a SHAPE
                    # channel -- what "expvarn" ranks on its own -- sharing the factor
                    # Dhat_b.  Under the within-bin Gaussian model the two relative sizes
                    # are
                    #   mass  : e^{sigma^2/2} - 1
                    #   shape : sqrt(e^{sigma^2} - 1) = CV_b(e^z), the weights'
                    #           coefficient of variation, out of which the bin mean
                    #           cancels exactly
                    # and Cauchy-Schwarz gives ||vtilde_b - vbar_b|| <= CV_b sqrt(tr Cov_b(v)).
                    # The shape channel is FIRST order in sigma and the mass channel
                    # second; in closed form their gap is
                    #   shape - mass = 0.5 log coth(sigma^2 / 4) > 0,
                    # so shape leads at EVERY sigma -- steeply for fine partitions, and
                    # the mass channel only catches up asymptotically.  logaddexp of the
                    # two therefore sits within log 2 of the shape channel, but on a
                    # different mass footing: the decomposition weights both by the PROXY
                    # mass n_b e^{zbar_b}, whereas "expvarn" uses the bin's true mass
                    # n_b e^{zbar_b + sigma^2/2}.
                    #   jgapv   the MASS channel alone, with its own value factor.  Set
                    #           vtilde_b = vbar_b -- i.e. assume the weights and the values
                    #           are uncorrelated INSIDE a bin, which is all (n_b, zbar_b,
                    #           sigma_b^2) can speak to -- and the two terms collapse to
                    #             eps_b ~ n_b e^{zbar_b}(e^{sigma^2/2} - 1) ||vbar_b - ohat||,
                    #           one closed form with no bound and no mixing of units.
                    #   errmix  both channels with BOTH value factors set to 1.  Cheapest,
                    #           but the two terms then carry different units and are added
                    #           with an implicit coefficient of 1 -- the same objection that
                    #           retired zbar + 3 sigma.  Kept as a measured data point.
                    #   errv    jgapv + the residual bound (units still mismatched on the
                    #           second term)
                    #   outerr  jgapv + the residual bound with its own value factor: units
                    #           consistent, at the cost of one cached float per bin
                    # The residual the bound covers is the within-bin weight/value
                    # correlation, which sigma_b^2 -- a statistic of the LOGITS -- cannot
                    # see at all; it is not a second estimate of the same quantity.
                    # None of the four has a free constant.
                    s2 = s2_b.clamp(min=0)
                    log_mass = _log_expm1(0.5 * s2)
                    log_shape = 0.5 * _log_expm1(s2)
                    if self.expand_rank != "errmix":
                        # ohat: the partial output over the tokens the selector has
                        # already made exact.  A fused kernel computes that partial
                        # softmax before it merges the proxies in, so reading it here
                        # costs a deployment nothing; this file takes it off the score
                        # matrix it already holds.
                        zs = raw.masked_fill(~sel, float("-inf"))
                        mx = zs.amax(-1, keepdim=True)
                        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
                        zs = zs.sub_(mx).exp_()
                        o_hat = torch.matmul(zs, v) / zs.sum(-1, keepdim=True).clamp(min=1e-30)
                        del zs
                        log_mass = log_mass + torch.log(
                            (vbar_b - o_hat.unsqueeze(-2)).norm(dim=-1).clamp(min=1e-6))
                    if self.expand_rank == "outerr":
                        if self._vspread is None:
                            raise RuntimeError("expand_rank='outerr' needs the scatter "
                                               "bin path (bin_mode='profile')")
                        log_shape = log_shape + torch.log(self._vspread.clamp(min=1e-6))
                    score_b = zbar_b + torch.log(n_b.clamp(min=1))
                    score_b = score_b + (log_mass if self.expand_rank == "jgapv"
                                         else torch.logaddexp(log_mass, log_shape))
                else:
                    # ours: an upper confidence bound on the bin's LARGEST logit.
                    # Under a sub-Gaussian model for the within-bin logits,
                    # max_i z_i <~ zbar_b + sigma_b*sqrt(2 log n_b), so expand_c plays
                    # the role of sqrt(2 log n_b) (2.63 at n_b=32).
                    score_b = zbar_b + self.expand_c * s2_b.clamp(min=0).sqrt()
                m_h = self._head_budget(score_b.shape[1], score_b.device)
                trunc_keep: Optional[torch.Tensor] = None
                if self.expand_budget_tokens > 0:
                    # TOKEN-budget equalised expansion for unequal-size partitions
                    # ("profile"): take bins in rank order while the cumulative
                    # token count is below the budget; the crossing bin is included
                    # (same accounting as the V19 diagnostic).
                    order = score_b.masked_fill(
                        ~live0, float("-inf")).argsort(-1, descending=True)
                    n_sorted = n_b.gather(-1, order)
                    took = n_sorted.cumsum(-1)
                    budget = float(self.expand_budget_tokens)
                    within = ((took - n_sorted) < budget) & (n_sorted > 0)
                    capped = (self.expand_truncate
                              and (self.expand_bin_cap > 0 or self.expand_alloc == "hier")
                              and key_scores is not None)
                    if capped:
                        sc = key_scores
                        if sc.shape[-2] != tail_f.shape[-2]:
                            sc = sc[..., -tail_f.shape[-2]:, :]
                        Kk = tail_f.shape[-1]
                        capped = (sc.shape[-1] >= Kk
                                  and sc.shape[:-1] == tail_f.shape[:-1])
                    if capped:
                        # V35d spread expansion: bins are taken in rank order at an
                        # effective cost min(n_b, t) each, and EVERY taken bin
                        # contributes only its top tokens by SELECTOR score up to its
                        # quota (t, or the remaining budget for the crossing bin);
                        # the rest of each taken bin stays a proxy, count reduced.
                        if self.expand_alloc == "hier":
                            # V47: quotas follow the ranked error (see config)
                            quota = self._hier_quota(score_b, n_b, live0, budget, order)
                        else:
                            cap = torch.full_like(n_sorted, float(self.expand_bin_cap))
                            ncap = torch.minimum(n_sorted, cap)
                            tookc = ncap.cumsum(-1)
                            withinc = ((tookc - ncap) < budget) & (n_sorted > 0)
                            fullc = withinc & (tookc <= budget)
                            crossc = withinc & ~fullc                # <= 1 bin per row
                            rem = (budget - (ncap * fullc).sum(-1, keepdim=True)).clamp(min=0)
                            quota_sorted = torch.where(fullc, ncap, torch.zeros_like(ncap))
                            quota_sorted = torch.where(crossc, rem.expand_as(ncap),
                                                       quota_sorted)
                            quota = torch.zeros_like(n_b).scatter_(-1, order, quota_sorted)
                        if os.environ.get("UTA_ERRLOG") and score_b.shape[2] == 1:
                            # V50: the sorted-bin error curve, log err_(r) - log err_(1)
                            ss = score_b.masked_fill(~(live0 & (n_b > 0)), float("-inf")
                                                     ).gather(-1, order)[..., :64].float()
                            ss = ss - ss[..., :1]
                            _ERRLOG.setdefault("curve", []).append(
                                ss.reshape(-1, ss.shape[-1]).cpu().half().numpy())
                        if os.environ.get("UTA_ALLOC_DEBUG"):
                            got = quota > 0
                            print(f"[alloc] mode={self.expand_alloc} "
                                  f"bins/row={got.float().sum(-1).mean().item():.1f} "
                                  f"max_q={quota.amax().item():.0f} "
                                  f"top1_q={quota.gather(-1, order[..., :1]).float().mean().item():.1f} "
                                  f"sum_q={quota.sum(-1).float().mean().item():.1f}", flush=True)
                        # per-token rank WITHIN its own bin by selector score: sort by a
                        # composite (bin id, descending score) key, then subtract each
                        # bin segment's start in the sorted order.  The score half of the
                        # key is the order-preserving unsigned image of the float32 score
                        # (flip the sign bit for positives, invert for negatives), so one
                        # sort replaces the rank-then-sort pair -- same ranks, ~2x faster.
                        scf = sc[..., :Kk].float().masked_fill(
                            ~(tail_f > 0), float("-inf")).contiguous()
                        bidx = bid.expand(tail_f.shape)
                        bid_t = torch.where(tail_f > 0, bidx,
                                            torch.full_like(bidx, nbins))
                        if os.environ.get("UTA_FAST_RANK"):
                            # One sort instead of two: the score half of the key is the
                            # order-preserving unsigned image of the float32 score, so the
                            # stage is ~2x faster.  Opt-in: the rule is the same and ranks
                            # match on synthetic data, but exactly tied scores are broken
                            # differently (neither sort is stable), which moves an
                            # occasional generated token -- every reported number comes
                            # from the reference path below.
                            u = scf.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
                            ukey = torch.where((u & 0x80000000) != 0, (~u) & 0xFFFFFFFF,
                                               u | 0x80000000)
                            pos = (bid_t * (1 << 32) + (0xFFFFFFFF - ukey)).argsort(-1)
                        else:
                            gr = scf.argsort(-1, descending=True).argsort(-1)
                            pos = (bid_t * (Kk + 1) + gr).argsort(-1)
                        sorted_bid = bid_t.gather(-1, pos)
                        ar = torch.arange(Kk, device=scf.device).expand_as(sorted_bid)
                        is_start = torch.ones_like(sorted_bid, dtype=torch.bool)
                        is_start[..., 1:] = sorted_bid[..., 1:] != sorted_bid[..., :-1]
                        start = torch.where(is_start, ar, torch.zeros_like(ar)
                                            ).cummax(-1).values
                        rib = torch.zeros_like(ar).scatter_(-1, pos, ar - start)
                        pad0q = torch.zeros_like(quota[..., :1])
                        quota_tok = torch.cat([quota, pad0q], -1).gather(-1, bid_t)
                        is_exp = torch.zeros_like(live0)         # no whole-bin reads
                        trunc_keep = (tail_f > 0) & (rib < quota_tok)
                    elif self.expand_truncate:
                        # V26 exact budget: whole bins only while they fit; the
                        # crossing bin contributes its top tokens by SELECTOR score.
                        full = within & (took <= budget)
                        cross = within & ~full                       # <= 1 bin per row
                        is_exp = torch.zeros_like(live0).scatter_(
                            -1, order, full.contiguous()) & live0
                        is_cross = torch.zeros_like(live0).scatter_(
                            -1, order, cross.contiguous()) & live0
                        remaining = (budget - (n_sorted * full).sum(-1)).clamp(min=0)
                        pad0 = torch.zeros_like(is_cross[..., :1])
                        tok_cross = (torch.cat([is_cross, pad0], -1).gather(-1, bid)
                                     & (tail_f > 0))
                        keep = torch.zeros_like(tok_cross)
                        if key_scores is not None and bool(tok_cross.any()):
                            Kk = tail_f.shape[-1]
                            sc = key_scores
                            if sc.shape[-2] != tail_f.shape[-2]:
                                sc = sc[..., -tail_f.shape[-2]:, :]
                            if sc.shape[-1] >= Kk and sc.shape[:-1] == tail_f.shape[:-1]:
                                sc = sc[..., :Kk].float().masked_fill(~tok_cross, float("-inf"))
                                rank = sc.argsort(-1, descending=True).argsort(-1)
                                keep = tok_cross & (rank < remaining.unsqueeze(-1))
                        trunc_keep = keep
                    else:
                        is_exp = torch.zeros_like(live0).scatter_(
                            -1, order, within.contiguous()) & live0
                elif m_h is None:
                    mm = min(self.expand_m, nbins)
                    idx = score_b.masked_fill(~live0, float("-inf")).topk(mm, dim=-1).indices
                    is_exp = torch.zeros_like(live0).scatter_(-1, idx, True) & live0
                else:
                    # per-head budget: one topk to the layer's maximum, then keep only
                    # each head's own prefix
                    mm = min(int(m_h.max()), nbins)
                    idx = score_b.masked_fill(~live0, float("-inf")).topk(mm, dim=-1).indices
                    rank = torch.arange(mm, device=score_b.device).view(1, 1, 1, mm)
                    keep = (rank < m_h.view(1, -1, 1, 1)).expand_as(idx).contiguous()
                    is_exp = torch.zeros_like(live0).scatter_(-1, idx, keep) & live0
                pad_col = torch.zeros_like(is_exp[..., :1])
                expand_tok = (torch.cat([is_exp, pad_col], -1).gather(-1, bid)
                              & (tail_f > 0))
                n_b = n_b.masked_fill(is_exp, 0)     # expanded bins leave the proxy set
                if trunc_keep is not None and bool(trunc_keep.any()):
                    # the crossing bin: its kept tokens become exact, the remainder
                    # stays a proxy whose count is reduced accordingly
                    expand_tok = expand_tok | trunc_keep
                    kept = torch.zeros_like(n_b).scatter_add_(
                        -1, bid.clamp(max=nbins - 1), trunc_keep.to(n_b.dtype))
                    n_full, n_b = n_b, (n_b - kept).clamp(min=0)
                    if self.residual_stats != "none":
                        # Rebuild the proxy's first moments over the tokens that are
                        # STILL a proxy.  Sums are additive, so subtracting the expanded
                        # tokens' own sums is exact and costs one scatter over them.
                        exp_f = trunc_keep.to(tail_f.dtype)
                        den_r = n_b.clamp(min=1)
                        live_r = n_b > 0
                        # value channel: a token the merge adds exactly must not ALSO
                        # sit in the centroid that stands for the unread remainder
                        v_gone = self._scatter_mean(v, exp_f, bid, nbins,
                                                    torch.ones_like(den_r))
                        vbar_b = torch.where(
                            live_r.unsqueeze(-1),
                            (vbar_b * n_full.unsqueeze(-1) - v_gone) / den_r.unsqueeze(-1),
                            vbar_b)
                        if self.residual_stats == "wexact":
                            # weight the residual by its own softmax weights: the exact
                            # vtilde_R the proxy is a stand-in for
                            res_f = tail_f * (~trunc_keep).to(tail_f.dtype)
                            w = torch.exp(raw - raw.amax(-1, keepdim=True)) * res_f
                            wsum = torch.zeros_like(n_b).scatter_add_(
                                -1, bid.clamp(max=nbins - 1), w.to(n_b.dtype))
                            vbar_b = torch.where(
                                live_r.unsqueeze(-1),
                                self._scatter_mean(v, w, bid, nbins,
                                                   wsum.clamp(min=1e-30)),
                                vbar_b)
                        if self.residual_stats in ("full", "fullvar"):
                            # mass channel: zbar_b likewise still carries the logits of
                            # the tokens just made exact, which are the bin's highest
                            z_gone = torch.zeros_like(n_b).scatter_add_(
                                -1, bid.clamp(max=nbins - 1), (raw * exp_f).to(n_b.dtype))
                            zbar_b = torch.where(
                                live_r, (zbar_b * n_full - z_gone) / den_r, zbar_b)
                        if self.residual_stats == "fullvar":
                            # V50: the within-bin spread of what is STILL a proxy.  Key
                            # sums are additive, so the residual first/second key moments
                            # are the cached ones minus the expanded tokens' own -- the
                            # same subtraction as above, one scatter over the residual.
                            # Only the Jensen correction (kappa) reads it: the ranking
                            # was decided on the full-bin spread before the expansion.
                            res_f = tail_f * (~trunc_keep).to(tail_f.dtype)
                            kmr = self._scatter_mean(k, res_f, bid, nbins, den_r)
                            ksr = self._scatter_mean(k * k, res_f, bid, nbins, den_r)
                            _, s2r = self._moments_from_key_stats(q, kmr, ksr, scaling)
                            del kmr, ksr
                            s2_b = torch.where(live_r & (n_b >= 2), s2r,
                                               torch.zeros_like(s2r))
                if MicroMetricLogger().is_metric_enabled("uta_expand_frac"):
                    MicroMetricLogger().log(
                        "uta_expand_frac",
                        float(expand_tok.float().sum(-1).mean().item()) / max(1, tail_f.shape[-1]))
                if MicroMetricLogger().is_metric_enabled("uta_proxy_bins"):
                    MicroMetricLogger().log("uta_proxy_bins",
                                            float((n_b > 0).float().sum(-1).mean().item()))

        # Ablation: zero every surviving bin so no proxy mass enters the softmax.  The
        # merge below then reduces to a plain sparse softmax over {selected} u {expanded}
        # -- the tail is discarded rather than approximated.
        if not self.use_aggregate or self.aggregate_mode == "none":
            n_b = torch.zeros_like(n_b)
        elif self.aggregate_mode == "single":
            # Collapse every surviving bin into one proxy, keeping the partition and the
            # expanded set exactly as they were:
            #   n = sum n_b,  zbar = sum n_b zbar_b / n,  vbar = sum n_b vbar_b / n
            live0 = n_b > 0
            nS = n_b.sum(-1, keepdim=True)
            den = nS.clamp(min=1)
            zS = (zbar_b * n_b).sum(-1, keepdim=True) / den
            s2S = (s2_b * n_b).sum(-1, keepdim=True) / den
            vS = (torch.einsum("bhqn,bhqnd->bhqd", n_b, vbar_b) / den).unsqueeze(-2)
            keep0 = torch.zeros_like(live0)
            keep0[..., :1] = True
            n_b = torch.where(keep0, nS, torch.zeros_like(n_b))
            zbar_b = torch.where(keep0, zS, zbar_b)
            s2_b = torch.where(keep0, s2S, s2_b)
            vbar_b = torch.where(keep0.unsqueeze(-1), vS, vbar_b)

        # Fused decode path: hand the merge to the Triton kernel.  Everything above --
        # bin statistics, ranking, expansion choice -- is unchanged, so any score
        # difference would be the kernel's doing and nothing else.
        if (self.use_triton_decode and expand_tok is not None
                and queries.shape[2] == 1 and bid is not None):
            fused = self._triton_decode(q, k, v, sel, expand_tok, tail_f, bid,
                                        nbins, n_b, vbar_b, scaling)
            if fused is not None:
                return fused.to(queries.dtype).transpose(1, 2).contiguous()

        # per-bin logit: mean logit + log of the bin's OWN token count
        with stage("mb/T5_kappa_bin_logit"):
            live = n_b > 0
            if self.kappa_mode in ("jsamp", "jsampm", "jsampmq") and bid is not None:
                # replaces (not adds to) the closed-form kappa: the gap is read
                # off m sampled keys per bin instead of the inflated diagonal
                # variance (JSAMPM_METHOD.md §5-3)
                kap_b = self._kappa_sampled(q, k, bid, nbins, n_b, zbar_b,
                                            scaling,
                                            int(getattr(self, "kappa_samples", 8)),
                                            pool=(self.kappa_mode != "jsamp"),
                                            pool_q=(self.kappa_mode == "jsampmq"))
            else:
                kap_b = self._kappa(s2_b)
            ks = float(getattr(self, "kappa_scale", 1.0))
            if ks != 1.0:
                kap_b = kap_b * ks
            ell_b = zbar_b + torch.log(n_b.clamp(min=1)) + kap_b
            ell_b = ell_b.masked_fill(~live, float("-inf"))

        # Two-point bins: the surviving bin contributes e^{z*}v* + (n-1)e^{z_r}v_r
        # instead of n*e^{zbar}vbar.  Both terms collapse into a single (ell, vbar)
        # pair, so the merge below -- and every consumer of it -- is untouched.
        if self.two_point and bid is not None:
            with stage("mb/T5b_two_point"):
                ell_b, vbar_b = self._two_point_merge(
                    q, k, v, tail_f, bid, nbins, n_b, zbar_b, vbar_b, scaling,
                    live, s2_b=s2_b, k_mean=k_mean_c, k_sq=k_sq_c, rep_k=self.rep_k,
                    sel_scores=key_scores)

        # single global softmax over {heavy tokens} U {bins}
        with stage("mb/T6_merge_softmax"):
            raw_max = raw.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True)
            bin_max = ell_b.amax(-1, keepdim=True)
            row_max = torch.maximum(raw_max, bin_max)
            row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

            exact = sel if expand_tok is None else (sel | expand_tok)
            e_sel = torch.exp(raw - row_max) * exact
            num = torch.matmul(e_sel, v)
            den = e_sel.sum(-1, keepdim=True)

            w_b = torch.exp(ell_b - row_max)                           # (B,H,Q,nb)
            w_b = torch.where(live, w_b, torch.zeros_like(w_b))
            num = num + torch.einsum("bhqn,bhqnd->bhqd", w_b, vbar_b)
            den = den + w_b.sum(-1, keepdim=True)

        if MicroMetricLogger().is_metric_enabled("uta_bin_count"):
            MicroMetricLogger().log("uta_bin_count", float(live.sum(-1).float().mean().item()))

        return (num / den.clamp(min=1e-30)).to(queries.dtype).transpose(1, 2).contiguous()

    @torch.no_grad()
    def _triton_decode(self, q, k, v, sel, expand_tok, tail_f, bid, nbins,
                       n_b, vbar_b, scaling):
        """Decode-step merge via the fused Triton kernel; None if the shapes do not fit.

        The kernel wants explicit token-index lists while the masker pipeline produces
        boolean masks.  At decode the per-row counts are constant (fixed sink/local/
        heavy budgets, fixed m), so the conversion is a topk rather than a ragged
        gather; if that ever fails to hold we fall back to the torch path rather than
        silently truncating.
        """
        try:
            from triton_expand_decode import decode_expand_attention
        except Exception:
            return None

        B, H, Q, K = sel.shape
        if Q != 1:
            return None
        selc, expc = sel[:, :, 0, :], expand_tok[:, :, 0, :]
        n_sel, n_exp = selc.sum(-1), expc.sum(-1)
        ns, ne = int(n_sel.max()), int(n_exp.max())
        if ns == 0 or ne == 0:
            return None

        def _idx(flags, n, counts):
            """Boolean row mask -> (B*H, n) int32 index list, -1 in the padding slots.

            Rows can legitimately hold different counts (a bin clipped by the end of
            the sequence), so the list is padded rather than truncated -- truncating
            would silently drop tokens from the attention.
            """
            idx = flags.float().topk(n, dim=-1).indices
            rank = torch.arange(n, device=idx.device).view(1, 1, n)
            idx = torch.where(rank < counts.unsqueeze(-1), idx, torch.full_like(idx, -1))
            return idx.int().reshape(B * H, n)

        sel_idx = _idx(selc, ns, n_sel)
        exp_idx = _idx(expc, ne, n_exp)

        # per-bin key mean: the query-independent statistic the kernel consumes, and
        # exactly what zbar_b was derived from (zbar_b = s * q . kbar_b)
        den = n_b.clamp(min=1)
        kbar = self._scatter_mean(k, tail_f, bid, nbins, den)[:, :, 0, :, :]

        BQ, D = B * H, q.shape[-1]
        log_n = torch.log(n_b[:, :, 0, :].clamp(min=1)).reshape(BQ, nbins).float()
        log_n = log_n.masked_fill(n_b[:, :, 0, :].reshape(BQ, nbins) <= 0, float("-inf"))

        out = decode_expand_attention(
            q[:, :, 0, :].reshape(BQ, D).contiguous(),
            k.reshape(BQ, K, D).contiguous(), v.reshape(BQ, K, D).contiguous(),
            sel_idx, exp_idx,
            kbar.reshape(BQ, nbins, D).contiguous(),
            vbar_b[:, :, 0, :, :].reshape(BQ, nbins, D).contiguous(),
            log_n, sm_scale=scaling,
        )
        return out.view(B, H, 1, D)

    # ------------------------------------------------------------ entrypoint --
    # ------------------------------------------------- KV-cache compression (V38) --
    @torch.no_grad()
    def _residency_bias(self, queries: torch.Tensor, keys: torch.Tensor,
                        attention_mask: Optional[torch.Tensor], scaling: float,
                        smd: Dict[Any, Any], layer_idx: int) -> Optional[torch.Tensor]:
        """Additive mask that hides the rows a SnapKV/PyramidKV pass evicted.

        The kept set is chosen once per request, at the first sparse call (the question
        chunk), exactly as `SnapKVMasker` chooses it: every prefix key is scored by the
        attention the last `evict_obs` query rows pay to it, max-pooled along the key
        axis and averaged over the query heads of its KV group; the top `evict_budget`
        fraction survives, under the PyramidKV per-layer allocation when asked.  Sink
        and local positions are always resident, and keys created after the selection
        stay resident, as they do for an eviction method.

        Returning the set as a *mask* rather than as a masker is what makes the two
        mechanisms compose: the rows are invisible to the selector (whose scores carry
        the same mask), to the partition's tail and to the proxies, so \rho is read
        against the surviving cache instead of the original one.
        """
        b, hq, nq, _ = queries.shape
        hkv, n_keys = keys.shape[1], keys.shape[2]
        ng = hq // hkv
        cache: Dict[int, Dict[str, Any]] = smd.setdefault("kvcompress", {})
        ent = cache.get(layer_idx)
        if ent is None:
            budget = (self.evict_tokens if self.evict_tokens > 0
                      else max(1, int(round(self.evict_budget * n_keys))))
            window = (self.evict_window if self.evict_window > 0
                      else default_window(self.evict_mode, budget, self.evict_alpha))
            keep = select_resident(
                queries, keys, attention_mask, scaling,
                mode=self.evict_mode, budget=budget, window=window,
                beta=self.evict_beta, kernel=self.evict_pool,
                layer_idx=layer_idx, num_layers=self.evict_layers)
            n_sink = int(round(self.evict_sink * n_keys))
            n_local = int(round(self.evict_local * n_keys))
            if n_sink > 0:
                keep[..., :n_sink] = True
            if n_local > 0:
                keep[..., -n_local:] = True
            cache[layer_idx] = ent = {"keep": keep, "n0": n_keys,
                                      "kept": int(keep[0, 0].sum())}
            # V39: the prompt is encoded densely and the compression applies only to
            # what is STORED for generation -- PyramidKV's own protocol (its paper,
            # Sec. 4.2, and `SnapKVMasker.paper_mode`, which returns a full mask on
            # this same call).  Biasing the selection call instead would encode the
            # question chunk against the 2k keys it just chose, which destroys the
            # very query representation the retrieval depends on (measured: qa_1
            # 78.0 -> 68.5 even when the whole resident set is read).
            return attention_mask

        keep = ent["keep"]
        n0 = ent["n0"]
        if n_keys > n0:            # tokens generated since the selection stay resident
            keep = torch.cat([keep, torch.ones((b, hkv, n_keys - n0), dtype=torch.bool,
                                               device=keys.device)], dim=-1)
        elif n_keys < n0:
            keep = keep[..., :n_keys]
        bias = torch.zeros((b, hkv, 1, n_keys), dtype=torch.float32, device=keys.device)
        bias.masked_fill_(~keep.unsqueeze(2), -1e9)
        bias = bias.expand(b, hkv, ng, n_keys).reshape(b, hq, 1, n_keys)
        if attention_mask is None:
            return bias
        return attention_mask.to(torch.float32) + bias

    def custom_attention(
        self,
        module: nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        **kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if "sparse_meta_data" not in kwargs:
            raise ValueError("sparse_meta_data must be provided in kwargs")
        sparse_meta_data: Dict[Any, Any] = kwargs.pop("sparse_meta_data")

        # V54: a KV press (CentroidKV) leaves the cache shorter than the context; the
        # rotated calibration metric must be built at the true decode position.
        from ..utils.centroidkv import kvpress_pos_offset
        self._pos_offset = kvpress_pos_offset(sparse_meta_data, kwargs.get("layer_idx"))

        if self.evict_mode != "none" and (self.evict_budget > 0 or self.evict_tokens > 0):
            # V38: hide the evicted rows from every stage below, selector included
            attention_mask = self._residency_bias(
                queries, keys, attention_mask, scaling, sparse_meta_data,
                int(kwargs.get("layer_idx") or 0))

        mask_shape = (queries.shape[0], queries.shape[1], queries.shape[2], keys.shape[2])
        sparse_mask = Mask.create_empty_mask(mask_shape, dtype=queries.dtype, device=queries.device)
        for masker in self.maskers:
            with stage(f"sel/{type(masker).__name__}"):
                sparse_mask = masker.add_mask(
                    keys=keys, queries=queries, values=values,
                    attention_mask=attention_mask, scaling=scaling, dropout=dropout,
                    sparse_meta_data=sparse_meta_data, previous_mask=sparse_mask, **kwargs,
                )

        if MicroMetricLogger().is_metric_enabled("research_attention_density"):
            MicroMetricLogger().log(
                "research_attention_density", sparse_mask.get_density(),
                metadata={"layer_idx": kwargs.get("layer_idx")},
            )

        if sparse_mask.is_full_mask():
            from ..utils.mask_attention_utils import get_true_attention_output
            return get_true_attention_output(
                module, queries, keys, values, attention_mask, scaling, dropout, **kwargs
            )

        key_scores: Optional[torch.Tensor] = None
        if self.bin_mode == "hat":
            layer_idx = kwargs.get("layer_idx")
            key_scores = sparse_meta_data.get("hat_scores", {}).get(layer_idx)
            if key_scores is None:
                raise ValueError(
                    "bin_mode='hat' needs HashAttention's approximate scores, but none "
                    f"were published for layer {layer_idx}. Put a HashAttentionTopKMasker "
                    "in the masker pipeline, or use bin_mode='score'/'fixed'."
                )

        if key_scores is None and self.expand_truncate:
            # V26: the selector's per-token score orders the tokens of the crossing bin
            layer_idx = kwargs.get("layer_idx")
            for nm in ("static_scores", "hat_scores"):
                key_scores = sparse_meta_data.get(nm, {}).get(layer_idx)
                if key_scores is not None:
                    break

        self._cur_layer = int(kwargs.get("layer_idx") or 0)
        output = self._multibin_output(
            queries, keys, values, attention_mask, scaling, sparse_mask, key_scores
        )

        if os.environ.get("UTA_ERRLOG"):
            # V50 tuning probe: per-row relative output error against dense attention
            from ..utils.mask_attention_utils import get_true_attention_output
            true_output, _ = get_true_attention_output(
                module, queries, keys, values, attention_mask, scaling, dropout, **kwargs
            )
            t = true_output.float()
            rel = (output.float() - t).norm(dim=-1) / t.norm(dim=-1).clamp(min=1e-30)
            key = "dec" if queries.shape[2] == 1 else "qc"
            _ERRLOG.setdefault(key, []).append(rel.reshape(-1).cpu().half().numpy())
            del t, true_output

        if MicroMetricLogger().is_metric_enabled("research_attention_output_error"):
            from ..utils.mask_attention_utils import get_true_attention_output
            true_output, _ = get_true_attention_output(
                module, queries, keys, values, attention_mask, scaling, dropout, **kwargs
            )
            err = torch.norm(true_output - output) / torch.norm(true_output)
            MicroMetricLogger().log(
                "research_attention_output_error", float(err.item()),
                metadata={"layer_idx": kwargs.get("layer_idx")},
            )

        return output, None

    @classmethod
    def create_from_config(cls, config: SparseAttentionConfig) -> "UTAMultiBinAttention":
        if not isinstance(config, UTAMultiBinConfig):
            raise TypeError(f"Expected UTAMultiBinConfig, got {type(config)}")
        maskers = [ResearchMasker.create_masker_from_config(mc) for mc in config.masker_configs]
        return cls(config, maskers)
