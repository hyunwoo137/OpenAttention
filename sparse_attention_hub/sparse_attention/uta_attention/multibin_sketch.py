"""Multi-bin UTA whose proxies carry a low-rank sketch of the block's keys.

WHAT THE PROXY GETS WRONG
------------------------
The multi-bin proxy estimates a block's mass as `n_b e^{zbar_b}` -- exp of the
mean logit -- where the truth is the mean of exp:

    S_b = n_b e^{zbar_b + K_b},   K_b = log mean_i e^{z_i - zbar_b} >= 0

Closing K_b from cached MOMENTS is measured to fail (V10 3.2: sigma^2/2 violates
the truth 68% of the time and falls 2.9x short on the block that carries the
mass), because the within-block logit law is one-token-dominated and the cumulant
series cannot be truncated.

WHAT THIS DOES INSTEAD
----------------------
Approximate the TOKENS, not the distribution.  Keep a rank-r sketch of the
block's key deviations, delta_i = k_i - kbar_blk ~= sum_j a_ij u_j, and take the
exp per token on the sketch:

    w_i = exp(sum_j t_j (a_ij - abar_j)),    t_j = s (q . u_j)
    S_b = e^{zbar_b} sum_{i in block & tail} w_i

`u_j` and `a_ij` are a PCA of the block's keys, so they are QUERY-INDEPENDENT and
cached once with the KV pages -- which is the whole reason this class uses fixed
position blocks rather than `equalcount`.  With runs of adjacent TAIL tokens the
membership moves whenever the selector's choice moves, and the eigendecomposition
would have to be redone every decode step (~32k of them per layer).

Measured, at bin 32 / d 128 / RULER 32K / HAT (V17):
    rank 1  ->  KV +3.9%,  relative output error -16.5% (5%) / -16.6% (3%)
    rank 2  ->  KV +7.8%,  -26.8% / -27.0%
The gain is carried by the MASS correction, not by the value-side one (V15 2.3:
direction-only is worth 1%); but correcting mass alone saturates and then
REVERSES (r=8 26.6% -> r=32 20.4%), because restoring the true mass while leaving
the direction at vbar_b puts the full weight behind the wrong vector.  So the two
corrections ship together, driven by the same weights w_i.

IMPLEMENTATION NOTE
-------------------
The projection is applied through the token-space identity

    q . (P_r delta_i) = [U_r U_r^T c]_i,     c_i = z_i - zbar_blk

where U_r are the top eigenvectors of the n x n key Gram.  This is numerically
identical to the deployable form above and avoids ever materialising `u_j` in
d-space, but it does read `c` from the score matrix -- fine here because the
prototype materialises it anyway (as does the parent class), and irrelevant to
the cost argument, which is settled by the cacheability of U_r.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..research_attention.maskers.base import ResearchMasker
from ..utils.kv_utils import _get_num_key_value_groups, repeat_kv
from ..utils.mask import Mask
from .multibin import NEG, UTAMultiBinAttention, UTAMultiBinConfig


@dataclass
class UTAMultiBinSketchConfig(UTAMultiBinConfig):
    """Config for the sketched multi-bin proxy.

    Attributes:
        sketch_rank: r.  0 reproduces the parent exactly; 32 (= bin_size) is exact
            attention on the tail.  1-2 are the measured operating points.
        sketch_values: also sketch the VALUE deviations, so the proxy's direction
            moves with the weights instead of staying at vbar_b.  Costs another
            r*(d + n) floats per block.  Off = "key-only", which V15 2.3 found
            better per byte at low rank.
    """

    sketch_rank: int = 1
    sketch_values: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sketch_rank < 0 or self.sketch_rank > self.bin_size:
            raise ValueError(
                f"sketch_rank must be in [0, bin_size={self.bin_size}], "
                f"got {self.sketch_rank}")
        if self.bin_mode != "fixed":
            raise ValueError(
                "sketch requires bin_mode='fixed': the sketch basis is cached with "
                f"the KV pages and must be query-independent, got {self.bin_mode!r}")


class UTAMultiBinSketchAttention(UTAMultiBinAttention):
    """Multi-bin UTA whose surviving proxies use a low-rank key/value sketch."""

    def __init__(self, config: UTAMultiBinSketchConfig,
                 maskers: List[ResearchMasker]) -> None:
        super().__init__(config, maskers)
        self.sketch_rank = int(getattr(config, "sketch_rank", 1))
        self.sketch_values = bool(getattr(config, "sketch_values", False))
        # (layer, n_kv_heads, n_blocks) -> top-r eigenvectors of the block Gram.
        # Query-independent, so one eigendecomposition per block serves every
        # query and every decode step; this cache is the deployable KV-page side
        # table, and its absence is what makes `equalcount` unusable here.
        self._basis: Dict[Any, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}

    # ------------------------------------------------------------------ basis --
    @staticmethod
    def _topr(d: torch.Tensor, r: int) -> torch.Tensor:
        """Top-r eigenvectors of the Gram `d d^T`, robust to degenerate blocks.

        A block whose tokens repeat (or that is mostly padding) has a near-zero,
        highly degenerate Gram, and LAPACK's divide-and-conquer eigensolver can
        fail to converge on it outright.  Guard in three steps: normalise each
        Gram by its own trace so the jitter is scale-relative, retry the failures
        in float64, and give any block that still fails an arbitrary orthonormal
        basis -- degenerate blocks have no preferred direction anyway, and their
        deviations are ~0, so the sketch weights come out ~1 (the plain proxy).
        """
        m = d.shape[-2]
        if r <= 0:                       # `[..., -0:]` would keep every column
            return d.new_zeros(d.shape[:-1] + (0,))
        G = d @ d.transpose(-1, -2)
        scale = G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None].clamp(min=1e-12)
        eye = torch.eye(m, device=d.device, dtype=G.dtype)
        Gn = G / scale + 1e-5 * eye
        try:
            return torch.linalg.eigh(Gn)[1][..., -r:]
        except torch.linalg.LinAlgError:
            pass
        flat = Gn.reshape(-1, m, m)
        out = eye.expand(flat.shape[0], m, m).clone()
        for i in range(flat.shape[0]):                    # only the failures pay
            try:
                out[i] = torch.linalg.eigh(flat[i])[1]
            except torch.linalg.LinAlgError:
                try:
                    out[i] = torch.linalg.eigh(
                        flat[i].to(torch.float64))[1].to(flat.dtype)
                except torch.linalg.LinAlgError:
                    pass                                   # keep the identity basis
        return out.reshape(Gn.shape)[..., -r:]

    @torch.no_grad()
    def _block_basis(self, keys: torch.Tensor, values: torch.Tensor,
                     layer_idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Top-r eigenvectors of each block's key (and value) Gram, cached.

        Computed on the KV heads BEFORE GQA expansion -- the keys are shared
        across a group, so the basis is too, and computing it here is `ngroups`
        times cheaper.
        """
        m = self.bin_size
        B, Hkv, K, D = keys.shape
        nb = (K + m - 1) // m
        key = (layer_idx, Hkv, nb, keys.device)
        hit = self._basis.get(key)
        if hit is not None:
            return hit

        pad = (-K) % m
        kp = torch.nn.functional.pad(keys, (0, 0, 0, pad)) if pad else keys
        kb = kp.to(torch.float32).reshape(B, Hkv, nb, m, D)
        valid = torch.ones((1, 1, nb, m, 1), dtype=torch.float32, device=keys.device)
        if pad:                                   # trailing block is partly empty
            valid = valid.clone()
            valid[:, :, -1, m - pad:, :] = 0.0
        n_blk = valid.sum(-2).clamp(min=1)
        dk = (kb - (kb * valid).sum(-2, keepdim=True) / n_blk.unsqueeze(-2)) * valid
        r = min(self.sketch_rank, m)
        Uk = self._topr(dk, r)

        Uv = None
        if self.sketch_values:
            vp = torch.nn.functional.pad(values, (0, 0, 0, pad)) if pad else values
            vb = vp.to(torch.float32).reshape(B, Hkv, nb, m, D)
            dv = (vb - (vb * valid).sum(-2, keepdim=True) / n_blk.unsqueeze(-2)) * valid
            Uv = self._topr(dv, r)

        self._basis.clear()                       # one context at a time
        self._basis[key] = (Uk, Uv)
        return Uk, Uv

    # ----------------------------------------------------------------- output --
    @torch.no_grad()
    def _multibin_output(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
        attention_mask: Optional[torch.Tensor], scaling: float, sparse_mask: Mask,
        key_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # NOTE: do NOT delegate rank 0 to the parent.  The parent leaves `bid` as
        # None for bin_mode="fixed" and its expansion stage is guarded on
        # `bid is not None`, so a fixed-block config SILENTLY IGNORES expand_m.
        # Rank 0 therefore runs through the path below, where the projector is
        # empty (p == 0, w == 1, sum_i w_i == n_b) and the proxy is bit-identical
        # to the parent's -- but the expansion stage below actually runs.  That is
        # the matched baseline the sketch has to beat.
        ngroups = _get_num_key_value_groups(queries, keys)
        k = repeat_kv(keys, ngroups).to(torch.float32)
        v = repeat_kv(values, ngroups).to(torch.float32)
        q = queries.to(torch.float32)

        raw = torch.matmul(q, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            raw = raw + attention_mask[:, :, :, : k.shape[-2]].to(torch.float32)
        valid = raw > NEG
        sel = (sparse_mask.get_dense_mask() != 0) & valid
        tail_f = ((~sel) & valid).to(torch.float32)

        m = self.bin_size
        B, H, Q, K = raw.shape
        D = v.shape[-1]
        pad = (-K) % m
        nb = (K + pad) // m
        dev, dt = raw.device, raw.dtype

        n_b, zbar_b, s2_b, vbar_b = self._bins_fixed_keyside(q, k, v, tail_f, scaling)
        live = n_b > 0

        # ---- sketch weights ------------------------------------------------
        Uk, Uv = self._block_basis(keys, values, self._cur_layer)
        rep = lambda t: (t.unsqueeze(2).expand(-1, -1, ngroups, -1, -1, -1)
                         .reshape(B, H, nb, m, -1).unsqueeze(2))     # -> (B,H,1,nb,m,r)
        Ukr = rep(Uk)
        tb = (torch.nn.functional.pad(tail_f, (0, pad)) if pad else tail_f
              ).reshape(B, H, Q, nb, m)
        rawb = (torch.nn.functional.pad(raw, (0, pad), value=0.0) if pad else raw
                ).reshape(B, H, Q, nb, m)
        blkv = (tb.new_ones((1, 1, 1, nb, m)) if not pad else
                torch.cat([tb.new_ones((1, 1, 1, nb - 1, m)),
                           torch.cat([tb.new_ones((1, 1, 1, 1, m - pad)),
                                      tb.new_zeros((1, 1, 1, 1, pad))], -1)], -2))

        # centre on the BLOCK (query-independent membership), project, then
        # re-centre on the block's TAIL members so the weights multiply e^{zbar_b}
        c = (rawb - (rawb * blkv).sum(-1, keepdim=True)
             / blkv.sum(-1, keepdim=True).clamp(min=1)) * blkv
        p = (Ukr @ (Ukr.transpose(-1, -2) @ c.unsqueeze(-1))).squeeze(-1)
        pbar = (p * tb).sum(-1, keepdim=True) / n_b.clamp(min=1).unsqueeze(-1)
        w = torch.exp((p - pbar).clamp(max=30.0)) * tb
        sw = w.sum(-1)                                    # replaces n_b in ell_b

        if self.sketch_values and Uv is not None:
            Uvr = rep(Uv)
            vb = (torch.nn.functional.pad(v, (0, 0, 0, pad)) if pad else v
                  ).reshape(B, H, nb, m, D)
            y = (Uvr @ (Uvr.transpose(-1, -2) @ w.unsqueeze(-1))).squeeze(-1) * tb
            # The BASIS is block-centred (query-independent, cacheable), but the
            # vector it reconstructs must be centred on the bin's TAIL members --
            # vbar_b is what the correction is added to.  Centring both on the
            # block instead leaves the proxy off by (vbar_b - vbar_blk), which is
            # exactly what a full-rank equivalence check catches.  Expand that
            # centring algebraically rather than materialising (v - vbar_b):
            #   sum_i y_i (v_i - vbar_b) = sum_i y_i v_i - (sum_i y_i) vbar_b
            # `vb` carries no query axis, so the (B,H,Q,nb,m,D) tensor the direct
            # form needs -- 26 GB at the question prefill -- never exists.
            corr = (torch.einsum("bhqnm,bhnmd->bhqnd", y, vb)
                    - y.sum(-1).unsqueeze(-1) * vbar_b)
            vbar_b = vbar_b + corr / sw.clamp(min=1e-30).unsqueeze(-1)

        # ---- expansion: unchanged ranking (V16: error-ranking loses) --------
        expand_tok = None
        if self.expand_m > 0:
            score_b = zbar_b + self.expand_c * s2_b.clamp(min=0).sqrt()
            mm = min(self.expand_m, nb)
            idx = score_b.masked_fill(~live, float("-inf")).topk(mm, dim=-1).indices
            is_exp = torch.zeros_like(live).scatter_(-1, idx, True) & live
            expand_tok = (is_exp.unsqueeze(-1).expand(-1, -1, -1, -1, m)
                          .reshape(B, H, Q, nb * m)[..., :K]) & (tail_f > 0)
            sw = sw.masked_fill(is_exp, 0.0)
            live = live & ~is_exp

        # ---- single global softmax over {heavy} u {expanded} u {proxies} ----
        ell_b = (zbar_b + torch.log(sw.clamp(min=1e-30)) + self._kappa(s2_b))
        ell_b = ell_b.masked_fill(~(live & (sw > 0)), float("-inf"))

        raw_max = raw.masked_fill(~valid, float("-inf")).amax(-1, keepdim=True)
        row_max = torch.maximum(raw_max, ell_b.amax(-1, keepdim=True))
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

        exact = sel if expand_tok is None else (sel | expand_tok)
        e_sel = torch.exp(raw - row_max) * exact
        num = torch.matmul(e_sel, v)
        den = e_sel.sum(-1, keepdim=True)

        w_b = torch.exp(ell_b - row_max)
        w_b = torch.where(torch.isfinite(w_b), w_b, torch.zeros_like(w_b))
        num = num + torch.einsum("bhqn,bhqnd->bhqd", w_b, vbar_b)
        den = den + w_b.sum(-1, keepdim=True)

        return (num / den.clamp(min=1e-30)).to(queries.dtype).transpose(1, 2).contiguous()


    @classmethod
    def create_from_config(
        cls, config: "UTAMultiBinSketchConfig"
    ) -> "UTAMultiBinSketchAttention":
        if not isinstance(config, UTAMultiBinSketchConfig):
            raise TypeError(f"Expected UTAMultiBinSketchConfig, got {type(config)}")
        maskers = [ResearchMasker.create_masker_from_config(mc)
                   for mc in config.masker_configs]
        return cls(config, maskers)
