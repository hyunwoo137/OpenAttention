"""Static-metric top-K masker (V25).

A heavy-hitter selector that replaces the learned HashAttention signatures by the
query second moment the BinAttention partition already relies on.  Per layer and
query head, M = E[q q^T] is calibrated offline in the pre-RoPE space, rotated to the
decode position (window-averaged over `rot_window` positions) and, when a question
chunk is available, shrunk toward the request's own question rows.  Its top-r
eigenvectors U give an r-dimensional key code  z_i = U^T k_i  computed ONCE per key
(at the question-chunk call for the context, incrementally for generated tokens).
For a query q the approximate logit is  (U^T q) . z_i  and the selector is top-k on it.

Footprint per token per query head: r * 16 bits (bf16 codes); with `share_kv` one
U per KV head (M averaged over the group's query heads) so the codes are stored per
KV head -- 1/4 of the per-query-head storage for GQA-4.  Optional int8 codes.

Everything is cached in sparse_meta_data["static_sel"][layer_idx] and is therefore
reset with the request, exactly like HashAttention's key signatures.  The cache is
also published for the BinAttention partition (multibin.py) so the projection is not
recomputed there.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class StaticTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for StaticTopKMasker."""

    static_file: str = ""            # torch file: M (L,H,D,D) pre-RoPE, inv_freq, attention_scaling
    rank: int = 16                   # r
    rot_window: int = 32             # w: decode positions averaged in the rotated metric
    source: str = "static+question"  # "static" | "static+question"
    profile_P: int = 64              # question rows behind M_q
    share_kv: bool = False           # one U per KV head (codes stored per KV head)
    code_bits: int = 16              # 16 (bf16 codes) or 8 (int8 codes, per-coordinate scale)
    rebuild_every: int = 0           # >0: rebuild U and re-project every N generated tokens
    basis: str = "eig"               # "eig" (top-r eigenvectors of M) | "random" (V49 control:
                                     # a random orthonormal r-frame, same codes/cost, no query geometry)
    search_space: Dict[str, Any] = field(default_factory=dict)


@MaskerRegistry.register(StaticTopKMaskerConfig)
class StaticTopKMasker(TopKMasker):
    """Top-k on the rank-r static-metric approximation of the logits."""

    def __init__(self, config: StaticTopKMaskerConfig) -> None:
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.rank = int(config.rank)
        self.rot_window = int(config.rot_window)
        self.source = config.source
        self.profile_P = int(config.profile_P)
        self.share_kv = bool(config.share_kv)
        self.code_bits = int(config.code_bits)
        self.rebuild_every = int(config.rebuild_every)
        self.basis = str(getattr(config, "basis", "eig"))
        if self.basis not in ("eig", "random"):
            raise ValueError(f"basis must be 'eig' or 'random', got {self.basis}")
        if self.source not in ("static", "static+question"):
            raise ValueError(f"source must be 'static' or 'static+question', got {self.source}")
        if not config.static_file:
            raise ValueError("static_file is required")
        d = torch.load(config.static_file, map_location="cpu")
        self._M_pre: torch.Tensor = d["M"].float()                       # (L,H,D,D)
        self._inv_freq: torch.Tensor = d["inv_freq"].float()
        self._rope_scaling: float = float(d.get("attention_scaling", 1.0))
        self._dev: Optional[torch.device] = None

    # ------------------------------------------------------------------ RoPE --
    def _to_device(self, device: torch.device) -> None:
        if self._dev != device:
            self._M_pre = self._M_pre.to(device)
            self._inv_freq = self._inv_freq.to(device)
            self._dev = device

    @staticmethod
    def _rope_cs(inv_freq: torch.Tensor, scaling: float, positions: torch.Tensor):
        freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], -1)
        return emb.cos() * scaling, emb.sin() * scaling

    @staticmethod
    def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin

    @torch.no_grad()
    def _rot_avg(self, M_pre: torch.Tensor, p: int, w: int) -> torch.Tensor:
        """mean_{t in [p,p+w)} R_t M R_t^T in closed form (see run_v25_static_topk_diag)."""
        inv, scl = self._inv_freq, self._rope_scaling
        D = M_pre.shape[-1]
        dev = M_pre.device
        th = torch.cat([inv, inv]).double()
        a = th.unsqueeze(1); b = th.unsqueeze(0)

        def mean_exp(alpha: torch.Tensor) -> torch.Tensor:
            if w <= 64:
                t = torch.arange(p, p + w, device=dev, dtype=torch.float64)
                ph = alpha.unsqueeze(-1) * t
                return torch.complex(ph.cos().mean(-1), ph.sin().mean(-1))
            z = torch.polar(torch.ones_like(alpha), alpha)
            den = 1 - z
            small = den.abs() < 1e-9
            s = torch.where(small, torch.full_like(z, float(w)),
                            (1 - z ** w) / torch.where(small, torch.ones_like(den), den))
            return s * torch.polar(torch.ones_like(alpha), alpha * p) / w

        Em = mean_exp(a - b); Ep = mean_exp(a + b)
        cc = 0.5 * (Em.real + Ep.real); ss = 0.5 * (Em.real - Ep.real)
        cs = 0.5 * (Ep.imag - Em.imag); sc = 0.5 * (Ep.imag + Em.imag)
        cc, ss, cs, sc = [x.float() * (scl * scl) for x in (cc, ss, cs, sc)]
        h = D // 2
        J = torch.zeros(D, D, device=dev)
        J[:h, h:] = -torch.eye(h, device=dev); J[h:, :h] = torch.eye(h, device=dev)
        M = M_pre.float()
        return cc * M + cs * (M @ J.T) + sc * (J @ M) + ss * (J @ M @ J.T)

    # ----------------------------------------------------------------- build --
    @torch.no_grad()
    def _build(self, queries: torch.Tensor, keys: torch.Tensor, layer_idx: int, ng: int,
               Mq: Optional[torch.Tensor], pos_offset: int = 0) -> Dict[str, Any]:
        """U for this layer at the current decode start, and the codes of all keys.

        `pos_offset` (V54): after a KV press the cache holds fewer entries than the
        context had tokens; the decode position is `K + pos_offset`, not `K`.
        """
        B, H, Q, D = queries.shape
        K = keys.shape[2]
        Hkv = keys.shape[1]
        Mst = self._rot_avg(self._M_pre[layer_idx], K + int(pos_offset), max(self.rot_window, 1))     # (H,D,D)
        if Mq is not None and self.source == "static+question":
            def _nt(x):
                return x / torch.diagonal(x, dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12)[..., None, None]
            M = 0.5 * _nt(Mst) + 0.5 * _nt(Mq)
        else:
            M = Mst
        if self.share_kv:
            M = M.view(Hkv, ng, D, D).mean(1)                                        # (Hkv,D,D)
        if self.basis == "random":
            # V49 control: replace the query eigenvectors by a random orthonormal frame
            # (fixed per layer), keeping rank, codes and arithmetic identical.
            g = torch.Generator(device="cpu"); g.manual_seed(1234 + int(layer_idx))
            G = torch.randn(M.shape[0], D, self.rank, generator=g)
            U = torch.linalg.qr(G)[0].to(M.device).contiguous()                     # (Hs,D,r)
        else:
            evals, evecs = torch.linalg.eigh(M.float())
            U = evecs[..., -self.rank:].contiguous()                                 # (Hs,D,r)
        ent: Dict[str, Any] = {"U": U, "K": 0, "Kbuild": K, "codes": None, "scale": None, "Mq": Mq}
        return self._extend(ent, keys, ng)

    @torch.no_grad()
    def _extend(self, ent: Dict[str, Any], keys: torch.Tensor, ng: int) -> Dict[str, Any]:
        K0, K = ent["K"], keys.shape[2]
        if K <= K0:
            return ent
        knew = keys[:, :, K0:, :]
        if not self.share_kv:
            knew = repeat_kv(knew, ng)                                               # (B,H,dK,D)
        z = torch.matmul(knew.to(torch.float32), ent["U"].unsqueeze(0))              # (B,Hs,dK,r)
        if self.code_bits == 8:
            if ent["scale"] is None:                                                 # fixed at the first build
                ent["scale"] = (z.abs().amax(dim=2, keepdim=True) / 127.0).clamp(min=1e-12)
            z = (z / ent["scale"]).round().clamp(-127, 127).to(torch.int8)
        else:
            z = z.to(torch.bfloat16)
        ent["codes"] = z if ent["codes"] is None else torch.cat([ent["codes"], z], 2)
        ent["K"] = K
        return ent

    @torch.no_grad()
    def _question_moment(self, queries: torch.Tensor, K: int) -> torch.Tensor:
        """M_q from P rows of the question chunk, every row rotated to position K."""
        B, H, Q, D = queries.shape
        P = min(self.profile_P, max(Q - 2, 1))
        pi = torch.linspace(0, max(Q - 3, 0), P, device=queries.device).long().unique()
        qsel = queries[0].index_select(1, pi).float()                                # (H,P,D)
        cos, sin = self._rope_cs(self._inv_freq, self._rope_scaling, (Q - pi).to(queries.device))
        qrot = self._rope_apply(qsel, cos, sin)
        return torch.matmul(qrot.transpose(-1, -2), qrot) / qrot.shape[1]           # (H,D,D)

    # --------------------------------------------------------------- scores --
    @torch.no_grad()
    def _scores(self, queries: torch.Tensor, ent: Dict[str, Any], ng: int) -> torch.Tensor:
        B, H, Q, D = queries.shape
        U = ent["U"]
        if self.share_kv:
            U = U.repeat_interleave(ng, dim=0)                                       # (H,D,r)
        zq = torch.matmul(queries.to(torch.float32), U.unsqueeze(0))                 # (B,H,Q,r)
        codes = ent["codes"]
        if self.code_bits == 8:
            # int8 codes carry a per-(head, coordinate) scale: fold it into the query side
            zk = codes.to(torch.float32)
            sc = ent["scale"]                                                        # (B,Hs,1,r)
            if self.share_kv:
                zq = zq * sc.repeat_interleave(ng, dim=1)
                zk = zk.repeat_interleave(ng, dim=1)
            else:
                zq = zq * sc
        else:
            zk = codes.to(torch.float32)
            if self.share_kv:
                zk = zk.repeat_interleave(ng, dim=1)
        return torch.matmul(zq, zk.transpose(-1, -2))                                # (B,H,Q,K)

    # ------------------------------------------------------------- add_mask --
    def add_mask(self, keys: torch.Tensor, queries: torch.Tensor, values: torch.Tensor,
                 attention_mask: torch.Tensor, scaling: float, dropout: float,
                 sparse_meta_data: Dict[str, Any], previous_mask: Mask, **kwargs: Any) -> Mask:
        if sparse_meta_data is None:
            raise ValueError("sparse_meta_data cannot be None")
        layer_idx: Optional[int] = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")
        self._to_device(keys.device)
        if previous_mask.is_full_mask():
            return previous_mask
        dims: AttentionTensorDimensions = self._extract_tensor_dimensions(keys, queries)
        k_eff: int = self._calculate_effective_size(self.heavy_size, dims.seq_len_keys)
        if dims.seq_len_keys <= k_eff:
            return self._create_full_mask(dims, previous_mask.dtype, previous_mask.device)
        ng = _get_num_key_value_groups(queries, keys)
        cache = sparse_meta_data.setdefault("static_sel", {})
        ent = cache.get(layer_idx)
        Q, K = queries.shape[2], keys.shape[2]
        from sparse_attention_hub.sparse_attention.utils.centroidkv import kvpress_pos_offset
        _off = kvpress_pos_offset(sparse_meta_data, layer_idx)     # V54: 0 unless a KV press ran
        if Q >= 8:
            # question chunk (or any multi-row call with a partition-worthy query block): rebuild
            Mq = self._question_moment(queries, K) if self.source == "static+question" else None
            ent = self._build(queries, keys, layer_idx, ng, Mq, _off)
        elif ent is None or K < ent["K"]:
            # V54b: a shrunk cache (prompt-time KV press) keeps the question moment the
            # question chunk produced; only the codes are rebuilt over the new keys
            ent = self._build(queries, keys, layer_idx, ng,
                              ent.get("Mq") if ent is not None else None, _off)
        elif self.rebuild_every > 0 and K - ent["Kbuild"] >= self.rebuild_every:
            ent = self._build(queries, keys, layer_idx, ng, ent.get("Mq"), _off)
        else:
            ent = self._extend(ent, keys, ng)
        cache[layer_idx] = ent
        scores = self._scores(queries, ent, ng)
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :K].to(scores.dtype)
        sparse_meta_data.setdefault("static_scores", {})[layer_idx] = scores
        idx = self._get_topk_indices_from_inactive_positions(scores, previous_mask, k_eff)
        mask = self._create_mask_from_rowise_indices(dims, idx, keys.device, previous_mask.dtype)
        return previous_mask.merge_mask(mask, inplace=False)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "StaticTopKMasker":
        if not isinstance(config, StaticTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
