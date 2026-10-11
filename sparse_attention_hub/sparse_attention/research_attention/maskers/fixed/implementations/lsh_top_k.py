"""LSH top-K masker (V49, paper v13 selector comparison).

A heavy-hitter selector built from MagicPIG's locality-sensitive hashing: keys and
queries are centred by the key mean, lifted by the Neyshabur--Srebro MIPS transform,
and hashed by L tables of K signed random projections, exactly as the MagicPIG
sampling masker does.  Instead of sampling the colliding keys with importance
weights, the selector ranks every key by the number of tables in which its code
equals the query's code (the LSH estimate of the query--key angle) and keeps the
top-k.  No learned weights; the footprint is K*L bits per token per KV head.

State lives in sparse_meta_data["lsh_sel"] so it never collides with a MagicPIG
sampling masker in the same pipeline.
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

from ...sampling.implementations.magic_pig import MagicPig, MagicPigConfig
from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class LSHTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for LSHTopKMasker."""

    lsh_k: int = 10          # bits per table
    lsh_l: int = 150         # tables
    center: bool = True
    seed: int = 42
    search_space: Dict[str, Any] = field(default_factory=dict)


@MaskerRegistry.register(LSHTopKMaskerConfig)
class LSHTopKMasker(TopKMasker):
    """Top-k on the LSH collision count (MagicPIG hashing, deterministic selection)."""

    def __init__(self, config: LSHTopKMaskerConfig) -> None:
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self._mp = MagicPig(MagicPigConfig(lsh_l=int(config.lsh_l), lsh_k=int(config.lsh_k),
                                           center=bool(config.center), packing="int64",
                                           seed=int(config.seed)))

    @torch.no_grad()
    def _counts(self, qc: torch.Tensor, kc: torch.Tensor) -> torch.Tensor:
        """Number of equal table codes, (B,H,Q,L) x (B,H,K,L) -> (B,H,Q,K), chunked."""
        B, H, Q, L = qc.shape
        K = kc.shape[2]
        out = torch.empty(B, H, Q, K, device=qc.device, dtype=torch.float32)
        step_k = max(1, int(2e8 // max(1, B * H * L)))
        for q0 in range(Q):
            qrow = qc[:, :, q0:q0 + 1].unsqueeze(3)                          # (B,H,1,1,L)
            for k0 in range(0, K, step_k):
                kk = kc[:, :, k0:k0 + step_k].unsqueeze(2)                    # (B,H,1,k,L)
                out[:, :, q0:q0 + 1, k0:k0 + step_k] = (qrow == kk).sum(-1).float()
        return out

    @torch.no_grad()
    def lsh_scores(self, keys: torch.Tensor, queries: torch.Tensor,
                   sparse_meta_data: Dict[str, Any], layer_idx: int) -> torch.Tensor:
        """Collision counts (B,H,Q,K) of every key with every query row."""
        smd = sparse_meta_data.setdefault("lsh_sel", {})
        ng = _get_num_key_value_groups(queries, keys)
        K = keys.shape[2]
        # codes are per KV head (the projection is shared by all heads of a layer)
        kcent, _ = self._mp._center_KQ(keys.float(), keys.float()[:, :, :1], smd, layer_idx)
        key_mean = smd["key_mean"][layer_idx]                                # (B,Hkv,1,D)
        qcent = queries.float() - repeat_kv(key_mean, ng)
        # MIPS lift: keys scaled by the max key norm (fixed at the first build, as the
        # cached codes of old keys cannot be recomputed), queries normalised.
        mx = smd.setdefault("max_norm", {})
        if layer_idx not in mx:
            mx[layer_idx] = kcent.norm(dim=-1).amax()
        m = mx[layer_idx]
        kn = kcent.norm(dim=-1, keepdim=True).clamp(max=m)
        kt = torch.cat([kcent / m, torch.sqrt((m * m - kn * kn).clamp(min=0)) / m], -1)
        qt = torch.cat([qcent / (qcent.norm(dim=-1, keepdim=True) + 1e-8),
                        torch.zeros_like(qcent[..., :1])], -1)
        self._mp._initialize_projection_cache(smd, layer_idx)
        if smd["projections"][layer_idx] is None:
            g = torch.Generator(device=keys.device)
            g.manual_seed(self._mp.seed + layer_idx)
            smd["projections"][layer_idx] = torch.randn(kt.shape[-1], self._mp.lsh_k * self._mp.lsh_l,
                                                        device=keys.device, dtype=torch.float32, generator=g)
        # incremental key codes (B,Hkv,K,L)
        self._mp._initialize_key_signature_cache(smd, layer_idx)
        cached = smd["key_signature"][layer_idx]
        if cached is None or cached.shape[2] > K:
            cached = self._mp._compute_signatures(kt.contiguous(), smd, layer_idx)
        elif cached.shape[2] < K:
            new = self._mp._compute_signatures(kt[:, :, cached.shape[2]:].contiguous(), smd, layer_idx)
            cached = torch.cat([cached, new], 2)
        smd["key_signature"][layer_idx] = cached
        qcodes = self._mp._compute_signatures(qt.contiguous(), smd, layer_idx)    # (B,H,Q,L)
        return self._counts(qcodes, repeat_kv(cached, ng))

    def add_mask(self, keys: torch.Tensor, queries: torch.Tensor, values: torch.Tensor,
                 attention_mask: torch.Tensor, scaling: float, dropout: float,
                 sparse_meta_data: Dict[str, Any], previous_mask: Mask, **kwargs: Any) -> Mask:
        if sparse_meta_data is None:
            raise ValueError("sparse_meta_data cannot be None")
        layer_idx: Optional[int] = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")
        if previous_mask.is_full_mask():
            return previous_mask
        dims: AttentionTensorDimensions = self._extract_tensor_dimensions(keys, queries)
        k_eff: int = self._calculate_effective_size(self.heavy_size, dims.seq_len_keys)
        if dims.seq_len_keys <= k_eff:
            return self._create_full_mask(dims, previous_mask.dtype, previous_mask.device)
        scores = self.lsh_scores(keys, queries, sparse_meta_data, layer_idx)
        K = keys.shape[2]
        # collision counts are small integers: break ties at random (fixed per layer)
        # rather than by position, which topk would otherwise do implicitly
        g = torch.Generator(device=scores.device)
        g.manual_seed(7 + int(layer_idx))
        scores = scores + 1e-3 * torch.rand(scores.shape, device=scores.device, generator=g)
        # publish the per-token score (as the static selector does) so a cluster
        # expansion can order the tokens of a cluster by it
        sparse_meta_data.setdefault("static_scores", {})[layer_idx] = scores
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :K].to(scores.dtype)
        idx = self._get_topk_indices_from_inactive_positions(scores, previous_mask, k_eff)
        mask = self._create_mask_from_rowise_indices(dims, idx, keys.device, previous_mask.dtype)
        return previous_mask.merge_mask(mask, inplace=False)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "LSHTopKMasker":
        if not isinstance(config, LSHTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
