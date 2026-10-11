"""SnapKV / PyramidKV maskers (V35, paper v10 Table `main` KV-compression block).

KV-cache *compression* baselines under the hub protocol (context prefilled with
full attention, sparse attention on the question chunk and generation):

  SnapKV     At the FIRST sparse call of a request (the question chunk), score
             every prefix key by the attention the observation window (the last
             `obs_window` rows of that call's query block) pays to it, summed
             over the window and max-pooled along the key axis (kernel
             `pool_kernel`); keep the top `heavy_size` keys per KV head.  The
             kept set is FROZEN: every later question chunk and decode step
             reads only the kept keys (plus whatever earlier maskers -- sink and
             local -- activate, and keys created after the selection, which an
             eviction method also keeps resident).
  PyramidKV  Same, with the per-layer budget following the PyramidKV
             arithmetic allocation: layer 0 gets `2b - b/beta`, the last layer
             `b/beta`, linear in between (same total as SnapKV).

This reproduces the decode-time READ pattern of the eviction methods at a
matched budget; it does not physically free memory (the harness never does).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.kv_compress import (
    default_window,
    select_resident,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class SnapKVMaskerConfig(TopKMaskerConfig):
    """Configuration for the SnapKV / PyramidKV masker."""

    obs_window: int = 32
    pool_kernel: int = 7
    pyramid: bool = False        # PyramidKV per-layer budget allocation
    pyramid_beta: int = 20
    num_layers: int = 32
    # V39: the protocol of the PyramidKV paper (Cai et al., 2024, S4.2).  Their budget
    # is an ABSOLUTE average cache size per layer (pass heavy_size as an int), the
    # observation window is the last alpha=8 tokens of the input, those alpha tokens
    # are retained in every layer, and the compression applies to what is STORED --
    # the prompt itself is encoded with full attention and only the generation reads
    # the compressed cache.  `paper_mode` switches all three on.
    paper_mode: bool = False
    alpha: int = 8
    # V40: which compressor to reproduce -- "snapkv", "pyramidkv" or "streamingllm"
    # (see `utils.kv_compress`).  `pyramid=True` is kept as an alias for "pyramidkv".
    mode: str = ""
    search_space: Dict[str, Any] = field(default_factory=dict)


@MaskerRegistry.register(SnapKVMaskerConfig)
class SnapKVMasker(TopKMasker):
    """SnapKV-style one-shot key selection, frozen for the rest of the request."""

    heavy_size: Union[float, int]

    def __init__(self, config: SnapKVMaskerConfig) -> None:
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.obs_window = int(config.obs_window)
        self.pool_kernel = int(config.pool_kernel)
        self.pyramid = bool(config.pyramid)
        self.pyramid_beta = int(config.pyramid_beta)
        self.num_layers = int(config.num_layers)
        self.paper_mode = bool(getattr(config, "paper_mode", False))
        self.alpha = int(getattr(config, "alpha", 8))
        self.mode = str(getattr(config, "mode", "") or
                        ("pyramidkv" if self.pyramid else "snapkv"))
        if self.paper_mode:
            # KVCache-Factory `run_ruler.py` fixes the window per method
            self.obs_window = default_window(self.mode, int(self.heavy_size), self.alpha)

    # -- selection (first sparse call of the request) ---------------------
    def _select(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        layer_idx: int,
    ) -> Dict[str, Any]:
        """Delegate to the shared, reference-faithful selector (`utils.kv_compress`).

        The composition path (`UTAMultiBinAttention._residency_bias`) calls the same
        function with the same arguments, so the stand-alone baseline and the
        compressed cache BinAttention reads from are the identical key set.
        """
        budget = (int(self.heavy_size) if self.paper_mode
                  else self._calculate_effective_size(self.heavy_size, keys.shape[2]))
        keep = select_resident(
            queries, keys, attention_mask, scaling,
            mode=self.mode, budget=budget, window=self.obs_window,
            beta=self.pyramid_beta, kernel=self.pool_kernel,
            layer_idx=layer_idx, num_layers=self.num_layers)
        return {"keep": keep, "n0": keys.shape[2]}

    # -- masker interface -------------------------------------------------
    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        if previous_mask.is_full_mask():
            return previous_mask
        layer_idx = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")
        cache: Dict[int, Dict[str, Any]] = sparse_meta_data.setdefault("snapkv", {})
        ent = cache.get(layer_idx)
        if ent is None:
            ent = self._select(queries, keys, attention_mask, scaling, layer_idx)
            cache[layer_idx] = ent
            if self.paper_mode:
                # the prompt is encoded densely; compression applies to what is stored
                return Mask.create_full_mask(
                    previous_mask.shape, dtype=previous_mask.dtype,
                    device=queries.device)

        b, hq, nq, _ = queries.shape
        hkv, n_keys = keys.shape[1], keys.shape[2]
        ng = hq // hkv
        dense = previous_mask.get_dense_mask()
        keep = ent["keep"]
        if n_keys > ent["n0"]:   # keys created after the selection stay resident
            keep = torch.cat([keep, torch.ones((b, hkv, n_keys - ent["n0"]),
                                               dtype=torch.bool, device=keys.device)], -1)
        else:
            keep = keep[..., :n_keys]
        keep = keep.unsqueeze(2).expand(b, hkv, ng, n_keys).reshape(b, hq, 1, n_keys)
        dense = torch.where(keep.expand(b, hq, nq, n_keys), torch.ones_like(dense), dense)
        return Mask.create_mask_from_dense_mask(dense.shape, dense, dtype=previous_mask.dtype)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "SnapKVMasker":
        if not isinstance(config, SnapKVMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
