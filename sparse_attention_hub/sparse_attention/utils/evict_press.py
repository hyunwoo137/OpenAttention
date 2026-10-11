"""V54c: the eviction compressors (SnapKV / PyramidKV / StreamingLLM) as a *context* press.

The table's eviction rows follow KVCache-Factory: the whole prompt, question included, is
encoded with full attention and the last 8 prompt tokens (the question) score what stays.
CentroidKV's authors evaluate under the kvpress split instead: the context is compressed
before the question is seen, and their eviction baselines score the context by its own
last tokens.  This press reproduces that split for the eviction methods with the same
hyper-parameters as the table (`kv_compress.select_resident`: absolute per-layer budget,
maxpool kernel 7, window 8 / budget-4, PyramidKV's arithmetic allocation), so that a
compressor can be compared to CentroidKV under both splits at the same budget.

The kept rows are physically compacted out of the HuggingFace cache (as kvpress does), so
the cache really holds `budget` entries per layer; question and generated tokens are then
positioned as if the context were uncompressed (`adapters/huggingface.py`).  The scoring
needs the post-RoPE queries of the last `window` context tokens, which the adapter captures
from each layer's `q_proj` during the context prefill (`needs_queries`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .kv_compress import MODES, default_window, select_resident


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rope_queries(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to `(b, H, w, d)` queries with `(b, w, d)` cos/sin (HF convention)."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin)


@dataclass
class EvictPress:
    """SnapKV / PyramidKV / StreamingLLM applied to the context cache (kvpress split)."""

    mode: str = "pyramidkv"
    budget: int = 8000
    window: int = 0            # 0 = the method's default (8, or budget-4 for streamingllm)
    kernel: int = 7
    beta: int = 20
    num_layers: int = 32
    when: str = "context"
    logcount: bool = False     # no count term for eviction; kept for interface parity

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.when != "context":
            raise ValueError("EvictPress runs on the context only; the prompt-time variant "
                             "is the residency path (--evict-mode) of the table")
        if self.window <= 0:
            self.window = default_window(self.mode, int(self.budget))

    @property
    def needs_queries(self) -> bool:
        return self.mode != "streamingllm"

    def tag(self) -> str:
        return f"{self.mode}ctx{int(self.budget)}"

    @torch.no_grad()
    def apply_to_cache(self, cache: Any, n_orig: int, queries: Optional[Dict[int, torch.Tensor]] = None,
                       scaling: Optional[float] = None) -> Dict[str, Any]:
        """Compact every layer of a HF cache in place to its resident rows.

        `queries[layer]`: `(b, H, w, d)` post-RoPE queries of the last `w` context tokens
        (unused by streamingllm).  Returns the record `sparse_meta_data["kvpress"]` expects.
        """
        n_new: Dict[int, int] = {}
        layers = list(enumerate(cache.layers)) if hasattr(cache, "layers") else None
        if layers is None:
            layers = [(i, None) for i in range(len(cache.key_cache))]
        for i, layer in layers:
            keys = layer.keys if layer is not None else cache.key_cache[i]
            values = layer.values if layer is not None else cache.value_cache[i]
            if keys is None or keys.numel() == 0:
                continue
            b, hkv, n, d = keys.shape
            if self.needs_queries:
                if queries is None or i not in queries:
                    raise ValueError(f"EvictPress({self.mode}) needs the window queries of layer {i}")
                q = queries[i].to(keys.device)
                sc = float(scaling) if scaling is not None else d ** -0.5
            else:
                q = torch.zeros((b, hkv, 1, d), dtype=keys.dtype, device=keys.device)
                sc = d ** -0.5
            keep = select_resident(q, keys, None, sc, mode=self.mode, budget=int(self.budget),
                                   window=int(self.window), beta=self.beta, kernel=self.kernel,
                                   layer_idx=i, num_layers=self.num_layers)        # (b,hkv,n)
            n_keep = int(keep[0, 0].sum())
            if int(keep.sum()) != b * hkv * n_keep:
                raise RuntimeError("select_resident returned an uneven kept count across heads")
            pos = torch.arange(n, device=keys.device).view(1, 1, n).expand(b, hkv, n)
            idx = pos[keep].view(b, hkv, n_keep)                                  # sorted per head
            gi = idx.unsqueeze(-1).expand(b, hkv, n_keep, d)
            k2, v2 = keys.gather(2, gi), values.gather(2, gi)
            if layer is not None:
                layer.keys, layer.values = k2, v2
            else:
                cache.key_cache[i], cache.value_cache[i] = k2, v2
            n_new[i] = n_keep
        return {"press": self.mode, "tag": self.tag(), "n_orig": int(n_orig), "n_new": n_new,
                "logcount": None}
