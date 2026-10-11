r"""CentroidKV (Hu et al., 2025, arXiv:2506.11418) as a cache *press* for the hub.

The merging algorithm below is the released `CentroidKVPress.compress` of
https://github.com/jessica-hujie/CentroidKV (a kvpress `BasePress`), transcribed
line for line so that the merged keys and values are bit-identical to the release
(`tests/unit/test_centroidkv_press.py` checks this against the released file).  Two
things are added, neither of which changes the merged cache:

* the per-entry **count** (the paper's cluster degree $n_t$) is returned instead of
  being discarded, so that the paper's attention over centroids,
  $\mathrm{softmax}(qK^\top/\sqrt d + \log N)\,V$ (their Eq. 3 / Algorithm 1, line 11),
  can be applied as an additive bias when `logcount=True`; the release itself drops
  the count and attends to every centroid with weight one, which is what
  `logcount=False` reproduces;
* `apply_to_cache` rewrites a HuggingFace `DynamicCache` in place after the context
  prefill, which is where kvpress runs a press (its forward hook fires once per layer
  on the prefill call, before the question is seen).

Protocol (kvpress / CentroidKV paper, Sec. 6.1): the context is compressed before the
question is introduced; question and generated tokens are then positioned as if the
context were uncompressed (`kvpress.pipeline.generate_answer` starts their
`position_ids` at the *original* context length), and every later token is appended
to the compressed cache uncompressed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


# --------------------------------------------------------------------------- release --
def bipartite_soft_matching(key: torch.Tensor, n_ex: int, n_chunks: int, chunk_size: int, r: int):
    """Verbatim from the release (see module docstring)."""
    bsz, n_head, _, head_dim = key.shape
    key = key / key.norm(dim=-1, keepdim=True)

    if n_ex == 0:
        key_chunks = key[:, :, :, :].view(bsz, n_head, n_chunks, chunk_size, head_dim)
        ex_node_max = None
        ex_node_idx = None
    else:
        key_chunks = key[:, :, :-n_ex, :].view(bsz, n_head, n_chunks, chunk_size, head_dim)
        key_ex = key[:, :, -n_ex:, :]
        ex_a, ex_b = key_ex[:, :, ::2, :], key_ex[:, :, 1::2, :]
        scores_ex = ex_a @ ex_b.transpose(-1, -2)
        ex_node_max, ex_node_idx = scores_ex.max(dim=-1)

    a, b = key_chunks[:, :, :, ::2, :], key_chunks[:, :, :, 1::2, :]
    scores = a @ b.transpose(-1, -2)

    node_max, node_idx = scores.max(dim=-1)
    sort_score = node_max.reshape(bsz, n_head, -1)
    if n_ex != 0:
        sort_score = torch.cat((sort_score, ex_node_max), dim=2)
    edge_value, edge_idx = sort_score.sort(dim=-1, descending=True)

    edge_idx = edge_idx.unsqueeze(-1)

    unm_idx = edge_idx[:, :, r:, :]
    src_idx = edge_idx[:, :, :r, :]

    chunk_offset = src_idx // (chunk_size // 2)
    local_idx = node_idx.reshape(bsz, n_head, -1, 1)
    if n_ex != 0:
        local_idx = torch.cat((local_idx, ex_node_idx.unsqueeze(-1)), dim=2)
    local_offset = local_idx.gather(dim=-2, index=src_idx)
    dst_idx = chunk_offset * chunk_size // 2 + local_offset

    return unm_idx, src_idx, dst_idx


def _bsmerge(step: int, key: torch.Tensor, value: torch.Tensor, count: torch.Tensor,
             n_local: int, n_sink: int, n_total: int, chunk_size: int, max_merge: int,
             ratio: float):
    """Verbatim from the release (`CentroidKVPress._bsmerge`)."""
    bsz, n_head, _, head_dim = key.shape

    n_ex = (n_total - n_sink - n_local) % chunk_size
    if n_ex % 2 != 0:
        n_ex -= 1
        n_local += 1
    n_chunks = (n_total - n_sink - n_local) // chunk_size
    if n_chunks == 0:
        chunk_size = n_total - n_sink - n_local
        n_chunks = 1
        n_ex = 0

    sink_key = key[:, :, :n_sink, :]
    sink_value = value[:, :, :n_sink, :]
    sink_count = count[:, :, :n_sink, :]

    local_key = key[:, :, -n_local:, :]
    local_value = value[:, :, -n_local:, :]
    local_count = count[:, :, -n_local:, :]

    merge_key = key[:, :, n_sink:-n_local, :]
    merge_value = value[:, :, n_sink:-n_local, :]
    merge_count = count[:, :, n_sink:-n_local, :]

    n_merge = min(max_merge, int(merge_key.shape[-2] * ratio / 2))

    unm_idx, src_idx, dst_idx = bipartite_soft_matching(merge_key, n_ex, n_chunks, chunk_size, n_merge)

    unm_idx = unm_idx.expand(*unm_idx.shape[:-1], head_dim)
    src_idx = src_idx.expand(*src_idx.shape[:-1], head_dim)
    dst_idx = dst_idx.expand(*dst_idx.shape[:-1], head_dim)

    preserved_key = merge_key[:, :, ::2, :].gather(dim=-2, index=unm_idx)
    preserved_value = merge_value[:, :, ::2, :].gather(dim=-2, index=unm_idx)

    if step == 0:
        preserved_count = torch.ones((bsz, n_head, unm_idx.shape[-2], 1), dtype=count.dtype, device=count.device)
    else:
        preserved_count = merge_count[:, :, ::2, :].gather(dim=-2, index=unm_idx[..., 0].unsqueeze(-1))

    src_key = merge_key[:, :, ::2, :].gather(dim=-2, index=src_idx)
    src_value = merge_value[:, :, ::2, :].gather(dim=-2, index=src_idx)

    if step == 0:
        src_count = torch.ones((bsz, n_head, src_idx.shape[-2], 1), dtype=count.dtype, device=count.device)
    else:
        src_count = merge_count[:, :, ::2, :].gather(dim=-2, index=src_idx[..., 0].unsqueeze(-1))

    src_key.mul_(src_count)
    src_value.mul_(src_count)

    if step == 0:
        dst_key = merge_key[:, :, 1::2, :]
        dst_value = merge_value[:, :, 1::2, :]
    else:
        dst_key = merge_key[:, :, 1::2, :] * merge_count[:, :, 1::2, :]
        dst_value = merge_value[:, :, 1::2, :] * merge_count[:, :, 1::2, :]

    dst_count = merge_count[:, :, 1::2, :].scatter_reduce(-2, dst_idx[..., 0].unsqueeze(-1), src_count, reduce="sum")
    dst_key = dst_key.scatter_reduce(-2, dst_idx, src_key, reduce="sum")
    dst_value = dst_value.scatter_reduce(-2, dst_idx, src_value, reduce="sum")

    dst_key.div_(dst_count)
    dst_value.div_(dst_count)

    count = torch.cat((sink_count, preserved_count, dst_count, local_count), dim=-2)
    key = torch.cat((sink_key, preserved_key, dst_key, local_key), dim=-2)
    value = torch.cat((sink_value, preserved_value, dst_value, local_value), dim=-2)

    n_total = count.shape[-2]

    return key, value, count, n_total


@torch.no_grad()
def centroidkv_compress(
    keys: torch.Tensor,
    values: torch.Tensor,
    compression_ratio: float,
    *,
    sink_budget: int = 16,
    local_budget: int = 64,
    chunk_size: int = 256,
    merge_ratio: float = 0.8,
    decrease_rate: float = 0.2,
    max_iterations: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`CentroidKVPress.compress` of the release, returning the counts as well.

    keys/values: (b, n_kv_heads, n, d) post-RoPE, as stored in the cache.  Returns
    (keys', values', count) with count of shape (b, n_kv_heads, n', 1) in the dtype
    of `keys` (the release keeps its running count in that dtype, and the
    count-weighted means are formed in it; we do not change that arithmetic).
    """
    if compression_ratio == 0:
        return keys, values, torch.ones(keys.shape[:3] + (1,), dtype=keys.dtype, device=keys.device)

    bsz, n_head, k_len, head_dim = keys.shape
    count = torch.ones((bsz, n_head, k_len, 1), dtype=keys.dtype, device=keys.device)
    cache_budget = int(k_len * (1 - compression_ratio))
    if k_len <= cache_budget:
        return keys, values, count

    merge_step = 0
    current_key = keys.clone()
    current_value = values.clone()
    current_count = count.clone()
    current_length = k_len
    num_local = int(local_budget)
    num_sink = int(sink_budget)

    while current_length > cache_budget and merge_step < max_iterations:
        if decrease_rate == 0:
            effective_step = merge_step
        else:
            max_linear_steps = max(0, math.floor((merge_ratio - 1e-6) / decrease_rate))
            effective_step = min(merge_step, max_linear_steps)
        ratio = merge_ratio - effective_step * decrease_rate
        max_merge = current_length - cache_budget
        current_key, current_value, current_count, current_length = _bsmerge(
            merge_step, current_key, current_value, current_count, num_local, num_sink,
            current_length, int(chunk_size), max_merge, ratio)
        merge_step += 1

    return current_key, current_value, current_count


# ----------------------------------------------------------------------- press --
@dataclass
class CentroidKVPress:
    r"""Configuration + cache rewrite.  `logcount` adds the paper's $\log N$ term."""

    compression_ratio: float = 0.75
    sink_budget: int = 16
    local_budget: int = 64
    chunk_size: int = 256
    merge_ratio: float = 0.8
    decrease_rate: float = 0.2
    logcount: bool = False
    # when the press runs (V54b):
    #   "context": kvpress protocol -- the context cache is merged before the question is
    #              seen, so the question itself is encoded against the merged cache;
    #   "prompt":  the protocol of the table's eviction rows (KVCache-Factory) -- the whole
    #              prompt, question included, is encoded with full attention, the STORED
    #              cache is merged, and only generation reads the merged cache.
    when: str = "context"

    def __post_init__(self) -> None:
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be in [0, 1)"
        assert self.when in ("context", "prompt"), "when must be 'context' or 'prompt'"

    def tag(self) -> str:
        """Short name for job directories: `ckv0.75` / `ckv0.75+prompt` / `...+logN`."""
        t = f"ckv{self.compression_ratio:g}"
        if (self.sink_budget, self.local_budget, self.chunk_size, self.merge_ratio,
                self.decrease_rate) != (16, 64, 256, 0.8, 0.2):
            t += (f"s{self.sink_budget}l{self.local_budget}c{self.chunk_size}"
                  f"m{self.merge_ratio:g}d{self.decrease_rate:g}")
        if self.when == "prompt":
            t += "+prompt"
        if self.logcount:
            t += "+logN"
        return t

    def compress(self, keys: torch.Tensor, values: torch.Tensor):
        return centroidkv_compress(
            keys, values, self.compression_ratio, sink_budget=self.sink_budget,
            local_budget=self.local_budget, chunk_size=self.chunk_size,
            merge_ratio=self.merge_ratio, decrease_rate=self.decrease_rate)

    @torch.no_grad()
    def apply_to_cache(self, cache: Any, n_orig: int) -> Dict[str, Any]:
        r"""Rewrite every layer of a HF cache in place; return the per-layer record.

        The record is stored under `sparse_meta_data["kvpress"]` and read by the
        adapter (position ids, $\log N$ bias) and by the position-aware maskers
        (`kvpress_pos_offset`).
        """
        n_new: Dict[int, int] = {}
        logcount: Dict[int, torch.Tensor] = {}
        if hasattr(cache, "layers"):                      # transformers >= 4.56
            layers = list(enumerate(cache.layers))
            for i, layer in layers:
                k = getattr(layer, "keys", None)
                if k is None or k.numel() == 0:
                    continue
                k2, v2, c = self.compress(k, layer.values)
                layer.keys, layer.values = k2, v2
                n_new[i] = int(k2.shape[2])
                if self.logcount:
                    logcount[i] = c[..., 0].to(torch.float32).clamp(min=1.0).log()
        else:                                             # legacy key_cache / value_cache lists
            for i in range(len(cache.key_cache)):
                k = cache.key_cache[i]
                if k is None or k.numel() == 0:
                    continue
                k2, v2, c = self.compress(k, cache.value_cache[i])
                cache.key_cache[i], cache.value_cache[i] = k2, v2
                n_new[i] = int(k2.shape[2])
                if self.logcount:
                    logcount[i] = c[..., 0].to(torch.float32).clamp(min=1.0).log()
        return {"press": "centroidkv", "tag": self.tag(), "n_orig": int(n_orig),
                "n_new": n_new, "logcount": logcount if self.logcount else None}


# --------------------------------------------------------------------- helpers --
def kvpress_pos_offset(sparse_meta_data: Optional[Dict[Any, Any]], layer_idx: Any) -> int:
    """How far the true decode position is ahead of the cache length in this layer.

    Position-aware stages (the rotated calibration metric of the static selector and
    of the query-geometry clustering) take the decode position from the number of
    cached keys.  After a press the cache is shorter than the context it encodes, so
    the position of the current token is `n_keys + (n_orig - n_new)`.  Zero when no
    press ran.
    """
    if not sparse_meta_data:
        return 0
    kp = sparse_meta_data.get("kvpress")
    if not kp:
        return 0
    n_new = kp["n_new"].get(int(layer_idx or 0))
    if n_new is None:
        return 0
    return int(kp["n_orig"]) - int(n_new)


def kvpress_logcount_bias(
    sparse_meta_data: Optional[Dict[Any, Any]], layer_idx: Any, n_keys: int, n_q_heads: int,
) -> Optional[torch.Tensor]:
    r"""`(b, H, 1, n_keys)` additive bias $\log N$ for this layer, or None.

    Entries appended after the press (question, generation) have count one.  The
    per-KV-head counts are repeated over the query heads of each group.
    """
    if not sparse_meta_data:
        return None
    kp = sparse_meta_data.get("kvpress")
    if not kp or not kp.get("logcount"):
        return None
    lc = kp["logcount"].get(int(layer_idx or 0))
    if lc is None:
        return None
    b, hkv, n_new = lc.shape
    if n_keys < n_new:
        lc = lc[..., :n_keys]
    elif n_keys > n_new:
        lc = torch.cat([lc, torch.zeros((b, hkv, n_keys - n_new), dtype=lc.dtype, device=lc.device)], -1)
    ng = max(1, n_q_heads // hkv)
    return lc.unsqueeze(2).expand(b, hkv, ng, n_keys).reshape(b, n_q_heads, 1, n_keys)
