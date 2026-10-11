"""Faithful re-implementations of the KV-cache compression baselines.

The reference is `pyramidkv/pyramidkv_utils.py` + `run_ruler.py` of
Zefan-Cai/KVCache-Factory, the codebase of the PyramidKV paper, which also ships the
StreamingLLM and SnapKV baselines that paper compares against.  Their eval script
fixes `kernel_size = 7`, `pooling = "maxpool"`, `window_size = 8` for
SnapKV/PyramidKV and `window_size = budget - 4` for StreamingLLM (so that the
retained prefix is exactly the 4 attention sinks of Xiao et al.).

`budget` is what they call `max_capacity_prompt`: an ABSOLUTE average cache size per
layer, INCLUSIVE of the retained window.

One function serves both call sites in this repo -- the stand-alone masker
(`SnapKVMasker`) and the residency mask that composes a compressor with BinAttention
(`UTAMultiBinAttention._residency_bias`).  Keeping a single implementation is
deliberate: when the two were written separately they silently disagreed (the
composition route biased the selection call itself, which encodes the question chunk
against the keys it has just chosen -- worth -9.5 points on RULER qa_1).
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .kv_utils import _get_num_key_value_groups, repeat_kv

MODES = ("snapkv", "pyramidkv", "streamingllm")


def default_window(mode: str, budget: int, alpha: int = 8) -> int:
    """The observation/retained window each method uses at this budget.

    KVCache-Factory `run_ruler.py`: `window_sizes = 8` for snapkv/pyramidkv,
    `max_capacity_prompts - 4` for streamingllm.
    """
    if mode == "streamingllm":
        return max(1, int(budget) - 4)
    return int(alpha)


def layer_prefix_budget(
    mode: str, budget: int, window: int, layer_idx: int, num_layers: int,
    beta: int, n_keys: int,
) -> int:
    """How many PREFIX keys this layer keeps, excluding the retained window.

    PyramidKV's arithmetic allocation, verbatim from `PyramidKVCluster.update_kv`:
    the pyramid is built over `budget - window`, the steps are integer-divided, and
    the top of the pyramid is clamped when the context is shorter than it.
    """
    base = int(budget) - int(window)
    if mode != "pyramidkv" or num_layers <= 1:
        return max(1, base)
    min_num = base // max(1, beta)
    max_num = base * 2 - min_num
    if max_num >= n_keys - window:                      # clamp, as upstream does
        max_num = n_keys - window
        min_num = base * 2 - max_num
    steps = (max_num - min_num) // (num_layers - 1)
    return max(1, max_num - layer_idx * steps)


@torch.no_grad()
def select_resident(
    queries: torch.Tensor,
    keys: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    *,
    mode: str,
    budget: int,
    window: int,
    beta: int = 20,
    kernel: int = 7,
    layer_idx: int = 0,
    num_layers: int = 32,
) -> torch.Tensor:
    """Boolean `(b, n_kv_heads, n_keys)`: the keys this layer keeps for generation.

    Mirrors `update_kv` of the corresponding cluster class: the retained window is
    the last `window` keys and is never a candidate; the remaining
    `layer_prefix_budget` slots go to the first keys (StreamingLLM's sinks) or to the
    top of the pooled observation-window attention (SnapKV / PyramidKV).
    """
    if mode not in MODES:
        raise ValueError(f"unknown KV-compression mode {mode!r}; expected one of {MODES}")

    b, hq, nq, _ = queries.shape
    hkv, n_keys = keys.shape[1], keys.shape[2]
    keep = torch.zeros((b, hkv, n_keys), dtype=torch.bool, device=keys.device)

    # upstream: `if q_len < self.max_capacity_prompt: return key_states` -- nothing
    # is evicted from a context that already fits the budget
    if n_keys <= int(budget):
        keep[:] = True
        return keep

    w = min(int(window), n_keys - 1)
    n_cand = n_keys - w                                  # candidates exclude the window
    cap = min(layer_prefix_budget(mode, budget, w, layer_idx, num_layers, beta, n_keys),
              n_cand)

    if mode == "streamingllm":
        # `indices = torch.arange(max_capacity_prompt - window_size)`: the first `cap`
        # keys, i.e. the attention sinks.  No scores are consulted.
        keep[..., :cap] = True
    else:
        ng = _get_num_key_value_groups(queries, keys)
        q_obs = queries[:, :, -min(w, nq):, :].to(torch.float32)
        krep = repeat_kv(keys, ng).to(torch.float32)
        scores = torch.matmul(q_obs, krep.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, -q_obs.shape[2]:, :n_keys].to(torch.float32)
        # softmax over the FULL key range, then sum the observation rows over the
        # candidate prefix only (upstream: `attn_weights[..., -w:, :-w].sum(dim=-2)`)
        attn = torch.softmax(scores, dim=-1)[..., :n_cand].sum(dim=2)     # (b,hq,n_cand)
        pad = kernel // 2
        attn = F.max_pool1d(attn.reshape(b * hq, 1, n_cand), kernel_size=kernel,
                            stride=1, padding=pad).reshape(b, hq, n_cand)
        attn = attn.view(b, hkv, ng, n_cand).mean(dim=2)                  # GQA: mean
        keep[..., :n_cand].scatter_(-1, attn.topk(cap, dim=-1).indices, True)

    keep[..., n_keys - w:] = True
    return keep
