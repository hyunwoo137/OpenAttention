"""UTA (Unified Tail Aggregation) attention mechanism.

UTA extends sparse attention by recovering information from "tail" tokens
(those not selected by top-k) via mean-pooled proxy aggregation with
log-count compensation. This provides a mathematically principled way to
approximate full attention while maintaining sparsity.

Algorithm:
    1. Apply masker pipeline (Sink + Local + TopK) to get sparse_attention_mask
    2. For each query row, identify "tail" positions (mask == 0)
    3. Mean-pool tail keys and values: K_mean, V_mean
    4. Compute proxy logit: z_proxy = Q · K_mean * scale + log(N_tail)
    5. Merge sparse output with proxy via numerator/denominator aggregation
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from ..base import SparseAttentionConfig
from ..research_attention.base import (
    ResearchAttention,
    ResearchAttentionConfig,
)
from ..research_attention.maskers.base import ResearchMasker
from ..research_attention.maskers.sampling.base import SamplingMasker
from ..utils.kv_utils import _get_num_key_value_groups, repeat_kv
from ..utils.mask import Mask

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from sparse_attention_hub.metric_logging.stage_timer import stage


@dataclass
class UTAAttentionConfig(ResearchAttentionConfig):
    """Configuration for UTA attention mechanism.

    Inherits masker_configs from ResearchAttentionConfig.
    The masker pipeline should typically be: Sink + Local + TopK.
    UTA handles tail aggregation automatically.
    """

    pass


class UTAAttention(ResearchAttention):
    """UTA attention mechanism with tail token aggregation.

    Extends ResearchAttention by adding a proxy for tokens not selected
    by the sparse masker pipeline. After the maskers produce a sparse mask,
    UTA computes a mean-pooled proxy for the unmasked ("tail") tokens and
    merges it with the sparse attention output using log-sum-exp weighting.

    This is mathematically equivalent to:
        output = softmax(QK^T)[sparse] @ V[sparse]  +  proxy_weight * V_mean[tail]

    where the proxy weight is calibrated via +log(N_tail) compensation.
    """

    def __init__(
        self,
        sparse_attention_config: SparseAttentionConfig,
        maskers: List[ResearchMasker],
    ) -> None:
        """Initialize UTA attention mechanism.

        Args:
            sparse_attention_config: Configuration for the sparse attention mechanism.
            maskers: List of research maskers to apply (Sink + Local + TopK).
        """
        super().__init__(sparse_attention_config, maskers)

    def _compute_uta_proxy(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        scaling: float,
        sparse_attention_mask: Mask,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute UTA proxy for tail (unmasked) tokens.

        For each query row, mean-pools K and V over positions where
        the sparse mask is 0, then computes a compensated proxy logit.

        Args:
            queries: (B, H, Q, D) query tensor
            keys: (B, H_kv, K, D) key tensor (before GQA repeat)
            values: (B, H_kv, K, D) value tensor (before GQA repeat)
            scaling: attention scaling factor (1/sqrt(d))
            sparse_attention_mask: mask from masker pipeline

        Returns:
            proxy_v_mean: (B, H, Q, D) mean-pooled V for tail tokens
            proxy_logit: (B, H, Q, 1) compensated logit = Q·K_mean*scale + log(N)
            tail_count: (B, H, Q, 1) number of tail tokens per query row
        """
        # GQA: repeat keys/values to match query heads
        with stage("uta/T1_tail_mask"):
            ngroups = _get_num_key_value_groups(queries, keys)
            key_states = repeat_kv(keys, ngroups)
            value_states = repeat_kv(values, ngroups)

            # Get the dense mask: 1 = active (sparse), 0 = tail
            dense_mask = sparse_attention_mask.get_dense_mask()  # (B, H, Q, K)

            # Tail mask: 1 where token is NOT selected by sparse maskers
            tail_mask = (dense_mask == 0).float()  # (B, H, Q, K)

            # Count tail tokens per query row
            tail_count = tail_mask.sum(dim=-1, keepdim=True)  # (B, H, Q, 1)

        # Mean-pool keys over tail positions
        # tail_mask: (B, H, Q, K), key_states: (B, H, K, D)
        # We need to compute per-query-row weighted mean of keys
        # k_weighted_sum: (B, H, Q, D)
        # Both operands must be float32 for matmul compatibility
        with stage("uta/T3_tail_kv_mean"):
            k_weighted_sum = torch.matmul(
                tail_mask,                       # (B, H, Q, K) — already float32
                key_states.to(torch.float32),    # (B, H, K, D)
            )
            k_mean = k_weighted_sum / tail_count.clamp(min=1)  # (B, H, Q, D)

            # Mean-pool values over tail positions
            v_weighted_sum = torch.matmul(
                tail_mask,                         # (B, H, Q, K) — already float32
                value_states.to(torch.float32),    # (B, H, K, D)
            )
            v_mean = v_weighted_sum / tail_count.clamp(min=1)  # (B, H, Q, D)

        # Compute proxy logit: Q · K_mean * scale + log(N_tail)
        # q: (B, H, Q, D), k_mean: (B, H, Q, D) → element-wise dot then sum
        with stage("uta/T5_proxy_logit"):
            q_float = queries.to(torch.float32)
            proxy_logit = (q_float * k_mean).sum(dim=-1, keepdim=True) * scaling  # (B, H, Q, 1)

            # Add log(N) compensation
            log_n = torch.log(tail_count.clamp(min=1).to(torch.float32))  # (B, H, Q, 1)
            proxy_logit = proxy_logit + log_n

        return v_mean, proxy_logit, tail_count

    def _merge_sparse_and_proxy(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        sparse_attention_mask: Mask,
        module: nn.Module,
        proxy_v_mean: torch.Tensor,
        proxy_logit: torch.Tensor,
        tail_count: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Merge sparse attention output with UTA proxy at numerator/denominator level.

        Computes:
            exp_w = exp(QK^T * scale - max) ⊙ sparse_mask
            num = exp_w @ V + exp(proxy_logit - max) * V_mean
            den = sum(exp_w) + exp(proxy_logit - max)
            output = num / den

        Args:
            queries, keys, values: attention tensors
            attention_mask: causal/padding mask
            scaling: 1/sqrt(d)
            dropout: dropout probability
            sparse_attention_mask: mask from masker pipeline
            module: attention module (for training flag)
            proxy_v_mean: (B, H, Q, D) mean-pooled V for tail
            proxy_logit: (B, H, Q, 1) compensated proxy logit
            tail_count: (B, H, Q, 1) count of tail tokens

        Returns:
            attention_output: (B, Q, H, D) transposed output
            attention_weights: None (not computed for efficiency)
        """
        # Compute masked exp attention weights (sparse part)
        with stage("uta/T2_scores_fullQK"):
            ngroups = _get_num_key_value_groups(queries, keys)
            key_states = repeat_kv(keys, ngroups)
            value_states = repeat_kv(values, ngroups)

            q = queries.to(torch.float32)
            k = key_states.to(torch.float32)
            raw_scores = torch.matmul(q, k.transpose(2, 3)) * scaling  # (B, H, Q, K)

            if attention_mask is not None:
                raw_scores = raw_scores + attention_mask[
                    :, :, :, : key_states.shape[-2]
                ].to(torch.float32)

        with stage("uta/T6_merge_softmax"):
            # Row-wise max for numerical stability (include proxy logit in max computation)
            sparse_max = torch.max(raw_scores, dim=-1, keepdim=True)[0]  # (B, H, Q, 1)
            row_max = torch.maximum(sparse_max, proxy_logit.to(torch.float32))

            # Sparse part: exp weights
            exp_scores = torch.exp(raw_scores - row_max)

            # Apply sparse mask (zero out tail positions in exp_scores)
            exp_scores = sparse_attention_mask.apply_mask_dense(exp_scores)

            # Apply dropout if needed
            training = module.training if module is not None else False
            if dropout > 0.0 and training:
                exp_scores = torch.nn.functional.dropout(
                    exp_scores, p=dropout, training=training
                )

            # Sparse numerator and denominator
            v = value_states.to(torch.float32)
            num_sparse = torch.matmul(exp_scores, v)  # (B, H, Q, D)
            den_sparse = exp_scores.sum(dim=-1, keepdim=True)  # (B, H, Q, 1)

            # Proxy part: exp weight
            proxy_exp = torch.exp(proxy_logit.to(torch.float32) - row_max)  # (B, H, Q, 1)

            # Zero out proxy for rows with no tail tokens
            proxy_exp = proxy_exp * (tail_count > 0).float()

            # Proxy numerator: proxy_exp * V_mean
            num_proxy = proxy_exp * proxy_v_mean.to(torch.float32)  # (B, H, Q, D)

            # Merge
            num = num_sparse + num_proxy
            den = den_sparse + proxy_exp

            # Compute output
            attention_output = (num / den.clamp(min=1e-12)).to(queries.dtype)
            attention_output = attention_output.transpose(1, 2).contiguous()  # (B, Q, H, D)

        return attention_output, None

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
        """Compute UTA attention with sparse masking + tail aggregation.

        Overrides ResearchAttention.custom_attention() to add UTA proxy merge
        after the masker pipeline produces the sparse mask.

        Args:
            module: The attention module
            queries: (B, H, Q, D) query tensor
            keys: (B, H_kv, K, D) key tensor
            values: (B, H_kv, K, D) value tensor
            attention_mask: Optional causal/padding mask
            scaling: attention scaling factor
            dropout: dropout probability
            **kwargs: must include sparse_meta_data

        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Extract sparse_meta_data from kwargs
        if "sparse_meta_data" not in kwargs:
            raise ValueError(
                "sparse_meta_data must be provided in kwargs while calling custom_attention()"
            )
        sparse_meta_data: Dict[Any, Any] = kwargs.pop("sparse_meta_data")

        # Step 1: Build sparse mask via masker pipeline (Sink + Local + TopK)
        mask_shape = (
            queries.shape[0],
            queries.shape[1],
            queries.shape[2],
            keys.shape[2],
        )
        sparse_attention_mask = Mask.create_empty_mask(
            mask_shape, dtype=queries.dtype, device=queries.device
        )

        for masker in self.maskers:
            with stage(f"sel/{type(masker).__name__}"):
                sparse_attention_mask = masker.add_mask(
                    keys=keys,
                    queries=queries,
                    values=values,
                    attention_mask=attention_mask,
                    scaling=scaling,
                    dropout=dropout,
                    sparse_meta_data=sparse_meta_data,
                    previous_mask=sparse_attention_mask,
                    **kwargs,
                )

        # Log density if enabled
        if MicroMetricLogger().is_metric_enabled("research_attention_density"):
            MicroMetricLogger().log(
                "research_attention_density",
                sparse_attention_mask.get_density(),
                metadata={"layer_idx": kwargs.get("layer_idx")},
            )

        # If mask is full (sequence too short), fall back to dense attention
        if sparse_attention_mask.is_full_mask():
            from ..utils.mask_attention_utils import get_true_attention_output
            return get_true_attention_output(
                module, queries, keys, values, attention_mask, scaling, dropout, **kwargs
            )

        # Step 2: Compute UTA proxy for tail tokens
        proxy_v_mean, proxy_logit, tail_count = self._compute_uta_proxy(
            queries, keys, values, scaling, sparse_attention_mask
        )

        # Step 3: Merge sparse attention with proxy
        attention_output, attention_weights = self._merge_sparse_and_proxy(
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            scaling=scaling,
            dropout=dropout,
            sparse_attention_mask=sparse_attention_mask,
            module=module,
            proxy_v_mean=proxy_v_mean,
            proxy_logit=proxy_logit,
            tail_count=tail_count,
        )

        # Log output error if enabled
        if MicroMetricLogger().is_metric_enabled("research_attention_output_error"):
            from ..utils.mask_attention_utils import get_true_attention_output
            true_output, _ = get_true_attention_output(
                module, queries, keys, values, attention_mask, scaling, dropout, **kwargs
            )
            error = torch.norm(true_output - attention_output) / torch.norm(true_output)
            MicroMetricLogger().log(
                "research_attention_output_error",
                float(error.item()),
                metadata={"layer_idx": kwargs.get("layer_idx")},
            )

        return attention_output, attention_weights

    @classmethod
    def create_from_config(cls, config: SparseAttentionConfig) -> "UTAAttention":
        """Create UTAAttention instance from configuration.

        Args:
            config: UTAAttentionConfig with masker_configs.

        Returns:
            Instance of UTAAttention.

        Raises:
            TypeError: If config is not a UTAAttentionConfig.
        """
        if not isinstance(config, UTAAttentionConfig):
            raise TypeError(f"Expected UTAAttentionConfig, got {type(config)}")

        maskers: List[ResearchMasker] = []
        for masker_config in config.masker_configs:
            masker = ResearchMasker.create_masker_from_config(masker_config)
            maskers.append(masker)

        return cls(config, maskers)
