"""Advanced UTA variants: Pure Jensen Gap Correction (UTAJensen).

Adds σ²/2 = (scale² · Σ_d q_d² · Var(k_d)) / 2 to the proxy logit.
Denominator scale is corrected for Jensen's inequality bias,
while numerator value is kept as simple unweighted mean v_mean.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from ..base import SparseAttentionConfig
from ..research_attention.base import ResearchAttentionConfig
from ..research_attention.maskers.base import ResearchMasker
from ..utils.kv_utils import _get_num_key_value_groups, repeat_kv
from ..utils.mask import Mask

from .base import UTAAttention, UTAAttentionConfig

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from sparse_attention_hub.metric_logging.stage_timer import stage


@dataclass
class UTAJensenConfig(UTAAttentionConfig):
    """Config for UTA with Jensen Gap Correction (Log-Normal variance).

    Same masker pipeline as UTAAttention (Sink + Local + TopK).
    Adds alpha * σ²/2 bias correction to proxy logit, keeping value as simple v_mean.
    """
    alpha: float = 1.0


class UTAJensenAttention(UTAAttention):
    """UTA with Jensen Gap Correction on denominator scale."""

    def __init__(self, config: UTAJensenConfig, maskers: List[ResearchMasker]):
        super().__init__(config, maskers)
        self.alpha = getattr(config, "alpha", 1.0)

    def _compute_jensen_proxy(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        scaling: float,
        sparse_attention_mask: Mask,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with stage("utaj/T1_tail_mask"):
            ngroups = _get_num_key_value_groups(queries, keys)
            key_states = repeat_kv(keys, ngroups)
            value_states = repeat_kv(values, ngroups)

            dense_mask = sparse_attention_mask.get_dense_mask()
            tail_mask = (dense_mask == 0).float()
            tail_count = tail_mask.sum(dim=-1, keepdim=True)

            q_float = queries.to(torch.float32)
            k_float = key_states.to(torch.float32)
            v_float = value_states.to(torch.float32)

        # 1. Mean-pool keys and values over tail
        with stage("utaj/T3_tail_kv_mean"):
            k_weighted_sum = torch.matmul(tail_mask, k_float)
            k_mean = k_weighted_sum / tail_count.clamp(min=1)

            v_weighted_sum = torch.matmul(tail_mask, v_float)
            v_mean = v_weighted_sum / tail_count.clamp(min=1)

        # 2. Per-dimension Key variance
        with stage("utaj/T4_jensen_var"):
            with stage("utaj/T4b_var_k_sq_mean"):
                k_sq_weighted_sum = torch.matmul(tail_mask, k_float ** 2)
                k_sq_mean = k_sq_weighted_sum / tail_count.clamp(min=1)
                k_var = (k_sq_mean - k_mean ** 2).clamp(min=0)

            # 3. Logit variance: σ² = scale² · Σ_d q_d² · Var(k_d)
            with stage("utaj/T4c_var_q2_dot"):
                variance = (scaling ** 2) * (q_float ** 2 * k_var).sum(
                    dim=-1, keepdim=True
                ).clamp(min=0)

        with stage("utaj/T5_proxy_logit"):
            z_bar = (q_float * k_mean).sum(dim=-1, keepdim=True) * scaling
            log_n = torch.log(tail_count.clamp(min=1).to(torch.float32))

        # Log variance metric if enabled
        if MicroMetricLogger().is_metric_enabled("jensen_tail_variance"):
            MicroMetricLogger().log(
                "jensen_tail_variance",
                float(variance.mean().item()),
            )

        # Jensen corrected proxy logit: z̄ + alpha * (σ²/2) + log(N)
        proxy_logit = z_bar + self.alpha * (variance / 2) + log_n

        return v_mean, proxy_logit, tail_count

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
            raise ValueError(
                "sparse_meta_data must be provided in kwargs while calling custom_attention()"
            )
        sparse_meta_data: Dict[Any, Any] = kwargs.pop("sparse_meta_data")

        mask_shape = (
            queries.shape[0], queries.shape[1],
            queries.shape[2], keys.shape[2],
        )
        sparse_attention_mask = Mask.create_empty_mask(
            mask_shape, dtype=queries.dtype, device=queries.device
        )
        for masker in self.maskers:
            with stage(f"sel/{type(masker).__name__}"):
                sparse_attention_mask = masker.add_mask(
                    keys=keys, queries=queries, values=values,
                    attention_mask=attention_mask, scaling=scaling, dropout=dropout,
                    sparse_meta_data=sparse_meta_data,
                    previous_mask=sparse_attention_mask, **kwargs,
                )

        if MicroMetricLogger().is_metric_enabled("research_attention_density"):
            MicroMetricLogger().log(
                "research_attention_density",
                sparse_attention_mask.get_density(),
                metadata={"layer_idx": kwargs.get("layer_idx")},
            )

        if sparse_attention_mask.is_full_mask():
            from ..utils.mask_attention_utils import get_true_attention_output
            return get_true_attention_output(
                module, queries, keys, values, attention_mask, scaling, dropout, **kwargs
            )

        v_mean, proxy_logit, tail_count = self._compute_jensen_proxy(
            queries, keys, values, scaling, sparse_attention_mask
        )

        attention_output, attention_weights = self._merge_sparse_and_proxy(
            queries=queries, keys=keys, values=values,
            attention_mask=attention_mask, scaling=scaling, dropout=dropout,
            sparse_attention_mask=sparse_attention_mask, module=module,
            proxy_v_mean=v_mean,
            proxy_logit=proxy_logit,
            tail_count=tail_count,
        )

        # Log output error
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
    def create_from_config(cls, config: SparseAttentionConfig) -> "UTAJensenAttention":
        if not isinstance(config, UTAJensenConfig):
            raise TypeError(f"Expected UTAJensenConfig, got {type(config)}")
        maskers = [ResearchMasker.create_masker_from_config(mc) for mc in config.masker_configs]
        return cls(config, maskers)



