"""HuggingFace adapter implementation for sparse attention."""

import random
import string
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..sparse_attention.base import SparseAttention, SparseAttentionConfig
from ..sparse_attention.research_attention.base import ResearchAttention
from .base import ModelAdapter, Request, RequestResponse
from .model_servers.huggingface import ModelServerHF
from .utils.config import ModelServerConfig

INT_MAX = 2**31 - 1


class ModelAdapterHF(ModelAdapter):
    """ModelAdapter for HuggingFace integration. Provides concrete implementations for huggingface's
    transformer library.
    """

    def __init__(
        self,
        model_name: str,
        sparse_attention_config: Optional[SparseAttentionConfig],
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        hybrid: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HuggingFace adapter.

        Args:
            model_name: Name of the HuggingFace model to use
            sparse_attention_config: Configuration for sparse attention. If None, adapter runs in dense-only mode.
            model_kwargs: Additional keyword arguments for model creation
            device: Device to run the model on TODO: support dynamic and multipledevice placement
            tokenizer_kwargs: Additional keyword arguments for tokenizer creation
        """
        super().__init__(model_name, sparse_attention_config, **kwargs)
        self._registered_attention_name: Optional[str] = None
        self._custom_attention_fn: Optional[Callable] = None
        self.model_kwargs: Dict[str, Any] = model_kwargs or {}
        self.tokenizer_kwargs: Dict[str, Any] = tokenizer_kwargs or {}

        raw_registry_path: Any = kwargs.get("model_registry_path", "")
        self.model_registry_path: str = (
            raw_registry_path if isinstance(raw_registry_path, str) else ""
        )

        raw_allow_unregistered: Any = kwargs.get("allow_unregistered_models", True)
        self.allow_unregistered_models: bool = (
            raw_allow_unregistered if isinstance(raw_allow_unregistered, bool) else True
        )

        # more useful parameters to store
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.torch_dtype = self.model_kwargs.get("torch_dtype", torch.float32)

        # Handle dense-only mode when sparse_attention_config is None
        self._sparse_attention_available: bool = sparse_attention_config is not None
        # Control token-by-token question processing (for hybrid models)
        self.hybrid = hybrid if hybrid is not None else False
        # Convert device string to GPU ID for ModelServer
        gpu_id: Optional[int] = None
        if self.device.startswith("cuda"):
            if self.device == "cuda":
                gpu_id = 0 if torch.cuda.is_available() else None
            else:
                # Extract GPU ID from "cuda:0", "cuda:1", etc.
                try:
                    gpu_id = int(self.device.split(":")[1])
                except (IndexError, ValueError):
                    gpu_id = None

        model_server = ModelServerHF(
            ModelServerConfig(
                model_registry_path=self.model_registry_path,
                allow_unregistered_models=self.allow_unregistered_models,
            )
        )
        self.model = model_server.get_model(self.model_name, gpu_id, self.model_kwargs)
        self.tokenizer = model_server.get_tokenizer(
            self.model_name, self.tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.random_separator = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=100)
        )

    def __del__(self) -> None:
        """Clean up registered attention functions when the adapter is destroyed."""
        self._cleanup_attention_registration()

    def process_request(
        self,
        request: Request,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> RequestResponse:
        """Processes request with optimized tokenization but independent question processing.
        Context is tokenized once but each question is processed independently to avoid KV cache contamination.

        Args:
            request: The request to process

        Returns:
            response: The response to the request
        """
        max_context_length: int = request_kwargs.get("max_context_length", INT_MAX)
        max_new_tokens: int = generation_kwargs.get("max_new_tokens", INT_MAX)
        print(
            " Processing request with max_context_length: ",
            max_context_length,
            " and max_new_tokens: ",
            max_new_tokens,
            flush=True,
        )

        questions: List[str] = (
            request.questions
            if isinstance(request.questions, list)
            else [request.questions]
        )
        context: str = request.context
        answer_prefix: str = request.answer_prefix

        context, questions = self._preprocess_context_and_questions(
            context, questions, answer_prefix
        )

        context_tokens = self.tokenizer.encode(context, return_tensors="pt")
        context_tokens = context_tokens[
            :, :max_context_length
        ]  # truncate context to max_context_length

        input_device = (
            self.model.device if hasattr(self.model, "device") else self.device
        )
        if input_device is not None:
            context_tokens = context_tokens.to(input_device)
        print(f"Context tokens: {context_tokens.shape}")
        responses: List[str] = []

        self.model.eval()
        with torch.no_grad():
            for question in questions:
                sparse_meta_data: Dict[str, Any] = {}

                question_tokens = self.tokenizer.encode(question, return_tensors="pt")
                if input_device is not None:
                    question_tokens = question_tokens.to(input_device)

                self._press_next_pos = None
                captured: Dict[int, torch.Tensor] = {}
                if self._sparse_attention_available:
                    with self._press_capture(captured, context_tokens):
                        context_outputs = self.model(
                            context_tokens,
                            past_key_values=None,
                            use_cache=True,
                            sparse_meta_data=sparse_meta_data,
                            **self._prefill_logits_kwargs(),
                        )
                    self._maybe_apply_kv_press(context_outputs, sparse_meta_data, context_tokens, captured)

                    with self.enable_sparse_mode():
                        response_text = self._generate_response(
                            question_tokens,
                            context_outputs,
                            sparse_meta_data,
                            generation_kwargs,
                            **kwargs,
                        )
                else:
                    with self._press_capture(captured, context_tokens):
                        context_outputs = self.model(
                            context_tokens,
                            past_key_values=None,
                            use_cache=True,
                            **self._prefill_logits_kwargs(),
                        )
                    self._maybe_apply_kv_press(context_outputs, sparse_meta_data, context_tokens, captured)

                    response_text = self._generate_response(
                        question_tokens,
                        context_outputs,
                        sparse_meta_data,
                        generation_kwargs,
                        **kwargs,
                    )

                responses.append(response_text)

        if isinstance(request.questions, str):
            return RequestResponse(responses=responses[0])
        else:
            return RequestResponse(responses=responses)

    @contextmanager
    def _press_capture(self, captured: Dict[int, torch.Tensor], context_tokens: torch.Tensor):
        """V54c: for a press that scores by the last `window` context queries (SnapKV /
        PyramidKV under the kvpress split), hook every attention layer's `q_proj` during
        the context prefill and keep those rows, RoPE'd at their positions."""
        press = getattr(self, "kv_press", None)
        if press is None or not getattr(press, "needs_queries", False) \
                or getattr(press, "when", "context") != "context":
            yield
            return
        w = int(press.window)
        T = int(context_tokens.shape[1])
        base = getattr(self.model, "model", self.model)
        layers = list(base.layers)
        pos = torch.arange(max(T - w, 0), T, device=context_tokens.device).unsqueeze(0)
        cs: Dict[str, torch.Tensor] = {}
        handles = []

        def _mk(i, attn):
            def hook(mod, inp, out):
                q = out[:, -w:, :]
                b, ww, _ = q.shape
                q = q.view(b, ww, -1, attn.head_dim).transpose(1, 2)            # (b,H,w,d)
                if "cos" not in cs:
                    cs["cos"], cs["sin"] = base.rotary_emb(q, pos)
                from ..sparse_attention.utils.evict_press import rope_queries
                captured[i] = rope_queries(q.float(), cs["cos"].float(), cs["sin"].float()).to(q.dtype)
            return hook

        for i, layer in enumerate(layers):
            handles.append(layer.self_attn.q_proj.register_forward_hook(_mk(i, layer.self_attn)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _maybe_apply_kv_press(
        self, context_outputs: Any, sparse_meta_data: Dict[str, Any], context_tokens: torch.Tensor,
        captured: Optional[Dict[int, torch.Tensor]] = None,
    ) -> None:
        """V54: run a KV-cache *press* (CentroidKV) on the prefilled context cache.

        kvpress protocol: the context is compressed before the question is seen, the
        cache is rewritten in place, and every later token is positioned as if the
        context were uncompressed (`_generate_response` starts the question at the
        original context length).  A press is set as ``adapter.kv_press`` (an object
        with ``apply_to_cache(cache, n_orig) -> record``); the record is stored under
        ``sparse_meta_data["kvpress"]`` for the attention path.  No-op without a press.
        """
        press = getattr(self, "kv_press", None)
        if press is None or getattr(press, "when", "context") != "context":
            return          # "prompt" presses run in _generate_response, after the question
        n_orig = int(context_tokens.shape[1])
        if getattr(press, "needs_queries", False):
            base = getattr(self.model, "model", self.model)
            record = press.apply_to_cache(context_outputs.past_key_values, n_orig, queries=captured,
                                          scaling=float(base.layers[0].self_attn.scaling))
        else:
            record = press.apply_to_cache(context_outputs.past_key_values, n_orig)
        sparse_meta_data["kvpress"] = record
        self._press_next_pos = n_orig
        self._log_press(record, n_orig)

    def _apply_prompt_press(self, past_key_values: Any, sparse_meta_data: Dict[str, Any],
                            n_prompt: int) -> bool:
        """V54b: run a `when="prompt"` press once the whole prompt (question included) is in
        the cache.  Every per-request structure the sparse attention cached over the
        uncompressed keys is now stale, so the adapter clears the shared ones here and the
        maskers rebuild from what they kept (question moments / profile queries) at the
        next call.  Returns True when a press ran."""
        press = getattr(self, "kv_press", None)
        if press is None or getattr(press, "when", "context") != "prompt":
            return False
        record = press.apply_to_cache(past_key_values, int(n_prompt))
        sparse_meta_data["kvpress"] = record
        for key in ("static_sel", "static_scores", "hat_scores", "kvcompress", "snapkv"):
            ent = sparse_meta_data.get(key)
            if isinstance(ent, dict):
                for lid, e in list(ent.items()):
                    # static selector: keep the question moment, drop the codes (rebuilt)
                    if key == "static_sel" and isinstance(e, dict):
                        e["K"] = 10 ** 9         # forces `K < ent["K"]` -> rebuild with Mq
                    else:
                        ent.pop(lid, None)
        self._log_press(record, int(n_prompt))
        return True

    def _log_press(self, record: Dict[str, Any], n_orig: int) -> None:
        if not getattr(self, "_press_logged", False):
            n_new = sorted(set(record["n_new"].values()))
            print(f"[kvpress] {record.get('tag')}: {n_orig} tokens -> cache {n_new} entries/layer"
                  f" (log-count bias {'on' if record.get('logcount') else 'off'})", flush=True)
            self._press_logged = True

    def _prefill_logits_kwargs(self) -> Dict[str, Any]:
        """Restrict the context forward to the last position's logits.

        Only ``context_outputs.past_key_values`` is consumed (see ``_generate_response``); the logits
        over the context are discarded.  Computing them costs ``len(context) x vocab`` activations,
        which is 23 GiB at a 96K context with Llama-3.1's 128K vocabulary and made long-context runs
        OOM on a shared GPU.  Returns an empty dict when the model does not accept the argument, so
        that behaviour is unchanged on models whose forward lacks it.
        """
        if not hasattr(self, "_prefill_logits_kw"):
            import inspect

            try:
                params = inspect.signature(self.model.forward).parameters
            except (TypeError, ValueError):
                params = {}
            if "logits_to_keep" in params:
                self._prefill_logits_kw = {"logits_to_keep": 1}
            elif "num_logits_to_keep" in params:
                self._prefill_logits_kw = {"num_logits_to_keep": 1}
            else:
                self._prefill_logits_kw = {}
        return self._prefill_logits_kw

    def _preprocess_context_and_questions(
        self, context: str, questions: List[str], answer_prefix: str
    ) -> Tuple[str, List[str]]:
        """Preprocess the context and questions -- apply chat template if needed

        Args:
            context: The context to preprocess
            questions: The questions to preprocess
        """
        context = context + self.random_separator
        if self.tokenizer.chat_template is not None:
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context}],
                tokenize=False,
                add_generation_prompt=True,
            )
        new_context = context.split(self.random_separator)[0]
        new_questions = [
            question + context.split(self.random_separator)[1] + answer_prefix
            for question in questions
        ]
        return new_context, new_questions

    def get_custom_attention_function(
        self, sparse_attention: SparseAttention
    ) -> Callable:
        """Returns custom_attention_fn callable with the correct signature required for HuggingFace.

        Args:
            sparse_attention: The sparse attention instance

        Returns:
            custom_attention_fn: Callable with correct signature for HuggingFace
        """

        def custom_attention_callable(
            module: torch.nn.Module,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float = 1.0,
            dropout: float = 0.0,
            **kwargs: Dict[str, Any],
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Custom attention callable for HuggingFace integration."""
            if hasattr(module, "layer_idx"):
                layer_idx = getattr(module, "layer_idx", None)
                if layer_idx is not None:
                    kwargs["layer_idx"] = layer_idx

            if "sparse_meta_data" in kwargs:
                sparse_meta_data: Dict[Any, Any] = kwargs["sparse_meta_data"]
                kwargs.pop("sparse_meta_data", None)
            else:
                raise ValueError(
                    "sparse_meta_data must be provided while calling model.forward()"
                )

            kp = sparse_meta_data.get("kvpress") if sparse_meta_data else None
            if kp and attention_mask is not None and attention_mask.shape[-1] != keys.shape[2]:
                # V54c: a press that keeps a different number of entries per layer (PyramidKV)
                # leaves the cache shorter in some layers than the layer-0 length HF sized its
                # mask for; slicing that mask would drop the causal block of a multi-row call.
                # Batch 1, no padding: rebuild the causal mask for this layer's key length.
                nq, nk = queries.shape[2], keys.shape[2]
                ar_q = torch.arange(nq, device=keys.device).view(nq, 1)
                ar_k = torch.arange(nk, device=keys.device).view(1, nk)
                allowed = ar_k <= (nk - nq) + ar_q
                attention_mask = torch.zeros((1, 1, nq, nk), dtype=attention_mask.dtype,
                                             device=keys.device).masked_fill(
                    ~allowed, torch.finfo(attention_mask.dtype).min)
            if kp and kp.get("logcount"):
                # V54 CentroidKV, paper Eq. 3: softmax(qK^T/sqrt(d) + log N) V over the
                # centroids.  The count enters as an additive bias on the same channel
                # the causal mask uses, so every stage below (selector scores, exact
                # reads, tail) sees the same weighted logits.
                from ..sparse_attention.utils.centroidkv import kvpress_logcount_bias

                bias = kvpress_logcount_bias(
                    sparse_meta_data, kwargs.get("layer_idx"), keys.shape[2], queries.shape[1]
                )
                if bias is not None:
                    if attention_mask is None:
                        attention_mask = bias.to(queries.dtype)
                    else:
                        attention_mask = (
                            attention_mask[:, :, :, : keys.shape[2]] + bias.to(attention_mask.dtype)
                        )

            return sparse_attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                **kwargs,
            )

        return custom_attention_callable

    def _generate_unique_attention_name(self) -> str:
        """Generate a unique name not present in ALL_ATTENTION_FUNCTIONS."""
        base_name: str = "sparse_attention"
        existing_keys: List[str] = (
            ALL_ATTENTION_FUNCTIONS.valid_keys()
            + ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        )

        while True:
            suffix: str = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=8)
            )
            name: str = f"{base_name}_{suffix}"

            if name not in existing_keys:
                return name

    def _ensure_attention_registered(self) -> str:
        """Ensure custom attention function is registered and return the name.
        Caches the registration to avoid repeated registration overhead.

        Returns:
            The name of the registered attention function
        """
        if self._registered_attention_name is None:
            if not self._sparse_attention_available or self.sparse_attention is None:
                raise RuntimeError(
                    "Cannot register attention function: sparse attention is not available"
                )
            self._custom_attention_fn = self.get_custom_attention_function(
                self.sparse_attention
            )
            self._registered_attention_name = self._generate_unique_attention_name()

            from transformers.masking_utils import eager_mask

            ALL_ATTENTION_FUNCTIONS.register(
                self._registered_attention_name, self._custom_attention_fn
            )
            if isinstance(self.sparse_attention, ResearchAttention):
                ALL_MASK_ATTENTION_FUNCTIONS.register(
                    self._registered_attention_name, eager_mask
                )
            else:
                raise NotImplementedError(
                    "Sparse attention is not supported for this model yet"
                )

        return self._registered_attention_name

    def _cleanup_attention_registration(self) -> None:
        """Clean up registered attention functions."""
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name in ALL_ATTENTION_FUNCTIONS.valid_keys()
        ):
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(self._registered_attention_name)
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name
            in ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        ):
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(
                self._registered_attention_name
            )
        self._registered_attention_name = None
        self._custom_attention_fn = None

    @contextmanager
    def enable_sparse_mode(self) -> Generator[None, None, None]:
        """Context manager to temporarily enable sparse attention mode.

        Yields:
            None
        """
        # If sparse attention is not available, raise an error
        if not self._sparse_attention_available:
            raise RuntimeError(
                "Cannot enable sparse mode: sparse attention is not available"
            )

        # Store original implementations to restore later
        original_implementations: Dict[str, str] = {}

        # First, store the original implementations before registering custom attention
        for name, module in self.model.named_modules():
            if hasattr(module, "config") and hasattr(
                module.config, "_attn_implementation"
            ):
                original_implementations[name] = module.config._attn_implementation

        # Ensure custom attention function is registered (reuse if already registered)
        custom_attention_name: str = self._ensure_attention_registered()

        try:
            # Switch to sparse attention
            for name, module in self.model.named_modules():
                if hasattr(module, "config") and hasattr(
                    module.config, "_attn_implementation"
                ):
                    module.config._attn_implementation = custom_attention_name

            yield

        finally:
            # Restore original implementations
            for name, module in self.model.named_modules():
                if name in original_implementations:
                    module.config._attn_implementation = original_implementations[name]

    def _generate_response(
        self,
        question_tokens: torch.Tensor,
        context_outputs: Any,
        sparse_meta_data: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate text response.

        For hybrid models, after context prefill, feed the question tokens one-by-one so the
        linear-attention cached path uses seq_len == 1. For dense models, process all question tokens at once.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        max_new_tokens: int = generation_kwargs.get("max_new_tokens", 50)

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        # Start from cached context state
        past_key_values = context_outputs.past_key_values

        # V54: after a KV press the cache is shorter than the context it encodes;
        # position ids continue from the ORIGINAL context length (kvpress convention).
        _press_pos = [getattr(self, "_press_next_pos", None)]

        def _position_kwargs(n_tokens: int, device: torch.device) -> Dict[str, Any]:
            if _press_pos[0] is None:
                return {}
            pos = torch.arange(_press_pos[0], _press_pos[0] + n_tokens, device=device).unsqueeze(0)
            _press_pos[0] += n_tokens
            return {"position_ids": pos}

        last_outputs = None

        if self.hybrid:
            # TODO(sahil, P1): add regression coverage for hybrid models that
            # verifies question/prefix tokens are consumed one-by-one after
            # context prefill, while non-hybrid models keep the batched path.
            # ------------------------------------------------------------------
            # Step 1: consume the question/prefix ONE TOKEN AT A TIME
            # ------------------------------------------------------------------
            for idx in range(question_tokens.shape[1]):
                step_input = question_tokens[:, idx : idx + 1]

                model_kwargs = {
                    "input_ids": step_input,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "logits_to_keep": 1,
                    **_position_kwargs(step_input.shape[1], step_input.device),
                }

                # Only pass sparse_meta_data when sparse mode is actually enabled
                if self._sparse_attention_available:
                    model_kwargs["sparse_meta_data"] = sparse_meta_data

                with torch.no_grad():
                    last_outputs = self.model(**model_kwargs)

                past_key_values = last_outputs.past_key_values
        else:
            # Dense models: process all question tokens at once.  V29: HUB_QUESTION_CHUNK=<n> (env)
            # feeds the question chunk in slices of n tokens instead (same KV cache, same causal
            # attention); long multiple-choice prompts (LongBench-v2) otherwise materialise
            # (H, Q, K)-shaped fp32 buffers in the research attention path and run out of memory.
            import os as _os
            _qc = int(_os.environ.get("HUB_QUESTION_CHUNK", "0") or 0)
            _nq = question_tokens.shape[1]
            for _s0 in (range(0, _nq, _qc) if _qc > 0 else [0]):
                _inp = question_tokens[:, _s0:_s0 + _qc] if _qc > 0 else question_tokens
                model_kwargs = {
                    "input_ids": _inp,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    **_position_kwargs(_inp.shape[1], _inp.device),
                }
                if self._sparse_attention_available:
                    model_kwargs["sparse_meta_data"] = sparse_meta_data
                with torch.no_grad():
                    last_outputs = self.model(**model_kwargs)
                past_key_values = last_outputs.past_key_values

        if last_outputs is None:
            raise ValueError("Question tokens are empty; cannot generate response")

        # V54b: a "prompt" press merges the stored cache now that the whole prompt has been
        # encoded with full attention; generation continues at the original positions.
        if self._apply_prompt_press(past_key_values, sparse_meta_data,
                                    past_key_values.get_seq_length() if _press_pos[0] is None
                                    else _press_pos[0]):
            _press_pos[0] = int(sparse_meta_data["kvpress"]["n_orig"])

        generated_ids = [last_outputs.logits[0, -1].argmax()]

        # ------------------------------------------------------------------
        # Step 2: continue normal autoregressive generation, one token at a time
        # ------------------------------------------------------------------
        for _ in tqdm(range(max_new_tokens - 1), disable=(max_new_tokens < 1000)):
            model_kwargs = {
                "input_ids": generated_ids[-1].unsqueeze(0).unsqueeze(0),
                "past_key_values": past_key_values,
                "use_cache": True,
                **_position_kwargs(1, generated_ids[-1].device),
            }

            if self._sparse_attention_available:
                model_kwargs["sparse_meta_data"] = sparse_meta_data

            with torch.no_grad():
                outputs = self.model(**model_kwargs)

            past_key_values = outputs.past_key_values
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)

            if new_id.item() in should_stop_token_ids:
                break

        answer: str = self.tokenizer.decode(
            torch.stack(generated_ids), skip_special_tokens=True
        )

        return answer
