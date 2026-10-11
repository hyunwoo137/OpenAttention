"""Shared RULER-32K data path for the official-code baselines.

Reproduces the sparse-attention-hub protocol *exactly* so that the official
MagicPIG / HashAttention implementations see the same token sequences as every
hub accuracy runs (dense 87.04, HashAttention, vAttention, OpenAttention):

  * dataset  : xAlg-AI/att-hub-ruler-32k, 7 subsets, all 200 rows each
  * prompt   : hub `_preprocess_context_and_questions` -- Llama-3.1 chat template
               applied to  context+<sep>, split at <sep>;  context part and
               question part (question + assistant header + answer_prefix) are
               tokenised SEPARATELY with add_special_tokens=True (so both carry a
               leading BOS, as in the hub), context truncated to 32000 tokens.
  * decode   : greedy, max_new_tokens = min(120, row.max_new_tokens), stop on any
               generation_config.eos_token_id.
  * metric   : benchmark/ruler32k/calculate_metrics.py (string_match part/all).
"""
import json
import os
import re
from typing import Dict, Iterator, List, Optional

SUBSETS = ["qa_1", "qa_2", "vt", "fwe", "niah_multikey_2", "niah_multikey_3", "niah_multivalue"]
DATASET = "xAlg-AI/att-hub-ruler-32k"
MAX_CONTEXT_TOKENS = 32000
MAX_NEW_TOKENS_CAP = 120
SEP = "<<<SPARSE_HUB_SEP_7f3a9c>>>"



def load_rows(subsets: List[str], max_requests: int = 200) -> List[Dict]:
    from datasets import load_dataset
    rows = []
    for s in subsets:
        d = load_dataset(DATASET, s, split=s)
        for i, r in enumerate(d):
            if i >= max_requests:
                break
            rows.append({
                "task": s, "index": i,
                "context": r["context"], "question": r["question"],
                "answer_prefix": r["answer_prefix"],
                "answer": list(r["answer"]),
                "max_new_tokens": min(MAX_NEW_TOKENS_CAP, int(r["max_new_tokens"])),
            })
    return rows


def build_prompt_ids(tokenizer, row: Dict):
    """Return (context_ids, question_ids) exactly as the hub builds them."""
    tmpl = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["context"] + SEP}],
        tokenize=False, add_generation_prompt=True)
    ctx_str, tail = tmpl.split(SEP)
    q_str = row["question"] + tail + row["answer_prefix"]
    ctx_ids = tokenizer.encode(ctx_str)[:MAX_CONTEXT_TOKENS]
    q_ids = tokenizer.encode(q_str)
    return ctx_ids, q_ids


def eos_ids(model) -> List[int]:
    e = model.generation_config.eos_token_id
    if e is None:
        e = model.config.eos_token_id
    return list(e) if isinstance(e, (list, tuple)) else [e]


class PredWriter:
    """Append-only jsonl with resume: keyed by (task, index)."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.done = set()
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        o = json.loads(line)
                        self.done.add((o["task"], o["index"]))
                    except Exception:
                        pass
        self.f = open(path, "a", buffering=1)

    def is_done(self, row) -> bool:
        return (row["task"], row["index"]) in self.done

    def write(self, row, pred: str, extra: Optional[Dict] = None):
        o = {"task": row["task"], "index": row["index"], "answer": row["answer"],
             "predicted_answer": pred, "max_new_tokens": row["max_new_tokens"]}
        if extra:
            o.update(extra)
        self.f.write(json.dumps(o) + "\n")
        self.done.add((row["task"], row["index"]))


# ---- metric: verbatim from sparse-attention-hub/benchmark/ruler32k/calculate_metrics.py
_np_pattern = re.compile(r"[\x00-\x1f]")


def _clean(x: str) -> str:
    return _np_pattern.sub("", x.strip()).strip()


def string_match_part(preds, refs):
    score = sum(max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref])
                for pred, ref in zip(preds, refs)) / len(preds) * 100
    return round(score, 2)


def string_match_all(preds, refs):
    score = sum(sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
                for pred, ref in zip(preds, refs)) / len(preds) * 100
    return round(score, 2)


def score_jsonl(path: str) -> Dict[str, Dict]:
    by_task: Dict[str, List] = {}
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            by_task.setdefault(o["task"], []).append(o)
    out = {}
    for task, rows in by_task.items():
        preds = [_clean(r["predicted_answer"]) for r in rows]
        refs = [r["answer"] for r in rows]
        fn = string_match_part if task.split("_")[0] == "qa" else string_match_all
        dens = [r["density"] for r in rows if "density" in r]
        out[task] = {"string_match": fn(preds, refs), "n": len(rows),
                     "density": (sum(dens) / len(dens)) if dens else None}
    return out
