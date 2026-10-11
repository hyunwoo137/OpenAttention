"""V32: RULER at context lengths without a hub dataset (64K, 96K), generated locally with the
NVIDIA RULER scripts and the same conversion as benchmark/ruler32k/create_huggingface_dataset.py.
Parquet files live in $RULER_LOCAL_DIR/<ctx_len>/<task>.parquet (default: data/ruler_local)."""
import os
from pathlib import Path
from typing import List

import pandas as pd

from ..ruler32k.ruler32k import Ruler32K

DEFAULT_DIR = Path(os.environ.get("RULER_LOCAL_DIR", Path(__file__).resolve().parents[2] / "data" / "ruler_local"))


def make_ruler_local(ctx_len: int, subsets_to_run: List[str], data_dir: Path = DEFAULT_DIR):
    class RulerLocal(Ruler32K):
        benchmark_name = f"ruler{ctx_len // 1024}k"
        huggingface_dataset_id = f"local:{data_dir}/{ctx_len}"

        def _load_datasets(self) -> pd.DataFrame:
            dfs = []
            for subset in self.subsets_to_run:
                f = Path(data_dir) / str(ctx_len) / f"{subset}.parquet"
                if not f.exists():
                    raise FileNotFoundError(f"RULER-{ctx_len} subset {subset} not generated: {f}")
                d = pd.read_parquet(f)
                d["context_length"] = ctx_len
                print(f"  ✓ Loaded {len(d)} samples from {subset} || context_length = {ctx_len}")
                dfs.append(d)
            return pd.concat(dfs, ignore_index=True)

    return RulerLocal(subsets_to_run=subsets_to_run)
