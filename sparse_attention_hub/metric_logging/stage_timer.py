"""CUDA-event stage timer for per-stage latency breakdown.

Off by default.  ``stage()`` degenerates to a bare ``yield`` unless
``UTA_STAGE_TIMING=1`` is in the environment (or ``StageTimer.enable()`` is
called), so instrumented code paths keep exactly their original cost in the
accuracy runs -- no events, no ``record_function``, no dict lookups beyond one
class-attribute read.

Usage::

    from sparse_attention_hub.metric_logging.stage_timer import StageTimer, stage

    with stage("mb/jensen_var"):
        ...

    StageTimer.reset()
    run_the_thing()
    StageTimer.flush()          # one cudaSynchronize, then drain the events
    print(StageTimer.summary())

Regions may nest; each ``stage()`` owns an independent pair of events, so a
parent region's time includes its children's.  Elapsed times are read only in
``flush()``, which keeps the measured code free of mid-stream synchronisation.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple

import torch

_TRUTHY = {"1", "true", "yes", "on"}


class StageTimer:
    """Process-wide accumulator of CUDA-event-delimited stage timings."""

    _enabled: bool = os.environ.get("UTA_STAGE_TIMING", "0").lower() in _TRUTHY
    _pending: List[Tuple[str, Any, Any]] = []
    _totals: Dict[str, float] = {}
    _calls: Dict[str, int] = {}

    # ------------------------------------------------------------------ state
    @classmethod
    def enable(cls, on: bool = True) -> None:
        """Turn instrumentation on or off for the rest of the process."""
        cls._enabled = on

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def reset(cls) -> None:
        """Drop every pending event and accumulated total."""
        cls._pending.clear()
        cls._totals.clear()
        cls._calls.clear()

    # ------------------------------------------------------------- collection
    @classmethod
    def record(cls, name: str, start: Any, end: Any) -> None:
        cls._pending.append((name, start, end))

    @classmethod
    def flush(cls) -> None:
        """Synchronise once, then convert every pending event pair to millis."""
        if not cls._pending:
            return
        torch.cuda.synchronize()
        for name, start, end in cls._pending:
            ms = start.elapsed_time(end)
            cls._totals[name] = cls._totals.get(name, 0.0) + ms
            cls._calls[name] = cls._calls.get(name, 0) + 1
        cls._pending.clear()

    @classmethod
    def summary(cls) -> Dict[str, Dict[str, float]]:
        """Per-stage {total_ms, calls, mean_ms}, flushing anything outstanding."""
        cls.flush()
        return {
            name: {
                "total_ms": total,
                "calls": float(cls._calls[name]),
                "mean_ms": total / max(1, cls._calls[name]),
            }
            for name, total in cls._totals.items()
        }


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time the enclosed region on the current CUDA stream.

    A no-op when the timer is disabled, which is the default.
    """
    if not StageTimer._enabled:
        yield
        return

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.profiler.record_function(name):
        try:
            yield
        finally:
            end.record()
            StageTimer.record(name, start, end)
