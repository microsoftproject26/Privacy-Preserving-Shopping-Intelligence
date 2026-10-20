"""v4base.guard — the resource rule of `contracts/PREG1_AMENDMENT_02.md` §4 ("Baselines... low priority, <= 16 GiB
RSS, cores left after the trainers; if it breaches 16 GiB or lowers a trainer's step rate by > 5%, Ops stops it and
runs it after P3"): `os.nice(10)` (best-effort; POSIX only, never fatal on a platform without it), a capped thread
count, and a hard 16 GiB RSS guard that aborts (never silently continues).

`RssGuard` reads current RSS through `v4int.cache.rss_bytes` (read-only import; not edited) so "current RSS" means
exactly what every other v4int/v4train receipt already means by it — the same `/proc/self/status` VmRSS on Linux,
`None` (unknown, no server-only assumption) on a platform without `/proc` (this dev host is Windows). `None` is
never treated as a pass or a fail: a guard that cannot measure RSS cannot enforce the limit and says so plainly in
the receipt, it does not pretend to.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

GiB = 1024 ** 3
RSS_LIMIT_BYTES = 16 * GiB
MAX_THREADS = 4
NICE_INCREMENT = 10


class ResourceGuardError(RuntimeError):
    """The 16 GiB RSS guard tripped: the caller must abort, not continue."""


def _default_rss_fn() -> Optional[int]:
    from v4int.cache import rss_bytes           # read-only import of v4int's own measurement (not edited)
    return rss_bytes()


def configure_low_priority(n_threads: int = MAX_THREADS) -> dict:
    """`os.nice(10)` (best-effort; caught and recorded, never raised) + `torch.set_num_threads(<= 4)`. Returns what
    was actually applied, for the run receipt."""
    n = max(1, min(int(n_threads), MAX_THREADS))
    torch.set_num_threads(n)
    has_nice = hasattr(os, "nice")
    nice_result: object = None
    if has_nice:
        try:
            nice_result = os.nice(NICE_INCREMENT)
        except OSError as e:
            nice_result = f"OSError: {e} (best-effort, not fatal)"
    return {"torch_num_threads": n, "os_nice_available": has_nice, "os_nice_result": nice_result,
            "nice_increment_requested": NICE_INCREMENT}


@dataclass
class RssGuard:
    """Call `.check()` periodically from every fit / scoring loop (TRAIN chunk, eval batch). Raises
    `ResourceGuardError` the first time RSS is at or over `limit_bytes`; the caller must let that propagate (no
    partial baseline output is written after an abort). `rss_fn` is injectable so tests can drive the guard
    deterministically without needing real 16 GiB memory pressure or a Linux host."""
    limit_bytes: int = RSS_LIMIT_BYTES
    rss_fn: Callable[[], Optional[int]] = field(default=_default_rss_fn)
    peak_seen_bytes: Optional[int] = field(default=None, init=False)
    n_checks: int = field(default=0, init=False)
    rss_measurable: Optional[bool] = field(default=None, init=False)

    def check(self) -> Optional[int]:
        self.n_checks += 1
        rss = self.rss_fn()
        self.rss_measurable = rss is not None if self.rss_measurable is None else (self.rss_measurable or rss is not None)
        if rss is None:
            return None
        if self.peak_seen_bytes is None or rss > self.peak_seen_bytes:
            self.peak_seen_bytes = rss
        if rss >= self.limit_bytes:
            raise ResourceGuardError(f"RSS {rss} bytes >= the 16 GiB guard ({self.limit_bytes} bytes): aborting "
                                     "per the Amendment 02 §4 resource rule")
        return rss

    def report(self) -> dict:
        return {"limit_bytes": self.limit_bytes, "peak_seen_bytes": self.peak_seen_bytes, "n_checks": self.n_checks,
                "rss_measurable_on_this_host": bool(self.rss_measurable)}


__all__ = ["GiB", "RSS_LIMIT_BYTES", "MAX_THREADS", "NICE_INCREMENT", "ResourceGuardError", "configure_low_priority",
           "RssGuard"]
