"""BASE-01: the resource rule (Amendment 02 §4) — os.nice(10) / torch.set_num_threads(<= 4) applied best-effort,
and the hard 16 GiB RSS guard that aborts. `RssGuard.rss_fn` is injected so the guard is tested deterministically
(no real 16 GiB memory pressure needed, and this dev host cannot measure RSS at all — Windows has no `/proc`)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from basekit import K, FixtureTrainProvider, assert_raises, nc
from v4base import pop as base_pop
from v4base.guard import GiB, RSS_LIMIT_BYTES, MAX_THREADS, ResourceGuardError, RssGuard, configure_low_priority


def test_configure_low_priority_is_best_effort_and_caps_threads():
    prior = torch.get_num_threads()
    try:
        report = configure_low_priority(n_threads=99)                  # way over the cap
        assert report["torch_num_threads"] == MAX_THREADS
        assert torch.get_num_threads() == MAX_THREADS
        # never raises even where os.nice does not exist (this dev host: Windows)
        assert "os_nice_available" in report and "os_nice_result" in report
    finally:
        torch.set_num_threads(prior)


def test_guard_passes_under_the_limit_and_tracks_the_peak():
    g = RssGuard(rss_fn=lambda: 1 * GiB)
    for _ in range(3):
        g.check()
    assert g.peak_seen_bytes == 1 * GiB
    assert g.n_checks == 3
    assert g.report()["rss_measurable_on_this_host"] is True


def test_guard_aborts_at_or_over_16_gib():
    g = RssGuard(rss_fn=lambda: RSS_LIMIT_BYTES)
    try:
        g.check()
    except ResourceGuardError:
        pass
    else:
        raise AssertionError("RssGuard did not abort at exactly the 16 GiB limit")


def test_guard_never_aborts_or_passes_on_an_unmeasurable_host():
    g = RssGuard(rss_fn=lambda: None)
    assert g.check() is None
    assert g.report()["rss_measurable_on_this_host"] is False


def test_guard_actually_aborts_a_fit_loop_when_it_trips():
    """A guard that is already over the limit before the first chunk must stop `pop.fit` before it returns a
    (partial, silently-wrong) scorer."""
    tp = FixtureTrainProvider(n=40, K=K, seed=0)
    tripped_guard = RssGuard(rss_fn=lambda: RSS_LIMIT_BYTES)
    assert_raises(ResourceGuardError, base_pop.fit, tp, K, chunk=4, guard=tripped_guard)


@nc("without the >= (only a strict >) the guard would let RSS exactly AT the 16 GiB limit through")
def test_nc_guard_would_allow_the_exact_limit_without_the_ge_comparison():
    g = RssGuard(rss_fn=lambda: RSS_LIMIT_BYTES)
    tripped = False
    try:
        g.check()
    except ResourceGuardError:
        tripped = True
    assert not tripped
