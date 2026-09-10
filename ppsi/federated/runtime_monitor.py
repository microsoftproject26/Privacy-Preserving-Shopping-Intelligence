"""Small sampled process-tree monitor for the S2-PR-01/03 runtime benchmark.

This is a `psutil` sampler, not a monitoring framework. It answers one question:
while a supervised child ran, what is the largest *simultaneous* sum of resident
memory across that child and its descendants, and did the machine ever fall below
a safe amount of available RAM?

Two properties matter for honesty:

* The peak is the maximum over time of a simultaneous sum. Summing each process's
  individual maximum from different instants would overstate the peak and is never
  done here.
* Every process is identified by ``(pid, create_time)`` so a recycled PID can never
  silently attach this monitor to an unrelated process.

Resident set size can double-count pages shared between processes, so the reported
peak is a sampled upper bound on the tree's footprint, not unique physical RAM and
not an exact continuous maximum.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class MonitorLimits:
    """Conservative plan thresholds, not measured hardware capability."""

    max_tree_rss_fraction_total: float = 0.70
    min_available_ram_gib: float = 2.0
    min_available_ram_fraction_total: float = 0.10

    def available_floor_bytes(self, total_bytes: int) -> float:
        return max(
            self.min_available_ram_gib * GIB,
            self.min_available_ram_fraction_total * total_bytes,
        )


@dataclass
class MonitorSample:
    """One instant: the simultaneous RSS sum and the machine's available RAM."""

    monotonic: float
    tree_rss_bytes: int
    available_bytes: int
    process_count: int
    inaccessible_count: int


@dataclass
class MonitorReport:
    """What the sampler actually observed. Absent data is never reported as zero."""

    status: str
    peak_tree_rss_bytes: int | None
    min_available_bytes: int | None
    sample_count: int
    max_sample_gap_seconds: float | None
    poll_seconds: float
    total_ram_bytes: int
    inaccessible_process_events: int
    samples_with_inaccessible_processes: int
    guard_triggered: bool
    guard_reason: str | None
    discovery_inaccessible_events: int = 0
    observed_pids: list[int] = field(default_factory=list)

    @property
    def measurement_complete(self) -> bool:
        """A sampled profile is only fully verified when nothing was unreadable."""
        return (
            self.status == "COMPLETE"
            and self.sample_count > 0
            and self.samples_with_inaccessible_processes == 0
            # A root or descendant we could never read is missing from every sum, so
            # the profile is incomplete even if no individual sample recorded an error.
            and self.discovery_inaccessible_events == 0
        )

    def to_public_dict(self) -> dict[str, Any]:
        """Public-safe summary: counts and byte totals only, never process names."""
        return {
            "schema": "runtime_process_tree_measurement_v1",
            "version": "1",
            "status": self.status,
            "measurement_complete": self.measurement_complete,
            "peak_tree_rss_bytes": self.peak_tree_rss_bytes,
            "min_available_ram_bytes": self.min_available_bytes,
            "total_ram_bytes": self.total_ram_bytes,
            "sample_count": self.sample_count,
            "poll_interval_seconds": self.poll_seconds,
            "max_sample_gap_seconds": self.max_sample_gap_seconds,
            "observed_process_count": len(self.observed_pids),
            "inaccessible_process_events": self.inaccessible_process_events,
            "samples_with_inaccessible_processes": self.samples_with_inaccessible_processes,
            "discovery_inaccessible_events": self.discovery_inaccessible_events,
            "guard_triggered": self.guard_triggered,
            "guard_reason": self.guard_reason,
            "peak_definition": (
                "maximum over time of the simultaneous sum of root and descendant RSS"
            ),
            "limitations": [
                "Sampled at a fixed interval; a shorter spike between samples is not captured.",
                "RSS can double-count shared pages, so this is not unique physical RAM.",
            ],
        }


class ProcessTreeMonitor:
    """Sample a child's process tree from the parent until it is stopped."""

    def __init__(
        self,
        root_pid: int,
        *,
        poll_seconds: float = 0.1,
        limits: MonitorLimits | None = None,
    ) -> None:
        self.root_pid = int(root_pid)
        self.poll_seconds = float(poll_seconds)
        self.limits = limits or MonitorLimits()
        self._known: dict[int, float] = {}
        self._samples: list[MonitorSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._inaccessible_events = 0
        self._discovery_inaccessible = 0
        self._normal_exits_during_discovery = 0
        self._samples_with_inaccessible = 0
        self._guard_reason: str | None = None
        self._total_ram = psutil.virtual_memory().total
        self._started = False

    # -- process discovery -------------------------------------------------

    def _register(self, proc: psutil.Process) -> None:
        """Remember ``(pid, create_time)`` so a recycled PID cannot be confused.

        A process that exits while being registered is a normal exit, not an
        unreadable process, and must not mark the measurement incomplete.
        """
        try:
            self._known.setdefault(proc.pid, proc.create_time())
        except psutil.NoSuchProcess:
            self._normal_exits_during_discovery += 1
        except psutil.AccessDenied:
            self._inaccessible_events += 1
            self._discovery_inaccessible += 1

    def _discover(self) -> None:
        try:
            root = psutil.Process(self.root_pid)
        except psutil.NoSuchProcess:
            return
        except psutil.AccessDenied:
            self._inaccessible_events += 1
            self._discovery_inaccessible += 1
            return
        self._register(root)
        try:
            for child in root.children(recursive=True):
                self._register(child)
        except psutil.NoSuchProcess:
            self._normal_exits_during_discovery += 1
        except psutil.AccessDenied:
            self._inaccessible_events += 1
            self._discovery_inaccessible += 1

    def _live_processes(self) -> tuple[list[psutil.Process], int]:
        """Return currently readable known processes and this instant's error count."""
        live: list[psutil.Process] = []
        inaccessible = 0
        for pid, create_time in list(self._known.items()):
            try:
                proc = psutil.Process(pid)
                if proc.create_time() != create_time:
                    # PID reuse: a different process now owns this id.
                    continue
                live.append(proc)
            except psutil.NoSuchProcess:
                continue  # Normal exit is not an incomplete measurement.
            except psutil.AccessDenied:
                inaccessible += 1
        return live, inaccessible

    # -- sampling ----------------------------------------------------------

    def _sample_once(self) -> MonitorSample | None:
        self._discover()
        live, inaccessible = self._live_processes()
        total_rss = 0
        counted = 0
        for proc in live:
            try:
                total_rss += proc.memory_info().rss
                counted += 1
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                inaccessible += 1
        if inaccessible:
            self._inaccessible_events += inaccessible
            self._samples_with_inaccessible += 1
        if counted == 0 and not live:
            return None
        return MonitorSample(
            monotonic=time.monotonic(),
            tree_rss_bytes=total_rss,
            available_bytes=psutil.virtual_memory().available,
            process_count=counted,
            inaccessible_count=inaccessible,
        )

    def _check_guard(self, sample: MonitorSample) -> None:
        if self._guard_reason is not None:
            return
        rss_ceiling = self.limits.max_tree_rss_fraction_total * self._total_ram
        if sample.tree_rss_bytes > rss_ceiling:
            self._guard_reason = (
                f"sampled tree RSS {sample.tree_rss_bytes} exceeded "
                f"{self.limits.max_tree_rss_fraction_total:.0%} of total RAM"
            )
            return
        floor = self.limits.available_floor_bytes(self._total_ram)
        if sample.available_bytes < floor:
            self._guard_reason = (
                f"system available RAM {sample.available_bytes} fell below the "
                f"{int(floor)} byte safety floor"
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._sample_once()
            if sample is not None:
                self._samples.append(sample)
                self._check_guard(sample)
            self._stop.wait(self.poll_seconds)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._started:
            raise RuntimeError("monitor already started")
        self._started = True
        self._discover()
        self._thread = threading.Thread(target=self._run, name="runtime-tree-monitor", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> MonitorReport:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return self.report()

    @property
    def guard_reason(self) -> str | None:
        return self._guard_reason

    def report(self) -> MonitorReport:
        if not self._samples:
            return MonitorReport(
                status="NO_SAMPLES",
                peak_tree_rss_bytes=None,
                min_available_bytes=None,
                sample_count=0,
                max_sample_gap_seconds=None,
                poll_seconds=self.poll_seconds,
                total_ram_bytes=self._total_ram,
                inaccessible_process_events=self._inaccessible_events,
                samples_with_inaccessible_processes=self._samples_with_inaccessible,
                guard_triggered=self._guard_reason is not None,
                guard_reason=self._guard_reason,
                observed_pids=sorted(self._known),
                discovery_inaccessible_events=self._discovery_inaccessible,
            )
        gaps = [
            b.monotonic - a.monotonic
            for a, b in zip(self._samples, self._samples[1:], strict=False)
        ]
        return MonitorReport(
            status="COMPLETE",
            peak_tree_rss_bytes=max(s.tree_rss_bytes for s in self._samples),
            min_available_bytes=min(s.available_bytes for s in self._samples),
            sample_count=len(self._samples),
            max_sample_gap_seconds=max(gaps) if gaps else None,
            poll_seconds=self.poll_seconds,
            total_ram_bytes=self._total_ram,
            inaccessible_process_events=self._inaccessible_events,
            samples_with_inaccessible_processes=self._samples_with_inaccessible,
            guard_triggered=self._guard_reason is not None,
            guard_reason=self._guard_reason,
            observed_pids=sorted(self._known),
            discovery_inaccessible_events=self._discovery_inaccessible,
        )

    # -- cleanup -----------------------------------------------------------

    def terminate_owned_tree(self, *, grace_seconds: float = 5.0) -> dict[str, Any]:
        """Stop only processes this monitor verified it owns.

        Never kills by image name and never touches a PID whose creation time no
        longer matches the one recorded when it was discovered.
        """
        owned: list[psutil.Process] = []
        for pid, create_time in sorted(self._known.items()):
            try:
                proc = psutil.Process(pid)
                if proc.create_time() == create_time:
                    owned.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        for proc in owned:
            try:
                proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _, alive = psutil.wait_procs(owned, timeout=grace_seconds)
        killed = 0
        for proc in alive:
            try:
                proc.kill()
                killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _, still_alive = psutil.wait_procs(alive, timeout=grace_seconds)
        return {
            "owned_process_count": len(owned),
            "force_killed_count": killed,
            "surviving_owned_count": len(still_alive),
            "cleanup_complete": not still_alive,
        }


def summarize_samples(values: list[float]) -> dict[str, Any]:
    """Descriptive statistics for a small sample of round times.

    ``p95`` uses the nearest-rank definition ``sorted[ceil(0.95 * n) - 1]``. With
    the handful of warm rounds this benchmark produces, a p95 is descriptive and
    is not a confident production tail bound.
    """
    import math
    import statistics

    if not values:
        return {"n": 0, "p50": None, "p95": None, "min": None, "max": None, "mean": None}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "n": n,
        "p50": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * n) - 1],
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }
