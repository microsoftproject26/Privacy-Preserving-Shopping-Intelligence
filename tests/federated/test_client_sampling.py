"""Tests for ppsi.federated.sampling — deterministic client sampling."""

from __future__ import annotations

import pytest

from ppsi.federated.sampling import (
    DEFAULT_SAMPLER_VERSION,
    build_trace,
    derive_round_seed,
    sample_clients,
    selection_digest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_POOL_SMALL = [f"client-v1-{i:064x}" for i in range(100)]
_POOL_LARGE = [f"client-v1-{i:064x}" for i in range(10_000)]


# ---------------------------------------------------------------------------
# Seed derivation tests
# ---------------------------------------------------------------------------


class TestDeriveRoundSeed:
    """Tests for the SHA-256-based seed derivation."""

    def test_deterministic(self) -> None:
        """Same inputs produce the same seed."""
        s1 = derive_round_seed("client_sampler_v1", 13, 0)
        s2 = derive_round_seed("client_sampler_v1", 13, 0)
        assert s1 == s2

    def test_different_seed_different_output(self) -> None:
        """Different experiment seeds produce different round seeds."""
        s1 = derive_round_seed("client_sampler_v1", 13, 0)
        s2 = derive_round_seed("client_sampler_v1", 42, 0)
        assert s1 != s2

    def test_different_round_different_output(self) -> None:
        """Different round indices produce different round seeds."""
        s1 = derive_round_seed("client_sampler_v1", 13, 0)
        s2 = derive_round_seed("client_sampler_v1", 13, 1)
        assert s1 != s2


# ---------------------------------------------------------------------------
# Sampling tests
# ---------------------------------------------------------------------------


class TestSampleClients:
    """Tests for the sample_clients function."""

    def test_reproducibility(self) -> None:
        """Same pool/seed/round produces identical result."""
        r1 = sample_clients(_POOL_SMALL, 13, 0, 10)
        r2 = sample_clients(_POOL_SMALL, 13, 0, 10)
        assert r1.selected_client_ids == r2.selected_client_ids
        assert r1.selected_digest == r2.selected_digest

    def test_different_seed_different_sample(self) -> None:
        """Different experiment seed changes the sample (on a large pool)."""
        r1 = sample_clients(_POOL_LARGE, 13, 0, 10)
        r2 = sample_clients(_POOL_LARGE, 42, 0, 10)
        assert r1.selected_client_ids != r2.selected_client_ids

    def test_different_round_different_sample(self) -> None:
        """Different round index changes the sample."""
        r1 = sample_clients(_POOL_LARGE, 13, 0, 10)
        r2 = sample_clients(_POOL_LARGE, 13, 1, 10)
        assert r1.selected_client_ids != r2.selected_client_ids

    def test_within_round_uniqueness(self) -> None:
        """One round contains unique client IDs (without replacement)."""
        r = sample_clients(_POOL_SMALL, 13, 0, 50)
        assert len(set(r.selected_client_ids)) == 50

    def test_oversized_request_raises(self) -> None:
        """Requesting more clients than pool size raises ValueError."""
        small = _POOL_SMALL[:5]
        with pytest.raises(ValueError, match="exceeds"):
            sample_clients(small, 13, 0, 10)

    def test_exact_pool_size_ok(self) -> None:
        """Requesting exactly pool size is allowed."""
        small = _POOL_SMALL[:10]
        r = sample_clients(small, 13, 0, 10)
        assert len(r.selected_client_ids) == 10
        assert set(r.selected_client_ids) == set(small)

    def test_zero_clients_raises(self) -> None:
        """Requesting zero clients raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            sample_clients(_POOL_SMALL, 13, 0, 0)

    def test_negative_round_raises(self) -> None:
        """Negative round index raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            sample_clients(_POOL_SMALL, 13, -1, 10)

    def test_non_list_raises(self) -> None:
        """Non-list eligible_client_ids raises TypeError."""
        with pytest.raises(TypeError):
            sample_clients(tuple(_POOL_SMALL), 13, 0, 10)  # type: ignore[arg-type]

    def test_cross_round_overlap_allowed(self) -> None:
        """The same client may appear in different rounds."""
        r0 = sample_clients(_POOL_SMALL, 13, 0, 50)
        r1 = sample_clients(_POOL_SMALL, 13, 1, 50)
        # With 100 clients and 50 per round, overlap is very likely
        overlap = set(r0.selected_client_ids) & set(r1.selected_client_ids)
        assert len(overlap) > 0, "Expected some overlap across rounds"


# ---------------------------------------------------------------------------
# Digest tests
# ---------------------------------------------------------------------------


class TestSelectionDigest:
    """Tests for deterministic digest computation."""

    def test_determinism(self) -> None:
        """Same IDs produce the same digest."""
        ids = ["client-v1-aaa", "client-v1-bbb", "client-v1-ccc"]
        d1 = selection_digest(ids)
        d2 = selection_digest(ids)
        assert d1 == d2

    def test_order_independent(self) -> None:
        """Digest is order-independent (sorts internally)."""
        ids1 = ["client-v1-bbb", "client-v1-aaa"]
        ids2 = ["client-v1-aaa", "client-v1-bbb"]
        assert selection_digest(ids1) == selection_digest(ids2)

    def test_different_ids_different_digest(self) -> None:
        """Different IDs produce different digest."""
        d1 = selection_digest(["client-v1-aaa"])
        d2 = selection_digest(["client-v1-bbb"])
        assert d1 != d2


# ---------------------------------------------------------------------------
# Trace tests
# ---------------------------------------------------------------------------


class TestClientSamplingTrace:
    """Tests for the trace dataclass."""

    def test_trace_build(self) -> None:
        """Build a trace from a SamplingResult."""
        result = sample_clients(_POOL_SMALL, 13, 0, 5)
        trace = build_trace(
            result,
            sampler_version=DEFAULT_SAMPLER_VERSION,
            experiment_seed=13,
            round_index=0,
            eligible_pool_count=len(_POOL_SMALL),
            clients_per_round=5,
        )
        assert trace.trace_version == "client_sampling_trace_v1"
        assert trace.sampler_version == DEFAULT_SAMPLER_VERSION
        assert trace.experiment_seed == 13
        assert trace.round_index == 0
        assert trace.eligible_pool_count == len(_POOL_SMALL)
        assert trace.clients_per_round == 5
        assert len(trace.selected_client_ids) == 5
        assert trace.selected_digest == result.selected_digest
        assert trace.failed_client_ids == []
        assert trace.retried_client_ids == []

    def test_trace_to_json_deterministic(self) -> None:
        """Trace JSON is deterministic across calls."""
        result = sample_clients(_POOL_SMALL, 13, 0, 5)
        trace = build_trace(
            result,
            sampler_version=DEFAULT_SAMPLER_VERSION,
            experiment_seed=13,
            round_index=0,
            eligible_pool_count=len(_POOL_SMALL),
            clients_per_round=5,
        )
        assert trace.to_json() == trace.to_json()


# ---------------------------------------------------------------------------
# Sampling determinism tests
# ---------------------------------------------------------------------------


class TestSamplingDeterminism:
    """Verify deterministic sampling behavior across runs and environments.

    Tests actual behavioral reproducibility rather than source-text inspection.
    """

    def test_repeated_sampling_identical(self) -> None:
        """Calling sample_clients multiple times gives identical selections and digests."""
        results = [sample_clients(_POOL_SMALL, 13, 0, 10) for _ in range(10)]
        first = results[0]
        for r in results[1:]:
            assert r.selected_client_ids == first.selected_client_ids
            assert r.selected_digest == first.selected_digest

    def test_independent_of_python_hash_seed(self) -> None:
        """Seed derivation and selection digest are deterministic across subprocesses
        with different PYTHONHASHSEED values, proving independence from Python's hash()."""
        import os
        import subprocess
        import sys

        code = (
            "from ppsi.federated.sampling import derive_round_seed, sample_clients\n"
            "pool = [f'client-v1-{i:064x}' for i in range(50)]\n"
            "seed = derive_round_seed('client_sampler_v1', 42, 5)\n"
            "res = sample_clients(pool, 42, 5, 10)\n"
            "print(f'{seed}:{res.selected_digest}')\n"
        )

        outputs = []
        for hash_seed in ["0", "42", "random", "999999"]:
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            outputs.append(proc.stdout.strip())

        assert len(set(outputs)) == 1, f"Outputs varied across PYTHONHASHSEED: {outputs}"
