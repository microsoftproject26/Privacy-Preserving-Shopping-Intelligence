"""v4train.order — the deterministic per-seed data-order policy of central training.

Canonical order = ascending decision_id (MANIFEST_SPEC_V4 ordering.central_canonical_order; GRU1 FULL_RUN_PROTOCOL
L33). Pass p (0-based) of a run with seed s visits the canonical positions in the order

    numpy.random.Generator(PCG64(SeedSequence([ORDER_TAG, s, p]))).permutation(N)

and consecutive effective batches are taken from it, the last batch of a pass at its own size (the tail). The order
depends on (seed, pass) only — never on the family, the process, the worker or a resume — so both families with the
same seed see identical data (package 05 §B "identical data order policy by seed"), and a resume recomputes the
in-progress pass exactly from its index. The GRU1 rule default_rng(seed + pass) is NOT used: with seeds 2026 and 2027
it gives seed 2026 pass 1 the same permutation as seed 2027 pass 0 (a test shows it). Each pass's permutation is
hashed (sha256 of its int64 little-endian bytes) and the hash is recorded, so numpy stream drift would be detected.
"""
from __future__ import annotations

import hashlib

import numpy as np

ORDER_TAG = 0x5634_5452          # "V4TR"
ORDER_POLICY = ("positions = Generator(PCG64(SeedSequence([0x56345452, seed, pass]))).permutation(N) over the "
                "canonical ascending-decision_id order; consecutive effective batches; tail batch at its own size")


def pass_permutation(seed: int, pass_index: int, N: int) -> np.ndarray:
    if N <= 0 or pass_index < 0:
        raise ValueError("N must be > 0 and pass_index >= 0")
    ss = np.random.SeedSequence([ORDER_TAG, int(seed), int(pass_index)])
    return np.random.Generator(np.random.PCG64(ss)).permutation(int(N)).astype(np.int64)


def permutation_sha256(perm: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(perm, dtype="<i8").tobytes()).hexdigest()


def naive_gru1_permutation(seed: int, pass_index: int, N: int) -> np.ndarray:
    """The GRU1 rule (default_rng(seed + pass)); kept only so a test can show its cross-seed collision."""
    return np.random.default_rng(int(seed) + int(pass_index)).permutation(int(N)).astype(np.int64)


__all__ = ["ORDER_TAG", "ORDER_POLICY", "pass_permutation", "permutation_sha256", "naive_gru1_permutation"]
