"""Secure-aggregation simulation (ppsi.fedsim.secagg): the decoded sum is bit-identical to the plain fixed-point sum
under dropout, for the complete graph and a k-regular graph; threshold failures are reported, never a wrong sum;
Shamir sharing, the graph, determinism and the byte accounting."""
from __future__ import annotations

import random

import numpy as np
import pytest

from ppsi.fedsim import secagg as sa

N = 64
D = 512
CLIP = 0.75
SCALE = sa.fixed_point_scale(CLIP, N)
VARIANTS = [None, 32]          # full, k-regular (k = 32 keeps p = 0.3 recoverable for these seeds)


def edge_enc(cfg, seed):
    """Clients: random in range, exactly +/-clip, zeros, and values straddling the rounding boundary."""
    def enc(i):
        g = np.random.default_rng([seed, i])
        x = g.uniform(-CLIP, CLIP, cfg.d)
        if i % 5 == 0:
            x[:] = CLIP
        elif i % 5 == 1:
            x[:] = -CLIP
        elif i % 5 == 2:
            x[: cfg.d // 2] = CLIP
            x[cfg.d // 2:] = -CLIP
        elif i % 5 == 3:
            x[:] = 0.5 / SCALE * (2 * g.integers(0, 3, cfg.d) + 1)      # exact .5 ties
        return sa.encode_fixed(x, SCALE, CLIP)
    return enc


@pytest.mark.parametrize("k", VARIANTS)
@pytest.mark.parametrize("p", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("seed", [1, 2])
def test_bit_identical_sum(k, p, seed):
    cfg = sa.SecAggConfig(n=N, d=D, k=k, clip=CLIP, seed=seed)
    drop = sa.sample_dropped(N, p, seed)
    r = sa.run_round(cfg, edge_enc(cfg, seed), drop)
    assert r.success, r.failure_reason
    assert r.bit_identical and np.array_equal(r.total, r.plain)
    assert r.survivors == [i for i in range(N) if i not in set(drop)]
    assert len(r.survivors) == N - len(drop)


def test_dropout_levels_actually_drop():
    assert sa.sample_dropped(N, 0.0, 3) == []
    assert 0 < len(sa.sample_dropped(N, 0.3, 3)) < N // 2


def test_wraparound_and_clip_bound():
    # all clients at +clip: the sum is n * clip * scale < 2^31, no signed wrap; all at -clip likewise
    for sign in (1.0, -1.0):
        cfg = sa.SecAggConfig(n=N, d=64, clip=CLIP, seed=5)
        r = sa.run_round(cfg, lambda i, s=sign: sa.encode_fixed(np.full(64, s * 10.0), SCALE, CLIP), [])
        assert r.bit_identical
        dec = sa.decode_fixed(r.total, SCALE)
        assert np.all(np.abs(dec) <= N * CLIP) and np.allclose(dec, sign * N * CLIP, atol=N / SCALE)
    assert N * CLIP * SCALE < 2 ** 31 <= 2 * N * CLIP * SCALE * 2
    # masked words are uniform mod 2^32 (wrap is exercised): raw masked input wraps, yet the sum is exact
    e = sa.encode_fixed(np.array([-CLIP, CLIP, 0.0]), SCALE, CLIP)
    assert e[0] > 2 ** 31 and e[1] < 2 ** 31 and e[2] == 0
    assert sa.decode_fixed(e, SCALE)[0] == -CLIP


def test_encoded_sum_matches_float_sum():
    cfg = sa.SecAggConfig(n=N, d=128, clip=CLIP, seed=9)
    g = np.random.default_rng(1)
    xs = [g.uniform(-CLIP, CLIP, 128) for _ in range(N)]
    r = sa.run_round(cfg, lambda i: sa.encode_fixed(xs[i], SCALE, CLIP), [3, 7])
    ref = sum(xs[i] for i in r.survivors)
    assert np.max(np.abs(sa.decode_fixed(r.total, SCALE) - ref)) <= len(r.survivors) * 0.5 / SCALE + 1e-12


@pytest.mark.parametrize("k", VARIANTS)
def test_threshold_failure(k):
    t = sa.threshold(N)
    assert t == 43
    cfg = sa.SecAggConfig(n=N, d=64, k=k, clip=CLIP, seed=4)
    enc = edge_enc(cfg, 4)
    r = sa.run_round(cfg, enc, list(range(N - t + 1)))          # t - 1 survivors
    assert not r.success and "survivors" in r.failure_reason and r.total is None
    ok = sa.run_round(cfg, enc, list(range(N - t)))             # exactly t survivors
    if k is None:
        assert ok.success and ok.bit_identical


def test_kreg_neighbourhood_failure_is_reported():
    # a k-regular graph with a tiny degree and heavy dropout cannot recover every secret: failure, never a wrong sum
    cfg = sa.SecAggConfig(n=N, d=64, k=4, clip=CLIP, seed=6)
    r = sa.run_round(cfg, edge_enc(cfg, 6), sa.sample_dropped(N, 0.3, 6))
    assert (not r.success and r.failure_reason) or r.bit_identical


def test_shamir_roundtrip_and_threshold():
    rng = random.Random(1)
    sec = (1 << 255) + 12345
    sh = sa.shamir_split(sec, list(range(1, 11)), 6, rng)
    assert sa.shamir_reconstruct(sh[2:8], 6) == sec % sa.PRIME
    with pytest.raises(sa.SecAggError):
        sa.shamir_reconstruct(sh[:5], 6)


def test_graph_is_regular_symmetric():
    g = sa.build_graph(N, 16, 3)
    assert all(len(v) == 16 for v in g)
    assert all(i in g[j] for i in range(N) for j in g[i])
    assert g == sa.build_graph(N, 16, 3)
    assert len(sa.build_graph(N, None, 3)[0]) == N - 1


@pytest.mark.parametrize("k", VARIANTS)
def test_determinism(k):
    cfg = sa.SecAggConfig(n=N, d=D, k=k, clip=CLIP, seed=8)
    drop = sa.sample_dropped(N, 0.1, 8)
    a = sa.run_round(cfg, edge_enc(cfg, 8), drop)
    b = sa.run_round(cfg, edge_enc(cfg, 8), drop)
    assert np.array_equal(a.total, b.total) and a.bytes_up == b.bytes_up and a.bytes_down == b.bytes_down
    assert sa.sample_dropped(N, 0.1, 8) == drop


def test_byte_accounting():
    d = 1000
    plain = 4 * d + 8
    assert sa.plain_upload_bytes(d) == plain
    cfg = sa.SecAggConfig(n=N, d=d, clip=1.0, seed=2)
    r = sa.run_round(cfg, lambda i: np.zeros(d, dtype=np.uint32), [])
    deg = N - 1
    assert r.bytes_up[0] == 64 and r.bytes_up[1] == deg * sa.CT_BYTES
    assert r.bytes_up[2] == plain                               # masked input = one full-length vector + header
    assert r.bytes_up[3] == N * sa.UNMASK_ENTRY_BYTES
    assert r.bytes_down[0] == deg * 68 and r.rounds == 4
    cs = sa.cost_summary(cfg, r)
    assert cs["overhead_ratio_up"] > 1.0 and cs["overhead_ratio_up_down"] > cs["overhead_ratio_up"]
    assert sa.CT_BYTES == 8 + 72 + 28 and sa.SHARE_BYTES == 36
    # k-regular costs fewer setup bytes than the full graph; the masked vector is the same size
    ck = sa.SecAggConfig(n=N, d=d, k=16, clip=1.0, seed=2)
    rk = sa.run_round(ck, lambda i: np.zeros(d, dtype=np.uint32), [])
    assert rk.bytes_up[1] == 16 * sa.CT_BYTES < r.bytes_up[1] and rk.bytes_up[2] == plain
    # at a large update length the overhead ratio approaches 1
    big = 265162
    rb = (4 * big + 8 + 64 + deg * sa.CT_BYTES + N * sa.UNMASK_ENTRY_BYTES) / (4 * big + 8)
    assert 1.0 < rb < 1.05


def test_dropped_client_bytes_stop_after_round_1():
    cfg = sa.SecAggConfig(n=N, d=64, clip=1.0, seed=2)
    r = sa.run_round(cfg, lambda i: np.zeros(64, dtype=np.uint32), [0, 1])
    nb = sa.build_graph(N, None, 2)
    assert sa.per_client_bytes(cfg, nb, set(r.survivors), 0)["up"].keys() == {0, 1}
    assert sa.per_client_bytes(cfg, nb, set(r.survivors), 5)["up"].keys() == {0, 1, 2, 3}
