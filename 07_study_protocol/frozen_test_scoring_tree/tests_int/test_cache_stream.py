"""The manifest-scoped cache equals the streaming store bitwise (>= 20,000 random rows + whole trainer batches + the
tail; every evaluation batch), and batch() is a pure function, so a resumed run is served identical bytes."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from intkit import REAL_DS, batches_equal, nc, real_train_tables, write_env
from v4data.common import VALIDATION_SPLITS
from v4data.store import CleanV2Store
from v4eval.predict import stream_rows
from v4int.cache import merged_decision_set
from v4int.providers import StoreTrainProvider
from v4train.order import pass_permutation

Q = lambda *_: None  # noqa: E731


@pytest.fixture(scope="module")
def big(real_train_store, tmp_path_factory):
    """A TRAIN manifest of 1,500 real clients (>= 20,000 decisions) and the streaming reference store over the same
    bucket, tagged with the same TRANSFORM_STATE self-hash as the provider's batches."""
    tabs = real_train_tables(real_train_store, n_clients=1500, seed=7)
    paths = write_env(tmp_path_factory.mktemp("big"), {"M05_CALIB_TRAIN": tabs})
    tp = StoreTrainProvider("M05_CALIB_TRAIN", history_policy="SESSION_L50", paths=paths, ds_root=REAL_DS, buckets=[0], log=Q)
    ref = CleanV2Store(real_train_store.cs, real_train_store.spec, tp.verified.transform_state_self_sha256)
    return tp, ref, paths


def _stream_batch(ref, tp, pos):
    ds = ref.decision_set("TRAIN", "loss")
    return ref.batch(ds, ref.positions_for(ds, tp.decision_ids()[pos]))


def test_cache_equals_stream_on_20k_random_rows(big):
    tp, ref, _ = big
    assert tp.n_decisions >= 20_000, tp.n_decisions
    rng = np.random.default_rng(20260926)
    rows = 0
    for _ in range(80):                                                  # 80 x 256 = 20,480 random rows
        pos = rng.choice(tp.n_decisions, 256, replace=False)
        ok, why = batches_equal(tp.batch(pos), _stream_batch(ref, tp, pos))
        assert ok, why
        rows += pos.size
    assert rows >= 20_000


def test_cache_equals_stream_on_whole_trainer_batches(big):
    tp, ref, _ = big
    perm = pass_permutation(2026, 0, tp.n_decisions)                     # the trainer's pass-0 order (seed 2026)
    B = 256
    for s in (0, B, 2 * B, (tp.n_decisions // B) * B):                   # three whole batches + the tail
        pos = perm[s:min(tp.n_decisions, s + B)]
        ok, why = batches_equal(tp.batch(pos), _stream_batch(ref, tp, pos))
        assert ok, (s, why)


def test_eval_cache_equals_stream_on_every_batch(providers, real_val_store):
    _, ep = providers
    ref = CleanV2Store(real_val_store.cs, real_val_store.spec, ep.verified.transform_state_self_sha256)
    ds = merged_decision_set(ref, VALIDATION_SPLITS, "eval")
    pos = ref.positions_for(ds, ep.manifest.decision_id[stream_rows(ep.manifest)])
    n = 0
    for k, b in enumerate(ep.batches(ep.eval_batch)):
        ok, why = batches_equal(b, ref.batch(ds, pos[k * ep.eval_batch:(k + 1) * ep.eval_batch]))
        assert ok, (k, why)
        n += int(b["decision_id"].shape[0])
    assert n == pos.size


def test_rebuilt_provider_serves_identical_bytes(big):
    """Resume = a fresh process rebuilds the provider: the same positions give the same bytes."""
    tp, _, paths = big
    again = StoreTrainProvider("M05_CALIB_TRAIN", history_policy="SESSION_L50", paths=paths, ds_root=REAL_DS, buckets=[0], log=Q)
    assert again.fingerprint == tp.fingerprint
    rng = np.random.default_rng(3)
    for _ in range(5):
        pos = rng.choice(tp.n_decisions, 256, replace=False)
        assert batches_equal(tp.batch(pos), again.batch(pos))[0]


def test_cache_holds_only_manifest_windows(big, real_train_store):
    tp, _, _ = big
    assert tp.cache_info.n_decisions == tp.n_decisions
    assert tp.cache_info.n_event_rows < real_train_store.cs.n_rows          # a strict subset of the bucket's rows
    assert tp.cache_info.cache_nbytes > 0


@nc("a cache whose window index is off by one row is not bitwise equal to the streaming store")
def test_nc_shifted_window_index_differs(big):
    tp, ref, _ = big
    bad = copy.copy(tp.ds)
    bad.end_row = np.maximum(tp.ds.end_row - 1, 0)
    pos = np.random.default_rng(1).choice(tp.n_decisions, 256, replace=False)
    got = tp.store.batch(bad, pos)                                   # (an RC refusal would also be an AssertionError)
    ok, why = batches_equal(got, _stream_batch(ref, tp, pos))
    assert ok, why
