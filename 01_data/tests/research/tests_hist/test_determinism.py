"""Determinism under USER_HISTORY_L50: batch() is a pure function, so the manifest-scoped cache (v4int) equals the
streaming HistoryStore bitwise (>= 20,000 random TRAIN rows + whole trainer batches + the tail; every evaluation
batch), a rebuilt provider serves identical bytes, and the SESSION_L50 provider still equals the INT-01 streaming path.
Negative control: a cache that kept the session lengths is not the USER_HISTORY stream."""
from __future__ import annotations

import copy

import numpy as np

from histkit import REAL_DS, Q, batches_equal, nc
from v4data.common import VALIDATION_SPLITS
from v4eval.predict import stream_rows
from v4int.cache import merged_decision_set
from v4int.providers import StoreTrainProvider
from v4train.order import pass_permutation

B = 256


def _stream_train(hs, tp, pos):
    ds = hs.decision_set("loss")
    return hs.batch(ds, hs.positions_for(ds, tp.decision_ids()[pos]))


def _tagged(hs, tp):
    """The streaming reference tagged with the provider's TRANSFORM_STATE self-hash (a provenance field only)."""
    from v4data.store import CleanV2Store
    from v4hist.stream import HistoryStore
    st = CleanV2Store(hs.store.cs, hs.store.spec, tp.verified.transform_state_self_sha256)
    return HistoryStore(st, hs.index, hs.policy, hs.decision_splits)


def test_train_cache_equals_user_stream_on_20k_random_rows(providers_user, hist_train):
    tp, _ = providers_user
    ref = _tagged(hist_train, tp)
    assert tp.n_decisions >= 20_000
    rng = np.random.default_rng(20260926)
    for _ in range(80):
        pos = rng.choice(tp.n_decisions, B, replace=False)
        ok, why = batches_equal(tp.batch(pos), _stream_train(ref, tp, pos))
        assert ok, why


def test_train_cache_equals_user_stream_on_whole_trainer_batches(providers_user, hist_train):
    tp, _ = providers_user
    ref = _tagged(hist_train, tp)
    perm = pass_permutation(2026, 0, tp.n_decisions)
    for s in (0, B, 2 * B, (tp.n_decisions // B) * B):
        pos = perm[s:min(tp.n_decisions, s + B)]
        ok, why = batches_equal(tp.batch(pos), _stream_train(ref, tp, pos))
        assert ok, (s, why)


def test_eval_cache_equals_user_stream_on_every_batch(providers_user, hist_val):
    _, ep = providers_user
    ref = _tagged(hist_val, ep)
    ds = ref.decision_set("eval")
    pos = ref.positions_for(ds, ep.manifest.decision_id[stream_rows(ep.manifest)])
    n = 0
    for k, b in enumerate(ep.batches(ep.eval_batch)):
        ok, why = batches_equal(b, ref.batch(ds, pos[k * ep.eval_batch:(k + 1) * ep.eval_batch]))
        assert ok, (k, why)
        n += int(b["decision_id"].shape[0])
    assert n == pos.size


def test_eval_cache_holds_train_period_rows(providers_user, providers_session):
    _, eu = providers_user
    _, es = providers_session
    assert eu.cache_info.store_splits == ("TRAIN",) + tuple(VALIDATION_SPLITS)
    assert eu.cache_info.n_event_rows > es.cache_info.n_event_rows                   # the users' TRAIN tails
    assert eu.cache_info.mean_window_length > es.cache_info.mean_window_length


def test_rebuilt_provider_serves_identical_bytes_and_batch_is_pure(providers_user, real_env):
    tp, _ = providers_user
    again = StoreTrainProvider("M05_CALIB_TRAIN", history_policy="USER_HISTORY_L50", paths=real_env["paths"],
                               ds_root=REAL_DS, buckets=[0], log=Q)
    assert again.fingerprint == tp.fingerprint
    rng = np.random.default_rng(3)
    for _ in range(5):
        pos = rng.choice(tp.n_decisions, B, replace=False)
        a = tp.batch(pos)
        tp.batch(rng.choice(tp.n_decisions, 200, replace=False))                    # an unrelated call in between
        assert batches_equal(a, tp.batch(pos.copy()))[0] and batches_equal(a, again.batch(pos))[0]


def test_session_provider_still_equals_the_int01_stream(providers_session, real_train_store, real_val_store):
    from v4data.store import CleanV2Store
    tp, ep = providers_session
    ref = CleanV2Store(real_train_store.cs, real_train_store.spec, tp.verified.transform_state_self_sha256)
    ds = ref.decision_set("TRAIN", "loss")
    rng = np.random.default_rng(8)
    for _ in range(40):
        pos = rng.choice(tp.n_decisions, B, replace=False)
        ok, why = batches_equal(tp.batch(pos), ref.batch(ds, ref.positions_for(ds, tp.decision_ids()[pos])))
        assert ok, why
    vref = CleanV2Store(real_val_store.cs, real_val_store.spec, ep.verified.transform_state_self_sha256)
    vds = merged_decision_set(vref, VALIDATION_SPLITS, "eval")
    vpos = vref.positions_for(vds, ep.manifest.decision_id[stream_rows(ep.manifest)])
    for k, b in enumerate(ep.batches(ep.eval_batch)):
        ok, why = batches_equal(b, vref.batch(vds, vpos[k * ep.eval_batch:(k + 1) * ep.eval_batch]))
        assert ok, (k, why)


def test_identical_builder_for_train_and_eval(real_env, providers_user, hist_train_on_full, monkeypatch):
    """ONE window builder serves both roles. (a) Both providers build their windows through the same function,
    v4int.cache.build_subset_store, with the same history_policy (a spy on the real function records the calls).
    (b) TRAIN decisions served through the EVALUATION provider's store configuration (TRAIN + VALIDATION_SELECT +
    VALIDATION_CONFIRM) are bitwise the TRAIN provider's batches. (c) Both caches carry the same policy record."""
    import v4int.cache as C
    import v4int.providers as P
    from v4int.providers import StoreEvalProvider
    calls = []
    real = C.build_subset_store

    def spy(ids, **kw):
        calls.append({"fn": real, "history_policy": kw["history_policy"], "splits": tuple(kw["splits"]),
                      "kind": kw["kind"]})
        return real(ids, **kw)
    monkeypatch.setattr(P, "build_subset_store", spy)
    tp = StoreTrainProvider("M05_CALIB_TRAIN", history_policy="USER_HISTORY_L50", paths=real_env["paths"],
                            ds_root=REAL_DS, buckets=[0], log=Q)
    ep = StoreEvalProvider("M06_CALIB_VALIDATION_WARM", history_policy="USER_HISTORY_L50", paths=real_env["paths"],
                           ds_root=REAL_DS, buckets=[0], log=Q)
    assert [c["kind"] for c in calls] == ["loss", "eval"]
    assert all(c["fn"] is real and c["history_policy"] == "USER_HISTORY_L50" for c in calls)
    assert calls[0]["splits"] == ("TRAIN",) and calls[1]["splits"] == tuple(VALIDATION_SPLITS)
    ref = _tagged(hist_train_on_full, tp)                     # the evaluation provider's store configuration
    assert tuple(ref.store.cs.splits) == ("TRAIN",) + tuple(VALIDATION_SPLITS)
    rng = np.random.default_rng(12)
    for _ in range(40):
        pos = rng.choice(tp.n_decisions, B, replace=False)
        ok, why = batches_equal(tp.batch(pos), _stream_train(ref, tp, pos))
        assert ok, why
    tu, eu = providers_user
    for a_, b_ in ((tp, tu), (ep, eu)):
        assert a_.fingerprint == b_.fingerprint
    assert tp.history["resolver"] == ep.history["resolver"] and tp.history["history_policy"] == ep.history[
        "history_policy"] and tp.history["window_rule"] == ep.history["window_rule"]


@nc("a cache that kept the SESSION lengths (the forgotten policy step) is not the USER_HISTORY stream")
def test_nc_cache_with_session_lengths_is_not_the_user_stream(providers_session, hist_train):
    tp, _ = providers_session                                  # its cached windows are the session windows
    ref = _tagged(hist_train, tp)
    bad = copy.copy(tp)
    pos = np.random.default_rng(1).choice(tp.n_decisions, B, replace=False)
    ok, why = batches_equal(bad.batch(pos), _stream_train(ref, tp, pos))
    assert ok, why
