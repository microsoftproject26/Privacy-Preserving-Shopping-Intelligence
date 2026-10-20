"""Protocol conformance, canonical order, loss-eligibility, target agreement and batch purity (real rows, bucket 0)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from intkit import REAL_DS, assert_raises, batches_equal, nc
from v4int.providers import ProviderError, ProviderIntegrityError, StoreEvalProvider, StoreTrainProvider
from v4train.provider import BatchProvider, EvalProvider

Q = lambda *_: None  # noqa: E731


def test_protocol_conformance(providers):
    tp, ep = providers
    assert isinstance(tp, BatchProvider) and isinstance(ep, EvalProvider)
    assert tp.split == "TRAIN" and tp.manifest_id == "M05_CALIB_TRAIN"
    assert ep.manifest.manifest_id == "M06_CALIB_VALIDATION_WARM"
    assert isinstance(tp.fingerprint, str) and len(tp.fingerprint) == 64
    assert isinstance(ep.fingerprint, str) and len(ep.fingerprint) == 64
    assert tp.n_decisions == tp.decision_ids().shape[0]


def test_canonical_order_is_strictly_ascending_decision_id(providers, real_env):
    tp, _ = providers
    did = tp.decision_ids()
    assert did.dtype == np.int64 and np.all(did[1:] > did[:-1])
    table = np.asarray(real_env["tables"]["M05_CALIB_TRAIN"]["decisions"]["decision_id"], dtype=np.int64)
    assert not np.all(table[1:] > table[:-1]), "the manifest table order is (client_ord, decision_id), not canonical"
    assert np.array_equal(did, np.sort(table))
    d2 = tp.decision_ids()
    d2[0] = -1
    assert tp.decision_ids()[0] != -1, "decision_ids() must return a copy"


def test_every_row_loss_eligible_and_targets_agree(providers, real_env):
    tp, _ = providers
    dec = real_env["tables"]["M05_CALIB_TRAIN"]["decisions"]
    tgt_of = dict(zip(np.asarray(dec["decision_id"]).tolist(), np.asarray(dec["target_class"]).tolist()))
    for s in range(0, tp.n_decisions, 256):
        pos = np.arange(s, min(tp.n_decisions, s + 256))
        b = tp.batch(pos)
        assert bool(b["loss_mask"].all()) and bool((b["target_class"] >= 0).all())
        assert b["decision_id"].tolist() == tp.decision_ids()[pos].tolist()
        assert b["target_class"].tolist() == [tgt_of[d] for d in b["decision_id"].tolist()]
        assert b["feature_view"] == "clean_v2_rich" and b["event_numeric_features"].shape[-1] == 15
        assert all(v.device.type == "cpu" for v in b.values() if torch.is_tensor(v))


def test_batch_serves_exact_positions_in_order(providers):
    tp, _ = providers
    pos = np.array([7, 3, 3, 0, tp.n_decisions - 1], dtype=np.int64)          # any order, repeats allowed
    b = tp.batch(pos)
    assert b["decision_id"].tolist() == tp.decision_ids()[pos].tolist()
    for bad in (np.array([], dtype=np.int64), np.array([tp.n_decisions]), np.array([-1]), np.array([[0, 1]]),
                np.array([0.5])):
        assert_raises(ProviderError, tp.batch, bad)


def test_batch_is_a_pure_function(providers):
    tp, _ = providers
    rng = np.random.default_rng(11)
    pos = rng.choice(tp.n_decisions, 64, replace=False)
    a = tp.batch(pos)
    tp.batch(rng.choice(tp.n_decisions, 200, replace=False))                   # an unrelated call in between
    b = tp.batch(pos.copy())
    assert batches_equal(a, b)[0]


def test_target_disagreement_refused_at_build(real_env, tmp_path):
    from intkit import write_env
    tabs = {k: {t: dict(v) for t, v in tb.items()} for k, tb in real_env["tables"].items() if k == "M05_CALIB_TRAIN"}
    dec = tabs["M05_CALIB_TRAIN"]["decisions"]
    tc = np.asarray(dec["target_class"]).copy()
    tc[5] = (tc[5] + 1) % 158486                                              # one planted wrong label
    dec["target_class"] = tc
    paths = write_env(tmp_path, tabs)                                         # hashes are consistent with the bad label
    assert_raises(ProviderIntegrityError, StoreTrainProvider, "M05_CALIB_TRAIN", history_policy="SESSION_L50", paths=paths, ds_root=REAL_DS,
                  buckets=[0], log=Q)


def test_batch_integrity_check_is_live(providers):
    """A corrupted internal target table is caught on the next batch (the per-batch assertion is not dead code)."""
    tp, _ = providers
    saved = tp._target.copy()
    try:
        tp._target[0] = saved[0] + 1
        assert_raises(ProviderIntegrityError, tp.batch, np.array([0]))
    finally:
        tp._target[:] = saved
    tp.batch(np.array([0]))


def test_manifest_roles_come_from_the_locked_layout(real_env):
    """A training provider on an evaluation-layout manifest (and vice versa) is refused by the layout, not the id."""
    from v4int.verify import ManifestVerificationError
    assert_raises(ManifestVerificationError, StoreTrainProvider, "M06_CALIB_VALIDATION_WARM", history_policy="SESSION_L50", paths=real_env["paths"],
                  log=Q)
    assert_raises(ManifestVerificationError, StoreEvalProvider, "M05_CALIB_TRAIN", history_policy="SESSION_L50", paths=real_env["paths"], log=Q)
    assert_raises(ManifestVerificationError, StoreTrainProvider, "MX_NOT_REGISTERED", history_policy="SESSION_L50", paths=real_env["paths"], log=Q)


def test_population_agnostic_any_registered_manifest(real_env, tmp_path):
    """Manifests under new ids (a future population) are served with no code change; counts come from the header."""
    from intkit import write_env
    tabs = {"MX_POP5_100_TRAIN": real_env["tables"]["M01_COHORT_TRAIN"],
            "MX_POP5_100_VALIDATION": real_env["tables"]["M02_VALIDATION_WARM"]}
    paths = write_env(tmp_path, tabs)
    tp = StoreTrainProvider("MX_POP5_100_TRAIN", history_policy="SESSION_L50", paths=paths, ds_root=REAL_DS, buckets=[0], log=Q)
    ep = StoreEvalProvider("MX_POP5_100_VALIDATION", history_policy="SESSION_L50", paths=paths, ds_root=REAL_DS, buckets=[0], log=Q)
    assert tp.manifest_id == "MX_POP5_100_TRAIN" and tp.n_decisions == tp.verified.header["n_decisions"]
    assert tp.n_decisions == len(tabs["MX_POP5_100_TRAIN"]["decisions"]["decision_id"])
    assert ep.manifest.manifest_id == "MX_POP5_100_VALIDATION" and ep.n_stream_rows == ep.verified.header["n_eval_rows"]
    assert isinstance(tp, BatchProvider) and isinstance(ep, EvalProvider)
    assert bool(tp.batch(np.arange(min(64, tp.n_decisions)))["loss_mask"].all())


@nc("a provider whose canonical order is the manifest table order (client_ord, decision_id) is not ascending")
def test_nc_table_order_is_not_canonical(real_env):
    dec = real_env["tables"]["M05_CALIB_TRAIN"]["decisions"]
    did = np.asarray(dec["decision_id"], dtype=np.int64)
    assert np.all(did[1:] > did[:-1])


@nc("the loss-eligibility check applied to a batch with one masked row must fail")
def test_nc_masked_row_is_not_loss_eligible(providers):
    from v4train.objective import host_contributing
    tp, _ = providers
    b = dict(tp.batch(np.arange(32)))
    b["loss_mask"] = b["loss_mask"].clone()
    b["loss_mask"][3] = False
    assert host_contributing(b) == 32, "every served row must contribute to the loss"
