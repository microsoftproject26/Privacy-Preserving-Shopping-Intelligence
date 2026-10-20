"""INT-07 - the gated TEST evaluation path on a SYNTHETIC four-partition dataset (tests_int/int07kit.py; no real
RETRO_TEST row / label / feature / metric is read anywhere).

  * store / windows: the gated row index equals v4hist's on the non-TEST partitions; a RETRO_TEST query's USER_HISTORY_L50
    window holds its prior VALIDATION and TEST events in causal order and never a later event (check_windows,
    check_served_batch, check_session_offsets); VALIDATION windows are byte-identical in a TEST-inclusive store and, in
    fresh interpreters, with and without the TEST path imported and exercised (no side effect); the gated cache branch
    merges VS + VC + TEST in ascending decision_id, refuses a non-TEST decision, re-validates the gate;
  * provider: the RETRO_TEST StoreEvalProvider (DATA-09 shim + gated store) serves the manifest stream, names its gate
    and content sha in describe(), and its comparison key has split RETRO_TEST;
  * consumers end to end: reeval (evaluated_test_role -> role_evaluator rows), launch_device (LO engine on the RETRO_TEST
    provider; per-client assembly == predict_adapter; TEST cutoff rule), v4base (LAST rows), dump_rows_retro (NPZ + meta);
    every row carries split RETRO_TEST, the TEST role, the register entry id;
  * negative controls (strict xfail) for the causality, key and TEST-partition properties.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import int07kit as K
from intkit import batches_equal, nc

pytestmark = pytest.mark.skipif(not K.have_real_bucket0(), reason="the frozen v1 bucket 0 (F4 donor rows) is absent")

HERE = Path(__file__).resolve().parent
USER = "USER_HISTORY_L50"
MAIN_TRAIN = "MAIN_TRAIN_5_100_131072"
Q = lambda *_: None  # noqa: E731


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return K.make_world(tmp_path_factory.mktemp("int07_world"))


def _index_ids(ds, splits, *, eval_only=True):
    import pyarrow.parquet as pq
    out = []
    for s in splits:
        t = pq.ParquetFile(ds / "decisions_index" / f"split={s}" / "user_bucket=0" / "data_0.parquet").read()
        d = np.asarray(t["decision_id"].to_numpy(), dtype=np.int64)
        m = np.asarray(t["eval_mask"].to_numpy(zero_copy_only=False), dtype=bool)
        out.append(d[m] if eval_only else d)
    return np.sort(np.concatenate(out))


@pytest.fixture(scope="module")
def full(world):
    """The gated four-partition store of the synthetic dataset and its gated row index."""
    from v4int import retro_store as RS
    st = RS.GatedCleanV2Store.build(gate=world.gate, splits=RS.TEST_ORDER, buckets=[0], ds_root=world.ds, log=Q)
    ri = RS.build_gated_row_index(world.gate, world.ds, st.cs.buckets, RS.TEST_ORDER)
    from v4hist.windows import align_to_store
    align_to_store(ri, st.cs)
    return st, ri


def _policy_windows(st, ri, splits):
    from v4hist.windows import policy_decision_set, window_rows
    from v4int.cache import merged_decision_set
    ds = policy_decision_set(merged_decision_set(st, splits, "eval"), ri, USER)
    idx, valid = window_rows(ds.end_row, ds.length)
    return ds, idx, valid


# ------------------------------------------------------------------------------------------------ store / windows
def test_gated_index_equals_v4hist_index_on_non_test_partitions(world):
    from v4hist.policy import NON_TEST_ORDER
    from v4hist.windows import build_row_index
    from v4int import retro_store as RS
    a = build_row_index(world.ds, [0], NON_TEST_ORDER)
    b = RS.build_gated_row_index(world.gate, world.ds, [0], NON_TEST_ORDER)
    for f in ("decision_id", "user_ord", "user_rank", "position", "ts_us", "split_code", "session_id"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert (a.bucket_rows, a.n_users, a.order_checks) == (b.bucket_rows, b.n_users, b.order_checks)


def test_gated_history_rule_extends_history_splits_by_one_partition(world):
    from v4hist.policy import history_splits
    from v4int import retro_store as RS
    for ds in (("TRAIN",), ("VALIDATION_SELECT",), ("VALIDATION_SELECT", "VALIDATION_CONFIRM")):
        assert RS.gated_history_splits(world.gate, ds) == history_splits(ds)
    assert RS.gated_history_splits(world.gate, ("TEST",)) == RS.TEST_ORDER
    assert RS.gated_store_splits(world.gate, USER, RS.TEST_DECISION_SPLITS) == RS.TEST_ORDER
    assert RS.gated_store_splits(world.gate, "SESSION_L50", RS.TEST_DECISION_SPLITS) == RS.TEST_DECISION_SPLITS


def test_retro_test_windows_are_causal_and_hold_prior_validation_and_test_events(full):
    from v4hist.windows import SPLIT_CODES, check_served_batch, check_session_offsets, check_windows, \
        session_offset_tokens
    st, ri = full
    ds, idx, valid = _policy_windows(st, ri, ("TEST",))
    assert ds.decision_id.size > 0
    rows = ri.rows_of(ds.decision_id)
    assert np.array_equal(rows, ds.end_row) and bool(np.all(ri.split_code[rows] == SPLIT_CODES["TEST"]))
    rec = check_windows(ri, rows, idx, valid, policy=USER)
    assert rec["decisions_with_earlier_partition_rows"] > 0 and rec["rows_from_earlier_partitions"] > 0
    ic = np.where(valid, idx, 0)
    sc = np.where(valid, ri.split_code[ic], -1)
    has_val = ((sc == 1) | (sc == 2)).any(axis=1)
    has_prior_test = ((sc == 3).sum(axis=1) >= 2)
    assert bool(np.any(has_val & has_prior_test))                  # VALIDATION + earlier TEST events in one window
    ts = np.where(valid, ri.ts_us[ic], np.iinfo(np.int64).min)
    assert bool(np.all(np.diff(np.where(valid, ts, np.iinfo(np.int64).max), axis=1)[valid[:, 1:]] >= 0))
    assert bool(np.all(ts.max(axis=1) <= ri.ts_us[rows]))           # never a later event
    b = st.batch(ds, np.arange(ds.decision_id.size))
    check_served_batch(b, st.cs, st.spec, idx, valid)
    check_session_offsets(ri, rows, idx, valid, session_offset_tokens(ri.session_id, ds.end_row, ds.length))


@nc("a RETRO_TEST window extended one event into the future must fail the causality proof")
def test_nc_future_event_in_a_retro_test_window(full):
    from v4hist.windows import check_windows, window_rows
    st, ri = full
    ds, idx, valid = _policy_windows(st, ri, ("TEST",))
    rows = ri.rows_of(ds.decision_id)
    nxt = np.minimum(ds.end_row + 1, ri.n_rows - 1)                 # the next stored row (a later event or user)
    ln = np.minimum(np.asarray(ds.length, dtype=np.int64) + 1, 50)
    idx2, valid2 = window_rows(nxt, ln)
    check_windows(ri, rows, idx2, valid2, policy=USER)


def test_validation_windows_are_byte_identical_in_a_test_inclusive_store(world, full):
    from v4data.common import VALIDATION_SPLITS
    from v4hist.windows import add_session_features
    from v4int.cache import build_subset_store
    from v4int.providers import add_session_offsets
    st, ri = full
    gds, _idx, _valid = _policy_windows(st, ri, VALIDATION_SPLITS)
    ids = _index_ids(world.ds, VALIDATION_SPLITS)
    assert np.array_equal(gds.decision_id, ids)
    cs_, cds, info = build_subset_store(ids, splits=VALIDATION_SPLITS, kind="eval", state_sha256=None,
                                        history_policy=USER, ds_root=world.ds, buckets=[0], log=Q)
    assert info.store_splits == ("TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM")
    pos = np.arange(ids.size)
    a = add_session_offsets(cs_.batch(cds, pos), cs_, cds, pos)
    b = add_session_features(st.batch(gds, pos), ri.session_id, gds.end_row, gds.length)
    ok, why = batches_equal(a, b)
    assert ok, why


def _probe(world, mode):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    args = [sys.executable, "-B", str(HERE / "int07_probe.py"), str(world.ds), mode]
    if mode == "after_test":
        args += [str(world.lock), str(world.register)]
    p = subprocess.run(args, capture_output=True, text=True, env=env, cwd=str(HERE.parent))
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_validation_store_has_no_side_effect_from_the_test_path(world):
    plain, after = _probe(world, "plain"), _probe(world, "after_test")
    assert plain["retro_store_imported"] is False and after["retro_store_imported"] is True
    assert after["test_decisions"] > 0 and after["test_store_splits"] == ["TRAIN", "VALIDATION_SELECT",
                                                                        "VALIDATION_CONFIRM", "TEST"]
    assert plain["store"] == after["store"] and plain["batch"] == after["batch"] and plain["n"] == after["n"] > 0


def test_gated_cache_branch_merges_ascending_and_keeps_test_decisions_only(world, full):
    from v4hist.windows import user_lengths
    from v4int import retro_store as RS
    from v4int.cache import CacheBuildError, build_subset_store
    ids = _index_ids(world.ds, ("TEST",))
    st, ds, info = build_subset_store(ids, splits=RS.TEST_DECISION_SPLITS, kind="eval", state_sha256=None,
                                      history_policy=USER, ds_root=world.ds, buckets=[0], log=Q, test_gate=world.gate)
    assert np.array_equal(ds.decision_id, ids) and ds.split == "RETRO_TEST"
    assert info.store_splits == RS.TEST_ORDER and isinstance(st, RS.GatedCleanV2Store)
    _st, ri = full
    assert np.array_equal(np.asarray(ds.length, dtype=np.int64),
                          user_lengths(ri, ri.rows_of(ids)).astype(np.int64))
    mixed = np.sort(np.concatenate([ids, _index_ids(world.ds, ("VALIDATION_SELECT",))[:1]]))
    with pytest.raises(CacheBuildError, match="not a TEST-partition decision"):
        build_subset_store(mixed, splits=RS.TEST_DECISION_SPLITS, kind="eval", state_sha256=None,
                           history_policy=USER, ds_root=world.ds, buckets=[0], log=Q, test_gate=world.gate)


@nc("a RETRO_TEST subset store that silently accepts a VALIDATION decision")
def test_nc_gated_cache_accepts_a_validation_decision(world):
    from v4int import retro_store as RS
    from v4int.cache import CacheBuildError, build_subset_store
    ids = np.sort(np.concatenate([_index_ids(world.ds, ("TEST",)), _index_ids(world.ds, ("VALIDATION_SELECT",))[:1]]))
    try:
        build_subset_store(ids, splits=RS.TEST_DECISION_SPLITS, kind="eval", state_sha256=None, history_policy=USER,
                           ds_root=world.ds, buckets=[0], log=Q, test_gate=world.gate)
        accepted = True
    except CacheBuildError:
        accepted = False
    assert accepted


def test_gate_closing_refuses_every_store_entry_point(world, tmp_path):
    import shutil
    from v4data.common import TestGateError
    from v4data09.d9_gate import check_test_gate
    from v4int import retro_store as RS
    from v4int.cache import build_subset_store
    lock, reg = tmp_path / "L.json", tmp_path / "R.json"
    shutil.copyfile(world.lock, lock)
    shutil.copyfile(world.register, reg)
    g = check_test_gate(lock, reg)
    RS.require_gate(g)
    lock.unlink()
    ids = _index_ids(world.ds, ("TEST",))
    for fn in (lambda: RS.require_gate(g), lambda: RS.gated_history_splits(g, ("TEST",)),
               lambda: RS.GatedCleanV2Store.build(gate=g, splits=RS.TEST_ORDER, buckets=[0], ds_root=world.ds, log=Q),
               lambda: RS.build_gated_row_index(g, world.ds, [0], RS.TEST_ORDER),
               lambda: build_subset_store(ids, splits=RS.TEST_DECISION_SPLITS, kind="eval", state_sha256=None,
                                          history_policy=USER, ds_root=world.ds, buckets=[0], log=Q, test_gate=g)):
        with pytest.raises(TestGateError):
            fn()


# ------------------------------------------------------------------------------------------------ provider
@pytest.fixture(scope="module")
def shim(world):
    """The DATA-09 wrappers installed for the synthetic extension (role=None: the eval-only consumers' mode) over the
    verification stubs; removed after the module."""
    from v4int import test_roles as TR
    mp = pytest.MonkeyPatch()
    K.verification_stubs(world, mp)
    TR.install(world.ext_path, role=None, lock_path=world.lock, register_path=world.register, eager=True)
    yield world
    TR.uninstall()
    mp.undo()


def _provider(world, mid=None, **kw):
    from v4data09 import d9_common as D9
    from v4int.providers import StoreEvalProvider
    return StoreEvalProvider(mid or D9.MAIN_TEST_ID, history_policy=USER, manifest_registry=world.ext_path,
                             registry_root=world.root, ds_root=world.ds, buckets=[0], prefetch=0, log=Q, **kw)


@pytest.fixture(scope="module")
def rep(shim):
    return _provider(shim)


def test_retro_test_provider_serves_the_manifest_stream(shim, rep):
    from v4eval.predict import stream_rows
    from v4int import retro_store as RS
    m = rep.manifest
    assert m.header["v4_split"] == "RETRO_TEST" and rep.test_gate is not None
    stream = stream_rows(m)
    got_ids, got_t = [], []
    for b in rep.batches(rep.eval_batch):
        got_ids.append(b["decision_id"].numpy())
        got_t.append(b["target_class"].numpy())
    assert np.array_equal(np.concatenate(got_ids), m.decision_id[stream])
    assert np.array_equal(np.concatenate(got_t), m.target_class[stream])
    d = rep.describe()
    assert d["split"] == "RETRO_TEST" and d["test_exposure_register_entry_id"] == shim.gate.entry_id == 2
    assert d["manifest_content_sha256"] == rep.manifest_binding["content_sha256"]
    assert d["history"]["store_splits"] == list(RS.TEST_ORDER) and tuple(d["cache"]["store_splits"]) == RS.TEST_ORDER
    assert "session_offset_tokens" in next(iter(rep.batches(rep.eval_batch)))


def test_every_retro_test_manifest_is_served(shim):
    from v4data09 import d9_common as D9
    for mid in D9.TEST_IDS:
        p = _provider(shim, mid)
        assert p.n_stream_rows == int(p.manifest.eval_mask.sum()) and p.describe()["split"] == "RETRO_TEST"


def test_comparison_key_split_is_retro_test_for_the_gated_provider(rep):
    from types import SimpleNamespace
    from v4int.providers import ProviderError, comparison_key_base
    key = comparison_key_base(None, K.support_key_standin(rep), rep)
    assert key["split"] == "RETRO_TEST" and key["eval_manifest_id"] == rep.manifest_id
    no_gate = SimpleNamespace(manifest=rep.manifest, test_gate=None, frozen=rep.frozen, history_policy=USER,
                              transform_state=rep.transform_state, verified=rep.verified, manifest_id=rep.manifest_id)
    with pytest.raises(ProviderError, match="TEST gate"):
        comparison_key_base(None, K.support_key_standin(rep), no_gate)


@nc("the RETRO_TEST provider's comparison key keeps split VALIDATION")
def test_nc_retro_test_key_says_validation(rep):
    from v4int.providers import comparison_key_base
    assert comparison_key_base(None, K.support_key_standin(rep), rep)["split"] == "VALIDATION"


def test_provider_under_a_foreign_gate_is_refused(shim, rep, tmp_path):
    from v4data09.d9_gate import check_test_gate
    from v4int import retro_store as RS
    other = check_test_gate(*K.d9c().make_gate_files(tmp_path, lock_text='{"document": "ALL_METHODS_LOCK", "v": 9}'))
    with pytest.raises(RS.TestPathRefused, match="another TEST gate"):
        RS.gate_for_provider(rep.manifest.header, other)


# ------------------------------------------------------------------------------------------------ consumers
def _tiny(K_=7, seed=1):
    from fl.synthetic import make_tiny_adapter
    a = make_tiny_adapter(K=K_, seed=seed, tied=False)
    a.module.eval()
    return a


def _frozen():
    from v4eval import frozen_config as fc
    fz = fc.load()
    fc.apply_runtime_flags(fz)
    return fz


def _val_key(rep):
    from v4int.providers import comparison_key_base
    kb = dict(comparison_key_base(None, K.support_key_standin(rep), rep))
    kb.update(split="VALIDATION", eval_manifest_id="MAIN_VALIDATION_5_100_131072", candidate_order_hash="VAL_CAND",
              initialization_block="SYN_THETA0", topology_hash="SYN_TINY", objective_id="SYN_CE", precision_id="FP32")
    return kb                                         # a harness key base (the model fields a trainer key carries)


def _reeval_args(world, role="MAIN_TEST", **kw):
    from v4int import reeval as R
    a = {"--ckpt": str(world.root / "x.pt"), "--ckpt-kind": "central", "--expect-sha256": "0" * 64,
         "--variant": "SASREC_D64_B2", "--recipe-id": "SMALL_SASREC_D64_B2", "--seed": "2026", "--role": role,
         "--arm-label": "C_SMALL", "--selection-rule": "PRACTICAL_BEST", "--checkpoint-id": "best", "--gpu": "0",
         "--out-dir": str(world.root / "out"), "--history-policy": USER, "--registry-ext": str(world.ext_path)}
    a.update(kw)
    return R.parse_args([x for kv in a.items() for x in kv])


def test_reeval_test_role_rows_carry_retro_test(shim, rep, monkeypatch):
    from v4data09 import d9_common as D9
    from v4eval import evaluator_code_sha256
    from v4eval.evaluate import evaluate
    from v4eval.predict import predict_adapter
    from v4int import reeval as R
    from v4int import retro_store as RS
    monkeypatch.setattr(RS, "train_key_ref", lambda mid, **kw: K.support_key_standin(rep, mid))
    args = _reeval_args(shim)
    reg_kw = dict(history_policy=USER, paths=None, log=Q, manifest_registry=shim.ext_path, registry_root=shim.root,
                  ds_root=shim.ds, buckets=[0])
    base_kb = _val_key(rep)
    rep2, kb, over, key_train = R.evaluated_test_role(args, gate=shim.gate, pair={"train": MAIN_TRAIN,
                                                                                   "eval": D9.MAIN_TEST_ID},
                                                      reg_kw=reg_kw, paths=None, base_kb=base_kb,
                                                      main_train_id=MAIN_TRAIN)
    assert rep2.fingerprint == rep.fingerprint and key_train["key_only"] is True
    assert kb["split"] == "RETRO_TEST" and kb["eval_manifest_id"] == D9.MAIN_TEST_ID
    assert kb["candidate_order_hash"] == rep.manifest.header["candidate_order_hash"]
    assert kb["data_protocol_hash"] == base_kb["data_protocol_hash"]
    fz, ad = _frozen(), _tiny()
    h = R.central_harness(ad, frozen=fz, device=torch.device("cpu"), key_base=kb, labels={}, seed=2026,
                          evaluator_sha=evaluator_code_sha256(), evalp=rep2)
    _v, rows, ranks, res = R.evaluate_role(h, rep2, kb, over)
    art = predict_adapter(ad, rep2.batches(fz.eval_batch), rep2.manifest, config=fz, model_sha256="x")
    assert np.array_equal(ranks, art.ranks)
    ref = evaluate(rep2.manifest, art, config=fz, evaluator_code_sha256=evaluator_code_sha256())
    assert ref["E2E"]["macro"]["mrr@20"] == res["E2E"]["macro"]["mrr@20"]
    rows = R.relabel(rows, arm_label="C_SMALL", selection_rule="PRACTICAL_BEST", checkpoint_id="best",
                     checkpoint_sha256="s", kind="central", source_run_id="C_SMALL_s2026", efe=6.0)
    R.check_test_rows(rows)
    assert {r["split"] for r in rows} == {"RETRO_TEST"} and {r["manifest_role"] for r in rows} == {"MAIN_TEST"}
    assert {r["test_exposure_register_entry_id"] for r in rows} == {shim.gate.entry_id}
    assert {r["eval_manifest_content_sha256"] for r in rows} == {rep.manifest_binding["content_sha256"]}
    with pytest.raises(R.ReevalRefused):
        R.check_test_rows([dict(rows[0], split="VALIDATION")])


def test_reeval_eval_only_test_role_binds_the_source_train_manifest(shim, rep, monkeypatch):
    from v4data09 import d9_common as D9
    from v4int import reeval as R
    from v4int import retro_store as RS
    monkeypatch.setattr(RS, "train_key_ref", lambda mid, **kw: K.support_key_standin(rep, mid))
    reg_kw = dict(history_policy=USER, paths=None, log=Q, manifest_registry=shim.ext_path, registry_root=shim.root,
                  ds_root=shim.ds, buckets=[0])
    kw = dict(gate=shim.gate, pair={"eval": D9.TRUE_COLD_TEST_ID}, reg_kw=reg_kw, paths=None, base_kb=_val_key(rep),
              main_train_id=MAIN_TRAIN)
    args = _reeval_args(shim, role="TRUE_COLD_TEST")
    _rep, kb, over, kt = R.evaluated_test_role(args, source_train_binding={"content_sha256": "SYNTHETIC_SUPPORT"}, **kw)
    assert kb["split"] == "RETRO_TEST" and kb["train_manifest_id"] == MAIN_TRAIN and over["manifest_role"] == \
        "TRUE_COLD_TEST"
    with pytest.raises(R.ReevalRefused, match="source run's train manifest"):
        R.evaluated_test_role(args, source_train_binding={"content_sha256": "OTHER"}, **kw)


def test_device_arm_rows_on_retro_test(shim, rep, tmp_path):
    import v4int.launch_device as D
    from fl.numerics import state_digest
    from v4eval import evaluator_code_sha256
    from v4eval.predict import predict_adapter
    from v4int import retro_store as RS
    from v4int.providers import comparison_key_base
    kb, over = RS.retro_role_key(_val_key(rep), comparison_key_base(None, K.support_key_standin(rep), rep), shim.gate,
                                 role="MAIN_TEST", eval_manifest_content_sha256=rep.manifest_binding["content_sha256"])
    row_key, row_dts, _hex = D.eval_row_arrays(rep, shim.mdir)
    plans = D.build_client_plans(rep.manifest, row_key, row_decision_ts=row_dts, support_pos_by_key=None,
                                 warm="MAIN_TEST" in D.WARM_ROLES)
    worker = _tiny()
    start = worker.broadcast_state(clone=True)
    spec = D.device_spec("LO", seed=2026, peak_lr=1e-2, role="MAIN_TEST", start_digest=state_digest(start))
    fz = _frozen()

    def row_fn(res, cell, selected):
        return D.device_rows(res, kb, {}, spec=spec, cell=cell, selected=selected, down_bytes=0,
                             overrides=dict(over, device_run_id="DEV_LO_MAIN_TEST_s2026"))

    out = D.execute(spec=spec, run_dir=tmp_path, worker=worker, start_state=start, plans=plans,
                    get_support=lambda p: None, get_eval_batch=lambda sp: D.eval_batch_at(rep, sp),
                    manifest=rep.manifest, frozen=fz, row_fn=row_fn, evaluator_sha=evaluator_code_sha256(),
                    n_shards=2, say=Q, eval_manifest_content_sha256=rep.manifest_binding["content_sha256"])
    assert out["status"] == "DONE"
    rows = json.loads((tmp_path / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert rows and {r["split"] for r in rows} == {"RETRO_TEST"} and {r["manifest_role"] for r in rows} == {"MAIN_TEST"}
    assert {r["eval_manifest_id"] for r in rows} == {rep.manifest_id}
    assert {r["test_exposure_register_entry_id"] for r in rows} == {shim.gate.entry_id}
    worker.load_state_(start)
    worker.module.eval()
    art = predict_adapter(worker, rep.batches(fz.eval_batch), rep.manifest, config=fz, model_sha256="x")
    got = np.load(tmp_path / "cells" / "LO_P1.ranks_int32.npy")
    assert np.array_equal(got[rep.manifest.rankable], art.ranks[rep.manifest.rankable].astype(np.int32))


def test_device_test_cutoff_rule(rep, shim):
    import v4int.launch_device as D
    from v4data.common import TRAIN_END_US
    row_key, row_dts, _hex = D.eval_row_arrays(rep, shim.mdir)
    sup = {str(k): np.arange(2) for k in set(row_key.tolist())}
    warm = D.build_client_plans(rep.manifest, row_key, row_decision_ts=row_dts, support_pos_by_key=sup, warm=True)
    cold = D.build_client_plans(rep.manifest, row_key, row_decision_ts=row_dts, support_pos_by_key=sup, warm=False)
    for p in warm:
        assert p.cutoff == TRAIN_END_US                              # MAIN_TEST: min(earliest RETRO_TEST ts, TRAIN_END)
    for p in cold:                                                   # the other TEST roles: the earliest RETRO_TEST ts
        assert p.cutoff == int(row_dts[row_key == p.key].min()) and p.cutoff > TRAIN_END_US


def test_launch_device_test_role_rules():
    import v4int.launch_device as D
    from v4int.launch_central import LaunchRefused
    assert "MAIN_TEST" in D.WARM_ROLES and set(D.TEST_ROLES) == {"MAIN_TEST", "UNSEEN_HOLDOUT_TEST", "SPARSE_PANEL_TEST",
                                                                 "HEAVY_PANEL_TEST", "TRUE_COLD_TEST"}
    for role in D.TEST_ROLES:
        D.check_role(role, select=False, arm="LO", selection=None)
        D.check_role(role, select=False, arm="FT_C", selection={"arm": "FT_C"})
        with pytest.raises(LaunchRefused, match="--select"):
            D.check_role(role, select=True, arm="LO", selection=None)
        with pytest.raises(LaunchRefused, match="needs --selection"):
            D.check_role(role, select=False, arm="FT_FA", selection=None)
    with pytest.raises(LaunchRefused, match="not a registered TEST role"):
        D.check_role("MAIN_RETRO_TEST", select=False, arm="LO", selection=None)


def test_baseline_rows_on_retro_test(shim, rep, tmp_path):
    from v4base import run as BR
    from v4int.providers import comparison_key_base
    key = comparison_key_base(None, K.support_key_standin(rep), rep)
    out = BR.execute("LAST", train_provider=None, eval_provider=rep, key_base=key, out_dir=tmp_path,
                     K=int(rep.manifest.catalogue_K), receipt_extra={"phase": "MAIN_TEST",
                                                                     "test_gate": shim.gate.binding()})
    assert out["rows"] and {r["split"] for r in out["rows"]} == {"RETRO_TEST"}
    assert {r["candidate_order_hash"] for r in out["rows"]} == {rep.manifest.header["candidate_order_hash"]}
    rec = json.loads((tmp_path / "RECEIPT.json").read_text(encoding="utf-8"))
    assert rec["phase"] == "MAIN_TEST" and rec["test_gate"]["test_exposure_register_entry_id"] == shim.gate.entry_id


def test_dump_rows_retro_writes_the_retro_test_rows(shim, rep, tmp_path):
    from v4base.identity import eval_client_key_hex_by_eval_user_key
    from v4dev.dump_rows import dump_from_provider, write_rows
    from v4eval.predict import stream_rows
    from v4int import dump_rows_retro as DR
    p, meta = DR.build_retro_eval_provider("MAIN_TEST", USER, ext_path=shim.ext_path, lock_path=shim.lock,
                                           register_path=shim.register, base_path=shim.base_path,
                                           registry_root=shim.root, ds_root=shim.ds, buckets=[0], log=Q)
    assert p.fingerprint == rep.fingerprint and meta["split"] == "RETRO_TEST"
    assert meta["test_exposure_register_entry_id"] == shim.gate.entry_id and Path(meta["manifest_dir"]) == shim.mdir
    hexes = eval_client_key_hex_by_eval_user_key(p, manifest_dir=meta["manifest_dir"])
    view = DR.ArrayFieldsView(p)
    rows, m = dump_from_provider(view, first_users=2, client_key_hex=hexes)
    m.update(meta, role="MAIN_TEST", dropped_non_array_fields=sorted(view.dropped))
    res = write_rows(rows, m, tmp_path / "rows.npz")
    assert res["meta"]["n_users"] == 2 and res["meta"]["manifest_id"] == rep.manifest_id
    assert {"item_tokens", "decision_id", "target_class", "session_offset_tokens"} <= set(rows)
    assert "feature_view" in view.dropped
    full, _ = dump_from_provider(DR.ArrayFieldsView(p))
    assert np.array_equal(full["manifest_row"], stream_rows(p.manifest))
    assert np.array_equal(full["decision_id"], p.manifest.decision_id[stream_rows(p.manifest)])


def test_dump_retro_records_the_real_batch_view_and_keeps_the_rows(shim, rep, tmp_path):
    """DEV-05: on the real RETRO_TEST provider, dump_retro's rows equal the r6 composition (ArrayFieldsView ->
    dump_from_provider) array for array, and meta.batch_view is the provider's own Clean V2 view (base names in order)."""
    from v4dev.checkpointio import load_schema
    from v4dev.dump_rows import dump_from_provider
    from v4int import dump_rows_retro as DR
    from v4models.schema import NAME_KEYS
    p, meta = DR.build_retro_eval_provider("MAIN_TEST", USER, ext_path=shim.ext_path, lock_path=shim.lock,
                                           register_path=shim.register, base_path=shim.base_path,
                                           registry_root=shim.root, ds_root=shim.ds, buckets=[0], log=Q)
    rows, m = DR.dump_retro(p, meta, "MAIN_TEST")
    ref, _ = dump_from_provider(DR.ArrayFieldsView(p))
    assert sorted(rows) == sorted(ref) and all(np.array_equal(rows[k], ref[k]) and rows[k].dtype == ref[k].dtype
                                               for k in ref)
    base = load_schema()
    bw = base.widths
    want = {"feature_view": bw.view_id, "clean_v2_schema_self_sha256": base.self_sha256}
    want.update({NAME_KEYS[k]: list(bw.names_of(k)) for k in NAME_KEYS if bw.names_of(k)})
    assert m["batch_view"] == want and m["split"] == "RETRO_TEST" and "feature_view_binding" not in m


@nc("a row keyed VALIDATION passes the TEST-row guard of reeval")
def test_nc_validation_row_passes_the_test_row_guard():
    from v4int import reeval as R
    try:
        R.check_test_rows([{"split": "VALIDATION", "test_exposure_register_entry_id": 2}])
        passed = True
    except R.ReevalRefused:
        passed = False
    assert passed
