"""INT-07 additions to the DATA-09 suite: the gated RETRO_TEST store / window path (v4int.retro_store, the gated branch of
v4int.cache.build_subset_store, the RETRO_TEST branch of v4int.providers) and every TEST consumer (reeval, launch_device,
v4base.run, v4int.dump_rows_retro, launch_fl) refuse without an open TEST gate - before any dataset / manifest file is
opened (the dataset paths below do not exist) - and pass the gate check under the synthetic tmp gate; importing the TEST
path has no side effect on the locked refusals. SYNTHETIC only: no dataset is opened at all. Strict-xfail negative
controls (test_nc_*) prove the checks bite.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import closed_gate_only, make_gate_files
from data_testkit import assert_raises, nc
from v4data.common import TestGateError

ABSENT = Path("Z:/int07_absent_dataset_root")
TEST_IDS = np.array([11, 12, 13], dtype=np.int64)


@pytest.fixture
def open_gate(gate_files, monkeypatch):
    """The synthetic tmp gate as the DEFAULT gate (d9_gate DEFAULT_LOCK / DEFAULT_REGISTER), so that every consumer's
    own default-path gate check sees an open gate."""
    import v4data09.d9_gate as G
    monkeypatch.setattr(G, "DEFAULT_LOCK", Path(gate_files[0]))
    monkeypatch.setattr(G, "DEFAULT_REGISTER", Path(gate_files[1]))
    return G.check_test_gate()


def _forged(g):
    import dataclasses
    return dataclasses.replace(g, entry_sha256="0" * 64)


# ------------------------------------------------------------------------------------------------ the store path
def test_require_gate_refusals(synth_gate, tmp_path):
    from v4data09.d9_gate import check_test_gate
    from v4int import retro_store as RS
    assert RS.require_gate(synth_gate) is synth_gate
    for bad in (None, "gate", {"entry_id": 2}):
        assert_raises(RS.TestPathRefused, RS.require_gate, bad)
    assert_raises(TestGateError, RS.require_gate, _forged(synth_gate))
    lock, reg = make_gate_files(tmp_path)
    g = check_test_gate(lock, reg)
    lock.unlink()
    assert_raises(TestGateError, RS.require_gate, g)


def test_gated_history_rule(synth_gate):
    from v4hist.policy import HistoryPolicyError, history_splits
    from v4int import retro_store as RS
    for ds in (("TRAIN",), ("VALIDATION_SELECT",), ("VALIDATION_CONFIRM",), ("VALIDATION_SELECT", "VALIDATION_CONFIRM")):
        assert RS.gated_history_splits(synth_gate, ds) == history_splits(ds)
    assert RS.gated_history_splits(synth_gate, ("TEST",)) == ("TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM", "TEST")
    assert RS.gated_store_splits(synth_gate, "SESSION_L50", ("TEST",)) == ("TEST",)
    for bad in (("RETRO_TEST",), ("LABEL_GRACE",), ()):
        assert_raises(HistoryPolicyError, RS.gated_history_splits, synth_gate, bad)
    assert_raises(RS.TestPathRefused, RS.gated_history_splits, None, ("TEST",))


def test_every_store_entry_point_refuses_before_opening_a_file(synth_gate):
    from v4int import retro_store as RS
    from v4int.cache import build_subset_store
    for g in (None, _forged(synth_gate)):
        for fn in (lambda: RS.GatedCleanV2Store.build(gate=g, splits=RS.TEST_ORDER, buckets=[0], ds_root=ABSENT),
                   lambda: RS.build_gated_row_index(g, ABSENT, [0], RS.TEST_ORDER),
                   lambda: RS.gated_path_builders(g, "USER_HISTORY_L50", RS.TEST_DECISION_SPLITS),
                   lambda: build_subset_store(TEST_IDS, splits=RS.TEST_DECISION_SPLITS, kind="eval", state_sha256=None,
                                              history_policy="USER_HISTORY_L50", ds_root=ABSENT, buckets=[0],
                                              test_gate=g)):
            assert_raises(TestGateError, fn)


def test_gated_cache_branch_serves_the_test_decision_rule_only(synth_gate):
    from v4int import retro_store as RS
    from v4int.cache import build_subset_store
    for splits in (("VALIDATION_SELECT", "VALIDATION_CONFIRM"), ("TEST",), ("TRAIN",)):
        assert_raises(RS.TestPathRefused, build_subset_store, TEST_IDS, splits=splits, kind="eval", state_sha256=None,
                      history_policy="USER_HISTORY_L50", ds_root=ABSENT, buckets=[0], test_gate=synth_gate)
    ss, *_fns = RS.gated_path_builders(synth_gate, "USER_HISTORY_L50", RS.TEST_DECISION_SPLITS)
    assert ss == RS.TEST_ORDER


def test_default_paths_still_refuse_test_after_importing_the_test_path(synth_gate):
    import v4hist.policy as HP
    import v4hist.windows as HW
    import v4int.cache as CA
    from v4data.store import CleanV2Store
    before = (HP.history_splits, HP.store_splits, HW.build_row_index, CA.refuse_test_split, CA.store_splits,
              CA.build_row_index, vars(CleanV2Store)["build"], vars(CleanV2Store)["decision_set"],
              vars(CleanV2Store)["__init__"])
    from v4int import retro_store as RS  # noqa: F401
    after = (HP.history_splits, HP.store_splits, HW.build_row_index, CA.refuse_test_split, CA.store_splits,
             CA.build_row_index, vars(CleanV2Store)["build"], vars(CleanV2Store)["decision_set"],
              vars(CleanV2Store)["__init__"])
    assert all(a is b for a, b in zip(before, after))
    assert_raises(HP.HistoryPolicyError, HP.history_splits, ("TEST",))
    assert_raises(TestGateError, CleanV2Store.build, splits=("TEST",), ds_root=ABSENT)
    assert_raises(TestGateError, CA.build_subset_store, TEST_IDS, splits=RS.TEST_DECISION_SPLITS, kind="eval",
                  state_sha256=None, history_policy="USER_HISTORY_L50", ds_root=ABSENT, buckets=[0])
    assert_raises(TestGateError, HW.build_row_index, ABSENT, [0], ("TEST",))


def test_retro_test_provider_refused_without_the_shim_and_key_split_rules(synth_gate):
    from v4int import retro_store as RS
    from v4int.providers import ProviderError, StoreEvalProvider, _key_split
    from v4int import test_roles as TR
    TR.uninstall()
    assert_raises(TestGateError, StoreEvalProvider, "MAIN_RETRO_TEST_5_100_131072", history_policy="USER_HISTORY_L50")
    hdr = lambda s: SimpleNamespace(header={"v4_split": s})  # noqa: E731
    assert _key_split(SimpleNamespace(manifest=hdr("VALIDATION"), test_gate=None)) == "VALIDATION"
    assert _key_split(SimpleNamespace(manifest=hdr("RETRO_TEST"), test_gate=synth_gate)) == "RETRO_TEST"
    assert_raises(ProviderError, _key_split, SimpleNamespace(manifest=hdr("RETRO_TEST"), test_gate=None))
    assert_raises(ProviderError, _key_split, SimpleNamespace(manifest=hdr("VALIDATION"), test_gate=synth_gate))
    assert_raises(TestGateError, _key_split, SimpleNamespace(manifest=hdr("RETRO_TEST"), test_gate=_forged(synth_gate)))
    h = {"manifest_id": "MAIN_RETRO_TEST_5_100_131072", **synth_gate.header_fields()}
    assert RS.gate_for_provider(h, synth_gate) is not None
    assert_raises(RS.TestPathRefused, RS.gate_for_provider, dict(h, test_exposure_register_entry_id=9), synth_gate)
    assert_raises(RS.TestPathRefused, RS.gate_for_provider, h, _forged(synth_gate))


def test_retro_role_key_rule(synth_gate):
    from v4int import retro_store as RS
    base = {"population_hash": "P0", "candidate_order_hash": "C0", "train_manifest_id": "T0", "eval_manifest_id": "E0",
            "split": "VALIDATION", "manifest_registry_self_sha256": "R0", "data_protocol_hash": "D"}
    rk = {"population_hash": "P1", "candidate_order_hash": "C1", "train_manifest_id": "T1", "eval_manifest_id": "E1",
          "split": "RETRO_TEST", "manifest_registry_self_sha256": "R1", "data_protocol_hash": "OTHER"}
    kb, over = RS.retro_role_key(base, rk, synth_gate, role="MAIN_TEST", eval_manifest_content_sha256="c")
    assert kb == dict(rk, data_protocol_hash="D")
    assert over["manifest_role"] == "MAIN_TEST" and over["test_exposure_register_entry_id"] == synth_gate.entry_id
    assert over["v4_split"] == "RETRO_TEST" and over["eval_manifest_content_sha256"] == "c"
    assert_raises(RS.TestPathRefused, RS.retro_role_key, base, dict(rk, split="VALIDATION"), synth_gate,
                  role="MAIN_TEST", eval_manifest_content_sha256="c")
    assert_raises(RS.TestPathRefused, RS.retro_role_key, base, rk, None, role="MAIN_TEST",
                  eval_manifest_content_sha256="c")


# ------------------------------------------------------------------------------------------------ consumers
def _reeval_args(role, *extra):
    from v4int import reeval as R
    return R.parse_args(["--ckpt", str(ABSENT / "best.pt"), "--ckpt-kind", "central", "--expect-sha256", "0" * 64,
                         "--variant", "SASREC_D64_B2", "--recipe-id", "SMALL_SASREC_D64_B2", "--seed", "2026",
                         "--role", role, "--arm-label", "C_SMALL", "--selection-rule", "PRACTICAL_BEST",
                         "--checkpoint-id", "best", "--gpu", "0", "--out-dir", str(ABSENT / "out"),
                         "--history-policy", "USER_HISTORY_L50", *extra])


def _device_args(role, *extra):
    import v4int.launch_device as D
    return D.parse_args(["--arm", "LO", "--seed", "2026", "--gpu", "0", "--run-dir", str(ABSENT / "run"),
                         "--history-policy", "USER_HISTORY_L50", "--role", role, "--smoke", "--dry-run", *extra])


@closed_gate_only
def test_every_consumer_refuses_a_test_role_with_the_real_gate_closed(capsys):
    import v4int.launch_device as D
    import v4int.launch_fl as F
    from v4base import run as BR
    from v4int import dump_rows_retro as DR
    from v4int import reeval as R
    from v4int.launch_central import LaunchRefused
    for role in ("MAIN_TEST", "UNSEEN_HOLDOUT_TEST", "SPARSE_PANEL_TEST", "HEAVY_PANEL_TEST", "TRUE_COLD_TEST"):
        assert R.test_gate_ok(role) is False
        with pytest.raises(R.ReevalRefused, match="TEST role"):
            R.run(_reeval_args(role, "--registry-ext", str(ABSENT / "ext.json")))
        with pytest.raises(LaunchRefused, match="TEST gate is closed"):
            D.preflight(_device_args(role))
    assert R.test_gate_ok("MAIN") is True
    with pytest.raises(BR.TestPhaseRefused, match="gate is closed"):
        BR.check_test_phase("MAIN_TEST", grid=False)
    BR.check_test_phase("MAIN", grid=True)                                  # not a TEST phase: untouched
    assert DR.main(["--role", "MAIN_TEST", "--history-policy", "USER_HISTORY_L50", "--out", str(ABSENT / "r.npz")]) == 2
    assert "gate is closed" in capsys.readouterr().err
    with pytest.raises(LaunchRefused, match="--role must be one of"):          # delegated to FL-09 (launch_fl owner)
        F.fl05_preflight(SimpleNamespace(role="MAIN_TEST", arm="FA", seed=2026, lr_mult=None, lr_selection=None))


def test_consumers_pass_the_gate_check_under_the_synthetic_gate(open_gate, capsys):
    import v4int.launch_device as D
    from v4base import run as BR
    from v4int import dump_rows_retro as DR
    from v4int import reeval as R
    from v4int.launch_central import LaunchRefused
    assert R.test_gate_ok("MAIN_TEST") is True and R.test_gate_ok("NOT_A_ROLE_TEST") is False
    R.check_args(_reeval_args("MAIN_TEST", "--registry-ext", str(ABSENT / "ext.json")))
    with pytest.raises(R.ReevalRefused, match="--registry-ext"):
        R.check_args(_reeval_args("MAIN_TEST"))
    with pytest.raises(R.ReevalRefused, match="TEST role"):
        R.check_args(_reeval_args("MAIN_RETRO_TEST", "--registry-ext", str(ABSENT / "ext.json")))
    rec = D.preflight(_device_args("SPARSE_PANEL_TEST"))
    assert rec["test_gate"] == {"required": True, "role": "SPARSE_PANEL_TEST"}
    with pytest.raises(LaunchRefused, match="--select"):
        D.preflight(_device_args("MAIN_TEST", "--select"))
    with pytest.raises(LaunchRefused, match="TEST roles only"):
        D.preflight(_device_args("MAIN", "--registry-ext", str(ABSENT / "ext.json")))
    BR.check_test_phase("MAIN_TEST", grid=False)
    for kw in (dict(grid=True), dict(grid=False, train_manifest="X"), dict(grid=False, eval_manifest="Y")):
        with pytest.raises(BR.TestPhaseRefused):
            BR.check_test_phase("MAIN_TEST", **kw)
    DR.check_role("MAIN_TEST")
    assert_raises(DR.DumpRowsRetroRefused, DR.check_role, "UNSEEN_HOLDOUT_TEST")


def test_reeval_test_role_needs_a_registered_role_even_under_the_gate(open_gate):
    from v4int import reeval as R
    assert "TEST" not in "".join(R.ROLES) and set(R.TEST_ROLES) == {
        "MAIN_TEST", "UNSEEN_HOLDOUT_TEST", "SPARSE_PANEL_TEST", "HEAVY_PANEL_TEST", "TRUE_COLD_TEST"}
    base = {"MAIN": {"train": "T", "eval": "E"}}
    with pytest.raises(R.ReevalRefused, match="TEST manifest"):
        R.resolve_role(dict(base, MAIN_RETRO_TEST={"eval": "MAIN_RETRO_TEST_5_100_131072"}), "MAIN_RETRO_TEST", base)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a forged GateRecord must not open the RETRO_TEST store path")
def test_nc_forged_gate_opens_the_store_path(synth_gate):
    from v4int import retro_store as RS
    try:
        RS.require_gate(_forged(synth_gate))
        opened = True
    except TestGateError:
        opened = False
    assert opened


@nc("the locked history rule must still refuse a TEST decision after the TEST path is imported")
def test_nc_history_splits_accepts_test_after_import():
    from v4hist.policy import HistoryPolicyError, history_splits
    from v4int import retro_store as RS  # noqa: F401
    try:
        history_splits(("TEST",))
        accepted = True
    except HistoryPolicyError:
        accepted = False
    assert accepted


@nc("reeval must refuse a TEST role while the real gate is closed")
@closed_gate_only
def test_nc_reeval_accepts_a_test_role_with_the_gate_closed():
    from v4int import reeval as R
    assert R.test_gate_ok("MAIN_TEST")


@nc("the gated cache branch must refuse the VALIDATION decision rule")
def test_nc_gated_cache_serves_validation_decisions(synth_gate):
    from v4int import retro_store as RS
    try:
        RS.gated_path_builders(synth_gate, "USER_HISTORY_L50", ("VALIDATION_SELECT", "VALIDATION_CONFIRM"))
        served = True
    except RS.TestPathRefused:
        served = False
    assert served
