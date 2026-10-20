"""ANA-06 — v4int.reeval and the FL-16 warm-line arms (CPU dry runs on synthetic FL run dirs; no data, no GPU):

  * a warm run (LAUNCH arm a launch_fl warm label, or arm_spec init_from) is re-evaluated only under its block arm
    (DP_WARM_FROZEN eps 8 = DP8_FROZEN, eps 1 = DP1_WARM_FROZEN, DP_WARM eps 8 / 1 = DP8_WARM / DP1_WARM, the other
    labels as is), a warm block label only on that run; theta_pre init_from and S8 frozen tables are required;
  * PF_WARM / PF_WARM_FROZEN are PF runs (--ckpt-kind pf); S_CAL_WARM* refuse;
  * a warm-line dry run carries `warm_line`; cold runs and FA_WARM keep their pre-ANA-06 dry-run record (no new key).
Negative controls (test_nc_*, strict xfail on AssertionError)."""
from __future__ import annotations

from types import SimpleNamespace

import v4int.launch_fl as F
from intkit import nc
from test_int06_reeval import _argv, _fl_run, _store_file
from v4int import reeval as RV

INIT = {"kind": "PRETRAIN_FOR_FL", "run_id": "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026"}
FROZEN = list(F.FROZEN_ITEM_KEYS)


def warm(tmp, arm, *, eps=None, init=True, frozen=None, **over):
    spec = {"cfg": {"seed": 2026}, "arm": arm, **({"init_from": INIT} if init else {})}
    fz = F.is_frozen_arm(arm) if frozen is None else frozen
    if fz:
        spec["frozen_item_tables"] = FROZEN
    kw = {"arm_spec": spec, **({"dp_z": {"epsilon": eps}} if eps is not None else {}), **over}
    return _fl_run(tmp, arm=arm, **kw)


def dry(cp, tmp, label, kind="fl", **kw):
    return RV.run(_argv(cp, tmp, kind=kind, arm_label=label, **kw))


def refused(fn, msg) -> None:
    """The refusal must happen with `msg` (an AssertionError otherwise, so the negative controls fail at an assert)."""
    got = None
    try:
        fn()
    except RV.ReevalRefused as e:
        got = str(e)
    assert got is not None and msg in got, (msg, got)


def test_warm_line_labels_admitted(tmp_path):
    cases = [("FP_WARM_FROZEN", None, "FP_WARM_FROZEN"), ("FA_Q8_WARM", None, "FA_Q8_WARM"),
             ("FA_1024_WARM_FROZEN", None, "FA_1024_WARM_FROZEN"), ("DP_WARM_FROZEN", 8.0, "DP8_FROZEN"),
             ("DP_WARM_FROZEN", 1.0, "DP1_WARM_FROZEN"), ("DP_WARM", 8.0, "DP8_WARM"), ("DP_WARM", 1.0, "DP1_WARM")]
    for i, (arm, eps, label) in enumerate(cases):
        d = tmp_path / f"w{i}"
        cp, rid, _ = warm(d, arm, eps=eps)
        got = dry(cp, d, label)
        wl = got["warm_line"]
        assert got["dry_run"] and wl["launch_arm"] == arm and wl["block_arm"] == label and wl["epsilon"] == eps
        assert wl["variant"] == F.warm_variant(arm) and wl["frozen_item_tables"] == F.is_frozen_arm(arm)


def test_pf_warm_is_a_pf_run(tmp_path):
    cp, rid, theta = warm(tmp_path, "PF_WARM_FROZEN")
    refused(lambda: dry(cp, tmp_path, "PF_WARM_FROZEN"), "--ckpt-kind pf")
    from fl.numerics import state_digest
    store = _store_file(tmp_path, rid, theta_digest=state_digest(theta))
    got = dry(cp, tmp_path, "PF_WARM_FROZEN", kind="pf", pf_store=store)
    assert got["warm_line"]["cold_arm"] == "PF"


@nc("the pre-ANA-06 pf rule (arm == 'PF') evaluates PF_WARM without its personal store")
def test_nc_pf_rule_cold_only(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "base_arm", lambda a: a)
    test_pf_warm_is_a_pf_run(tmp_path)


def check_warm_refusals(tmp_path):
    cases = [(dict(arm="FP_WARM_FROZEN"), "FP", "keeps its own label"),                 # a warm run as the cold arm
             (dict(arm="DP_WARM_FROZEN", eps=8.0), "DP1_WARM_FROZEN", "keeps its own label"),
             (dict(arm="DP_WARM_FROZEN", eps=8.0), "DP8", "keeps its own label"),
             (dict(arm="FA_WARM"), "FA", "keeps its own label"),                        # FA_WARM as FA on MAIN
             (dict(arm="FP_WARM", init=False), "FP_WARM", "no init_from"),
             (dict(arm="FP_WARM_FROZEN", frozen=False), "FP_WARM_FROZEN", "frozen_item_tables"),
             (dict(arm="FA_Q8_WARM", frozen=True), "FA_Q8_WARM", "frozen_item_tables"),
             (dict(arm="S_CAL_WARM"), "S_CAL_WARM", "keeps its own label"),
             (dict(arm="DP_WARM", eps=4.0), "DP8_WARM", "keeps its own label")]
    for i, (kw, label, msg) in enumerate(cases):
        d = tmp_path / f"r{i}"
        cp, _, _ = warm(d, **kw)
        refused(lambda: dry(cp, d, label), msg)
    d = tmp_path / "cold"                                                             # a cold FP run as FP_WARM
    cp, _, _ = _fl_run(d, arm="FP")
    refused(lambda: dry(cp, d, "FP_WARM"), "keeps its own label")


def test_warm_line_refusals(tmp_path):
    check_warm_refusals(tmp_path)


@nc("a reeval without the warm-line label binding (the pre-ANA-06 reeval)")
def test_nc_warm_label_unbound(tmp_path, monkeypatch):
    monkeypatch.setattr(RV, "check_warm_line", lambda args, src: None)
    check_warm_refusals(tmp_path)


def test_cold_and_fa_warm_dry_runs_unchanged(tmp_path):
    cp, rid, _ = _fl_run(tmp_path / "c")
    got = dry(cp, tmp_path / "c", "FA")
    assert set(got) == {"dry_run", "checkpoint_sha256", "checkpoint_kind", "source_run", "source_run_id", "fl_round",
                        "fl_efe"}
    cp, _, _ = warm(tmp_path / "w", "FA_WARM")
    assert "warm_line" not in dry(cp, tmp_path / "w", "FA_WARM")
    assert RV.check_warm_line(SimpleNamespace(ckpt_kind="central", arm_label="PRE"), {}) is None
