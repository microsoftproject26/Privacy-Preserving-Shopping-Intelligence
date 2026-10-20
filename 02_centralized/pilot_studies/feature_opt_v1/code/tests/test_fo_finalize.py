"""finalize_fo.py (owner: Opus) -- gate 2(b), hygiene_parity(). Never modifies the code under
test. Every write goes through FO_ROOT_OVERRIDE tmp dirs via the set_fo_root fixture.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import conftest_fo  # noqa: F401
from conftest_fo import set_fo_root  # noqa: F401

import feature_view as fv
import finalize_fo as ff


# =========================================================================== hygiene_parity (pure function)
def _ok_lean_dupf():
    lean = {"max_abs_logit_diff_first_1024": 1e-5, "ranks_identical_fraction": 0.9999}
    zero_checks = {"a==b": 0, "local_hour_rotation_max_residual": 0.0}
    dupf = {"label": "FULL_POPULATION", "checks": {"TRAIN": dict(zero_checks), "VALIDATION_SELECT": dict(zero_checks)}}
    return lean, dupf


def test_hygiene_parity_ok_when_everything_holds():
    lean, dupf = _ok_lean_dupf()
    p = ff.hygiene_parity(lean, dupf)
    assert p["status"] == "PARITY_OK"
    assert all(p["checks"].values())


def _assert_only_this_check_false(p, key):
    assert p["status"] == "PARITY_FAILED"
    assert p["checks"][key] is False
    others = {k: v for k, v in p["checks"].items() if k != key}
    assert all(others.values()), f"expected only {key} False, got {p['checks']}"


def test_hygiene_parity_negative_control_label_not_full_population():
    lean, dupf = _ok_lean_dupf()
    dupf["label"] = "SAMPLE_NOT_POPULATION"
    p = ff.hygiene_parity(lean, dupf)
    _assert_only_this_check_false(p, "duplicates_full_population_label")


def test_hygiene_parity_negative_control_nonzero_duplicate_count():
    lean, dupf = _ok_lean_dupf()
    dupf["checks"]["TRAIN"]["a==b"] = 3
    p = ff.hygiene_parity(lean, dupf)
    _assert_only_this_check_false(p, "all_duplicate_and_dead_counts_zero")
    assert p["nonzero_counts"] == {"TRAIN:a==b": 3}


def test_hygiene_parity_negative_control_rotation_residual_too_large():
    lean, dupf = _ok_lean_dupf()
    dupf["checks"]["VALIDATION_SELECT"]["local_hour_rotation_max_residual"] = 5e-6
    p = ff.hygiene_parity(lean, dupf)
    _assert_only_this_check_false(p, "rotation_residual_le_1e-6")


def test_hygiene_parity_negative_control_max_abs_logit_too_large():
    lean, dupf = _ok_lean_dupf()
    lean["max_abs_logit_diff_first_1024"] = 5e-3
    p = ff.hygiene_parity(lean, dupf)
    _assert_only_this_check_false(p, "max_abs_logit_le_1e-3")


def test_hygiene_parity_negative_control_rank_identical_fraction_too_low():
    lean, dupf = _ok_lean_dupf()
    lean["ranks_identical_fraction"] = 0.5
    p = ff.hygiene_parity(lean, dupf)
    _assert_only_this_check_false(p, "rank_identical_ge_0.999")


# =========================================================================== finalize_fo.main() end-to-end
N = 10


def _write_fixture(root, *, sigma_hat, arms, corrupt_decision_id_arm=None, missing_lean=False):
    for d in ("config", "audit", "state"):
        (root / d).mkdir(exist_ok=True)
    (root / "runs" / "probes").mkdir(parents=True, exist_ok=True)

    decision_id = np.arange(500, 500 + N, dtype=np.int64)
    target_class = np.full(N, 5, dtype=np.int64)
    eval_mask = np.ones(N, dtype=bool)
    eval_user_key = np.arange(N, dtype=np.int64)
    client_idx = np.full(N, 3, dtype=np.int64)
    np.savez_compressed(root / "runs" / "probes" / "COHORT_DS.npz", decision_id=decision_id,
                       target_class=target_class, eval_mask=eval_mask, eval_user_key=eval_user_key,
                       client_idx=client_idx)

    (root / "config" / "NOISE_REFERENCE.json").write_text(json.dumps({"sigma_hat": sigma_hat}))
    (root / "config" / "BASELINE_PIN.json").write_text(json.dumps(
        {"chosen": {"seed": 2026, "epoch": 3, "sha256": "deadbeef", "macro_mrr20": 0.5}}))

    fit = {f: {"valid_mean": 5.0, "valid_std": 1.0} for f in fv.M1_FEATURES}
    (root / "audit" / "M1_FIT.json").write_text(json.dumps({"features": fit}))

    zero_checks = {"a==b": 0, "local_hour_rotation_max_residual": 0.0}
    dupf = {"label": "FULL_POPULATION", "checks": {"TRAIN": dict(zero_checks), "VALIDATION_SELECT": dict(zero_checks)}}
    (root / "audit" / "DUPLICATES_FULL.json").write_text(json.dumps(dupf))

    if not missing_lean:
        lean = {"lean": {"macro_mrr@20": 0.501, "micro_recall@20": 0.601}, "delta_macro_mrr@20": 0.001,
                "max_abs_logit_diff_first_1024": 1e-5, "ranks_identical_fraction": 0.9999}
        (root / "audit" / "LEAN_PARITY.json").write_text(json.dumps(lean))

    probes = {"baseline": {"metrics": {"macro_mrr@20": 0.5, "micro_recall@20": 0.6}},
              "probes": {"P2_MAIN_CATEGORY_UNAVAILABLE": {
                  "metrics": {"macro_mrr@20": 0.4, "micro_recall@20": 0.5}, "delta_macro_mrr@20": -0.1,
                  "bootstrap_macro_mrr@20": {"ci95": [-0.12, -0.08]}, "label": "coherent absence"}},
              "gate_2a": {"rule": "x", "passing": [], "C2": None}}
    (root / "audit" / "GROUP_PROBE_RESULTS.json").write_text(json.dumps(probes))

    for arm, spec in arms.items():
        d = root / "runs" / "screen" / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps({"cohort_metrics": spec["cohort_metrics"]}))
        rid = decision_id.copy()
        if arm == corrupt_decision_id_arm:
            rid = rid[::-1].copy()
        np.savez_compressed(d / "cohort_ranks.npz", decision_id=rid,
                           ranks=np.asarray(spec["ranks"], dtype=np.int64))
    return decision_id


def _final_decision(root):
    return json.loads((root / "state" / "FINAL_DECISION.json").read_text())


def test_finalize_pair_required_sigma_low_branch(tmp_path, set_fo_root):
    set_fo_root(tmp_path)
    _write_fixture(tmp_path, sigma_hat=0.0005, arms={
        "CONTROL": {"cohort_metrics": {"macro_mrr@20": 0.500, "micro_recall@20": 0.600}, "ranks": [50] * N},
        "C1_D3": {"cohort_metrics": {"macro_mrr@20": 0.520, "micro_recall@20": 0.610}, "ranks": [1] * N}})
    rc = ff.main()
    assert rc == 0
    fin = _final_decision(tmp_path)
    assert fin["gate_2b_threshold"] == pytest.approx(0.002, abs=1e-12)   # max(0.002, 2*0.0005) = 0.002
    assert fin["decision"] == "PAIR_REQUIRED"
    assert fin["finalist"] == "C1_D3"


def test_finalize_pair_required_sigma_high_branch(tmp_path, set_fo_root):
    set_fo_root(tmp_path)
    _write_fixture(tmp_path, sigma_hat=0.003, arms={
        "CONTROL": {"cohort_metrics": {"macro_mrr@20": 0.500, "micro_recall@20": 0.600}, "ranks": [50] * N},
        "C1_D3": {"cohort_metrics": {"macro_mrr@20": 0.520, "micro_recall@20": 0.610}, "ranks": [1] * N}})
    rc = ff.main()
    assert rc == 0
    fin = _final_decision(tmp_path)
    assert fin["gate_2b_threshold"] == pytest.approx(0.006, abs=1e-12)   # max(0.002, 2*0.003) = 0.006
    assert fin["decision"] == "PAIR_REQUIRED"
    assert fin["finalist"] == "C1_D3"


def test_finalize_two_passing_arms_larger_delta_wins(tmp_path, set_fo_root):
    set_fo_root(tmp_path)
    _write_fixture(tmp_path, sigma_hat=0.0005, arms={
        "CONTROL": {"cohort_metrics": {"macro_mrr@20": 0.500, "micro_recall@20": 0.600}, "ranks": [50] * N},
        "C1_D3": {"cohort_metrics": {"macro_mrr@20": 0.530, "micro_recall@20": 0.610}, "ranks": [3] * N},
        "C2": {"cohort_metrics": {"macro_mrr@20": 0.550, "micro_recall@20": 0.620}, "ranks": [2] * N}})
    rc = ff.main()
    assert rc == 0
    fin = _final_decision(tmp_path)
    assert fin["decision"] == "PAIR_REQUIRED"
    assert fin["finalist"] == "C2"   # 0.05 delta > 0.03 delta, both pass
    assert fin["screen"]["C1_D3"]["passes"] is True
    assert fin["screen"]["C2"]["passes"] is True


def test_finalize_negative_control_just_below_threshold_keeps_current(tmp_path, set_fo_root):
    set_fo_root(tmp_path)
    _write_fixture(tmp_path, sigma_hat=0.0005, arms={
        "CONTROL": {"cohort_metrics": {"macro_mrr@20": 0.500, "micro_recall@20": 0.600}, "ranks": [50] * N},
        "C1_D3": {"cohort_metrics": {"macro_mrr@20": 0.5015, "micro_recall@20": 0.610}, "ranks": [1] * N}})
    rc = ff.main()
    assert rc == 0
    fin = _final_decision(tmp_path)
    g = fin["screen"]["C1_D3"]["gate_2b"]
    assert g["delta_ge_threshold"] is False   # 0.0015 < thr(0.002) -- the ONLY failing condition
    assert g["ci_lower_gt_0"] is True
    assert g["recall_guard"] is True
    assert fin["decision"] == "KEEP_CURRENT"
    assert fin["finalist"] is None


def test_finalize_negative_control_above_threshold_but_fails_recall_guard(tmp_path, set_fo_root):
    set_fo_root(tmp_path)
    _write_fixture(tmp_path, sigma_hat=0.0005, arms={
        "CONTROL": {"cohort_metrics": {"macro_mrr@20": 0.500, "micro_recall@20": 0.600}, "ranks": [50] * N},
        "C1_D3": {"cohort_metrics": {"macro_mrr@20": 0.520, "micro_recall@20": 0.594}, "ranks": [1] * N}})
    rc = ff.main()
    assert rc == 0
    fin = _final_decision(tmp_path)
    g = fin["screen"]["C1_D3"]["gate_2b"]
    assert g["delta_ge_threshold"] is True
    assert g["ci_lower_gt_0"] is True
    assert g["recall_guard"] is False   # -0.006 < -0.001 -- the ONLY failing condition
    assert fin["decision"] == "KEEP_CURRENT"
    assert fin["finalist"] is None


def test_finalize_negative_control_cohort_decision_id_mismatch_raises(tmp_path, set_fo_root):
    set_fo_root(tmp_path)
    _write_fixture(tmp_path, sigma_hat=0.0005, arms={
        "CONTROL": {"cohort_metrics": {"macro_mrr@20": 0.500, "micro_recall@20": 0.600}, "ranks": [50] * N},
        "C1_D3": {"cohort_metrics": {"macro_mrr@20": 0.520, "micro_recall@20": 0.610}, "ranks": [1] * N}},
        corrupt_decision_id_arm="C1_D3")
    with pytest.raises(RuntimeError, match="cohort order mismatch"):
        ff.main()


def test_finalize_missing_lean_parity_raises(tmp_path, set_fo_root):
    set_fo_root(tmp_path)
    _write_fixture(tmp_path, sigma_hat=0.0005, arms={
        "CONTROL": {"cohort_metrics": {"macro_mrr@20": 0.500, "micro_recall@20": 0.600}, "ranks": [50] * N}},
        missing_lean=True)
    with pytest.raises(RuntimeError, match="LEAN_PARITY"):
        ff.main()
