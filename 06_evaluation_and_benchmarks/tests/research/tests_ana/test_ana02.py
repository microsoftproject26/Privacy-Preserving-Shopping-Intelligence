"""ANA-02: the REV-ANA-01 fixes (A-F1 .. A-F17), the 06D §3-§5 registrations (rule map, hierarchy, still-improving flag),
the ONNX_CPU sub-block (06E §3), the TEST gate (E10.1) and tools/ana_block_index.py — on synthetic blocks.

Each MAJOR item has a positive test and a negative control (test_nc_*, strict xfail on AssertionError: the same positive
check run against a mutant that re-introduces the defect)."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

import ana_block_index as BI
import anakit
import layoutkit
import retention_report as R
from anakit import macro
from test_retention_report import _set_rows_field, check_refuses, conf, nc, result, run, tab_row, zero_denominator
from v4train import bootstrap_ci as BC


@pytest.fixture(scope="module")
def done2(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("done2")
    blk = anakit.build(tmp)
    out = tmp / "out"
    rc, text = run(blk, out)
    assert rc == 0, text
    return blk, out, result(out)


@pytest.fixture
def blk(tmp_path):
    return anakit.build(tmp_path)


def edit_index(b, fn):
    idx = b.index()
    fn(idx)
    b.write_index(idx)


def _crec(res, kid):
    return next(c for c in res["confirmatory"] if c["id"] == kid)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _add_arm_seed(b, arm, seed, rules=R.RULES):
    """Add (arm, seed) MAIN entries with rows / ranks computed as anakit does."""
    m = b.manifests["MAIN"]
    rows, entries = [], []
    for rule in rules:
        rk = anakit.arm_ranks(arm, seed, rule, b.base["MAIN"], m.rankable)
        rel = f"MAIN/{arm}_s{seed}/ranks_{rule}.npy"
        (b.root / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(b.root / rel, rk.astype(np.int32))
        rows += anakit.rows_for(arm, seed, rule, rk, m)
        entries.append({"arm": arm, "seed": seed, "rule": rule, "role": "MAIN",
                        "rows": f"MAIN/{arm}_s{seed}/RESULT_ROWS.json", "ranks": rel})
    anakit.jdump(b.root / "MAIN" / f"{arm}_s{seed}" / "RESULT_ROWS.json", {"rows": rows})
    edit_index(b, lambda idx: idx["entries"].extend(entries))


# ================================================================================================ A-F9 (06D §3-§5)
def test_embedded_06d_text_matches_the_contract():
    a06d = _norm((R.V4 / "contracts" / "PREG1_AMENDMENT_06D.md").read_text(encoding="utf-8"))
    assert _norm(R.REGISTERED_06D_3) in a06d and _norm(R.REGISTERED_06D_4) in a06d
    assert R.rule_map_from_06d() == R.CONFIRMATORY_RULE == {"K12": "CONTROLLED_BUDGET", "K13": "CONTROLLED_BUDGET"}
    assert R.hierarchy_from_06d() == R.HIERARCHY_06D
    assert "best-of-12 selection on the users' own validation data is not covered by the accountant" in a06d
    got = R.load_contrasts(R.V4 / "company" / "gates" / "ANA_CONTRASTS_v2.json")  # the real registered file passes
    assert got["confirmatory_rule_map"] == R.CONFIRMATORY_RULE and len(got["items"]) == 19
    assert got["multiplicity"].startswith("19 pre-registered contrasts")
    v1 = R.V4 / "company" / "gates" / "ANA_CONTRASTS.json"                          # v1: a VALIDATION dry check only
    assert len(R.load_contrasts(v1, split="VALIDATION", allow_v1=True)["items"]) == 16


def check_rule_map(res):
    for kid in ("K12", "K13"):
        c = _crec(res, kid)
        assert c["confirmatory_rule"] == "CONTROLLED_BUDGET"
        assert c["by_rule"]["CONTROLLED_BUDGET"]["label"] == "CONFIRMATORY"
        assert c["by_rule"]["PRACTICAL_BEST"]["label"] == "DESCRIPTIVE"
        assert c["by_rule"]["PRACTICAL_BEST"]["disclosure"] == R.DISCLOSURE_06D_3
        assert "disclosure" not in c["by_rule"]["CONTROLLED_BUDGET"]
    for kid in ("K1", "K2", "K8", "K14"):
        c = _crec(res, kid)
        assert c["by_rule"]["PRACTICAL_BEST"]["label"] == "CONFIRMATORY"
        assert c["by_rule"]["CONTROLLED_BUDGET"]["label"] == "DESCRIPTIVE"


def test_rule_map_applied(done2):
    b, out, res = done2
    check_rule_map(res)
    k12 = conf(res, "K12", "CONTROLLED_BUDGET")                  # both sides CONTROLLED_BUDGET (06A §7 pairing)
    assert all(p["num"].endswith("CONTROLLED_BUDGET|MAIN") and p["den"].endswith("CONTROLLED_BUDGET|MAIN")
               for p in k12["per_seed"])
    rep = (out / R.REPORT_NAME).read_text(encoding="utf-8")
    assert R.DISCLOSURE_06D_3 in rep and "| **K12** |" in rep


@nc("the ANA-01 labelling: every PRACTICAL_BEST row confirmatory, no per-contrast rule map")
def test_nc_rule_map_ignored(blk, tmp_path, monkeypatch):
    real = R.confirmatory

    def pb_only(eng, book, contrasts):                                                  # MUTANT
        out = real(eng, book, contrasts)
        for c in out:
            for rule, b in c["by_rule"].items():
                b["label"] = "CONFIRMATORY" if rule == "PRACTICAL_BEST" else "DESCRIPTIVE"
                b.pop("disclosure", None)
        return out
    monkeypatch.setattr(R, "confirmatory", pb_only)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    check_rule_map(result(tmp_path / "o"))


def test_contrasts_need_the_rule_map_and_hierarchy(blk, tmp_path):
    obj = anakit.contrasts_obj()
    obj.pop("confirmatory_rule_map")
    anakit.jdump(blk.contrasts, obj)
    check_refuses(blk, tmp_path, "confirmatory_rule_map")
    anakit.jdump(blk.contrasts, anakit.contrasts_obj(confirmatory_rule_map={"K12": "CONTROLLED_BUDGET"}))
    check_refuses(blk, tmp_path, "confirmatory_rule_map")
    anakit.jdump(blk.contrasts, anakit.contrasts_obj(hierarchy={"primary": ["K1"], "secondary": [], "tertiary": []}))
    check_refuses(blk, tmp_path, "hierarchy")


def test_hierarchy_labels_and_multiplicity_line(done2):
    _b, out, res = done2
    for c in res["confirmatory"]:
        assert c["hierarchy"] == R.HIERARCHY.get(c["id"], "tertiary")
    assert res["labels"]["hierarchy"]["K2"] == "primary" and res["labels"]["hierarchy"]["K12"] == "secondary"
    assert res["labels"]["hierarchy"]["K16"] == "tertiary" and res["labels"]["multiplicity"] == R.MULTIPLICITY_06D_4
    assert res["labels"]["multiplicity"].startswith("19 pre-registered contrasts")
    assert res["labels"]["hierarchy"]["K17"] == "secondary" and res["labels"]["hierarchy"]["K18"] == "tertiary"
    rep = (out / R.REPORT_NAME).read_text(encoding="utf-8")
    head = rep.split("## 1.")[0]
    assert R.MULTIPLICITY_06D_4 in head
    assert re.search(r"\| \*\*K2\*\* \| .*\| \*\*primary\*\* \|", rep)


def check_flag_at(tmp_path, best_half_efe: int, want: bool, b):
    rd = b.root / "MAIN" / "FL_FA_s2027"
    summ = json.loads((rd / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    summ["best_half_efe"] = best_half_efe
    anakit.jdump(rd / "RUN_SUMMARY.json", summ)
    e = R.Entry("FA", 2027, "PRACTICAL_BEST", "MAIN", "x", "MAIN/FL_FA_s2027/evals/eval_5.5.ranks_int32.npy", {},
                "MAIN/FL_FA_s2027", None, False, {})
    R.load_marks(b.root, e, b.manifests["MAIN"])
    assert e.meta_auto["best_mark"] == best_half_efe / 2
    assert e.meta_auto["still_improving_at_6_0"] is want


def test_still_improving_flag(done2, blk, tmp_path):
    _b, _out, res = done2
    fa = tab_row(res, "FA")
    assert fa["still_improving_at_6.0"]["per_seed"] == {"2026": True, "2027": True}       # best mark 5.5 >= 5.5
    assert fa["still_improving_at_6.0"]["any_seed"] is True and fa["still_improving_at_6.0"]["label"] == "DESCRIPTIVE"
    assert tab_row(res, "FA", "CONTROLLED_BUDGET")["still_improving_at_6.0"]["any_seed"] is True
    assert fa["entries"]["2026"]["meta"]["still_improving_at_6.0"] is True
    assert "still_improving_at_6.0" not in tab_row(res, "C_BIG")
    check_flag_at(tmp_path, 10, False, blk)
    check_flag_at(tmp_path, 11, True, blk)


@nc("a still-improving flag that needs the best mark strictly after 5.5")
def test_nc_still_improving_threshold(blk, tmp_path, monkeypatch):
    real = R.load_marks

    def strict(block, entry, manifest):                                                  # MUTANT
        out = real(block, entry, manifest)
        entry.meta_auto["still_improving_at_6_0"] = entry.meta_auto["best_mark"] > 5.5
        return out
    monkeypatch.setattr(R, "load_marks", strict)
    check_flag_at(tmp_path, 11, True, blk)


# ================================================================================================ A-F4 (seed sets)
def check_seed_sets(res):
    k13 = conf(res, "K13", "CONTROLLED_BUDGET")
    assert k13["seeds"] == [2026] and k13["label"] == "CONFIRMATORY" and k13["seeds_status"] == "COMPLETE"
    extra = _crec(res, "K13")["extra_seeds"]["CONTROLLED_BUDGET"]
    assert extra["seeds"] == [2026, 2027] and extra["label"] == "DESCRIPTIVE"


def _seed_world(b):
    _add_arm_seed(b, "DP1", 2027)                                   # a NICE second seed of a single-seed contrast
    edit_index(b, lambda idx: idx.__setitem__("entries", [e for e in idx["entries"]
                                                          if not (e["arm"] == "FP" and e["seed"] == 2027)]))


def test_registered_seeds_extra_and_incomplete(blk, tmp_path):
    _seed_world(blk)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    res = result(tmp_path / "o")
    check_seed_sets(res)
    k4 = conf(res, "K4")
    assert k4["seeds"] == [2026] and k4["seeds_status"] == "INCOMPLETE_SEEDS" and k4["missing_registered_seeds"] == [2027]
    assert isinstance(k4["point"], float)                           # reported, never substituted (06D §8)
    rep = (tmp_path / "o" / R.REPORT_NAME).read_text(encoding="utf-8")
    assert "INCOMPLETE_SEEDS" in rep and "DESCRIPTIVE (all seeds)" in rep


@nc("every available seed averaged into the confirmatory row")
def test_nc_registered_seeds(blk, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "REGISTERED_SEEDS", {})                                         # MUTANT
    _seed_world(blk)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    check_seed_sets(result(tmp_path / "o"))


# ================================================================================================ A-F3 (RULE_FREE)
def test_rule_free_only_for_baselines(tmp_path):
    for i, (rule, needle) in enumerate((("RULE_FREE", "RULE_FREE is allowed only"),
                                        ("CONTROLLED_BUDGET", "PRACTICAL_BEST only"))):
        b = anakit.build(tmp_path / f"b{i}")

        def set_lo(idx, rule=rule):
            for e in idx["entries"]:
                if e["arm"] == "LO" and e["role"] == "MAIN":
                    e["rule"] = rule
                    if rule == "RULE_FREE":
                        e["seed"] = None if e["seed"] == 2026 else e["seed"]
        edit_index(b, set_lo)
        check_refuses(b, tmp_path / f"b{i}", needle)


def test_lo_controlled_budget_not_available(done2):
    _b, _out, res = done2
    for kid in ("K8", "K11"):
        assert conf(res, kid, "CONTROLLED_BUDGET")["status"] == "NOT_AVAILABLE"
        assert conf(res, kid, "CONTROLLED_BUDGET")["seeds_status"] == "NOT_AVAILABLE"
        assert isinstance(conf(res, kid)["point"], float)


def check_rule_free_den():
    fa = R.Entry("FA", 2026, "PRACTICAL_BEST", "MAIN", "x", None, {}, None, None, False, {})
    lo = R.Entry("POP", 2026, "RULE_FREE", "MAIN", "x", None, {}, None, None, False, {})   # the REV-ANA-01 probe shape
    book = R.Book([fa, lo])
    assert book.pairs("MAIN", "FA", "POP", "PRACTICAL_BEST") == []
    assert book.pairs("MAIN", "POP", "POP", "PRACTICAL_BEST") == [(lo, lo)]      # baseline vs baseline stays


def test_book_pairs_rule_free_denominator_is_not_available():
    check_rule_free_den()


@nc("Book.pairs pairing a checkpoint-rule numerator with a rule-free denominator (ANA-01)")
def test_nc_rule_free_denominator(monkeypatch):
    def old_pairs(self, role, num_arm, den_arm, rule):                                    # MUTANT (ANA-01 code)
        nums, dens = self.get(role, num_arm, rule), self.get(role, den_arm, rule)
        out = []
        for seed in sorted(s for s in dens if s is not None):
            n = nums.get(seed, nums.get(None))
            if n is not None:
                out.append((n, dens[seed]))
        if not out and None in dens:
            n = nums.get(None)
            if n is not None:
                out.append((n, dens[None]))
        return out
    monkeypatch.setattr(R.Book, "pairs", old_pairs)
    check_rule_free_den()


# ================================================================================================ A-F5 (comparison key)
def test_extended_key_and_pinned_status(tmp_path):
    for i, (field, value, needle) in enumerate((
            ("candidate_order_hash", "x" * 64, "candidate_order_hash"),
            ("eval_stream", "ALL_ROWS", "eval_stream"),
            ("status", "REHEARSAL_NON_FROZEN_DEVICE", "MEASURED_PINNED_DEVICE"),
            ("device_pin_sha256", "q" * 64, "ONE device_pin_sha256"))):
        b = anakit.build(tmp_path / f"k{i}")
        _set_rows_field(b, "MAIN/FP_s2026/RESULT_ROWS.json", field, value)
        check_refuses(b, tmp_path / f"k{i}", needle)


def test_intended_differences_recorded(blk, tmp_path):
    _set_rows_field(blk, "MAIN/FP_s2026/RESULT_ROWS.json", "eval_batch", 256)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    key = result(tmp_path / "o")["checks"]["comparison_key"]
    assert key["intended_differences"]["FP|2026|PRACTICAL_BEST|MAIN"]["eval_batch"] == ["256"]
    assert key["intended_differences"]["FP|2027|PRACTICAL_BEST|MAIN"]["eval_batch"] == ["512"]
    assert key["device_pin_sha256"] == anakit.PIN and key["values"]["candidate_order_hash"] == "c" * 64


@nc("the ANA-01 three-field comparison key")
def test_nc_narrow_comparison_key(blk, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "KEY_FIELDS", ("eval_device_class", "split", "evaluator_code_sha256"))   # MUTANT
    _set_rows_field(blk, "MAIN/FP_s2026/RESULT_ROWS.json", "candidate_order_hash", "x" * 64)
    check_refuses(blk, tmp_path, "candidate_order_hash")


# ================================================================================================ A-F6 (rank coverage)
def uncover(b):
    """C_SMALL s2026 PRACTICAL_BEST: one RANKABLE row loses its rank (-1); the rows are recomputed from the ranks."""
    m = b.manifests["MAIN"]
    rk = anakit.arm_ranks("C_SMALL", 2026, "PRACTICAL_BEST", b.base["MAIN"], m.rankable)
    rk[int(np.flatnonzero(m.rankable)[3])] = -1
    np.save(b.root / "MAIN" / "C_SMALL_s2026" / "ranks_PRACTICAL_BEST.npy", rk.astype(np.int32))
    p = b.root / "MAIN" / "C_SMALL_s2026" / "RESULT_ROWS.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    obj["rows"] = [r for r in obj["rows"] if r["selection_rule"] != "PRACTICAL_BEST"] + \
        anakit.rows_for("C_SMALL", 2026, "PRACTICAL_BEST", rk, m)
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_rank_coverage_refused(blk, tmp_path):
    uncover(blk)
    check_refuses(blk, tmp_path, "RANKABLE")


def test_onnx_subset_rules(tmp_path):
    b = anakit.build(tmp_path / "a")
    edit_index(b, lambda idx: [e.pop("users_subset") for e in idx["entries"] if e.get("comparison_block")])
    check_refuses(b, tmp_path / "a", "must declare users_subset")
    b = anakit.build(tmp_path / "b")
    edit_index(b, lambda idx: [e["users_subset"].__setitem__("set_sha256", "0" * 64) for e in idx["entries"]
                               if e.get("comparison_block")])
    check_refuses(b, tmp_path / "b", "first_users_set_sha256")


def test_users_subset_v2_is_the_dev04_set(done2):
    b, _out, res = done2
    meta = tab_row(res, "C_SMALL_INT8")["entries"]["2026"]["meta"]
    assert meta["users_subset_n"] == anakit.N_SUBSET and meta["users_subset_set_sha256"] == anakit.SUBSET_SHA
    keys, sha = R.first_users_v2(b.ctx["MAIN"], anakit.N_SUBSET)
    assert np.array_equal(keys, anakit.SUBSET_USERS) and sha == anakit.SUBSET_SHA
    assert not np.array_equal(keys, np.arange(anakit.N_SUBSET))      # not the v1 eval_user_key < N order


@nc("no rank-coverage guard (an unscored -1 scored as a miss)")
def test_nc_rank_coverage(blk, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "check_rank_coverage", lambda label, view, ranks, hint="": None)   # MUTANT
    uncover(blk)
    check_refuses(blk, tmp_path, "RANKABLE")


# ================================================================================================ A-F7 (arm identity)
def test_arm_identity(tmp_path):
    b = anakit.build(tmp_path / "a")
    _set_rows_field(b, "MAIN/C_FULL_s2026/RESULT_ROWS.json", "train_manifest_id", "MAIN_TRAIN_5_100_131072")
    check_refuses(b, tmp_path / "a", "ARM_IDENTITY")
    b = anakit.build(tmp_path / "b")
    _set_rows_field(b, "MAIN/LO_s2026/RESULT_ROWS.json", "method", "FT_C")
    check_refuses(b, tmp_path / "b", "ARM_IDENTITY")
    b = anakit.build(tmp_path / "c")
    _set_rows_field(b, "MAIN/FL_FA_s2026/evals/eval_3.0.rows.json", "method", "FP")
    check_refuses(b, tmp_path / "c", "ARM_IDENTITY")


@nc("arm identity taken from the index only (a C_FULL / C_BIG swap passes)")
def test_nc_arm_identity(blk, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "check_arm_identity", lambda entry, r: None)                    # MUTANT
    _set_rows_field(blk, "MAIN/C_FULL_s2026/RESULT_ROWS.json", "train_manifest_id", "MAIN_TRAIN_5_100_131072")
    check_refuses(blk, tmp_path, "ARM_IDENTITY")


# ================================================================================================ A-F2 (ONNX sub-block)
def test_k16_confirmatory_inside_the_onnx_block(blk, tmp_path):
    anakit.add_onnx_fp32(blk)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    res = result(tmp_path / "o")
    k16 = conf(res, "K16")
    assert k16["label"] == "CONFIRMATORY" and k16["seeds"] == [2026] and k16["point"] == pytest.approx(100.0)
    assert k16["n_users"] == anakit.N_SUBSET and "comparison" not in k16
    assert "descriptive_fallback" not in _crec(res, "K16")
    ob = res["checks"]["comparison_key"]["onnx_block"]
    assert ob["runtime_identity"]["onnxruntime_version"] == "1.24.1" and len(ob["entries"]) == 2
    assert tab_row(res, "C_SMALL_INT8")["vs_C_BIG"]["mrr@20"]["comparison"] == "DEVICE_MIXED"


def check_runtime_identity_refused(b, tmp_path):
    anakit.add_onnx_fp32(b)
    p = b.root / "ONNX" / "BENCH_RESULT.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    obj["retention"]["onnx_fp32"]["onnxruntime_version"] = "1.20.0"
    p.write_text(json.dumps(obj), encoding="utf-8")
    check_refuses(b, tmp_path, "runtime identity")


def test_onnx_refusals(tmp_path):
    check_runtime_identity_refused(anakit.build(tmp_path / "a"), tmp_path / "a")
    b = anakit.build(tmp_path / "b")
    p = b.root / "ONNX" / "BENCH_RESULT.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    obj["retention"]["onnx_int8"]["ranks_sha256"] = "0" * 64
    p.write_text(json.dumps(obj), encoding="utf-8")
    check_refuses(b, tmp_path / "b", "ranks_sha256")
    b = anakit.build(tmp_path / "c")
    edit_index(b, lambda idx: [e.pop("comparison_block") for e in idx["entries"] if e.get("comparison_block")])
    check_refuses(b, tmp_path / "c", "comparison_block")


@nc("an ONNX sub-block without its runtime-identity check")
def test_nc_onnx_runtime_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "check_onnx_block", lambda entries, ctx, idx, split: None)       # MUTANT
    check_runtime_identity_refused(anakit.build(tmp_path / "a"), tmp_path / "a")


# ================================================================================================ A-F1 (truncation)
def check_trunc_separate(curve):
    pts = {(t["arm"], t["panel_type"], t["seed"], t["h"]): t for t in curve}
    assert ("C_SMALL", "FIXED_20", 2026, 5) in pts and ("C_SMALL", "FIXED_50", 2026, 5) in pts
    assert pts[("C_SMALL", "FIXED_20", 2026, 5)]["M"] == 0.17
    assert pts[("C_SMALL", "FIXED_50", 2026, 5)]["M"] == 0.06
    assert pts[("C_SMALL", "FIXED_50", 2026, 5)]["C_ref"] == 0.2


def test_trunc_subpanels_are_separate():
    check_trunc_separate(R.trunc_curves(anakit.trunc_rows())[0])


@nc("trunc curves that merge the FIXED sub-panels (the ANA-01 field names: last row wins)")
def test_nc_trunc_subpanel_merge(monkeypatch):
    real = R.trunc_curves

    def merged(rows, key=None):                                                            # MUTANT
        seen, out = {}, []
        for r in rows:
            k = (r.get("method"), r.get("trunc_panel"), r.get("seed"), r.get("trunc_h"), r.get("trunc_full_history"),
                 r.get("device_cell"))
            seen[k] = r
        return real(list(seen.values()), key)
    monkeypatch.setattr(R, "trunc_curves", merged)
    check_trunc_separate(R.trunc_curves(anakit.trunc_rows())[0])


def test_trunc_refusals(tmp_path):
    rows = anakit.trunc_rows()
    with pytest.raises(R.AnaRefused, match="duplicate"):
        R.trunc_curves(rows + [dict(rows[2])])
    key = {"eval_device_class": anakit.DEV, "split": "VALIDATION", "evaluator_code_sha256": anakit.EVSHA}
    R.trunc_curves(rows, key)
    with pytest.raises(R.AnaRefused, match="evaluator_code_sha256"):
        R.trunc_curves([dict(r, evaluator_code_sha256="f" * 64) for r in rows], key)
    no_cref = [r for r in rows if not (r["method"] == "C_BIG" and r["trunc_full_history"])]
    curve, ev = R.trunc_curves(no_cref)
    assert all(R.is_undef(t["R"]) for t in curve) and all(e["events_to_90"] == "NO_C_REF" for e in ev)   # GATE-B B4
    b = anakit.build(tmp_path / "a")
    edit_index(b, lambda idx: idx["records"].__setitem__("TRUNC_ROWS", ["records/TRUNC.json"]))
    check_refuses(b, tmp_path / "a", "TRUNC_ROWS")
    b = anakit.build(tmp_path / "b")
    edit_index(b, lambda idx: idx["records"].__setitem__("TRUNC_ROWS", [{"rows": "records/TRUNC.json",
                                                                          "rule": "CONTROLLED_BUDGET"}]))
    check_refuses(b, tmp_path / "b", "PRACTICAL_BEST")


# ================================================================================================ A-F8 (TEST gate)
@pytest.fixture(scope="module")
def val3(tmp_path_factory):
    """FIX-DELTA-03 M5: a TEST run takes the v3 registry only; its VALIDATION_RESULT is the v3 VALIDATION run."""
    return anakit.validation_result_v3(tmp_path_factory.mktemp("val3"))


def test_test_split_needs_the_gate(val3, tmp_path):
    b = anakit.build_test(tmp_path, val3)
    b.test_args = []
    check_refuses(b, tmp_path, "requires --all-methods-lock")
    rc, text = run(b, tmp_path / "v", "--all-methods-lock", str(b.gate["lock"]), split="VALIDATION")
    assert rc == 2 and "TEST-only" in text


def test_test_gate_refusals(val3, tmp_path):
    b = anakit.build_test(tmp_path / "a", val3)               # the lock changed after its register entry
    b.gate["lock"].write_text(b.gate["lock"].read_text(encoding="utf-8") + " ", encoding="utf-8")
    check_refuses(b, tmp_path / "a", "no TEST_EXPOSURE_REGISTER entry")
    b = anakit.build_test(tmp_path / "b", val3)               # --contrasts is not the lock file
    b.contrasts = tmp_path / "b" / "copy.json"
    b.contrasts.write_text(b.gate["lock"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    check_refuses(b, tmp_path / "b", "must be the ALL_METHODS_LOCK file")
    b = anakit.build_test(tmp_path / "c", val3)               # the extension edits a REGISTRY_V2 entry
    from v4data08.d8_registry import with_self
    ext = json.loads(b.gate["ext"].read_text(encoding="utf-8"))
    ext["roles"]["MAIN"] = {"train": "X", "eval": "Y"}
    anakit.jdump(b.gate["ext"], with_self(ext))
    check_refuses(b, tmp_path / "c", "does not extend REGISTRY_V2")
    b = anakit.build_test(tmp_path / "d", val3)               # the extension's TEST role map
    ext = json.loads(b.gate["ext"].read_text(encoding="utf-8"))
    ext["roles"]["MAIN_TEST"]["train"] = "UNEW_HOLDOUT_SUPPORT_32768"
    anakit.jdump(b.gate["ext"], with_self(ext))
    check_refuses(b, tmp_path / "d", "registered pair")
    b = anakit.build_test(tmp_path / "e", val3)               # a header of another gate
    b.ctx["MAIN"].manifest.header["test_exposure_register_entry_id"] = 1
    check_refuses(b, tmp_path / "e", "not bound to this gate")
    b = anakit.build_test(tmp_path / "f", val3)               # no VALIDATION_RESULT
    edit_index(b, lambda idx: idx["records"].pop("VALIDATION_RESULT"))
    check_refuses(b, tmp_path / "f", "VALIDATION_RESULT")
    b = anakit.build_test(tmp_path / "g", val3)               # a VALIDATION_RESULT of another script
    p = b.root / "records" / "VAL_RESULT.json"
    val = json.loads(p.read_text(encoding="utf-8"))
    val["script"]["sha256"] = "0" * 64
    val = R.with_self_sha({k: v for k, v in val.items() if k != R.SELF_KEY})
    p.write_text(R.dumps(val), encoding="utf-8")
    check_refuses(b, tmp_path / "g", "another retention_report.py")
    b = anakit.build_test(tmp_path / "h", val3)               # a VALIDATION role id in a TEST block
    edit_index(b, lambda idx: idx["roles"]["MAIN"].__setitem__("eval_manifest_id", "MAIN_VALIDATION_5_100_131072"))
    check_refuses(b, tmp_path / "h", "registered TEST pair")


def test_test_hook_is_a_hard_refusal_without_the_data09_loader(val3, tmp_path):
    """No injected loader: the DATA-09 hook refuses cleanly (exit 2, never 5) — here the local tree has no data."""
    b = anakit.build_test(tmp_path, val3)
    lines = []
    rc = R.main(b.argv(tmp_path / "o"), context_loader=None, say=lines.append)
    assert rc == R.REFUSED, lines
    assert "TEST manifests" in lines[0] and "DATA-09" in lines[0]


def break_chain(b):
    p = b.gate["register"]
    reg = json.loads(p.read_text(encoding="utf-8"))
    reg["entries"][0]["observations_description"] = "edited after the fact"
    p.write_text(json.dumps(reg), encoding="utf-8")


def test_test_register_chain_verified(val3, tmp_path):
    b = anakit.build_test(tmp_path, val3)
    break_chain(b)
    check_refuses(b, tmp_path, "hash chain")


@nc("a TEST gate that reads the register without verifying its hash chain")
def test_nc_test_gate_chain(val3, tmp_path, monkeypatch):
    def lax(lock_path, register_path):                                                     # MUTANT
        reg = json.loads(Path(register_path).read_text(encoding="utf-8"))
        lsha = R._sha_file(lock_path)
        e = [x for x in reg["entries"] if x.get("all_methods_lock_sha256") == lsha][0]
        return {"all_methods_lock_sha256": lsha, "test_exposure_register_entry_id": int(e["entry_id"]),
                "test_exposure_register_entry_sha256": e["entry_sha256"]}
    monkeypatch.setattr(R, "check_test_gate", lax)
    b = anakit.build_test(tmp_path, val3)
    break_chain(b)
    check_refuses(b, tmp_path, "hash chain")


# ================================================================================================ MINORs
def test_rule_vs_checkpoint(tmp_path):
    cases = (("MAIN/C_BIG_s2026/RESULT_ROWS.json", {"selection_rule": "CONTROLLED_BUDGET"},
              {"checkpoint_id": "C_BIG_s2026:best@3.0", "efe": 3.0}),
             ("MAIN/C_BIG_s2026/RESULT_ROWS.json", {"selection_rule": "PRACTICAL_BEST"},
              {"checkpoint_id": "C_BIG_s2026:best@2.5", "efe": 2.5}),
             ("MAIN/C_SMALL_s2026/RESULT_ROWS.json", {"selection_rule": "PRACTICAL_BEST"},
              {"checkpoint_id": "C_SMALL_s2026:endpoint_6.0"}),
             ("MAIN/FL_FA_s2026/RESULT_ROWS.json", {"selection_rule": "PRACTICAL_BEST"},
              {"checkpoint_id": "FL_FA_SMALL_s2026:best@5.0", "efe": 5.0, "schedule_mark": 5.0}))
    for i, (rel, where, fields) in enumerate(cases):
        b = anakit.build(tmp_path / f"r{i}")
        for f, v in fields.items():
            _set_rows_field(b, rel, f, v, where)
        check_refuses(b, tmp_path / f"r{i}", "RULE_CHECKPOINT")
    b = anakit.build(tmp_path / "cell")
    edit_index(b, lambda idx: [e["row_filter"].__setitem__("device_cell", "P1_F0.3") for e in idx["entries"]
                               if e["arm"] == "LO" and e["role"] == "MAIN" and e["seed"] == 2026])
    check_refuses(b, tmp_path / "cell", "selected cell")


def test_check_only_runs_pairing_and_the_block_denominators(blk, tmp_path):
    rc, text = run(blk, tmp_path / "ok", "--check-only")
    assert rc == 0 and "pairs checked" in text and not re.search(r"0\.\d{4,}", text)
    zero_denominator(blk)
    rc, text = run(blk, tmp_path / "o", "--check-only")
    assert rc == 2 and "zero denominator" in text and not (tmp_path / "o").exists()


def test_evidence_integrity(tmp_path):
    b = anakit.build(tmp_path / "a")
    p = b.root / "MAIN" / "FL_FA_s2027" / "RUN_SUMMARY.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s["status"] = "RUNNING"
    anakit.jdump(p, s)
    check_refuses(b, tmp_path / "a", "DONE_MAIN")
    for i, (field, value, needle) in enumerate((("run_label", "SMOKE_NOT_EVIDENCE", "SMOKE_NOT_EVIDENCE"),
                                                ("row_status", "FL_LR_SCREEN", "FL_LR_SCREEN"),
                                                ("role", "CALIBRATION", "CALIBRATION"))):
        b = anakit.build(tmp_path / f"s{i}")
        _set_rows_field(b, "MAIN/FP_s2026/RESULT_ROWS.json", field, value)
        check_refuses(b, tmp_path / f"s{i}", needle)


def test_g2_run_sha_recorded_and_frozen(done2):
    _b, _out, res = done2
    freeze = json.loads((R.V4 / "company" / "gates" / "G2_SCRIPT_FREEZE.json").read_text(encoding="utf-8"))
    assert R.G2_RUN_SHA256 == freeze["g2_script"]["sha256"] == R._sha_file(R.HERE / "g2_run.py")
    assert res["library_files_sha256"]["tools/g2_run.py"] == R.G2_RUN_SHA256


def test_n_decisions_and_bootstrap_w(done2, tmp_path):
    b, _out, res = done2
    m = b.manifests["MAIN"]
    assert conf(res, "K2")["n_decisions"] == int(m.eval_mask.sum())
    sub = R._subset_view(m, anakit.SUBSET_USERS)
    assert _crec(res, "K16")["descriptive_fallback"]["PRACTICAL_BEST"]["n_decisions"] == int(sub.eval_mask.sum())
    b2 = anakit.build(tmp_path / "w")
    _set_rows_field(b2, "MAIN/FP_s2026/RESULT_ROWS.json", "bootstrap_W_sha256", BC._plan(m).W_sha256)
    rc, text = run(b2, tmp_path / "w" / "ok")
    assert rc == 0, text
    _set_rows_field(b2, "MAIN/FP_s2026/RESULT_ROWS.json", "bootstrap_W_sha256", "0" * 64)
    check_refuses(b2, tmp_path / "w", "bootstrap_W_sha256")


def test_strata_source_manifest(blk, tmp_path):
    from v4base import strata as ST
    p = blk.root / "records" / "STRATA_CUTPOINTS_MAIN.json"
    p.unlink()
    stats = {anakit.hexkey("MAIN", k): {"n_train": 6 + k, "n_distinct_main_categories": k % 7,
                                        "top_brand_share": (k % 10) / 10.0} for k in range(1, 40)}
    ST.write_registry(p, stats, source_manifest_id="CALIB_TRAIN_5_100_32768")
    check_refuses(blk, tmp_path, "A-F16")


def test_historical_reference_seed_mean(blk, tmp_path):
    edit_index(blk, lambda idx: [e.__setitem__("historical_reference", True) for e in idx["entries"] if e["arm"] == "POP"])
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    h = result(tmp_path / "o")["historical_reference"][0]
    m = blk.manifests["MAIN"]
    v = macro(anakit.arm_ranks("POP", 2026, "PRACTICAL_BEST", blk.base["MAIN"], m.rankable), m)
    want = np.mean([100 * v / macro(blk.ranks[("MAIN", "C_BIG", s, "PRACTICAL_BEST")], m) for s in (2026, 2027)])
    assert h["C_BIG_seeds"] == [2026, 2027] and h["R_vs_C_BIG_point"] == pytest.approx(want, abs=1e-12)


def test_historical_rows_must_be_the_split(blk, tmp_path):
    edit_index(blk, lambda idx: [e.__setitem__("historical_reference", True) for e in idx["entries"] if e["arm"] == "POP"])
    _set_rows_field(blk, "MAIN/POP_rows.json", "split", "RETRO_TEST")
    check_refuses(blk, tmp_path, "historical reference too")


def test_run_all_creates_basetemp():
    text = (R.V4 / "code" / "tests_trunc" / "run_all.py").read_text(encoding="utf-8")
    assert 'mkdir(parents=True, exist_ok=True)' in text and "TMP=" in text and "TEMP=" in text


# ================================================================================================ the index builder
def build_index(tmp: Path, extra=(), layout=None, blk_=None):
    b = blk_ or anakit.build(tmp / "a")
    lay = layout or layoutkit.build_layout(tmp / "lay", b)
    lines = []
    rc = BI.main(lay["args"] + ["--out", str(tmp / "idx")] + list(extra), say=lines.append)
    return rc, "\n".join(lines), b, lay


def test_index_builder_end_to_end(done2, tmp_path):
    rc, text, b, lay = build_index(tmp_path)
    assert rc == 0, text
    assert re.fullmatch(r"INDEX: written \{.*\}", text) and not re.search(r"0\.\d{4,}", text)
    idx = json.loads((tmp_path / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))
    assert idx[R.SELF_KEY] == R._canon_indent(idx)
    arms = {(e["arm"], e["seed"], e["rule"], e["role"]) for e in idx["entries"]}
    assert ("DP8", 2027, "CONTROLLED_BUDGET", "MAIN") in arms and ("LO", 2026, "PRACTICAL_BEST", "UNSEEN_HOLDOUT") in arms
    pop = next(e for e in idx["entries"] if e["arm"] == "POP")
    assert pop["historical_reference"] is True and pop["rule"] == "RULE_FREE" and pop["seed"] is None
    int8 = next(e for e in idx["entries"] if e["arm"] == "C_SMALL_INT8")
    assert int8["comparison_block"] == "ONNX_CPU" and int8["rows"] is None
    assert int8["users_subset"] == {"rule": "SELECTION_KEY_V2_FIRST", "n": anakit.N_SUBSET, "set_sha256": anakit.SUBSET_SHA}
    fa = [e for e in idx["entries"] if e["arm"] == "FA" and e["role"] == "MAIN"]
    assert {e["ranks"].rsplit("/", 1)[1] for e in fa} == {"eval_5.5.ranks_int32.npy", "eval_6.0.ranks_int32.npy"}
    assert idx["records"]["TRUNC_ROWS"] == [{"rows": "TRUNC/SA_C_SMALL_s2026/RESULT_ROWS.json", "rule": "PRACTICAL_BEST"}]
    assert any("T41_WT" in x["dir"] for x in idx["excluded"])
    shutil.copy(tmp_path / "idx" / R.INDEX_NAME, R.V4.parent / "_pytest_tmp" / "ANA-02" / "SAMPLE_BLOCK_INDEX.json") \
        if (R.V4.parent / "_pytest_tmp" / "ANA-02").is_dir() else None
    lines = []
    argv = ["--block", str(tmp_path / "idx"), "--split", "VALIDATION", "--contrasts", str(b.contrasts)]
    assert R.main(argv + ["--out", str(tmp_path / "chk"), "--check-only"], context_loader=b.loader,
                  say=lines.append) == 0, lines
    assert R.main(argv + ["--out", str(tmp_path / "o")], context_loader=b.loader, say=lines.append) == 0, lines
    res = result(tmp_path / "o")
    _b2, _out2, res2 = done2
    for kid in ("K1", "K2", "K8", "K12", "K13"):
        for rule in R.RULES:
            assert conf(res, kid, rule).get("point") == pytest.approx(conf(res2, kid, rule).get("point"), abs=1e-12)


def test_index_builder_deterministic(tmp_path):
    rc, text, b, lay = build_index(tmp_path / "one")
    assert rc == 0, text
    lines = []
    assert BI.main(lay["args"] + ["--out", str(tmp_path / "two")], say=lines.append) == 0, lines
    assert (tmp_path / "one" / "idx" / R.INDEX_NAME).read_bytes() == (tmp_path / "two" / R.INDEX_NAME).read_bytes()


def sentinel_rows(lay):
    """Replace a re-evaluation RESULT_ROWS.json by a non-JSON sentinel (its record's sha updated): a builder that parses
    a rows file would crash or read it; the builder must copy it byte for byte and succeed."""
    d = lay["R"] / "REEVAL_A40" / "C_BIG_s2026_PB"
    (d / "RESULT_ROWS.json").write_bytes(b'{"rows": [SENTINEL_VALUE_NEVER_PARSED 0.123456')
    rec = json.loads((d / "REEVAL_RECORD.json").read_text(encoding="utf-8"))
    rec["result_rows_sha256"] = R._sha_file(d / "RESULT_ROWS.json")
    rec["outputs"]["RESULT_ROWS.json"] = rec["result_rows_sha256"]
    anakit.jdump(d / "REEVAL_RECORD.json", rec)
    return d


def check_never_parses_rows(tmp_path, monkeypatch):
    b = anakit.build(tmp_path / "a")
    lay = layoutkit.build_layout(tmp_path / "lay", b)
    d = sentinel_rows(lay)
    opened = []
    real = BI._open_json

    def spy(p, **kw):
        opened.append(Path(p).name)
        return real(p, **kw)
    monkeypatch.setattr(BI, "_open_json", spy)
    lines = []
    rc = BI.main(lay["args"] + ["--out", str(tmp_path / "idx")], say=lines.append)
    assert rc == 0, lines
    assert opened and not [n for n in opened if n == "RESULT_ROWS.json" or n.endswith("rows.json")], opened
    assert (tmp_path / "idx" / "REEVAL" / "C_BIG_s2026_PB" / "RESULT_ROWS.json").read_bytes() == \
        (d / "RESULT_ROWS.json").read_bytes()


def test_index_builder_never_parses_a_rows_file(tmp_path, monkeypatch):
    check_never_parses_rows(tmp_path, monkeypatch)
    with pytest.raises(BI.IndexRefused, match="never parses"):
        BI._open_json(tmp_path / "RESULT_ROWS.json")


@nc("an index builder that opens the rows files it copies")
def test_nc_index_builder_reads_rows(tmp_path, monkeypatch):
    real_copy = BI.Builder.copy_file

    def peeking(self, src, rel):                                                           # MUTANT
        try:
            BI._open_json(src, pin=True)
        except BI.IndexRefused:
            pass
        return real_copy(self, src, rel)
    monkeypatch.setattr(BI.Builder, "copy_file", peeking)
    check_never_parses_rows(tmp_path, monkeypatch)


def test_index_builder_refusals(tmp_path):
    def fresh(name):
        b = anakit.build(tmp_path / name / "a")
        return b, layoutkit.build_layout(tmp_path / name / "lay", b)

    def refused(name, lay, needle, extra=()):
        lines = []
        out = tmp_path / name / "idx"
        rc = BI.main(lay["args"] + ["--out", str(out)] + list(extra), say=lines.append)
        assert rc == BI.REFUSED and needle in lines[0], lines
        assert not out.exists() or not any(out.rglob("*.*")), "a refusal leaves no file behind"
    _b, lay = fresh("notdone")
    p = lay["R"] / "FL" / "FA_s2027" / "RUN_SUMMARY.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s["status"] = "RUNNING"
    anakit.jdump(p, s)
    refused("notdone", lay, "DONE_MAIN")
    _b, lay = fresh("smoke")
    shutil.copytree(lay["R"] / "DEV" / "LO_s2026", lay["R"] / "DEV" / "SMOKE_LO_s2026")
    refused("smoke", lay, "smoke")
    _b, lay = fresh("seed")
    shutil.rmtree(lay["R"] / "REEVAL_A40" / "FP_s2027_PB")
    refused("seed", lay, "missing registered seed")
    lines = []
    assert BI.main(lay["args"] + ["--out", str(tmp_path / "seed" / "ok"), "--allow-single-seed", "FP"],
                   say=lines.append) == 0, lines
    _b, lay = fresh("pin")
    anakit.jdump(lay["pin"], dict(layoutkit.PIN_OBJ, self_sha256="q" * 64))
    refused("pin", lay, "device_pin_sha256")
    _b, lay = fresh("dup")
    shutil.copytree(lay["R"] / "REEVAL_A40" / "FP_s2026_PB", lay["R"] / "REEVAL_A40" / "FP_COPY_s2026_PB")
    refused("dup", lay, "duplicate entry")
    _b, lay = fresh("cell")
    p = lay["R"] / "DEV" / "LO_s2026" / "RUN_SUMMARY.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s["selected_cell"] = None
    anakit.jdump(p, s)
    refused("cell", lay, "selected_cell")
    _b, lay = fresh("hash")
    np.save(lay["R"] / "REEVAL_A40" / "C_SMALL_s2027_CB" / "evals" / "MAIN.ranks_int32.npy", np.zeros(3, np.int32))
    refused("hash", lay, "ranks_sha256")
