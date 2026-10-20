"""ANA-03: K17-K19 (06I §1), ANA_CONTRASTS_v2 / _3 formats (v1 = a VALIDATION dry check only), K20 / K21 and the DESCRIPTIVE
C_FULL rows beside K20 (06K, DL 206), the bytes-per-client column with the dropped visits charged (06I §7b), REV-DELTA-01
F1 (the index builder against the real REEVAL_RECORD format) and F9 (the deployed C_BIG `...:best` id) — synthetic blocks.

Known answers come from the block's own rank files (anakit.macro), never from the script's tables. Each MAJOR item has a
positive test and a negative control (test_nc_*, strict xfail on AssertionError: the same positive check run against a
mutant that re-introduces the defect)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

import ana_block_index as BI
import anakit
import layoutkit
import retention_report as R
from anakit import macro
from test_ana02 import build_index, edit_index
from test_retention_report import _set_rows_field, check_refuses, conf, nc, result, run

M = "MAIN"


def _crec(res, kid):
    return next(c for c in res["confirmatory"] if c["id"] == kid)


def _pb(blk, arm, seed):
    return macro(blk.ranks[(M, arm, seed, "PRACTICAL_BEST")], blk.manifests[M])


def _build(tmp, version, *, warm=True, **warm_kw):
    blk = anakit.build(tmp)
    anakit.jdump(blk.contrasts, anakit.contrasts_obj(version))
    if warm:
        anakit.add_warm_start(blk, **warm_kw)
    return blk


@pytest.fixture(scope="module")
def done_v2(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ana03_v2")
    blk = _build(tmp, 2)
    rc, text = run(blk, tmp / "out")
    assert rc == 0, text
    return blk, tmp / "out", result(tmp / "out")


@pytest.fixture(scope="module")
def done_v3(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ana03_v3")
    blk = _build(tmp, 3)
    rc, text = run(blk, tmp / "out")
    assert rc == 0, text
    return blk, tmp / "out", result(tmp / "out")


# ================================================================================================ K17-K19 (06I §1)
def test_k17_k19_known_answers(done_v2):
    blk, _out, res = done_v2
    c_full = _pb(blk, "C_FULL", 2026)
    for kid, arm, hier in (("K17", "FA", "secondary"), ("K18", "FT_C", "tertiary"), ("K19", "LO", "tertiary")):
        rec = _crec(res, kid)
        assert rec["hierarchy"] == hier and rec["registered_seeds"] == [2026] and rec["confirmatory_rule"] == "PRACTICAL_BEST"
        b = conf(res, kid)
        assert b["label"] == "CONFIRMATORY" and b["seeds"] == [2026] and b["seeds_status"] == "COMPLETE"
        want = 100.0 * _pb(blk, arm, 2026) / c_full
        assert b["point"] == pytest.approx(want, abs=1e-9)
        lo, hi = b["ci95"]
        assert lo <= b["point"] <= hi
        assert [p["den"] for p in b["per_seed"]] == ["C_FULL|2026|PRACTICAL_BEST|MAIN"]
    assert conf(res, "K17", "CONTROLLED_BUDGET")["label"] == "DESCRIPTIVE"
    assert conf(res, "K18", "CONTROLLED_BUDGET")["status"] == "NOT_AVAILABLE"       # device arms exist under PB only


def test_c_full_denominator_is_the_same_view_and_w_as_k15(done_v2):
    _blk, _out, res = done_v2
    k15, k17 = conf(res, "K15"), conf(res, "K17")
    assert k17["W_sha256"] == k15["W_sha256"] and k17["n_users"] == k15["n_users"]
    assert k17["population"] == k15["population"] == "MAIN/ALL"
    assert k15["per_seed"][0]["num"] == "C_BIG|2026|PRACTICAL_BEST|MAIN"
    assert k15["per_seed"][0]["den"] == k17["per_seed"][0]["den"]


def test_a_second_c_full_seed_is_descriptive_only(tmp_path):
    """C_FULL seed 2027 (running) is outside the registered seed (2026): it goes to extra_seeds, DESCRIPTIVE."""
    b = _build(tmp_path, 2, warm=False)
    from test_ana02 import _add_arm_seed
    _add_arm_seed(b, "C_FULL", 2027, ("PRACTICAL_BEST",))
    rc, text = run(b, tmp_path / "o")
    assert rc == 0, text
    rec = _crec(result(tmp_path / "o"), "K17")
    assert rec["by_rule"]["PRACTICAL_BEST"]["seeds"] == [2026]
    assert rec["extra_seeds"]["PRACTICAL_BEST"]["label"] == "DESCRIPTIVE"
    assert rec["extra_seeds"]["PRACTICAL_BEST"]["seeds"] == [2026, 2027]


# ================================================================================================ contrasts formats
def _refuses(fn, needle: str) -> None:
    """A plain-assert refusal check (an AssertionError, so the negative controls below can use it)."""
    try:
        fn()
    except R.AnaRefused as e:
        assert needle in str(e), str(e)
        return
    raise AssertionError("not refused")


def check_v1_refused_for_test(tmp_path):
    p = tmp_path / "v1.json"
    anakit.jdump(p, anakit.contrasts_obj(1))
    _refuses(lambda: R.load_contrasts(p), "superseded")                                # no flag: refused
    _refuses(lambda: R.load_contrasts(p, split="TEST", allow_v1=True), "superseded")   # TEST: refused even with the flag
    assert len(R.load_contrasts(p, split="VALIDATION", allow_v1=True)["items"]) == 16
    b = anakit.build(tmp_path / "b")
    anakit.jdump(b.contrasts, anakit.contrasts_obj(1))
    rc, text = run(b, tmp_path / "o1", "--allow-v1-contrasts", split="TEST")
    assert rc == R.REFUSED and "VALIDATION dry check only" in text, text
    rc, text = run(b, tmp_path / "o2", "--check-only")                                 # VALIDATION without the flag
    assert rc == R.REFUSED and "superseded" in text, text


def test_v1_is_a_validation_dry_check_only(tmp_path):
    check_v1_refused_for_test(tmp_path)
    b = anakit.build(tmp_path / "dry")
    anakit.jdump(b.contrasts, anakit.contrasts_obj(1))
    rc, text = run(b, tmp_path / "dry_out", "--allow-v1-contrasts")
    assert rc == 0, text
    res = result(tmp_path / "dry_out")
    assert res["inputs"]["contrasts_v1_dry_check"] is True and len(res["confirmatory"]) == 16
    assert res["labels"]["multiplicity"].startswith("16 pre-registered contrasts")
    assert "DRY CHECK ONLY" in (tmp_path / "dry_out" / R.REPORT_NAME).read_text(encoding="utf-8")


@nc("a loader that accepts the v1 contrasts for every split")
def test_nc_v1_accepted_for_test(tmp_path, monkeypatch):
    real = R.load_contrasts

    def lax(path, *, split="VALIDATION", allow_v1=False):                              # MUTANT
        return real(path, split="VALIDATION", allow_v1=True)
    monkeypatch.setattr(R, "load_contrasts", lax)
    check_v1_refused_for_test(tmp_path)


def check_test_needs_v3(tmp_path):
    """FIX-DELTA-03 M5 (REV-DELTA-03): a TEST run refuses every contrasts format but PPSI_V4_ANA_CONTRASTS_3 (the
    loader and the CLI); v2 stays a VALIDATION dry check."""
    p = tmp_path / "v2.json"
    anakit.jdump(p, anakit.contrasts_obj(2))
    _refuses(lambda: R.load_contrasts(p, split="TEST"), "a TEST run uses the v3 registry only")
    real = R.V4 / "company" / "gates" / "ANA_CONTRASTS_v2.json"
    _refuses(lambda: R.load_contrasts(real, split="TEST"), "superseded by PPSI_V4_ANA_CONTRASTS_3")
    assert R.load_contrasts(p, split="VALIDATION")["format"] == R.CONTRAST_FORMAT_V2
    b = anakit.build(tmp_path / "b")                                                   # the CLI: v2 on a TEST block
    rc, text = run(b, tmp_path / "o", split="TEST")
    assert rc == R.REFUSED and "v3 registry only" in text, text


def test_test_run_refuses_a_superseded_registry(tmp_path):
    check_test_needs_v3(tmp_path)


@nc("a loader that still accepts format v2 on TEST (the pre-FIX-DELTA-03 load_contrasts)")
def test_nc_v2_accepted_for_test(tmp_path, monkeypatch):
    real = R.load_contrasts

    def lax(path, *, split="VALIDATION", allow_v1=False):                              # MUTANT
        return real(path, split="VALIDATION" if split == "TEST" else split, allow_v1=allow_v1)
    monkeypatch.setattr(R, "load_contrasts", lax)
    check_test_needs_v3(tmp_path)


def test_v2_and_v3_formats_accepted(tmp_path):
    p = tmp_path / "c.json"
    anakit.jdump(p, anakit.contrasts_obj(2))
    v2 = R.load_contrasts(p, split="VALIDATION")                                       # v2: a VALIDATION dry check
    assert v2["format"] == R.CONTRAST_FORMAT_V2 and [c["id"] for c in v2["items"]] == [f"K{i}" for i in range(1, 20)]
    assert v2["multiplicity"].startswith("19 pre-registered contrasts") and "K17" in v2["multiplicity"]
    anakit.jdump(p, anakit.contrasts_obj(3))
    v3 = R.load_contrasts(p, split="TEST")                                             # ANA-04: the 06L-C §C4 registry
    assert len(v3["items"]) == 26 and v3["multiplicity"].startswith("26 pre-registered contrasts")
    assert "K20, K20b, K20c, K21" in v3["multiplicity"] and v3["content_sha256"] != v2["content_sha256"]
    assert [c["id"] for c in v3["descriptive_items"]] == ["S8-S7"] and v2["descriptive_items"] == []
    real = R.V4 / "company" / "gates" / "ANA_CONTRASTS_v2.json"
    assert R.load_contrasts(real, split="VALIDATION")["self_sha256_status"] == "VERIFIED"
    real3 = R.V4 / "company" / "gates" / "ANA_CONTRASTS_v3.json"
    assert R.load_contrasts(real3, split="TEST")["self_sha256_status"] == "VERIFIED"


def test_contrast_files_still_refuse_any_deviation(tmp_path):
    p = tmp_path / "c.json"

    def refused(obj, needle):
        anakit.jdump(p, obj)
        with pytest.raises(R.AnaRefused, match=needle):
            R.load_contrasts(p)
    obj = anakit.contrasts_obj(2)
    obj["confirmatory_contrasts"][16]["question"] = "federated retention"                # K17 question text
    refused(obj, "K17 differs")
    obj = anakit.contrasts_obj(2)
    obj["confirmatory_contrasts"][17]["contrast"] = "R(FT_C / C_BIG)"                     # K18 formula
    refused(obj, "K18 differs")
    obj = anakit.contrasts_obj(2)
    obj["confirmatory_contrasts"] = obj["confirmatory_contrasts"][:16]
    refused(obj, "not exactly K1-K19")
    obj = anakit.contrasts_obj(3)
    obj["confirmatory_contrasts"] = obj["confirmatory_contrasts"][:19]
    refused(obj, "not exactly K1-K21")
    obj = anakit.contrasts_obj(2)
    obj.pop("registered_seeds")
    refused(obj, "registered_seeds")
    refused(anakit.contrasts_obj(2, registered_seeds={"K13": [2026], "K15": [2026], "K16": [2026], "K17": [2026],
                                                      "K18": [2026, 2027], "K19": [2026]}), "registered_seeds")
    refused(anakit.contrasts_obj(3, registered_seeds={"K13": [2026], "K15": [2026], "K16": [2026], "K17": [2026],
                                                      "K18": [2026], "K19": [2026], "K20": [2026]}), "registered_seeds")
    obj = anakit.contrasts_obj(2)
    obj["hierarchy"]["secondary"].remove("K17")
    obj["hierarchy"]["tertiary"].append("K17")
    refused(obj, "hierarchy")
    refused(anakit.contrasts_obj(2, format="PPSI_V4_ANA_CONTRASTS_9"), "format")
    refused(anakit.contrasts_obj(2, multiplicity_statement="16 pre-registered contrasts"), "multiplicity_statement")
    obj = anakit.contrasts_obj(3)
    obj["confirmatory_contrasts"][19]["contrast"] = "R(FA_WARM / C_FULL)"                # K20 formula
    refused(obj, "K20 differs")


V3_ORDER = ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K9b", "K9c", "K10", "K11", "K11b", "K12", "K13", "K14",
            "K15", "K16", "K17", "K18", "K19", "K20", "K20b", "K20c", "K21"]                  # 06L-C §C4 (written out)


def test_multiplicity_line_and_labels_in_the_report(done_v2, done_v3):
    for (_b, out, res), n in ((done_v2, 19), (done_v3, 26)):
        ids = [f"K{i}" for i in range(1, 20)] if n == 19 else V3_ORDER
        assert res["labels"]["multiplicity"].startswith(f"{n} pre-registered contrasts")
        assert res["inputs"]["contrasts_format"] == f"PPSI_V4_ANA_CONTRASTS_{2 if n == 19 else 3}"
        assert len(res["confirmatory"]) == n and res["labels"]["CONFIRMATORY"] == ids
        head = (out / R.REPORT_NAME).read_text(encoding="utf-8").split("## 1.")[0]
        label = "K1-K19" if n == 19 else "K1-K21, K9b, K9c, K11b, K20b, K20c"
        assert f"{n} pre-registered contrasts" in head and f"Only section 1 ({label})" in head
        assert res["labels"]["hierarchy"]["K17"] == "secondary"
        assert res["labels"]["registered_seeds"]["K17"] == [2026] and res["labels"]["registered_seeds"]["K19"] == [2026]


# ================================================================================================ K20 / K21 (06K)
def _warm(res):
    return {c["id"]: c for c in res["warm_start"]["rows"]}


def test_k20_k21_known_answers_confirmatory_with_the_v3_file(done_v3):
    blk, _out, res = done_v3
    seeds = (2026, 2027)
    k20 = conf(res, "K20")
    want20 = np.mean([100.0 * _pb(blk, "FA_WARM", s) / _pb(blk, "C_BIG", s) for s in seeds])
    assert k20["label"] == "CONFIRMATORY" and k20["seeds"] == [2026, 2027] and k20["seeds_status"] == "COMPLETE"
    assert k20["point"] == pytest.approx(want20, abs=1e-9)
    k21 = conf(res, "K21")
    want21 = np.mean([_pb(blk, "FA_WARM", s) - _pb(blk, "PRE", 2026) for s in seeds])
    assert k21["label"] == "CONFIRMATORY" and k21["seeds"] == [2026, 2027] and k21["kind"] == "delta"
    assert k21["point"] == pytest.approx(want21, abs=1e-12) and k21["point"] > 0
    assert [p["b"] for p in k21["per_seed"]] == ["PRE|2026|PRACTICAL_BEST|MAIN"] * 2         # one theta_pre, both seeds
    assert [p["a"] for p in k21["per_seed"]] == ["FA_WARM|2026|PRACTICAL_BEST|MAIN", "FA_WARM|2027|PRACTICAL_BEST|MAIN"]
    assert 0.0 <= k21["harmed_share"] <= 1.0
    for kid in ("K20", "K21"):
        rec = _crec(res, kid)
        assert rec["hierarchy"] == "secondary" and rec["confirmatory_rule"] == "PRACTICAL_BEST"
        assert rec["registered_seeds"] == [2026, 2027]
        assert conf(res, kid, "CONTROLLED_BUDGET")["label"] == "DESCRIPTIVE"


def test_deployment_rows_k20b_k20c_are_secondary_confirmatory_with_v3(done_v3):
    """ANA-04 (06L-C §C7): R(FA_WARM / C_FULL) = K20b (2 FA_WARM seeds vs C_FULL s2026) and R(PRE / C_FULL) = K20c (PRE
    single seed) are secondary CONFIRMATORY rows of the v3 registry (ANA-03's DESCRIPTIVE K20-D1 / K20-D2)."""
    blk, out, res = done_v3
    rows = res["warm_start"]["rows"]
    assert [c["id"] for c in rows] == ["K20", "K20b", "K20c", "K21", "K9c"]
    assert all(r == {"id": r["id"], "in_confirmatory": True} for r in rows)
    c_full = _pb(blk, "C_FULL", 2026)
    d1, d2 = conf(res, "K20b"), conf(res, "K20c")
    want1 = np.mean([100.0 * _pb(blk, "FA_WARM", s) / c_full for s in (2026, 2027)])
    assert d1["point"] == pytest.approx(want1, abs=1e-9) and d1["seeds"] == [2026, 2027]
    assert [p["den"] for p in d1["per_seed"]] == ["C_FULL|2026|PRACTICAL_BEST|MAIN"] * 2        # one C_FULL seed, shared
    assert d2["point"] == pytest.approx(100.0 * _pb(blk, "PRE", 2026) / c_full, abs=1e-9) and d2["seeds"] == [2026]
    for kid in ("K20b", "K20c"):
        assert d1["label"] == d2["label"] == "CONFIRMATORY" and _crec(res, kid)["hierarchy"] == "secondary"
        assert conf(res, kid)["seeds_status"] == "COMPLETE"
    assert "K20-D1" not in json.dumps(res) and "K20-D2" not in json.dumps(res)          # the ANA-03 ids are gone
    rep = (out / R.REPORT_NAME).read_text(encoding="utf-8")
    sec = rep.split("### 1b.")[1].split("## 2.")[0]
    assert sec.index("| **K20** |") < sec.index("| **K20b** |") < sec.index("| **K20c** |") < sec.index("| **K21** |")
    for text in R.DISCLOSURES_06K:
        assert text in rep
    assert "1,217,610 users (9x MAIN)" in sec and "ONE seed (2026)" in sec and "can only understate PRE" in sec


def test_k20_k21_are_descriptive_when_the_file_does_not_carry_them(done_v2):
    blk, _out, res = done_v2
    assert "K20" not in {c["id"] for c in res["confirmatory"]}                      # the v2 file has 19 contrasts
    ws = _warm(res)
    assert list(ws) == ["K20", "K20b", "K20c", "K21", "K9c"]
    assert all(r.get("in_extra_registered") for r in ws.values())
    extra = {c["id"]: c for c in res["extra_registered"]}
    assert list(extra) == ["K9b", "K9c", "K11b", "K20", "K20b", "K20c", "K21"]     # every registered row v2 lacks
    for kid in extra:
        assert extra[kid]["label"] == "DESCRIPTIVE" and "does not carry it" in extra[kid]["note"]
        assert all(b["label"] == "DESCRIPTIVE" for b in extra[kid]["by_rule"].values())
    want = np.mean([100.0 * _pb(blk, "FA_WARM", s) / _pb(blk, "C_BIG", s) for s in (2026, 2027)])
    assert extra["K20"]["by_rule"]["PRACTICAL_BEST"]["point"] == pytest.approx(want, abs=1e-9)
    assert "superseded by PPSI_V4_ANA_CONTRASTS_3" in res["labels"]["superseded_note"]


def test_no_warm_start_entries_gives_a_not_available_section(tmp_path):
    b = _build(tmp_path, 3, warm=False)
    rc, text = run(b, tmp_path / "o")
    assert rc == 0, text
    res = result(tmp_path / "o")
    assert res["warm_start"]["status"] == "NOT_AVAILABLE" and res["warm_start"]["rows"] == []
    assert conf(res, "K20")["status"] == "NOT_AVAILABLE" and conf(res, "K21")["status"] == "NOT_AVAILABLE"


def test_pre_without_a_control_budget_entry_is_still_reported(tmp_path):
    b = _build(tmp_path, 3, pre_rules=("PRACTICAL_BEST",))
    rc, text = run(b, tmp_path / "o")
    assert rc == 0, text
    res = result(tmp_path / "o")
    assert conf(res, "K21")["label"] == "CONFIRMATORY"
    assert conf(res, "K21", "CONTROLLED_BUDGET")["status"] == "NOT_AVAILABLE"


def test_pre_is_the_endpoint_checkpoint(tmp_path):
    b = _build(tmp_path, 3, pre_cid="PRE_s2026:best@3.0", pre_efe=3.0)
    check_refuses(b, tmp_path, "RULE_CHECKPOINT")
    b2 = _build(tmp_path / "x", 3, pre_extra={"train_manifest_id": "FULL_TRAIN_5_100_TIEBREAK"})
    check_refuses(b2, tmp_path / "x", "PRE's rows carry train_manifest_id")


@nc("PRE treated like a selected arm: a PRACTICAL_BEST PRE row with the endpoint id passes")
def test_nc_pre_selected_like_other_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "NO_SELECTION_ARMS", ())                                     # MUTANT
    b = _build(tmp_path, 3)
    rc, text = run(b, tmp_path / "o")
    assert rc == 0, text


def test_k21_pairs_one_pre_seed_with_both_fa_warm_seeds(tmp_path):
    b = _build(tmp_path, 3)
    rc, text = run(b, tmp_path / "chk", "--check-only")
    assert rc == 0 and "PASS" in text, text
    b1 = _build(tmp_path / "one", 3, fa_warm_seeds=(2026,))                             # one FA_WARM seed: INCOMPLETE
    rc, text = run(b1, tmp_path / "one" / "o")
    assert rc == 0, text
    k21 = conf(result(tmp_path / "one" / "o"), "K21")
    assert k21["seeds"] == [2026] and k21["seeds_status"] == "INCOMPLETE_SEEDS" and k21["missing_registered_seeds"] == [2027]


@nc("the seed rule of every other arm applied to PRE: FA_WARM s2027 against PRE s2026 is a SEED_MISMATCH")
def test_nc_pre_not_shared_across_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SHARED_SEED_ARMS", ())                                      # MUTANT
    b = _build(tmp_path, 3)
    rc, text = run(b, tmp_path / "o")
    assert rc == 0, text
    assert conf(result(tmp_path / "o"), "K21")["seeds"] == [2026, 2027]                  # both FA_WARM seeds paired


# ================================================================================================ bytes (06I §7b)
# FIX-DELTA-02 M4 (REV-DELTA-02; DL 218): the registered charged column charges a dropped visit its DOWNLOAD only; the
# up + down variant is the DESCRIPTIVE `_up_down` column.
def _charged(seed, per_visit=anakit.FA_DOWN_BYTES):
    return anakit.FA_PER_CLIENT + anakit.FA_DROPPED[seed] * per_visit / anakit.FA_N_CLIENTS


def test_charged_bytes_known_answers_and_sources(done_v2):
    _b, out, res = done_v2
    row = next(r for r in res["retention"]["MAIN"]["PRACTICAL_BEST"] if r["arm"] == "FA")
    m26, m27 = row["entries"]["2026"]["meta"], row["entries"]["2027"]["meta"]
    assert m26["bytes_per_client"] == anakit.FA_PER_CLIENT                             # the registered mean is unchanged
    assert m26["bytes_per_client_charged_dropped"] == pytest.approx(_charged(2026), abs=1e-3)
    assert m26["charged_bytes_source"] == "PARTICIPATION_SUMMARY_COUNT" and m26["dropped_visits"] == 250
    assert m27["bytes_per_client_charged_dropped"] == pytest.approx(_charged(2027), abs=1e-3)
    assert m27["charged_bytes_source"] == "PARTICIPATION_LOG_IDS" and m27["dropped_visits"] == 100   # the log's ids
    assert m26["bytes_per_client_charged_dropped"] > m26["bytes_per_client"]
    assert m26["bytes_per_client_charged_dropped_up_down"] == pytest.approx(                   # DESCRIPTIVE variant
        _charged(2026, anakit.FA_VISIT_BYTES), abs=1e-3)
    assert (m26["bytes_per_client"] < m26["bytes_per_client_charged_dropped"]
            < m26["bytes_per_client_charged_dropped_up_down"])
    assert "bytes_per_client_charged_dropped_download_only" not in m26
    rep = (out / R.REPORT_NAME).read_text(encoding="utf-8")
    assert "bytes/client (dropped visits charged, 06I 7b)" in rep and R.DISCLOSURE_BYTES_06I_7B in rep
    assert "bytes/client (dropped visits charged down + up; DESCRIPTIVE)" in rep and "row 218" in rep
    pb_table = rep.split("### PRACTICAL_BEST")[1].split("### CONTROLLED_BUDGET")[0]
    head = next(ln for ln in pb_table.splitlines() if ln.startswith("| arm | seed |"))
    fa_row = next(ln for ln in pb_table.splitlines() if ln.startswith("| FA | 2026 |"))
    cells = [c.strip() for c in fa_row.strip("|").split("|")]
    hcells = [c.strip() for c in head.strip("|").split("|")]
    assert len(cells) == len(hcells)
    col = {h: c for h, c in zip(hcells, cells)}
    assert col["bytes/client"] == "2000000000"                                           # the registered mean
    assert col["bytes/client (dropped visits charged, 06I 7b)"] == f"{_charged(2026):.0f}"          # download-only
    assert col["bytes/client (dropped visits charged down + up; DESCRIPTIVE)"] == \
        f"{_charged(2026, anakit.FA_VISIT_BYTES):.0f}"
    ladder_fa = next(a for lv in res["privacy_ladder"] for a in lv["arms"] if a["arm"] == "FA")
    assert ladder_fa["bytes_per_client_charged_dropped"] == pytest.approx(_charged(2026), abs=1e-3)
    assert ladder_fa["bytes_per_client_charged_dropped_up_down"] == pytest.approx(
        _charged(2026, anakit.FA_VISIT_BYTES), abs=1e-3)


def test_bytes_curves_carry_the_charged_series_as_a_second_series(done_v2):
    _b, out, res = done_v2
    rows = [r for r in res["curves"]["bytes"] if r["arm"] == "FA" and r["seed"] == 2026]
    assert len(rows) == 12
    for r in rows:
        frac = r["round"] / anakit.FA_ROUNDS
        assert r["cum_bytes_per_client"] == pytest.approx(anakit.FA_PER_CLIENT * frac)             # registered series (06A §3)
        assert r["cum_bytes_per_client_charged_dropped"] == pytest.approx(_charged(2026) * frac)   # the second series
        assert r["cum_bytes_per_client_charged_dropped_up_down"] == pytest.approx(                  # DESCRIPTIVE
            _charged(2026, anakit.FA_VISIT_BYTES) * frac)
    csv = (out / "curves" / "bytes.csv").read_text(encoding="utf-8").splitlines()
    assert {"cum_bytes_per_client_charged_dropped", "cum_bytes_per_client_charged_dropped_up_down"} <= set(csv[0].split(","))


def test_charged_bytes_edge_cases():
    base = {"bytes": {"per_visit": {"send_plus_receive_bytes": 10, "download_bytes": 4}, "per_client": {"mean": 100.0}},
            "plan": {"n_clients": 10}}
    got = R.charged_bytes(dict(base, participation={"dropped": 5}))
    assert got["bytes_per_client_charged_dropped"] == 102.0 and got["bytes_per_client_charged_dropped_up_down"] == 105.0
    nodown = {"bytes": {"per_visit": {"send_plus_receive_bytes": 10}, "per_client": {"mean": 100.0}},
              "plan": {"n_clients": 10}, "participation": {"dropped": 5}}
    assert R.charged_bytes(nodown)["charged_bytes_source"] == "NO_BYTES_RECORD"            # download bytes required
    assert R.charged_bytes(nodown)["bytes_per_client_charged_dropped"] is None
    assert R.charged_bytes(dict(base, participation={"dropped": 0}))["bytes_per_client_charged_dropped"] == 100.0
    none = R.charged_bytes(dict(base))
    assert none["bytes_per_client_charged_dropped"] is None and none["charged_bytes_source"] == "NO_PARTICIPATION_RECORD"
    assert R.charged_bytes({"participation": {"dropped": 1}})["charged_bytes_source"] == "NO_BYTES_RECORD"
    assert R.charged_bytes({**base, "plan": {}, "participation": {"dropped": 1}})["charged_bytes_source"] == "NO_N_CLIENTS"
    log = [{"round": 0, "dropped": ["a", "b"]}, {"round": 1, "dropped": ["a"]}]
    assert R.charged_bytes(dict(base, participation_log=log))["dropped_visits"] == 3            # a client dropped twice: 2 visits
    with pytest.raises(R.AnaRefused, match="participation.dropped"):
        R.charged_bytes(dict(base, participation={"dropped": 2}, participation_log=log))
    with pytest.raises(R.AnaRefused, match="dropped"):
        R.charged_bytes(dict(base, participation_log=[{"round": 0}]))


@nc("a charged column that is the registered mean (dropped visits not added)")
def test_nc_charged_bytes_equal_the_registered_mean(monkeypatch):
    real = R.charged_bytes

    def mean_only(summary):                                                            # MUTANT
        out = real(summary)
        out["bytes_per_client_charged_dropped"] = (summary.get("bytes") or {}).get("per_client", {}).get("mean")
        return out
    monkeypatch.setattr(R, "charged_bytes", mean_only)
    base = {"bytes": {"per_visit": {"send_plus_receive_bytes": 10, "download_bytes": 4}, "per_client": {"mean": 100.0}},
            "plan": {"n_clients": 10}, "participation": {"dropped": 5}}
    assert R.charged_bytes(base)["bytes_per_client_charged_dropped"] > 100.0


def test_charged_primary_column_is_download_only():
    """FIX-DELTA-02 M4: the registered (primary) charged value = mean + dropped x download / N (DL 218)."""
    base = {"bytes": {"per_visit": {"send_plus_receive_bytes": 10, "download_bytes": 4}, "per_client": {"mean": 100.0}},
            "plan": {"n_clients": 10}, "participation": {"dropped": 5}}
    assert R.charged_bytes(base)["bytes_per_client_charged_dropped"] == 100.0 + 5 * 4 / 10


@nc("the up + down variant printed as the registered charged column (DL 218 says downloads only)")
def test_nc_charged_primary_is_up_down(monkeypatch):
    real = R.charged_bytes

    def up_down(summary):                                                              # MUTANT: the pre-FIX swap
        out = real(summary)
        out["bytes_per_client_charged_dropped"] = out["bytes_per_client_charged_dropped_up_down"]
        return out
    monkeypatch.setattr(R, "charged_bytes", up_down)
    test_charged_primary_column_is_download_only()


# ================================================================================================ F9 (deployed C_BIG id)
def _lock(tmp_path, monkeypatch, shas=None):
    shas = shas or {"2026": "a" * 64, "2027": "b" * 64}
    p = tmp_path / "ARCHITECTURE_LOCK.json"
    anakit.jdump(p, {"document": "ARCHITECTURE_LOCK",
                     "c_big": {"checkpoints": {s: {"best.pt": {"sha256": x, "bytes": 1}} for s, x in shas.items()}}})
    monkeypatch.setattr(R, "ARCH_LOCK_PATH", p)
    return shas


def _deployed(b, seed, sha, efe=2.5):
    """The C_BIG PRACTICAL_BEST re-evaluation as the deployed ds3r2 tool labels it: id `...:best`, the checkpoint sha."""
    _set_rows_field(b, f"MAIN/C_BIG_s{seed}/RESULT_ROWS.json", "checkpoint_id", f"02B_SASREC_MAIN_s{seed}:best",
                    where={"selection_rule": "PRACTICAL_BEST"})
    _set_rows_field(b, f"MAIN/C_BIG_s{seed}/RESULT_ROWS.json", "efe", efe, where={"selection_rule": "PRACTICAL_BEST"})
    if sha is not None:
        _set_rows_field(b, f"MAIN/C_BIG_s{seed}/RESULT_ROWS.json", "checkpoint_sha256", sha,
                        where={"selection_rule": "PRACTICAL_BEST"})


def check_deployed_best_id(tmp_path, monkeypatch):
    """The `...:best` id is accepted iff the row's checkpoint_sha256 is the lock's c_big best.pt sha of ITS seed."""
    shas = _lock(tmp_path, monkeypatch)
    b = anakit.build(tmp_path / "ok")
    anakit.jdump(b.contrasts, anakit.contrasts_obj(2))
    for seed in (2026, 2027):
        _deployed(b, seed, shas[str(seed)])
    rc, text = run(b, tmp_path / "ok" / "o")
    assert rc == 0, text                                                                # accepted: efe 2.5, sha == the lock's
    bad = anakit.build(tmp_path / "bad")
    anakit.jdump(bad.contrasts, anakit.contrasts_obj(2))
    _deployed(bad, 2026, "f" * 64)
    _deployed(bad, 2027, shas["2027"])
    check_refuses(bad, tmp_path / "bad", "RULE_CHECKPOINT")                             # another checkpoint file
    nosha = anakit.build(tmp_path / "nosha")
    anakit.jdump(nosha.contrasts, anakit.contrasts_obj(2))
    _deployed(nosha, 2026, None)
    _deployed(nosha, 2027, shas["2027"])
    check_refuses(nosha, tmp_path / "nosha", "RULE_CHECKPOINT")                         # no checkpoint_sha256 in the row
    swap = anakit.build(tmp_path / "swap")
    anakit.jdump(swap.contrasts, anakit.contrasts_obj(2))
    _deployed(swap, 2026, shas["2027"])                                                 # seed 2027's best.pt on seed 2026
    _deployed(swap, 2027, shas["2027"])
    check_refuses(swap, tmp_path / "swap", "RULE_CHECKPOINT")


def test_deployed_c_big_best_id_needs_the_lock_sha(tmp_path, monkeypatch):
    check_deployed_best_id(tmp_path, monkeypatch)


@nc("an id-form check that accepts any `...:best` checkpoint")
def test_nc_any_best_id_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "deployed_c_big_best", lambda entry, r, cid: cid.endswith(":best"))     # MUTANT
    check_deployed_best_id(tmp_path, monkeypatch)


def test_c_big_at_3_0_forms_unchanged_and_the_real_lock_is_read(tmp_path, monkeypatch):
    b = anakit.build(tmp_path / "a")
    _set_rows_field(b, "MAIN/C_BIG_s2026/RESULT_ROWS.json", "checkpoint_id", "02B_SASREC_MAIN_s2026:best@2.5",
                    where={"selection_rule": "PRACTICAL_BEST"})
    _set_rows_field(b, "MAIN/C_BIG_s2026/RESULT_ROWS.json", "efe", 2.5, where={"selection_rule": "PRACTICAL_BEST"})
    check_refuses(b, tmp_path / "a", "RULE_CHECKPOINT")                                 # best@2.5 is still not the lock's
    real = json.loads(R.ARCH_LOCK_PATH.read_text(encoding="utf-8"))["c_big"]["checkpoints"]
    for seed in ("2026", "2027"):
        assert R.c_big_best_sha256(int(seed)) == real[seed]["best.pt"]["sha256"] and len(real[seed]["best.pt"]["sha256"]) == 64
    monkeypatch.setattr(R, "ARCH_LOCK_PATH", tmp_path / "missing.json")
    assert R.c_big_best_sha256(2026) is None


# ================================================================================================ F1 (index vs reeval record)
def _reeval_dir(lay, name="C_BIG_s2026_PB"):
    return lay["R"] / "REEVAL_A40" / name


def test_index_reads_the_record_the_real_reeval_tool_writes(tmp_path):
    rc, text, b, lay = build_index(tmp_path)
    assert rc == 0, text
    rec = json.loads((_reeval_dir(lay) / "REEVAL_RECORD.json").read_text(encoding="utf-8"))
    assert rec["status"] == "COMPLETE" and "outputs" in rec and rec["n_rows"] > 0          # v4int.reeval.write_outputs
    assert set(rec["outputs"]) == {"RESULT_ROWS.json", "evals/MAIN.result.json", "evals/MAIN.ranks_int32.npy"}
    assert rec["outputs"]["RESULT_ROWS.json"] == rec["result_rows_sha256"]
    idx = json.loads((tmp_path / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))
    big = next(e for e in idx["entries"] if (e["arm"], e["seed"], e["rule"], e["role"]) ==
               ("C_BIG", 2026, "PRACTICAL_BEST", "MAIN"))
    assert big["checkpoint_sha256"] == layoutkit.ckpt_sha("C_BIG", 2026, "PRACTICAL_BEST")
    panel = next(e for e in idx["entries"] if e["role"] == "UNSEEN_HOLDOUT" and e["arm"] == "C_BIG")
    assert panel["ranks"].endswith("evals/UNSEEN_HOLDOUT.ranks_int32.npy")               # tag = the role, as the tool names it


def check_record_forms(tmp_path):
    for form in ("outputs_only", "top_level_only"):
        b = anakit.build(tmp_path / form / "a")
        lay = layoutkit.build_layout(tmp_path / form / "lay", b, record_form=form)
        rec = json.loads((_reeval_dir(lay) / "REEVAL_RECORD.json").read_text(encoding="utf-8"))
        assert ("result_rows_sha256" in rec) == (form == "top_level_only") and ("outputs" in rec) == (form == "outputs_only")
        lines = []
        rc = BI.main(lay["args"] + ["--out", str(tmp_path / form / "idx")], say=lines.append)
        assert rc == 0, (form, lines)


def test_index_accepts_top_level_and_outputs_only_records(tmp_path):
    check_record_forms(tmp_path)


@nc("the ANA-02 index builder: only the top-level hash fields are read")
def test_nc_index_reads_only_top_level_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(BI, "record_hashes", lambda rec, name, where: (rec.get("result_rows_sha256"),
                                                                       rec.get("ranks_sha256")))         # MUTANT
    check_record_forms(tmp_path)


def test_index_refuses_disagreeing_or_missing_record_hashes(tmp_path):
    def refused(name, edit, needle):
        b = anakit.build(tmp_path / name / "a")
        lay = layoutkit.build_layout(tmp_path / name / "lay", b)
        p = _reeval_dir(lay) / "REEVAL_RECORD.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        edit(rec)
        anakit.jdump(p, rec)
        lines = []
        rc = BI.main(lay["args"] + ["--out", str(tmp_path / name / "idx")], say=lines.append)
        assert rc == BI.REFUSED and needle in lines[0], lines
    refused("disagree", lambda r: r["outputs"].update({"evals/MAIN.ranks_int32.npy": "0" * 64}), "disagrees")
    refused("neither", lambda r: [r.pop("result_rows_sha256"), r.pop("ranks_sha256"), r.pop("outputs")], "carries no")
    refused("wrong_rows", lambda r: [r.update(result_rows_sha256="1" * 64), r["outputs"].update(
        {"RESULT_ROWS.json": "1" * 64})], "RESULT_ROWS.json sha256 differs")


def test_layoutkit_no_longer_fakes_the_old_record_format(tmp_path):
    src = (Path(layoutkit.__file__)).read_text(encoding="utf-8")
    assert "write_outputs" in src and '"result_rows_sha256": R._sha_file' not in src


# ================================================================================================ index: 06K arms
def test_index_maps_fa_warm_and_pre(tmp_path):
    b = anakit.build(tmp_path / "a")
    lay = layoutkit.build_layout(tmp_path / "lay", b)
    import shutil
    for seed in (2026, 2027):                                          # an FA run started from theta_pre: summary arm "FA"
        shutil.copytree(lay["R"] / "FL" / f"FA_s{seed}", lay["R"] / "FL" / f"FA_WARM_s{seed}")
        p = lay["R"] / "FL" / f"FA_WARM_s{seed}" / "RUN_SUMMARY.json"  # FIX-DELTA-03 M2: the theta_pre identity
        summ = json.loads(p.read_text(encoding="utf-8"))
        summ["arm_spec"]["init_from"] = dict(anakit.INIT_FROM)
        anakit.jdump(p, summ)
        lp = p.parent / "LAUNCH_0.json"                                # ANA-05: the 2-D S7 warm-selection binding
        anakit.jdump(lp, dict(json.loads(lp.read_text(encoding="utf-8")), warm_selection=anakit.warm_binding("FA_WARM")))
    src = b.root / "MAIN" / "C_SMALL_s2026"
    d = lay["R"] / "REEVAL_A40" / "PRE_s2026_PB"
    layoutkit.write_reeval_dir(d, src / "RESULT_ROWS.json", b.root / "MAIN" / "C_SMALL_s2026" / "ranks_PRACTICAL_BEST.npy", {
        "format": "PPSI_V4_REEVAL_RECORD_1", "arm_label": "PRE", "seed": 2026, "role": "MAIN",
        "selection_rule": "PRACTICAL_BEST", "checkpoint_id": "PRE_s2026:endpoint_6.0", "checkpoint_sha256": "9" * 64,
        "device_class": anakit.DEV, "device_pin_sha256": anakit.PIN,
        "eval_manifest_id": anakit.ROLES["MAIN"][0],
        "central_source": layoutkit.central_source_record("PRE")}, "MAIN")          # FIX-DELTA-03 M1
    lines = []
    assert BI.main(lay["args"] + ["--out", str(tmp_path / "idx")], say=lines.append) == 0, lines
    idx = json.loads((tmp_path / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))
    keys = {(e["arm"], e["seed"], e["rule"], e["role"]) for e in idx["entries"]}
    assert ("PRE", 2026, "PRACTICAL_BEST", "MAIN") in keys                       # PRE is single-seed by registration
    assert {("FA_WARM", s, r, "MAIN") for s in (2026, 2027) for r in R.RULES} <= keys
    assert ("FA", 2026, "PRACTICAL_BEST", "MAIN") in keys                        # the plain FA runs are untouched


def test_only_the_confirmatory_list_carries_the_confirmatory_label(done_v3):
    _blk, out, res = done_v3
    assert res["labels"]["CONFIRMATORY"] == V3_ORDER

    def walk(o, path=""):
        if isinstance(o, dict):
            if "label" in o and not path.startswith("confirmatory"):
                assert o["label"] != "CONFIRMATORY", path
            for k, v in o.items():
                walk(v, f"{path}/{k}" if path else k)
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
    walk(res)
