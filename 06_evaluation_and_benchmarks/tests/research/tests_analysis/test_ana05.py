"""ANA-05 — tools/retention_report.py and tools/ana_block_index.py admit the arms / files registered since ANA-04
(synthetic blocks only; no RESULT_ROWS value of a real run is read):

  * C_FULL s2027 = a DESCRIPTIVE source seed (06D §8; 06E §2 / DL 176): rows labelled by v4int.reeval (ANA-05) are
    admitted; K15 stays seed 2026 (s2027 only in the retention table / extra_seeds); unlabelled s2027 rows, labelled
    s2026 rows and labels under another arm refuse; the index checks REEVAL_RECORD central_source.seed_status;
  * FA_WARM / FA_WARM_FROZEN MAIN bound to the 2-D FL_WARM_SELECTION_S7|S8.json (06M_B B2; FL-15): the LAUNCH binding
    (entry warm_selection) with winner factors = arm_spec factors, one file per stage, the registered file (records)
    equal to it; every mismatch refuses (report and index);
  * FT_FA_WARM / FT_FA_WARM_FROZEN (06M_B B4): device arms whose start_base is the block's base run of the same seed;
    the paired delta FT_FA_WARM − FA_WARM with the harmed-client share (known answer);
  * WARM_PREVIEW / S7-S8 CALIB screens (06M M2): refused with a clear label (report and index);
  * 06M_B B5 quality during rollout (AUC, first mark reaching PRE / LAST; known answers from the rank files);
  * S8 bytes per visit / uploaded parameters from the run's own comm record;
  * the v3 confirmatory set is unchanged (26 contrasts) and every ANA-05 number is DESCRIPTIVE.
Negative controls (test_nc_*, strict xfail on AssertionError)."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import ana_block_index as BI
import anakit
import layoutkit
import retention_report as R
import test_gate_b as GB
from anakit import macro
from test_ana02 import edit_index
from test_ana04 import V3_ORDER, add_fl
from test_retention_report import _set_rows_field, check_refuses, nc, result, run

M, PB, CB = "MAIN", "PRACTICAL_BEST", "CONTROLLED_BUDGET"
FTB = dict(R.FT_WARM_BASE)                                  # the registered map at import (NC-proof fixtures)


# ------------------------------------------------------------------------------------------------ fixtures
def start_base(arm: str, seed: int, *, checkpoint="best.pt", **over) -> dict:
    """launch_device --base-kind fa_warm LAUNCH start (load_warm_base record minus the state)."""
    base = FTB[arm]
    sb = {"kind": base, "base_kind": "fa_warm", "run_id": f"FL_{base}_SMALL_s{seed}", "checkpoint": checkpoint,
          "file_sha256": R._sha_bytes(f"ckpt|{base}|{seed}|{checkpoint}".encode()),
          "base_arm_spec_sha256": anakit.warm_spec_sha(base, seed), "init_from": dict(anakit.INIT_FROM),
          "warm_lr_factor": 1.0, "warm_server_lr_factor": 1.0,
          "frozen_item_tables": (["item_embed.weight", "output_embed"] if base == "FA_WARM_FROZEN" else None),
          "base_round": 100, "base_efe": 5.5}
    sb.update(over)
    return sb


def add_ft_warm(blk, arm: str, seeds=(2026, 2027), sb_over=None, rows_over=None) -> None:
    """FT_FA_WARM(_FROZEN) device entries (launch_device rows: method = arm + the base identity labels)."""
    for seed in seeds:
        sb = start_base(arm, seed, **(sb_over or {}))
        anakit.add_plain(blk, arm, (seed,), (PB,), device_cell=True,
                         entry_extra={"start_base": sb, "start_run_id": sb["run_id"]})
        rel = f"MAIN/{arm}_s{seed}/RESULT_ROWS.json"
        labels = {"base_source": FTB[arm], "base_run_id": sb["run_id"],
                  "base_checkpoint_sha256": sb["file_sha256"], "base_init_run_id": anakit.INIT_FROM["run_id"],
                  "ft_procedure": "FT_C"}
        labels.update(rows_over or {})
        for k, v in labels.items():
            _set_rows_field(blk, rel, k, v)


def add_warm_record(blk, arm="FA_WARM", winner=(1.0, 1.0), tamper=False) -> None:
    st = R.WARM_SELECTION_STAGE[arm]
    p = anakit.write_warm_selection(blk.root / "records" / f"FL_WARM_SELECTION_{st}.json", arm, winner)
    if tamper:
        obj = json.loads(p.read_text(encoding="utf-8"))
        obj["values"] = {"f=1|sf=1": 0.2}
        p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    edit_index(blk, lambda idx: idx["records"].__setitem__(f"WARM_SELECTION_{st}", f"records/{p.name}"))


def warm_block(tmp: Path, *, ft=True, s8=True, record=True):
    """v3 registry; PRE (θ_pre, seed 2026); ranked LAST; FA_WARM MAIN run dirs bound to the S7 file (in records);
    S8 run dirs (LAUNCH binding only); FT_FA_WARM + FT_FA_WARM_FROZEN on their bases."""
    blk = anakit.build(tmp)
    anakit.jdump(blk.contrasts, anakit.contrasts_obj(3))
    anakit.add_warm_start(blk, fa_warm_seeds=())                        # PRE only (FA_WARM from run dirs below)
    anakit.add_baselines_ranked(blk, ("LAST",))
    add_fl(blk, "FA_WARM")
    if s8:
        GB.add_s8(blk)
    if record:
        add_warm_record(blk)
    if ft:
        add_ft_warm(blk, "FT_FA_WARM")
        if s8:
            add_ft_warm(blk, "FT_FA_WARM_FROZEN")
    return blk


@pytest.fixture(scope="module")
def done5(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ana05")
    blk = warm_block(tmp)
    rc, text = run(blk, tmp / "out")
    assert rc == 0, text
    return blk, tmp / "out", result(tmp / "out")


def pbv(blk, arm, seed, rule=PB):
    return macro(blk.ranks[(M, arm, seed, rule)], blk.manifests[M])


# ================================================================================================ the full block
def test_confirmatory_set_unchanged_and_ana05_descriptive(done5):
    _blk, out, res = done5
    assert res["labels"]["CONFIRMATORY"] == V3_ORDER and len(V3_ORDER) == 26
    a = res["ana05"]
    assert a["label"] == "DESCRIPTIVE" and a["production_stack"]["label"] == "DESCRIPTIVE"
    assert a["rollout_quality"]["label"] == "DESCRIPTIVE" and a["s8_comm"]["label"] == "DESCRIPTIVE"
    assert "ANA-05" in res["task"] and res["disclosures"]["ana05_methods"] == list(R.DISCLOSURES_ANA05)
    rep = (out / R.REPORT_NAME).read_text(encoding="utf-8")
    assert "## 12b. ANA-05 additions" in rep and (out / "curves" / "rollout.csv").is_file()


def test_warm_selection_status(done5):
    _blk, _out, res = done5
    st = res["ana05"]["warm_selection"]["stages"]
    assert st["S7"]["status"] == "BOUND_AND_FILE_VERIFIED", st["S7"]
    assert st["S7"]["warm_selection_sha256"] == anakit.warm_binding("FA_WARM")["sha256"]
    assert st["S8"]["status"] == "BOUND_AND_FILE_VERIFIED", st["S8"]                 # REV-ANA-05 F3: file required
    assert st["S7"]["record"]["winner_factor"] == 1.0 and "values" not in json.dumps(st["S7"]["record"])
    assert st["S7"]["bound_runs"] == [{"seed": s, "run_id": f"FL_FA_WARM_SMALL_s{s}",
                                       "arm_spec_sha256": anakit.warm_spec_sha("FA_WARM", s)} for s in (2026, 2027)]


def test_production_stack_known_answer(done5):
    blk, _out, res = done5
    rows = {r["contrast"]: r for r in res["ana05"]["production_stack"]["rows"]}
    for ft, base in R.PRODUCTION_STACK:
        r = rows[f"{ft} − {base}"]
        assert r["status"] == "COMPUTED" and r["label"] == "DESCRIPTIVE", r
        want = np.mean([pbv(blk, ft, s) - pbv(blk, base, s) for s in (2026, 2027)])
        assert r["delta"]["point"] == pytest.approx(want, abs=1e-12) and r["delta"]["seeds"] == [2026, 2027]
        assert r["delta"]["harmed_share"] == 0.0 and r["delta"]["ci95"][0] > 0      # FT ranks <= base ranks
    bases = {b["entry"]: b for b in res["ana05"]["ft_warm_bases"]}
    assert bases["FT_FA_WARM|2026|PRACTICAL_BEST|MAIN"]["base_entry"] == "FA_WARM|2026|PRACTICAL_BEST|MAIN"


def _mark_values(blk, arm, seed):
    m = blk.manifests[M]
    rd = blk.root / "MAIN" / f"FL_{arm}_s{seed}" / "evals"
    return [(mk, macro(np.load(rd / f"eval_{mk:.1f}.ranks_int32.npy").astype(np.int64), m)) for mk in anakit.MARKS]


def check_rollout(blk, res):
    rq = res["ana05"]["rollout_quality"]
    pre = pbv(blk, "PRE", 2026)
    last = macro(blk.ranks[(M, "LAST", None, "RULE_FREE")], blk.manifests[M])
    runs = {(r["arm"], r["seed"]): r for r in rq["runs"]}
    assert set(runs) == {(a, s) for a in R.ROLLOUT_ARMS for s in (2026, 2027)}
    for (arm, seed), r in runs.items():
        mv = _mark_values(blk, arm, seed)
        auc = sum(0.5 * (mv[i][1] + mv[i + 1][1]) * 0.5 for i in range(11))
        assert r["auc"] == pytest.approx(auc, abs=1e-12) and r["auc_mean"] == pytest.approx(auc / 5.5, abs=1e-12)
        for tag, ref in (("PRE", pre), ("LAST", last)):
            first = next((mk for mk, v in mv if v >= ref), "NOT_REACHED")
            assert r["reach"][tag]["first_mark_point_rule"] == first, (arm, seed, tag)
            ci = r["reach"][tag]["first_mark_ci_rule"]
            assert ci == "NOT_REACHED" or (first != "NOT_REACHED" and ci >= first)
    assert runs[("FA_WARM", 2026)]["reach"]["PRE"]["first_mark_point_rule"] == 4.5          # ranks c + 4 = PRE's
    assert runs[("FA", 2026)]["reach"]["PRE"]["first_mark_point_rule"] == 5.0
    assert len(rq["by_arm"]) == 3 and "curve" not in rq
    assert len(res["curves"]["rollout"]) == 12 * 6


def test_rollout_quality_known_answers(done5):
    blk, _out, res = done5
    check_rollout(blk, res)


def test_s8_comm_from_the_run_record(done5):
    _blk, _out, res = done5
    rows = res["ana05"]["s8_comm"]["rows"]
    assert [r["seed"] for r in rows] == [2026, 2027]
    for r in rows:
        assert r["bytes_per_visit"] == GB.S8_VISIT and r["bytes_up_per_visit"] == GB.S8_UP
        assert r["uploaded_numel_measured"] == 265162 and r["uploaded_numel_equals_registered_text"] is True
        assert r["frozen_bytes_once"] == GB.ONCE
        assert r["per_visit_ratio_vs_FA_WARM"] == pytest.approx(GB.S8_VISIT / anakit.FA_VISIT_BYTES)


def test_ft_warm_arms_in_tables_and_privacy_matrix(done5):
    _blk, _out, res = done5
    arms = {r["arm"] for r in res["retention"]["MAIN"][PB]}
    assert {"FT_FA_WARM", "FT_FA_WARM_FROZEN"} <= arms
    l4 = next(lv for lv in res["privacy_matrix"] if lv["level"].startswith("L4"))
    assert {"FT_FA_WARM", "FT_FA_WARM_FROZEN"} <= {a["arm"] for a in l4["arms"]}


def test_check_only_passes(tmp_path):
    blk = warm_block(tmp_path)
    rc, text = run(blk, tmp_path / "chk", "--check-only")
    assert rc == 0 and "CHECK-ONLY: PASS" in text, text


def test_endpoint_base_is_reported_not_paired(tmp_path):
    blk = warm_block(tmp_path, s8=False)
    edit_index(blk, lambda idx: [e.update(start_base=start_base(e["arm"], e["seed"], checkpoint="endpoint_6.0.pt",
                                                                 file_sha256="7" * 64))
                                 for e in idx["entries"] if e["arm"] == "FT_FA_WARM"])
    for s in (2026, 2027):
        _set_rows_field(blk, f"MAIN/FT_FA_WARM_s{s}/RESULT_ROWS.json", "base_checkpoint_sha256", "7" * 64)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    r = result(tmp_path / "o")["ana05"]["production_stack"]["rows"][0]
    assert r["status"] == "NOT_AVAILABLE" and all("BASE_NOT_BEST" in x["reason"] for x in r["skipped"])


# ================================================================================================ C_FULL s2027
SEED_LABELS = {"source_seed_status": "DESCRIPTIVE", "comparison_status": "DESCRIPTIVE",
               "source_seed_rule": "06D §8 NICE seed; 06E §2 / DL 176"}


def cfull27_block(tmp, labels=SEED_LABELS):
    """C_FULL s2027 entries (anakit rows carry reeval's labels; labels=None strips them: a pre-ANA-05 row)."""
    blk = anakit.build(tmp)
    anakit.jdump(blk.contrasts, anakit.contrasts_obj(3))
    anakit.add_plain(blk, "C_FULL", (2027,))
    if labels is None:
        labels = {"source_seed_status": None, "comparison_status": "REEVAL"}
    for k, v in labels.items():
        _set_rows_field(blk, "MAIN/C_FULL_s2027/RESULT_ROWS.json", k, v)
    return blk


def test_c_full_s2027_is_descriptive(tmp_path):
    blk = cfull27_block(tmp_path)
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text
    res = result(tmp_path / "o")
    k15 = next(c for c in res["confirmatory"] if c["id"] == "K15")
    assert k15["registered_seeds"] == [2026] and k15["by_rule"][PB]["seeds"] == [2026]
    assert k15["by_rule"][PB]["label"] == "CONFIRMATORY"
    assert k15["extra_seeds"][PB]["label"] == "DESCRIPTIVE" and k15["extra_seeds"][PB]["seeds"] == [2026, 2027]
    ds = res["ana05"]["descriptive_seeds"]["rows"]
    assert {(r["arm"], r["seed"], r["rule"]) for r in ds} == {("C_FULL", 2027, PB), ("C_FULL", 2027, CB)}
    row = next(r for r in res["retention"]["MAIN"][PB] if r["arm"] == "C_FULL")
    assert row["descriptive_seeds"]["seeds"] == [2027]
    k20b = next(c for c in res["confirmatory"] if c["id"] == "K20b")                  # the shared denominator: s2026
    assert "extra_seeds" not in k20b


def check_seed_label_refusals(tmp):
    blk = cfull27_block(tmp / "a", labels=None)                                      # unlabelled s2027 rows
    check_refuses(blk, tmp / "a", "DESCRIPTIVE source seed")
    blk = cfull27_block(tmp / "b")
    _set_rows_field(blk, "MAIN/C_FULL_s2026/RESULT_ROWS.json", "source_seed_status", "DESCRIPTIVE")
    check_refuses(blk, tmp / "b", "source_seed_status")                              # s2026 labelled DESCRIPTIVE
    blk = anakit.build(tmp / "c")
    _set_rows_field(blk, "MAIN/C_SMALL_s2026/RESULT_ROWS.json", "source_seed_status", "DESCRIPTIVE")
    check_refuses(blk, tmp / "c", "source_seed_status")                              # a label under another arm


def test_c_full_seed_label_refusals(tmp_path):
    check_seed_label_refusals(tmp_path)


@nc("a report without the descriptive-seed label rule (the ANA-04 check_arm_identity) must not pass the refusals")
def test_nc_seed_labels_unchecked(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "DESCRIPTIVE_SEEDS", {})
    real = R.check_arm_identity

    def no_seed_rule(entry, r):
        return real(entry, {k: v for k, v in r.items() if k != "source_seed_status"})
    monkeypatch.setattr(R, "check_arm_identity", no_seed_rule)
    check_seed_label_refusals(tmp_path)


# ================================================================================================ warm selection (report)
def _strip_ws(idx):
    for e in idx["entries"]:
        if e["arm"] == "FA_WARM":
            e.pop("warm_selection", None)


def _ws(**over):
    def f(idx):
        for e in idx["entries"]:
            if e["arm"] == "FA_WARM":
                e["warm_selection"] = dict(e["warm_selection"], **over)
    return f


def check_warm_selection_refusals(tmp):
    cases = [("none", _strip_ws, "not bound"),
             ("factor", _ws(winner_factor=0.25), "winner_factor"),
             ("format", _ws(format="PPSI_V4_FL_WARM_SELECTION_1"), "format"),
             ("stage", _ws(stage="S8"), "stage"),
             ("dry", _ws(dry_run=True), "dry_run")]
    for tag, fn, needle in cases:
        blk = warm_block(tmp / tag, ft=False, s8=False, record=False)
        edit_index(blk, fn)
        check_refuses(blk, tmp / tag, needle)
    blk = warm_block(tmp / "two", ft=False, s8=False, record=False)                  # two files in one stage
    edit_index(blk, lambda idx: [e.update(warm_selection=dict(e["warm_selection"], sha256="3" * 64))
                                 for e in idx["entries"] if e["arm"] == "FA_WARM" and e["seed"] == 2027])
    check_refuses(blk, tmp / "two", "different warm selections")
    blk = warm_block(tmp / "rec", ft=False, s8=False, record=False)                  # the file != the binding
    add_warm_record(blk, winner=(0.25, 1.0))
    check_refuses(blk, tmp / "rec", "is not the registered")
    blk = warm_block(tmp / "tamp", ft=False, s8=False, record=False)                 # a tampered file
    add_warm_record(blk, tamper=True)
    check_refuses(blk, tmp / "tamp", "not a verified")
    blk = anakit.build(tmp / "fa")                                                   # a binding on a cold arm
    edit_index(blk, lambda idx: [e.update(warm_selection=anakit.warm_binding("FA_WARM")) for e in idx["entries"]
                                 if e["arm"] == "FA"])
    check_refuses(blk, tmp / "fa", "FA_WARM / FA_WARM_FROZEN's only")


def test_warm_selection_refusals(tmp_path):
    check_warm_selection_refusals(tmp_path)


def test_warm_selection_with_matching_factors_passes(tmp_path):
    blk = anakit.build(tmp_path)
    add_fl(blk, "FA_WARM", patch=lambda s: dict(s, arm_spec=dict(s["arm_spec"], warm_lr_factor=0.25,
                                                                 warm_server_lr_factor=0.1)))
    edit_index(blk, lambda idx: [e.update(warm_selection=anakit.warm_binding("FA_WARM", (0.25, 0.1)))
                                 for e in idx["entries"] if e["arm"] == "FA_WARM"])
    add_warm_record(blk, winner=(0.25, 0.1))
    rc, text = run(blk, tmp_path / "o")
    assert rc == 0, text


@nc("a report without the 2-D warm-selection binding (the ANA-04 load_marks) must not pass the refusals")
def test_nc_warm_selection_unchecked(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "warm_selection_problem", lambda arm, ws, spec: None)
    monkeypatch.setattr(R, "check_ana05", lambda blk, records, *a: {"warm_selection": {}, "ft_warm_bases": []})
    check_warm_selection_refusals(tmp_path)


def check_preview_refused(tmp):
    blk = anakit.build(tmp)
    add_fl(blk, "FA_WARM", patch=lambda s: dict(s, arm_spec=dict(s["arm_spec"], role="CALIBRATION")))
    check_refuses(blk, tmp, "WARM_PREVIEW")


def test_warm_preview_run_refused(tmp_path):
    check_preview_refused(tmp_path)


@nc("a report that ignores the run's role (a CALIB warm preview run) must not pass the preview refusal")
def test_nc_preview_admitted(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "PREVIEW_LABEL", "not a MAIN run")
    real = R.load_marks

    def roleless(block, entry, manifest):
        p = R._inside(block, entry.run_dir_rel) / "RUN_SUMMARY.json"
        s = json.loads(p.read_text(encoding="utf-8"))
        s["arm_spec"].pop("role", None)
        p.write_text(json.dumps(s), encoding="utf-8")
        return real(block, entry, manifest)
    monkeypatch.setattr(R, "load_marks", roleless)
    check_preview_refused(tmp_path)


# ================================================================================================ FT_FA_WARM (report)
def check_ft_warm_refusals(tmp):
    cases = [("nobase", lambda b: edit_index(b, lambda idx: [e.pop("start_base") for e in idx["entries"]
                                                              if e["arm"] == "FT_FA_WARM"]), "no start_base"),
             ("kind", lambda b: edit_index(b, lambda idx: [e["start_base"].update(kind="FA_WARM_FROZEN")
                                                           for e in idx["entries"] if e["arm"] == "FT_FA_WARM"]),
              "'kind'"),
             ("init", lambda b: edit_index(b, lambda idx: [e["start_base"].update(
                 init_from=dict(anakit.INIT_FROM, train_manifest="MAIN_TRAIN_5_100_131072"))
                 for e in idx["entries"] if e["arm"] == "FT_FA_WARM"]), "init_from"),
             ("spec", lambda b: edit_index(b, lambda idx: [e["start_base"].update(base_arm_spec_sha256="4" * 64)
                                                           for e in idx["entries"] if e["arm"] == "FT_FA_WARM"]),
              "is not the block's"),
             ("rows", lambda b: _set_rows_field(b, "MAIN/FT_FA_WARM_s2026/RESULT_ROWS.json", "base_run_id", "X"),
              "base identity"),
             ("nofa", lambda b: edit_index(b, lambda idx: idx.__setitem__("entries", [
                 e for e in idx["entries"] if e["arm"] != "FA_WARM"])), "no FA_WARM MAIN entry"),
             ("ftfa", lambda b: _set_rows_field(b, "MAIN/FT_FA_s2026/RESULT_ROWS.json", "base_source", "FA_WARM"),
              "fine-tuned from a FA_WARM base")]
    for tag, fn, needle in cases:
        blk = warm_block(tmp / tag, s8=False, record=False)
        fn(blk)
        check_refuses(blk, tmp / tag, needle)
    blk = anakit.build(tmp / "cold")                                                 # start_base on another arm
    edit_index(blk, lambda idx: [e.update(start_base=start_base("FT_FA_WARM", 2026)) for e in idx["entries"]
                                 if e["arm"] == "FT_FA"])
    check_refuses(blk, tmp / "cold", "FT_FA_WARM / FT_FA_WARM_FROZEN's only")


def test_ft_warm_refusals(tmp_path):
    check_ft_warm_refusals(tmp_path)


def test_ft_warm_base_problem_known_answers():
    assert R.ft_warm_base_problem("FT_FA_WARM", start_base("FT_FA_WARM", 2026)) is None
    assert R.ft_warm_base_problem("FT_FA_WARM_FROZEN", start_base("FT_FA_WARM_FROZEN", 2027)) is None
    for over in ({"base_kind": "fl"}, {"checkpoint": "latest.pt"}, {"file_sha256": "x"},
                 {"run_id": "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026"}, {"frozen_item_tables": ["x"]}):
        assert R.ft_warm_base_problem("FT_FA_WARM", start_base("FT_FA_WARM", 2026, **over)), over
    assert R.ft_warm_base_problem("FT_FA_WARM", start_base("FT_FA_WARM", 2026), "OTHER_RUN")


@nc("a report without the FT_FA_WARM base binding (entry start_base / rows / block base run unchecked) must not pass")
def test_nc_ft_warm_base_unchecked(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ft_warm_base_problem", lambda arm, sb, start_run_id=None: None)
    monkeypatch.setattr(R, "check_ana05", lambda blk, records, *a: {"warm_selection": {}, "ft_warm_bases": []})
    real = R.check_arm_identity

    def no_base_rows(entry, r):
        return real(entry, {k: v for k, v in r.items() if k != "base_source"}) if entry.arm.startswith("FT_FA") \
            else real(entry, r)
    monkeypatch.setattr(R, "check_arm_identity", no_base_rows)
    monkeypatch.setattr(R, "FT_WARM_BASE", {})
    check_ft_warm_refusals(tmp_path)


@nc("a production-stack delta that pairs FT_FA_WARM with the wrong base arm (FA) must not pass the known answer")
def test_nc_production_stack_wrong_base(tmp_path_factory, monkeypatch):
    monkeypatch.setattr(R, "PRODUCTION_STACK", (("FT_FA_WARM", "FA"), ("FT_FA_WARM_FROZEN", "FA_WARM_FROZEN")))
    tmp = tmp_path_factory.mktemp("nc_ps")
    blk = warm_block(tmp)
    rc, text = run(blk, tmp / "out")
    assert rc == 0, text
    res = result(tmp / "out")
    rows = {r["contrast"]: r for r in res["ana05"]["production_stack"]["rows"]}
    r = rows.get("FT_FA_WARM − FA_WARM") or rows["FT_FA_WARM − FA"]
    want = np.mean([pbv(blk, "FT_FA_WARM", s) - pbv(blk, "FA_WARM", s) for s in (2026, 2027)])
    assert r["delta"]["point"] == pytest.approx(want, abs=1e-12)


@nc("a rollout first-mark rule with a strict '>' (reaching = exceeding) must not pass the known answer")
def test_nc_rollout_strict_first_mark(tmp_path_factory, monkeypatch):
    real = R.rollout_quality

    def strict(eng, book):
        out = real(eng, book)
        for r in out["runs"]:
            ref = r["reach"].get("PRE") or {}
            if ref.get("first_mark_point_rule") == 4.5:
                ref["first_mark_point_rule"] = 5.0          # MUTANT: the equal-value mark does not count
        return out
    monkeypatch.setattr(R, "rollout_quality", strict)
    tmp = tmp_path_factory.mktemp("nc_ro")
    blk = warm_block(tmp, ft=False, s8=True, record=False)
    rc, text = run(blk, tmp / "out")
    assert rc == 0, text
    check_rollout(blk, result(tmp / "out"))


# ================================================================================================ the index builder
def _index(lay, out, extra=()):
    lines = []
    rc = BI.main(lay["args"] + list(extra) + ["--out", str(out)], say=lines.append)
    return rc, "\n".join(lines)


def _add_warm_fl_dirs(lay, blk, arm="FA_WARM", binding=True, launch_over=None, legs=1):
    for seed in (2026, 2027):
        src = blk.root / "MAIN" / f"FL_{arm}_s{seed}"
        d = lay["R"] / "FL" / f"{arm}_s{seed}"
        shutil.copytree(src, d)
        p = d / "RUN_SUMMARY.json"
        s = json.loads(p.read_text(encoding="utf-8"))
        s["arm_spec"]["cfg"]["seed"] = seed
        anakit.jdump(p, s)
        for k in range(legs):
            rec = {"label": "PRODUCTION", "arm": "FA", "fl05": {"role": "MAIN"}, "device_pin_sha256": anakit.PIN}
            if binding:
                rec["warm_selection"] = anakit.warm_binding(arm)
            rec.update((launch_over or {}).get(k, {}))
            anakit.jdump(d / f"LAUNCH_{k}.json", rec)


def _add_ft_dev_dirs(lay, blk, arm="FT_FA_WARM", dirname=None, start_over=None, summary_arm=None):
    for seed in (2026, 2027):
        d = lay["R"] / "DEV" / ((dirname or arm) + f"_s{seed}")
        src = blk.root / "MAIN" / f"{arm}_s{seed}"
        layoutkit._cp(src / "RESULT_ROWS.json", d / "RESULT_ROWS.json")
        layoutkit._cp(src / "ranks_PRACTICAL_BEST.npy", d / "cells" / f"{anakit.DEVICE_CELL}.ranks_int32.npy")
        anakit.jdump(d / "RUN_SUMMARY.json", {"status": "DONE", "arm": summary_arm or arm, "role": "MAIN",
                                              "selected_cell": anakit.DEVICE_CELL,
                                              "spec": {"seed": seed, "select": False}, "device_pin_sha256": anakit.PIN})
        anakit.jdump(d / "LAUNCH_0.json", {"label": "PRODUCTION", "arm": summary_arm or arm,
                                           "start": dict(start_base(arm, seed), **(start_over or {}))})


def _pre_reeval(lay, blk):
    d = lay["R"] / "REEVAL_A40" / "PRE_s2026_PB"
    layoutkit.write_reeval_dir(d, blk.root / "MAIN" / "PRE_s2026" / "RESULT_ROWS.json",
                               blk.root / "MAIN" / "PRE_s2026" / f"ranks_{PB}.npy", {
                                   "format": "PPSI_V4_REEVAL_RECORD_1", "arm_label": "PRE", "seed": 2026,
                                   "role": "MAIN", "selection_rule": PB, "checkpoint_id": "PRE_s2026:endpoint_6.0",
                                   "checkpoint_sha256": "9" * 64, "device_class": anakit.DEV,
                                   "device_pin_sha256": anakit.PIN, "eval_manifest_id": anakit.ROLES["MAIN"][0],
                                   "central_source": layoutkit.central_source_record("PRE")}, "MAIN")


@pytest.fixture(scope="module")
def lay5(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lay5")
    blk = warm_block(tmp / "b", s8=False, record=False)
    lay = layoutkit.build_layout(tmp / "lay", anakit.build(tmp / "base"))
    _add_warm_fl_dirs(lay, blk)
    _add_ft_dev_dirs(lay, blk)
    _pre_reeval(lay, blk)
    rec = tmp / "wsel"
    anakit.write_warm_selection(rec / "FL_WARM_SELECTION_S7.json", "FA_WARM")
    return blk, lay, tmp, rec / "FL_WARM_SELECTION_S7.json"


def test_index_admits_bound_fa_warm_and_ft_fa_warm_end_to_end(lay5):
    blk, lay, tmp, wsel = lay5
    rc, text = _index(lay, tmp / "idx_ok", ["--records", str(wsel)])
    assert rc == 0, text
    idx = json.loads((tmp / "idx_ok" / R.INDEX_NAME).read_text(encoding="utf-8"))
    fw = [e for e in idx["entries"] if e["arm"] == "FA_WARM"]
    assert len(fw) == 4 and all(e["warm_selection"]["sha256"] == anakit.warm_binding("FA_WARM")["sha256"] for e in fw)
    ft = [e for e in idx["entries"] if e["arm"] == "FT_FA_WARM"]
    assert {e["seed"] for e in ft} == {2026, 2027} and all(e["start_base"]["base_kind"] == "fa_warm" for e in ft)
    assert idx["records"]["WARM_SELECTION_S7"] == "records/FL_WARM_SELECTION_S7.json"
    lines = []
    argv = ["--block", str(tmp / "idx_ok"), "--split", "VALIDATION", "--contrasts", str(blk.contrasts)]
    assert R.main(argv + ["--out", str(tmp / "rep_ok")], context_loader=blk.loader, say=lines.append) == 0, lines
    res = result(tmp / "rep_ok")
    assert res["ana05"]["warm_selection"]["stages"]["S7"]["status"] == "BOUND_AND_FILE_VERIFIED"
    assert res["ana05"]["production_stack"]["rows"][0]["status"] == "COMPUTED"


def check_index_refusals(lay5, tag=""):
    blk, lay, tmp, wsel = lay5
    other = tmp / f"wsel_other{tag}" / "FL_WARM_SELECTION_S7.json"
    anakit.write_warm_selection(other, "FA_WARM", (0.25, 1.0))
    rc, text = _index(lay, tmp / f"i_rec{tag}", ["--records", str(other)])          # the file != the binding
    assert rc == BI.REFUSED and "is not the warm selection" in text, text

    def with_fl(name, fn, needle, **kw):
        base = lay["R"] / "FL"
        saved = {s: (base / f"FA_WARM_s{s}") for s in (2026, 2027)}
        bak = tmp / f"bak_{name}{tag}"
        for s, d in saved.items():
            shutil.move(str(d), str(bak / d.name))
        try:
            _add_warm_fl_dirs(lay, blk, **kw)
            fn()
            rc, text = _index(lay, tmp / f"i_{name}{tag}")
            assert rc == BI.REFUSED and needle in text, (name, text)
        finally:
            for s, d in saved.items():
                shutil.rmtree(d, ignore_errors=True)
                shutil.move(str(bak / d.name), str(d))
    with_fl("nobind", lambda: None, "not bound", binding=False)
    with_fl("legs", lambda: None, "one warm_selection", legs=2,
            launch_over={1: {"warm_selection": anakit.warm_binding("FA_WARM", (0.25, 1.0))}})
    with_fl("factor", lambda: None, "winner_factor", launch_over={0: {"warm_selection": anakit.warm_binding(
        "FA_WARM", (0.25, 1.0))}})
    with_fl("calib", lambda: None, "WARM_PREVIEW", launch_over={0: {"fl05": {"role": "CALIBRATION"}}})


def test_index_warm_refusals(lay5):
    check_index_refusals(lay5)


@nc("an index builder without the LAUNCH warm-selection binding (the ANA-04 fl()) must not pass the refusals")
def test_nc_index_warm_binding_unchecked(lay5, monkeypatch):
    monkeypatch.setattr(BI, "warm_selection_of", lambda d, arm, summ: None)
    monkeypatch.setattr(BI, "check_warm_preview", lambda d, summ, launch: None)
    check_index_refusals(lay5, tag="nc")


def test_index_refuses_a_warm_preview_directory(tmp_path):
    b = anakit.build(tmp_path / "a")
    blk = warm_block(tmp_path / "w", ft=False, s8=False, record=False)
    lay = layoutkit.build_layout(tmp_path / "lay", b)
    prev = lay["R"] / "WARM_PREVIEW"
    shutil.copytree(blk.root / "MAIN" / "FL_FA_WARM_s2026", prev / "WP1_fa_warm_f1")
    anakit.jdump(prev / "WP1_fa_warm_f1" / "LAUNCH_0.json", {"label": "PRODUCTION", "fl05": {"role": "CALIBRATION"}})
    rc, text = _index(lay, tmp_path / "idx", ["--fl", str(prev)])
    assert rc == BI.REFUSED and "WARM_PREVIEW" in text and "06M M2" in text, text


def check_index_ft_refusals(lay5, tag=""):
    blk, lay, tmp, _w = lay5
    dev = lay["R"] / "DEV"

    def with_dev(name, needle, **kw):
        """The good FT_FA_WARM_s<S> dirs are set aside; the mutated dirs are the only FA_WARM-based device runs."""
        hold = tmp / f"hold_{name}{tag}"
        good = [dev / f"FT_FA_WARM_s{s}" for s in (2026, 2027)]
        for d in good:
            shutil.move(str(d), str(hold / d.name))
        before = set(dev.iterdir())
        try:
            _add_ft_dev_dirs(lay, blk, **kw)
            rc, text = _index(lay, tmp / f"f_{name}{tag}")
            assert rc == BI.REFUSED and needle in text, (name, text)
        finally:
            for d in set(dev.iterdir()) - before:
                shutil.rmtree(d, ignore_errors=True)
            for d in good:
                shutil.move(str(hold / d.name), str(d))
    with_dev("asftfa", "--base-kind fa_warm run", dirname="FT_FA_X", summary_arm="FT_FA")
    with_dev("frozendir", "--base-kind fa_warm run", dirname="FT_FA_WARM_FROZEN")
    with_dev("kind", "registered FA_WARM MAIN checkpoint", dirname="FT_FA_WARM", start_over={"kind": "FA"})


def test_index_ft_warm_refusals(lay5):
    check_index_ft_refusals(lay5)


@nc("an index builder that treats an FA_WARM-based device run like any device run must not pass the refusals")
def test_nc_index_ft_warm_unchecked(lay5, monkeypatch):
    monkeypatch.setattr(R, "ft_warm_base_problem", lambda arm, sb, start_run_id=None: None)
    monkeypatch.setattr(R, "FT_WARM_BASE", {})
    check_index_ft_refusals(lay5, tag="nc")


def _cfull_reeval(lay, blk, seed, seed_status):
    d = lay["R"] / "REEVAL_A40" / f"C_FULL_s{seed}_PB"
    cs = layoutkit.central_source_record("C_FULL")
    if seed_status is not None:
        cs["seed_status"] = seed_status
    layoutkit.write_reeval_dir(d, blk.root / "MAIN" / f"C_FULL_s{seed}" / "RESULT_ROWS.json",
                               blk.root / "MAIN" / f"C_FULL_s{seed}" / f"ranks_{PB}.npy", {
                                   "format": "PPSI_V4_REEVAL_RECORD_1", "arm_label": "C_FULL", "seed": seed,
                                   "role": "MAIN", "selection_rule": PB, "checkpoint_id": f"C_FULL_s{seed}:best",
                                   "checkpoint_sha256": "8" * 64, "device_class": anakit.DEV,
                                   "device_pin_sha256": anakit.PIN, "eval_manifest_id": anakit.ROLES["MAIN"][0],
                                   "central_source": cs}, "MAIN")
    return d


def check_index_cfull(tmp):
    blk = cfull27_block(tmp / "b")
    lay = layoutkit.build_layout(tmp / "lay", anakit.build(tmp / "base"))
    d = _cfull_reeval(lay, blk, 2027, "DESCRIPTIVE")
    rc, text = _index(lay, tmp / "ok")
    assert rc == 0, text
    shutil.rmtree(d)
    _cfull_reeval(lay, blk, 2027, None)
    rc, text = _index(lay, tmp / "no")
    assert rc == BI.REFUSED and "seed_status" in text, text
    shutil.rmtree(d)
    p = lay["R"] / "REEVAL_A40" / "C_FULL_s2026_PB" / "REEVAL_RECORD.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["central_source"]["seed_status"] = "DESCRIPTIVE"
    anakit.jdump(p, rec)
    rc, text = _index(lay, tmp / "s26")
    assert rc == BI.REFUSED and "seed_status" in text, text


def test_index_c_full_seed_status(tmp_path):
    check_index_cfull(tmp_path)


@nc("the ANA-04 check_central_source (no seed_status rule) must not pass the C_FULL seed refusals")
def test_nc_index_seed_status_unchecked(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "DESCRIPTIVE_SEEDS", {})
    monkeypatch.setattr(R, "SOURCE_TRAIN_MANIFEST", dict(R.SOURCE_TRAIN_MANIFEST))
    real = BI.check_central_source

    def old(arm, rec, where):
        rec = copy.deepcopy(rec)
        (rec.get("central_source") or {}).pop("seed_status", None)
        return real(arm, rec, where)
    monkeypatch.setattr(BI, "check_central_source", old)
    check_index_cfull(tmp_path)



# ================================================================================================ REV-ANA-05 review fixes
# F3: the S7 / S8 file is REQUIRED in records on a VALIDATION block with warm entries
def check_record_required(tmp):
    blk = warm_block(tmp / "a", ft=False, s8=False)                                  # run-folder entries
    edit_index(blk, lambda idx: idx["records"].pop("WARM_SELECTION_S7"))
    check_refuses(blk, tmp / "a", "LAUNCH_BINDING_ONLY")
    blk = anakit.build(tmp / "b")                                                    # REEVAL-shaped entries
    anakit.add_warm_start(blk)
    edit_index(blk, lambda idx: idx["records"].pop("WARM_SELECTION_S7"))
    check_refuses(blk, tmp / "b", "LAUNCH_BINDING_ONLY")


def test_f3_selection_file_required_on_validation(tmp_path):
    check_record_required(tmp_path)


@nc("REV-ANA-05 F3: a report that accepts a VALIDATION warm block without the S7 / S8 file must not pass")
def test_nc_f3_file_optional(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "require_warm_record", lambda arm, st, entries, rec: None)
    check_record_required(tmp_path)


# F2: a binding without `path` refuses
def check_path_required(tmp):
    blk = warm_block(tmp, ft=False, s8=False)
    edit_index(blk, lambda idx: [e["warm_selection"].pop("path") for e in idx["entries"] if e["arm"] == "FA_WARM"])
    check_refuses(blk, tmp, "'path'")
    assert R.warm_selection_problem("FA_WARM", {k: v for k, v in anakit.warm_binding("FA_WARM").items()
                                                if k != "path"}, {})


def test_f2_binding_without_path_refused(tmp_path):
    check_path_required(tmp_path)


@nc("REV-ANA-05 F2: the ANA-05 name check (skipped when `path` is absent) must not pass the path refusal")
def test_nc_f2_path_optional(tmp_path, monkeypatch):
    real = R.warm_selection_problem

    def skip_absent(arm, ws, spec):                                                  # MUTANT: the ANA-05 rule
        if isinstance(ws, dict) and not ws.get("path"):
            ws = dict(ws, path=f"x/FL_WARM_SELECTION_{R.WARM_SELECTION_STAGE[arm]}.json")
        return real(arm, ws, spec)
    monkeypatch.setattr(R, "warm_selection_problem", skip_absent)
    check_path_required(tmp_path)


# F1 (VALIDATION): a warm entry without a run folder binds through warm_run + its rows, else refuses
def _rows_only(tmp, fn=None):
    blk = anakit.build(tmp)
    anakit.jdump(blk.contrasts, anakit.contrasts_obj(3))
    anakit.add_warm_start(blk)
    if fn is not None:
        fn(blk)
    return blk


def _entries(fn):
    return lambda b: edit_index(b, lambda idx: [fn(e) for e in idx["entries"] if e["arm"] == "FA_WARM"])


def check_rows_only_binding(tmp):
    cases = [("nows", _entries(lambda e: [e.pop("warm_selection"), e.pop("warm_run")]), "without a run folder"),
             ("norun", _entries(lambda e: e.pop("warm_run")), "no warm_selection / warm_run"),
             ("rowid", lambda b: _set_rows_field(b, "MAIN/FA_WARM_s2026/RESULT_ROWS.json", "fl_run_id", "FL_OTHER"),
              "rows fl_run_id"),
             ("factor", _entries(lambda e: e["warm_run"].update(warm_lr_factor=0.25)), "winner_factor"),
             ("seed", _entries(lambda e: e["warm_run"].update(seed=2028)), "warm_run.seed")]
    for tag, fn, needle in cases:
        blk = _rows_only(tmp / tag, fn)
        check_refuses(blk, tmp / tag, needle)


def test_f1_rows_only_warm_entries_bound_or_refused(tmp_path):
    blk = _rows_only(tmp_path / "ok")
    rc, text = run(blk, tmp_path / "ok" / "out")
    assert rc == 0, text
    st = result(tmp_path / "ok" / "out")["ana05"]["warm_selection"]["stages"]["S7"]
    assert st["status"] == "BOUND_AND_FILE_VERIFIED" and [r["seed"] for r in st["bound_runs"]] == [2026, 2027]
    check_rows_only_binding(tmp_path)


@nc("REV-ANA-05 F1: the ANA-05 PARTLY_UNBOUND status (a run-folder-less warm entry admitted) must not pass")
def test_nc_f1_rows_only_unbound(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "warm_run_problem", lambda arm, e: None)
    check_rows_only_binding(tmp_path)


# F1 (TEST): every warm entry must be a run bound on VALIDATION
@pytest.fixture(scope="module")
def warm_val(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("warm_val")
    blk = _rows_only(tmp / "v")
    rc, text = run(blk, tmp / "out")
    assert rc == 0, text
    return tmp / "out" / R.RESULT_NAME


def check_test_binding(tmp, warm_val):
    b = anakit.build_test(tmp / "id", warm_val)
    anakit.add_warm_start(b)
    _set_rows_field(b, "MAIN/FA_WARM_s2027/RESULT_ROWS.json", "fl_run_id", "FL_FA_WARM_OTHER_s2027")
    check_refuses(b, tmp / "id", "bound on VALIDATION")
    b = anakit.build_test(tmp / "nov", anakit.validation_result_v3(tmp / "nov_v3"))  # VALIDATION bound no warm run
    anakit.add_warm_start(b)
    check_refuses(b, tmp / "nov", "bound on VALIDATION")


def test_f1_test_entries_bind_to_validation_runs(tmp_path, warm_val):
    b = anakit.build_test(tmp_path / "ok", warm_val)
    anakit.add_warm_start(b)
    rc, text = run(b, tmp_path / "ok" / "out")
    assert rc == 0, text
    st = result(tmp_path / "ok" / "out")["ana05"]["warm_selection"]["stages"]["S7"]
    assert st["status"].startswith("BOUND_TO_VALIDATION_RUNS") and len(st["bound_runs"]) == 2
    check_test_binding(tmp_path, warm_val)


class _Anything(set):
    def __contains__(self, x):
        return True


@nc("REV-ANA-05 F1: a TEST warm entry not checked against the VALIDATION-bound runs must not pass")
def test_nc_f1_test_unbound(tmp_path, warm_val, monkeypatch):
    monkeypatch.setattr(R, "runs_bound_on_validation", lambda validation, st: _Anything())
    check_test_binding(tmp_path, warm_val)


# F1 (index): a warm --reeval dir carries its FL run's LAUNCH binding + identity
def _warm_reeval_layout(tmp, *, launch_over=None, source=True, tamper=False):
    blk = _rows_only(tmp / "b")
    lay = layoutkit.build_layout(tmp / "lay", anakit.build(tmp / "base"))
    for seed in (2026, 2027):
        src = tmp / "flsrc" / f"FA_WARM_s{seed}"
        src.mkdir(parents=True)
        la = {"label": "PRODUCTION", "arm": "FA_WARM", "run_id": f"FL_FA_WARM_SMALL_s{seed}",
              "arm_spec_sha256": anakit.warm_spec_sha("FA_WARM", seed),
              "arm_spec": {"arm": "FA_WARM", "cfg": {"seed": seed}, "init_from": dict(anakit.INIT_FROM)},
              "warm_selection": anakit.warm_binding("FA_WARM"), "fl05": {"role": "MAIN"}}
        la.update(launch_over or {})
        anakit.jdump(src / "LAUNCH_0.json", la)
        rec = {"format": "PPSI_V4_REEVAL_RECORD_1", "arm_label": "FA_WARM", "seed": seed, "role": "MAIN",
               "selection_rule": PB, "checkpoint_id": f"FL_FA_WARM_SMALL_s{seed}:best@3.0",
               "checkpoint_sha256": R._sha_bytes(f"fw{seed}".encode()), "device_class": anakit.DEV,
               "device_pin_sha256": anakit.PIN, "eval_manifest_id": anakit.ROLES["MAIN"][0], "source_run_id":
               la["run_id"]}
        if source:
            rec["inputs"] = {"source": {"run_dir": str(src), "launch_files": {"LAUNCH_0.json": R._sha_file(
                src / "LAUNCH_0.json")}}}
        layoutkit.write_reeval_dir(lay["R"] / "REEVAL_A40" / f"FA_WARM_s{seed}_PB",
                                   blk.root / "MAIN" / f"FA_WARM_s{seed}" / "RESULT_ROWS.json",
                                   blk.root / "MAIN" / f"FA_WARM_s{seed}" / f"ranks_{PB}.npy", rec, "MAIN")
        if tamper:
            anakit.jdump(src / "LAUNCH_0.json", dict(la, note="edited after the re-evaluation"))
    wsel = anakit.write_warm_selection(tmp / "wsel" / "FL_WARM_SELECTION_S7.json", "FA_WARM")
    return blk, lay, wsel


def test_f1_index_binds_a_warm_reeval_entry(tmp_path):
    blk, lay, wsel = _warm_reeval_layout(tmp_path)
    rc, text = _index(lay, tmp_path / "idx", ["--records", str(wsel)])
    assert rc == 0, text
    idx = json.loads((tmp_path / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))
    fw = sorted((e for e in idx["entries"] if e["arm"] == "FA_WARM"), key=lambda e: e["seed"])
    assert [e["warm_run"]["run_id"] for e in fw] == ["FL_FA_WARM_SMALL_s2026", "FL_FA_WARM_SMALL_s2027"]
    assert all(e["warm_selection"]["sha256"] == anakit.warm_binding("FA_WARM")["sha256"] for e in fw)
    lines = []
    argv = ["--block", str(tmp_path / "idx"), "--split", "VALIDATION", "--contrasts", str(blk.contrasts)]
    assert R.main(argv + ["--out", str(tmp_path / "chk"), "--check-only"], context_loader=blk.loader,
                  say=lines.append) == 0, lines


def check_index_reeval_refusals(tmp):
    for tag, kw, needle in (("nosrc", dict(source=False), "needs its FL run's LAUNCH files"),
                            ("tamper", dict(tamper=True), "recorded sha256"),
                            ("nobind", dict(launch_over={"warm_selection": None}), "not bound"),
                            ("arm", dict(launch_over={"arm_spec": {"arm": "FA", "cfg": {"seed": 2026}}}),
                             "LAUNCH arm_spec")):
        _b, lay, _w = _warm_reeval_layout(tmp / tag, **kw)
        rc, text = _index(lay, tmp / tag / "idx")
        assert rc == BI.REFUSED and needle in text, (tag, text)


def test_f1_index_warm_reeval_refusals(tmp_path):
    check_index_reeval_refusals(tmp_path)


@nc("REV-ANA-05 F1: an index that admits a warm --reeval dir without its FL run's LAUNCH binding must not pass")
def test_nc_f1_index_reeval_unbound(tmp_path, monkeypatch):
    monkeypatch.setattr(BI, "reeval_warm_binding", lambda d, rec, arm: (None, None))
    check_index_reeval_refusals(tmp_path)


# F4: an S_CAL run is excluded before the preview check (never refused)
def check_scal_excluded(tmp):
    b = anakit.build(tmp / "a")
    lay = layoutkit.build_layout(tmp / "lay", b)
    d = lay["R"] / "FL" / "S_CAL_s2026"
    shutil.copytree(lay["R"] / "FL" / "FA_s2026", d)
    p = d / "RUN_SUMMARY.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s.update(status="DONE_S_CAL", arm="S_CAL")
    s["arm_spec"].update(role="CALIBRATION", init_from=dict(anakit.INIT_FROM))
    anakit.jdump(p, s)
    anakit.jdump(d / "LAUNCH_0.json", {"label": "PRODUCTION", "arm": "S_CAL", "fl05": {"role": "CALIBRATION"}})
    rc, text = _index(lay, tmp / "idx")
    assert rc == 0, text
    idx = json.loads((tmp / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))
    assert any("S_CAL_s2026" in x["dir"] and x["reason"].startswith("S_CAL") for x in idx["excluded"])


def test_f4_s_cal_excluded_before_the_preview_check(tmp_path):
    check_scal_excluded(tmp_path)


@nc("REV-ANA-05 F4: the ANA-05 order (preview check before the S_CAL exclusion) must not pass")
def test_nc_f4_preview_first(tmp_path, monkeypatch):
    real = BI.Builder._launch

    def preview_first(d):                                                           # MUTANT: the ANA-05 order
        launch = real(d)
        sp = Path(d) / "RUN_SUMMARY.json"
        if sp.is_file():
            BI.check_warm_preview(Path(d), BI._open_json(sp), launch)
        return launch
    monkeypatch.setattr(BI.Builder, "_launch", staticmethod(preview_first))
    check_scal_excluded(tmp_path)
