"""DEV-FTPRE item 1 (ANA side) — FT_PRE's base check uses the PRE init record, not a run-id prefix (06K §2, 06L-C §C7;
FIX-DELTA-03 "still open": the FT_PRE base check looked only at the run-id prefix).

Report (tools/retention_report.py): an FT_PRE entry needs start_init = the theta_pre identity (launch_device --arm FT_PRE
LAUNCH start.init_from) and the PRETRAIN_FOR_FL_INIT.json file sha256, and they must equal the PRE entry's pre_init (the PRE
REEVAL_RECORD's binding to the same record). Index (tools/ana_block_index.py): FT_PRE dirs need a launch_device --arm
FT_PRE run whose LAUNCH start is kind PRE with init_from + init_record_sha256; the PRE REEVAL_RECORD's
inputs.pre_init_record becomes pre_init; a mismatch refuses. NC: the pre-DEV-FTPRE prefix-only admission.
FIX-DEV-FTPRE M4: a --trunc dir's TRUNC_PER_USER.npz is copied only when its sha256 equals the run's recorded
per_user.sha256 (STATUS.json; launch_trunc writes the same record into RESULT_ROWS.json, which the index never parses).
NC: the file copied without that check.
"""
from __future__ import annotations

import json
import shutil

import ana_block_index as BI
import anakit
import layoutkit
import retention_report as R
from test_ana02 import edit_index
from test_ana04 import full_block
from test_retention_report import check_refuses, conf, nc, result, run

M = "MAIN"


def _ftpre_entries(idx):
    return [e for e in idx["entries"] if e["arm"] == "FT_PRE"]


def check_init_record_rule(tmp_path):
    blk = full_block(tmp_path / "ok", calib=False)
    rc, text = run(blk, tmp_path / "ok" / "o")
    assert rc == 0, text
    assert conf(result(tmp_path / "ok" / "o"), "K9c")["label"] == "CONFIRMATORY"
    cases = [
        ("prefix_only", lambda idx: [e.pop("start_init") for e in _ftpre_entries(idx)], "no start_init"),
        ("main_trained", lambda idx: [e["start_init"].update(train_manifest="MAIN_TRAIN_5_100_131072")
                                      for e in _ftpre_entries(idx)], "FT_PRE's base must be θ_pre"),
        ("other_ckpt", lambda idx: [e["start_init"].update(checkpoint_sha256="d" * 64) for e in _ftpre_entries(idx)],
         "FT_PRE's base must be θ_pre of the PRE init record"),
        ("other_record", lambda idx: [e.update(start_init_record_sha256="e" * 64) for e in _ftpre_entries(idx)],
         "FT_PRE's base must be θ_pre of the PRE init record"),
        ("no_record_sha", lambda idx: [e.pop("start_init_record_sha256") for e in _ftpre_entries(idx)],
         "start_init_record_sha256"),
        ("pre_unbound", lambda idx: [e.pop("pre_init") for e in idx["entries"] if e["arm"] == "PRE"],
         "carries no pre_init"),
        ("no_pre", lambda idx: idx.update(entries=[e for e in idx["entries"] if e["arm"] != "PRE"]),
         "no PRE entry"),
        ("ft_c_with_init", lambda idx: [e.update(start_init=dict(anakit.INIT_FROM)) for e in idx["entries"]
                                        if e["arm"] == "FT_C"], "a device run from θ_pre is FT_PRE"),
        ("pre_init_elsewhere", lambda idx: [e.update(pre_init=dict(anakit.PRE_INIT)) for e in idx["entries"]
                                            if e["arm"] == "FA_WARM"], "pre_init"),
    ]
    for name, fn, needle in cases:
        b = full_block(tmp_path / name, calib=False)
        edit_index(b, fn)
        check_refuses(b, tmp_path / name, needle)


def test_ft_pre_base_uses_the_init_record(tmp_path):
    check_init_record_rule(tmp_path)


def test_ft_pre_rows_may_carry_method_ft_c(tmp_path):
    blk = full_block(tmp_path / "a", calib=False)
    rows_p = blk.root / M / "FT_PRE_s2026" / "RESULT_ROWS.json"
    obj = json.loads(rows_p.read_text(encoding="utf-8"))
    assert {r["method"] for r in obj["rows"]} == {"FT_PRE"}                         # the launcher's label
    obj["rows"] = [dict(r, method="FT_C") for r in obj["rows"]]
    rows_p.write_text(json.dumps(obj), encoding="utf-8")
    rc, text = run(blk, tmp_path / "a" / "o")
    assert rc == 0, text


def test_ft_pre_problem_known_answers():
    ok = dict(anakit.INIT_FROM)
    assert R.ft_pre_base_problem(ok, anakit.PRE_RUN, anakit.PRE_INIT_RECORD_SHA) is None
    assert R.ft_pre_base_problem(None, anakit.PRE_RUN, anakit.PRE_INIT_RECORD_SHA)          # prefix alone: refused
    assert R.ft_pre_base_problem(ok, "PRETRAIN_FOR_FL_OTHER_s2026", anakit.PRE_INIT_RECORD_SHA)
    assert R.ft_pre_base_problem(dict(ok, run_label="PRODUCTION"), anakit.PRE_RUN, anakit.PRE_INIT_RECORD_SHA)
    ftp = [("FT_PRE|2026", ok, anakit.PRE_INIT_RECORD_SHA)]
    assert R.ft_pre_vs_pre_problem(ftp, [("PRE|2026", dict(anakit.PRE_INIT))]) is None
    assert R.ft_pre_vs_pre_problem([], []) is None
    assert "disagree" in R.ft_pre_vs_pre_problem(ftp, [("a", dict(anakit.PRE_INIT)),
                                                       ("b", dict(anakit.PRE_INIT, checkpoint_sha256="d" * 64))])
    assert R.ft_pre_vs_pre_problem(ftp, [("PRE|2026", dict(anakit.PRE_INIT, run_id="PRETRAIN_FOR_FL_X_s2026"))])


# ------------------------------------------------------------------------------------------------ the index builder
def _layout(tmp_path, *, pre_init=True, ftpre_init=True, summary_arm="FT_PRE", ckpt="a" * 64):
    b = anakit.build(tmp_path / "a")
    lay = layoutkit.build_layout(tmp_path / "lay", b)
    R_ = lay["R"]
    src = b.root / M / "C_SMALL_s2026"
    rec = {"format": "PPSI_V4_REEVAL_RECORD_1", "arm_label": "PRE", "seed": 2026, "role": M,
           "selection_rule": "PRACTICAL_BEST", "checkpoint_id": "PRE_s2026:endpoint_6.0",
           "checkpoint_sha256": anakit.INIT_FROM["checkpoint_sha256"], "source_run_id": anakit.PRE_RUN,
           "device_class": anakit.DEV, "device_pin_sha256": anakit.PIN, "eval_manifest_id": anakit.ROLES[M][0],
           "central_source": layoutkit.central_source_record("PRE")}
    if pre_init:
        rec["inputs"] = {"pre_init_record": {"path": "/srv/PRE/PRETRAIN_FOR_FL_INIT.json",
                                             "file_sha256": anakit.PRE_INIT_RECORD_SHA, "fl_init_sha256": ckpt}}
    layoutkit.write_reeval_dir(R_ / "REEVAL_A40" / "PRE_s2026_PB", src / "RESULT_ROWS.json",
                               src / "ranks_PRACTICAL_BEST.npy", rec, M)
    for seed in (2026, 2027):
        d = R_ / "DEV" / f"FT_PRE_s{seed}"
        shutil.copytree(R_ / "DEV" / f"FT_C_s{seed}", d)
        summ = json.loads((d / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
        anakit.jdump(d / "RUN_SUMMARY.json", dict(summ, arm=summary_arm))
        start = {"run_id": anakit.PRE_RUN, "kind": "PRE", "init_record_sha256": anakit.PRE_INIT_RECORD_SHA}
        if ftpre_init:
            start["init_from"] = dict(anakit.INIT_FROM)
        anakit.jdump(d / "LAUNCH_0.json", {"label": "PRODUCTION", "arm": summary_arm, "start": start})
    return lay


def _index(lay, out):
    lines = []
    rc = BI.main(lay["args"] + ["--out", str(out)], say=lines.append)
    return rc, "\n".join(lines)


def test_index_binds_ft_pre_to_the_pre_init_record(tmp_path):
    lay = _layout(tmp_path / "ok")
    rc, text = _index(lay, tmp_path / "ok" / "idx")
    assert rc == 0, text
    idx = json.loads((tmp_path / "ok" / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))
    pre = next(e for e in idx["entries"] if e["arm"] == "PRE")
    assert pre["pre_init"] == anakit.PRE_INIT
    ftp = _ftpre_entries(idx)
    assert len(ftp) == 2 and all(e["start_init"] == anakit.INIT_FROM and e["start_init_record_sha256"] ==
                                 anakit.PRE_INIT_RECORD_SHA for e in ftp)
    for name, kw, needle in (("noinit", dict(ftpre_init=False), "no start_init"),
                             ("ftc", dict(summary_arm="FT_C"), "launch_device --arm FT_PRE run"),
                             ("flinit", dict(ckpt="f" * 64), "pre_init_record fl_init_sha256"),
                             ("unbound", dict(pre_init=False), "carries no pre_init")):
        lay = _layout(tmp_path / name, **kw)
        rc, text = _index(lay, tmp_path / name / "idx")
        assert rc == BI.REFUSED and needle in text, (name, text)
    lay = _layout(tmp_path / "other")                                  # FT_PRE from another theta_pre checkpoint
    for seed in (2026, 2027):
        p = lay["R"] / "DEV" / f"FT_PRE_s{seed}" / "LAUNCH_0.json"
        la = json.loads(p.read_text(encoding="utf-8"))
        la["start"]["init_from"]["checkpoint_sha256"] = "d" * 64
        anakit.jdump(p, la)
    rc, text = _index(lay, tmp_path / "other" / "idx")
    assert rc == BI.REFUSED and "FT_PRE's base must be θ_pre of the PRE init record" in text, text


# ------------------------------------------------------------------------------------------------ TRUNC per-user (M4)
TRUNC_DIR = ("TRUNC", "SA_C_SMALL_s2026")


def _trunc_layout(tmp_path, *, data=b"per-user-evidence", recorded="file", with_file=True):
    """The layoutkit Study A --trunc dir + a TRUNC_PER_USER.npz (never parsed by the index: any bytes) and a STATUS.json
    per_user record: recorded = "file" (the file's sha256), None (no record), or an explicit sha256."""
    b = anakit.build(tmp_path / "a")
    lay = layoutkit.build_layout(tmp_path / "lay", b)
    d = lay["R"].joinpath(*TRUNC_DIR)
    f = d / BI.TRUNC_PER_USER_NAME
    f.write_bytes(data)
    st = {"status": "DONE"}
    if recorded is not None:
        st["per_user"] = {"file": BI.TRUNC_PER_USER_NAME, "sha256": BI.sha_file(f) if recorded == "file" else recorded}
    anakit.jdump(d / "STATUS.json", st)
    if not with_file:
        f.unlink()
    return lay, d


def test_index_copies_trunc_per_user_only_with_its_recorded_sha256(tmp_path):
    lay, d = _trunc_layout(tmp_path / "ok")
    rc, text = _index(lay, tmp_path / "ok" / "idx")
    assert rc == 0, text
    item = json.loads((tmp_path / "ok" / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))["records"]["TRUNC_ROWS"][0]
    assert (tmp_path / "ok" / "idx" / item["per_user"]).read_bytes() == (d / BI.TRUNC_PER_USER_NAME).read_bytes()
    lay, _ = _trunc_layout(tmp_path / "old", recorded=None, with_file=False)          # a pre-DEV-FTPRE run: neither
    rc, text = _index(lay, tmp_path / "old" / "idx")
    assert rc == 0, text
    assert "per_user" not in json.loads((tmp_path / "old" / "idx" / R.INDEX_NAME).read_text(encoding="utf-8"))[
        "records"]["TRUNC_ROWS"][0]
    for name, kw, needle in (("stale", dict(recorded="e" * 64), "differs from the recorded per_user.sha256"),
                             ("norecord", dict(recorded=None), "records no per_user.sha256"),
                             ("nofile", dict(with_file=False), "is absent")):
        lay, _ = _trunc_layout(tmp_path / name, **kw)
        rc, text = _index(lay, tmp_path / name / "idx")
        assert rc == BI.REFUSED and needle in text, (name, text)


# ------------------------------------------------------------------------------------------------ negative control
@nc("the pre-DEV-FTPRE FT_PRE base check (LAUNCH start run-id prefix only) admits a base the init record refuses")
def test_nc_prefix_only_base_check(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "ft_pre_base_problem",                                    # MUTANT: prefix only
                        lambda init, sr, sirs: None if str(sr or "").startswith(R.FT_PRE_BASE_RUN_PREFIX) else "x")
    monkeypatch.setattr(R, "ft_pre_vs_pre_problem", lambda *a: None)
    check_init_record_rule(tmp_path)


@nc("the index copies a replaced (stale) TRUNC_PER_USER.npz without checking the recorded sha256")
def test_nc_trunc_per_user_unchecked(tmp_path, monkeypatch):
    monkeypatch.setattr(BI, "trunc_per_user_sha",                                    # MUTANT: copy if present
                        lambda d, st: BI.sha_file(d / BI.TRUNC_PER_USER_NAME)
                        if (d / BI.TRUNC_PER_USER_NAME).is_file() else None)
    lay, _ = _trunc_layout(tmp_path, recorded="e" * 64)
    rc, text = _index(lay, tmp_path / "idx")
    assert rc == BI.REFUSED and "differs from the recorded per_user.sha256" in text, text
