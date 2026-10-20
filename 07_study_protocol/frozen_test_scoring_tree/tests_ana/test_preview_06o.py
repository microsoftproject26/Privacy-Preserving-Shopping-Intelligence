"""FIX-REEVAL-06O (PREG1_AMENDMENT_06O O1-O2) - the ANA tools never let a C_FULL / PRE preview row on UNSEEN_VAL into a
block (synthetic blocks only):
  * tools/retention_report.py refuses a C_FULL / PRE entry on UNSEEN_VAL with PREVIEW_REFERENCE_LABEL, and any row
    labelled preview_reference (check_arm_identity) under any entry;
  * tools/ana_block_index.py, on the full synthetic runs_ds3 layout (layoutkit), lists reeval-written
    <C_FULL|PRE>_UNSEEN_VAL_s<S>_<PB|CB> dirs as excluded: the index entries are exactly those of the layout without them.
Negative controls (test_nc_*, strict xfail on AssertionError).
"""
from __future__ import annotations

import json

import anakit
import ana_block_index as BI
import layoutkit
import retention_report as R
from test_retention_report import check_refuses, nc

PB, CB = "PRACTICAL_BEST", "CONTROLLED_BUDGET"
UV = "UNSEEN_VAL"
NEEDLE = "PREVIEW_REFERENCE NON_MAIN_UNSEEN_VAL"


def check_report_refuses_preview_entries(tmp_path):
    for arm, rule in (("C_FULL", PB), ("PRE", CB)):
        blk = anakit.build(tmp_path / arm)
        anakit.add_role(blk, UV)
        anakit.add_plain(blk, "C_SMALL", (2026,), (PB,), role=UV)
        anakit.add_plain(blk, arm, (2026,), (rule,), role=UV)
        check_refuses(blk, tmp_path / arm, NEEDLE)


def test_report_refuses_c_full_and_pre_on_unseen_val(tmp_path):
    check_report_refuses_preview_entries(tmp_path)


def _entry(arm="C_SMALL", role="MAIN"):
    return R.Entry(arm, 2026, PB, role, "rows.json", None, {}, None, None, False, {})


def check_row_rule():
    row = {"train_manifest_id": "MAIN_TRAIN_5_100_131072", "comparison_status": "DESCRIPTIVE",
           "preview_reference": "NON_MAIN_UNSEEN_VAL"}
    refused = None
    try:
        R.check_arm_identity(_entry(), row)
    except R.AnaRefused as e:
        refused = str(e)
    assert refused is not None and NEEDLE in refused and "ARM_IDENTITY" in refused, refused
    row.pop("preview_reference")
    R.check_arm_identity(_entry(), row)                                  # the same row without the label passes


def test_report_refuses_a_preview_row_under_any_entry():
    check_row_rule()
    assert R.preview_row_problem({}) is None and R.PREVIEW_REFERENCE_ARMS == ("C_FULL", "PRE")


def _add_previews(lay, blk):
    """reeval-written preview dirs (the real write_outputs, the 06O record labels) beside the layout's REEVAL dirs."""
    Rr, idx = lay["R"], blk.index()
    src = next(e for e in idx["entries"] if e["arm"] == "C_FULL" and e["role"] == "MAIN")
    names = []
    for arm, seed, rule, tag in (("C_FULL", 2026, PB, "PB"), ("C_FULL", 2027, CB, "CB"), ("PRE", 2026, CB, "CB")):
        name = f"{arm}_UNSEEN_VAL_s{seed}_{tag}"
        cs = layoutkit.central_source_record(arm)
        if seed == 2027:
            cs.update(seed=2027, seed_status="DESCRIPTIVE")
        layoutkit.write_reeval_dir(Rr / "REEVAL_A40" / name, blk.root / src["rows"], blk.root / src["ranks"], {
            "format": "PPSI_V4_REEVAL_RECORD_1", "arm_label": arm, "seed": seed, "role": UV, "selection_rule": rule,
            "checkpoint_id": f"{arm}_s{seed}:{tag}", "checkpoint_sha256": layoutkit.ckpt_sha(arm, seed, rule),
            "device_class": anakit.DEV, "device_pin_sha256": anakit.PIN,
            "eval_manifest_id": R.ROLE_MAP["VALIDATION"][UV][0], "central_source": cs,
            "preview_reference": "NON_MAIN_UNSEEN_VAL", "preview_reference_rule": "06O"}, UV)
        names.append(name)
    return names


def check_index_on_layout(tmp_path):
    blk = anakit.build(tmp_path / "blk")
    lay = layoutkit.build_layout(tmp_path / "lay", blk)
    lines = []
    assert BI.main(lay["args"] + ["--out", str(tmp_path / "idx0")], say=lines.append) == 0, lines
    names = _add_previews(lay, blk)
    lines = []
    assert BI.main(lay["args"] + ["--out", str(tmp_path / "idx1")], say=lines.append) == 0, lines
    i0, i1 = (json.loads((tmp_path / d / R.INDEX_NAME).read_text(encoding="utf-8")) for d in ("idx0", "idx1"))
    assert i1["entries"] == i0["entries"] and i1["roles"] == i0["roles"]
    new = [x for x in i1["excluded"] if x not in i0["excluded"]]
    assert sorted(x["dir"].replace("\\", "/").rsplit("/", 1)[-1] for x in new) == sorted(names), new
    assert {x["reason"] for x in new} == {R.PREVIEW_REFERENCE_LABEL}


def test_index_excludes_preview_dirs_on_the_full_layout(tmp_path):
    check_index_on_layout(tmp_path)


# ------------------------------------------------------------------------------------------------ negative controls
@nc("a report without the 06O entry rule refuses C_FULL / PRE on UNSEEN_VAL only with the generic no-FT text")
def test_nc_report_preview_label_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "PREVIEW_REFERENCE_ARMS", ())                               # MUTANT
    check_report_refuses_preview_entries(tmp_path)


@nc("a report without the 06O row rule admits a preview_reference row under an ordinary entry")
def test_nc_report_preview_row_admitted(monkeypatch):
    monkeypatch.setattr(R, "preview_row_problem", lambda r: None)                      # MUTANT
    check_row_rule()


@nc("an index without the 06O exclusion cannot index REEVAL_A40 once the preview dirs are there")
def test_nc_index_exclusion_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(BI.RR, "PREVIEW_REFERENCE_ARMS", ())                           # MUTANT
    check_index_on_layout(tmp_path)
